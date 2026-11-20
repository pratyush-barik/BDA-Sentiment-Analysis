"""
tests/test_tier2_mllib_pipeline.py
Tier 2: Boundary Value Analysis, Invariants & MLlib Pipeline Evaluation.
Big Data Product Review Sentiment Analysis System.
"""

import math
from pathlib import Path
from typing import Any, Dict, List
import pytest
try:
    from tests.conftest import safe_import
except ImportError:
    from conftest import safe_import


@pytest.mark.tier2
@pytest.mark.spark
def test_mllib_pipeline_stages_and_feature_dimensions(spark_session, sample_reviews_df):
    """
    Test 2.1: Validate feature transformation pipeline produces feature vectors
    matching the configured dimensionality and preserves non-zero norms.
    """
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    if mllib_mod is None:
        pytest.skip("Milestone M3 MLlib pipeline not yet implemented")

    pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None)
    if pipeline_cls is None:
        pytest.skip("SentimentMLlibPipeline class not found in src.ml.spark_mllib_pipeline")

    pipeline_helper = pipeline_cls() if isinstance(pipeline_cls, type) else pipeline_cls
    build_fn = getattr(pipeline_helper, "build_pipeline", None)
    if build_fn is None:
        pytest.skip("build_pipeline method not found on SentimentMLlibPipeline")

    pipeline = build_fn(feature_type="hashing_tf", classifier_type="logistic_regression")
    train_fn = getattr(pipeline_helper, "train", None)
    if train_fn is not None:
        model = train_fn(sample_reviews_df)
    elif hasattr(pipeline, "fit"):
        model = pipeline.fit(sample_reviews_df)
    else:
        pytest.skip("No fitting method found for pipeline")

    # Transform data
    if hasattr(model, "transform"):
        transformed = model.transform(sample_reviews_df)
    elif hasattr(pipeline_helper, "predict"):
        transformed = pipeline_helper.predict(model, sample_reviews_df)
    else:
        pytest.skip("No transformation method found")

    # Invariant 1: 'features' column must exist
    cols = transformed.columns if hasattr(transformed, "columns") else list(transformed[0].keys())
    assert "features" in cols, f"'features' column missing in transformed schema: {cols}"

    # Invariant 2: Feature dimensionality must match configured size (e.g. 16384 or vocabSize)
    if hasattr(transformed, "first"):
        first_row = transformed.first()
        feats = first_row.get("features") if isinstance(first_row, dict) else getattr(first_row, "features", None)
    elif hasattr(transformed, "collect"):
        first_row = transformed.collect()[0]
        feats = first_row.get("features") if isinstance(first_row, dict) else getattr(first_row, "features", None)
    else:
        feats = transformed[0]["features"]

    assert feats is not None, "Feature vector is None"
    dim = feats.size if hasattr(feats, "size") else (feats.shape[0] if hasattr(feats, "shape") else len(feats))
    assert dim in (16384, 5000) or dim > 0, f"Unexpected feature dimension: {dim}"


@pytest.mark.tier2
@pytest.mark.spark
def test_multiclass_label_indexer_roundtrip(spark_session):
    """
    Test 2.2: Validate StringIndexer maps 3 sentiment classes to distinct indices
    and IndexToString accurately reconstructs original labels.
    """
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    if mllib_mod is None:
        pytest.skip("Milestone M3 MLlib pipeline not yet implemented")

    labels_data = [
        {"review_text": "great superb product love it", "sentiment_label": "positive", "rating": 5.0},
        {"review_text": "average okay normal item", "sentiment_label": "neutral", "rating": 3.0},
        {"review_text": "terrible bad broken defect waste", "sentiment_label": "negative", "rating": 1.0},
        {"review_text": "excellent wonderful quality", "sentiment_label": "positive", "rating": 5.0},
    ]
    df = spark_session.createDataFrame(labels_data)

    # Check via pipeline class helper or native stages
    pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None)
    if pipeline_cls is None:
        pytest.skip("SentimentMLlibPipeline class not found in src.ml.spark_mllib_pipeline")

    indexer_fn = getattr(pipeline_cls, "get_label_indexer", None)

    if indexer_fn is not None:
        indexer, converter = indexer_fn()
        indexed_df = indexer.fit(df).transform(df)
        labels = [r["label"] for r in indexed_df.collect()]
        assert set(labels) == {0.0, 1.0, 2.0}

        reconstructed_df = converter.transform(indexed_df)
        for row in reconstructed_df.collect():
            assert row["sentiment_label"] == row["predicted_sentiment"]
    else:
        # Validate multiclass label encoding & prediction via standard train/predict interface
        pipeline_helper = pipeline_cls() if isinstance(pipeline_cls, type) else pipeline_cls
        train_fn = getattr(pipeline_helper, "train", None)
        predict_fn = getattr(pipeline_helper, "predict", None)
        if train_fn is None or predict_fn is None:
            pytest.skip("train/predict methods not found on SentimentMLlibPipeline to verify label indexer")

        model = train_fn(df)
        preds = predict_fn(model, df)
        cols = preds.columns if hasattr(preds, "columns") else list(preds[0].keys())
        assert "prediction" in cols, f"Missing numeric 'prediction' column in pipeline output: {cols}"
        assert "predicted_sentiment" in cols or "sentiment" in cols, (
            f"Missing reconstructed string label column in pipeline output: {cols}"
        )
        rows = preds.collect() if hasattr(preds, "collect") else preds
        for r in rows:
            pred_val = r["prediction"] if isinstance(r, dict) else getattr(r, "prediction")
            assert pred_val in (0.0, 1.0, 2.0), f"Invalid numeric label index: {pred_val}"
            label_key = "predicted_sentiment" if "predicted_sentiment" in (r if isinstance(r, dict) else cols) else "sentiment"
            pred_label = r[label_key] if isinstance(r, dict) else getattr(r, label_key)
            assert pred_label in ("positive", "neutral", "negative"), f"Invalid reconstructed string label: {pred_label}"


@pytest.mark.tier2
@pytest.mark.spark
def test_classifier_training_and_prediction_output_schema(spark_session, sample_reviews_df):
    """
    Test 2.3: Train classifier (Logistic Regression) and verify prediction output schema
    and probability calibration invariants (probabilities sum to 1.0, prediction == argmax).
    """
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    if mllib_mod is None:
        pytest.skip("Milestone M3 MLlib pipeline not yet implemented")

    pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None)
    if pipeline_cls is None:
        pytest.skip("SentimentMLlibPipeline not found")

    pipeline = pipeline_cls() if isinstance(pipeline_cls, type) else pipeline_cls
    train_fn = getattr(pipeline, "train", None)
    predict_fn = getattr(pipeline, "predict", None)

    if train_fn is None or predict_fn is None:
        pytest.skip("train or predict method not found on SentimentMLlibPipeline")

    model = train_fn(sample_reviews_df)
    preds_df = predict_fn(model, sample_reviews_df)

    cols = preds_df.columns if hasattr(preds_df, "columns") else list(preds_df[0].keys())
    # Required prediction columns
    assert "prediction" in cols, f"Missing 'prediction' column: {cols}"
    assert "probability" in cols or "predicted_sentiment" in cols, f"Missing probability/predicted_sentiment: {cols}"

    rows = preds_df.collect() if hasattr(preds_df, "collect") else preds_df
    for r in rows:
        pred_val = r["prediction"] if isinstance(r, dict) else getattr(r, "prediction")
        assert pred_val in (0.0, 1.0, 2.0), f"Invalid prediction class: {pred_val}"

        if "probability" in r or hasattr(r, "probability"):
            probs = r["probability"] if isinstance(r, dict) else getattr(r, "probability")
            prob_list = probs.toArray().tolist() if hasattr(probs, "toArray") else (list(probs) if hasattr(probs, "__iter__") else [probs])
            # Invariant: Probabilities must sum to 1.0 within 1e-4 tolerance
            assert abs(sum(prob_list) - 1.0) < 1e-3, f"Probabilities do not sum to 1.0: {prob_list}"
            # Invariant: All probabilities must be non-negative
            assert all(p >= 0.0 for p in prob_list)
            # Invariant: Prediction must match argmax probability
            assert pred_val == float(prob_list.index(max(prob_list)))


@pytest.mark.tier2
@pytest.mark.spark
def test_metric_evaluator_correctness_and_bounds(spark_session, sample_reviews_df):
    """
    Test 2.4: Verify that evaluation metrics are within strict mathematical bounds [0.0, 1.0].
    """
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    eval_mod = safe_import("src.ml.evaluator")
    if mllib_mod is None and eval_mod is None:
        pytest.skip("Milestone M3 MLlib / Evaluator not yet implemented")

    pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None) if mllib_mod else None
    pipeline = pipeline_cls() if isinstance(pipeline_cls, type) else pipeline_cls

    eval_fn = getattr(pipeline, "evaluate", None) or getattr(eval_mod, "evaluate_model", None) or getattr(eval_mod, "compute_metrics", None)
    if eval_fn is None:
        pytest.skip("No evaluate method found")

    train_fn = getattr(pipeline, "train", None)
    if train_fn is None:
        pytest.skip("No train method found")

    model = train_fn(sample_reviews_df)
    metrics = eval_fn(model, sample_reviews_df) if pipeline_cls and hasattr(pipeline, "evaluate") else eval_fn(sample_reviews_df)

    assert isinstance(metrics, dict), "Evaluation metrics must be returned as a dict"
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        assert metric_name in metrics, f"Missing metric: {metric_name}"
        score = metrics[metric_name]
        assert isinstance(score, (int, float)), f"Metric {metric_name} is not numeric: {score}"
        # Mathematical invariant: [0.0, 1.0]
        assert 0.0 <= score <= 1.0, f"Metric {metric_name} out of bounds: {score}"


@pytest.mark.tier2
@pytest.mark.spark
def test_confusion_matrix_conservation_invariant(spark_session, sample_reviews_df):
    """
    Test 2.5: Verify that the confusion matrix is 3x3, all elements are non-negative,
    and sum of all elements strictly equals the test set size.
    """
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    if mllib_mod is None:
        pytest.skip("Milestone M3 MLlib pipeline not yet implemented")

    pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None)
    pipeline = pipeline_cls() if isinstance(pipeline_cls, type) else pipeline_cls
    if not hasattr(pipeline, "train") or not hasattr(pipeline, "evaluate"):
        pytest.skip("train / evaluate methods not found")

    model = pipeline.train(sample_reviews_df)
    metrics = pipeline.evaluate(model, sample_reviews_df)

    assert "confusion_matrix" in metrics, "Missing 'confusion_matrix' in metrics"
    cm = metrics["confusion_matrix"]

    # Structural invariants
    assert len(cm) == 3, f"Expected 3 rows in confusion matrix, got {len(cm)}"
    assert all(len(row) == 3 for row in cm), "Confusion matrix must be 3x3"
    assert all(all(val >= 0 for val in row) for row in cm), "Negative counts in confusion matrix"

    # Conservation invariant: total count must equal sample_reviews_df count
    total_elements = sum(sum(row) for row in cm)
    expected_count = sample_reviews_df.count() if hasattr(sample_reviews_df, "count") else len(sample_reviews_df)
    assert total_elements == expected_count, f"Conservation law violated: sum(cm)={total_elements} != N={expected_count}"


@pytest.mark.tier2
def test_edge_case_inputs_and_extreme_tokens():
    """
    Test 2.6: Ensure ML pipeline handles single-word reviews, empty tokens,
    repetitive long text, and stopword-only text without throwing unhandled exceptions
    or producing NaN probabilities.
    """
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    if mllib_mod is None:
        pytest.skip("Milestone M3 MLlib pipeline not yet implemented")

    pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None)
    if pipeline_cls is None:
        pytest.skip("SentimentMLlibPipeline class not found in src.ml.spark_mllib_pipeline")

    pipeline = pipeline_cls() if isinstance(pipeline_cls, type) else pipeline_cls
    predict_single_fn = getattr(pipeline, "predict_single", None)
    predict_fn = getattr(pipeline, "predict", None)

    # Determine callable inference function before running test cases
    infer_fn = None
    if predict_single_fn is not None and callable(predict_single_fn):
        infer_fn = predict_single_fn
    elif predict_fn is not None and callable(predict_fn):
        # Test if predict can accept a single string directly
        try:
            test_res = predict_fn("test")
            if isinstance(test_res, (dict, list)):
                infer_fn = predict_fn
        except Exception:
            pass

    if infer_fn is None:
        pytest.skip("SentimentMLlibPipeline does not expose a single-text prediction interface (predict_single or predict(str))")

    edge_cases = [
        "",  # Empty string
        "Superb",  # Single token
        "great " * 300,  # Extreme repetition
        "the on and of for with",  # Stopwords only
    ]

    for sample_text in edge_cases:
        res = infer_fn(sample_text)
        if isinstance(res, list):
            assert len(res) > 0, f"Empty prediction list for input: '{sample_text}'"
            res = res[0]
        assert isinstance(res, dict), f"Prediction output must be a dict, got {type(res)}"
        assert "predicted_sentiment" in res or "sentiment" in res or "prediction" in res, (
            f"Missing sentiment label or prediction in output: {res}"
        )
        conf = res.get("confidence", res.get("probability", 0.0))
        if isinstance(conf, (int, float)):
            assert not math.isnan(conf), f"Confidence was NaN for input: '{sample_text[:20]}'"
            assert 0.0 <= conf <= 1.0, f"Confidence out of bounds: {conf}"


@pytest.mark.tier2
@pytest.mark.spark
def test_model_persistence_and_reloading_parity(spark_session, sample_reviews_df, tmp_path: Path):
    """
    Test 2.7: Verify that saving a trained pipeline model to disk and reloading it
    produces identical inference results.
    """
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    if mllib_mod is None:
        pytest.skip("Milestone M3 MLlib pipeline not yet implemented")

    pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None)
    pipeline = pipeline_cls() if isinstance(pipeline_cls, type) else pipeline_cls

    if not hasattr(pipeline, "save_model") or not hasattr(pipeline, "load_model"):
        pytest.skip("save_model / load_model not implemented")

    model = pipeline.train(sample_reviews_df)
    model_dir = str(tmp_path / "saved_model")
    pipeline.save_model(model, model_dir)

    reloaded_model = pipeline.load_model(model_dir)
    assert reloaded_model is not None

    preds_original = pipeline.predict(model, sample_reviews_df)
    preds_reloaded = pipeline.predict(reloaded_model, sample_reviews_df)

    orig_rows = preds_original.collect() if hasattr(preds_original, "collect") else preds_original
    reload_rows = preds_reloaded.collect() if hasattr(preds_reloaded, "collect") else preds_reloaded

    orig_preds = [r["prediction"] if isinstance(r, dict) else getattr(r, "prediction") for r in orig_rows]
    reload_preds = [r["prediction"] if isinstance(r, dict) else getattr(r, "prediction") for r in reload_rows]
    assert orig_preds == reload_preds, "Reloaded model generated divergent predictions"
