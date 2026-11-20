"""
tests/test_tier4_e2e.py
Tier 4: Real-World Application Scenarios (Full Pipeline CLI, Visualizations, Flask Web Dashboard).
Big Data Product Review Sentiment Analysis System.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
import pytest
try:
    from tests.conftest import safe_import
except ImportError:
    from conftest import safe_import


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.mark.tier4
@pytest.mark.slow
def test_full_batch_pipeline_orchestration(tmp_path: Path):
    """
    Test 4.1: Execute the master runner CLI (python main.py --mode full --sample-size 100)
    and verify end-to-end completion with exit code 0.
    """
    main_script = Path(__file__).resolve().parent.parent / "main.py"
    if not main_script.exists():
        pytest.skip("Milestone M5 master pipeline runner (main.py) not yet implemented")

    project_root = main_script.parent
    cmd = [sys.executable, str(main_script), "--mode", "full", "--sample-size", "100"]
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, f"Master CLI execution failed with stderr:\n{proc.stderr}"
    assert "Done" in proc.stdout or "success" in proc.stdout.lower() or proc.returncode == 0


@pytest.mark.tier4
def test_visual_artifacts_generation_completeness():
    """
    Test 4.2: Verify that all 8 publication-grade static visualization files exist
    in reports/visualizations/, exceed minimum file size (5KB), and contain valid PNG magic headers.
    """
    viz_dir = Path(__file__).resolve().parent.parent / "reports" / "visualizations"
    plot_gen_mod = safe_import("src.visualization.plot_generator")

    if not viz_dir.exists() or len(list(viz_dir.glob("*.png"))) == 0:
        if plot_gen_mod is None:
            pytest.skip("Milestone M4 plot generator and visual artifacts not yet generated")
        else:
            pytest.skip("Visualizations not generated yet; run 'python main.py --mode visualize' first")

    png_files = sorted(list(viz_dir.glob("*.png")))
    assert len(png_files) >= 8, f"Expected at least 8 visual plot files, found {len(png_files)}"

    # Validate each plot file integrity
    for plot in png_files:
        assert plot.stat().st_size > 5000, f"Plot {plot.name} appears truncated (size: {plot.stat().st_size} bytes)"
        with open(plot, "rb") as f:
            header = f.read(8)
            assert header == PNG_MAGIC, f"File {plot.name} does not have valid PNG header magic"


@pytest.mark.tier4
def test_exported_metrics_json_schema_and_validity():
    """
    Test 4.3: Parse reports/metrics.json and validate that benchmark metrics
    for classifiers are present, validly formatted, and bounded in [0.0, 1.0].
    """
    metrics_path = Path(__file__).resolve().parent.parent / "reports" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("Milestone M5 reports/metrics.json not yet generated; run full pipeline first")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "timestamp" in data or "date" in data, "Missing execution timestamp in metrics.json"
    assert "models" in data or "classifiers" in data or any(k in data for k in ["logistic_regression", "naive_bayes", "random_forest"])

    models_dict = data.get("models", data.get("classifiers", data))

    # Check metrics for each available classifier
    for model_name, metrics in models_dict.items():
        if not isinstance(metrics, dict):
            continue
        for metric_key in ["accuracy", "precision", "recall", "f1"]:
            if metric_key in metrics:
                score = metrics[metric_key]
                assert isinstance(score, (int, float)), f"Metric {metric_key} for {model_name} is not numeric"
                assert 0.0 <= score <= 1.0, f"Metric {metric_key} for {model_name} out of bounds: {score}"

        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            assert len(cm) == 3, f"Expected 3x3 confusion matrix for {model_name}, got {len(cm)} rows"
            assert all(len(row) == 3 for row in cm)


@pytest.mark.tier4
@pytest.mark.web
def test_flask_dashboard_predict_endpoint(flask_client):
    """
    Test 4.4: Validate the Flask Web Dashboard /api/predict endpoint for
    positive, negative, and malformed inputs.
    """
    # Case A: Positive review text
    pos_payload = {"text": "Exceptional sound quality, superb battery life, highly recommend!"}
    res_pos = flask_client.post("/api/predict", json=pos_payload)
    assert res_pos.status_code == 200
    data_pos = res_pos.get_json()
    sentiment_pos = data_pos.get("predicted_sentiment", data_pos.get("sentiment", "")).lower()
    assert sentiment_pos == "positive"
    assert data_pos.get("confidence", 0.0) >= 0.50

    # Case B: Negative review text
    neg_payload = {"text": "Defective item, broke after one hour, complete waste of money."}
    res_neg = flask_client.post("/api/predict", json=neg_payload)
    assert res_neg.status_code == 200
    data_neg = res_neg.get_json()
    sentiment_neg = data_neg.get("predicted_sentiment", data_neg.get("sentiment", "")).lower()
    assert sentiment_neg == "negative"
    assert data_neg.get("confidence", 0.0) >= 0.50

    # Case C: Boundary / Empty text (must handle gracefully with 400 or valid response, never 500)
    res_empty = flask_client.post("/api/predict", json={"text": ""})
    assert res_empty.status_code in [200, 400], f"Empty input returned unhandled error: {res_empty.status_code}"


@pytest.mark.tier4
@pytest.mark.web
def test_flask_dashboard_kpis_endpoint(flask_client):
    """
    Test 4.5: Validate the Flask Web Dashboard /api/kpis endpoint returns
    valid analytical counts and percentage sums.
    """
    res = flask_client.get("/api/kpis")
    assert res.status_code == 200
    kpis = res.get_json()

    assert "total_reviews" in kpis and kpis["total_reviews"] >= 0
    assert "average_rating" in kpis and 1.0 <= kpis["average_rating"] <= 5.0
    assert "positive_percentage" in kpis or "positive_pct" in kpis
    assert "negative_percentage" in kpis or "negative_pct" in kpis

    pos_pct = kpis.get("positive_percentage", kpis.get("positive_pct", 0.0))
    neu_pct = kpis.get("neutral_percentage", kpis.get("neutral_pct", 0.0))
    neg_pct = kpis.get("negative_percentage", kpis.get("negative_pct", 0.0))

    # Mathematical Invariant: Percentages must sum to 100% within +/- 1.0% rounding tolerance
    total_pct = pos_pct + neu_pct + neg_pct
    assert abs(total_pct - 100.0) <= 1.0, f"KPI percentages do not sum to 100%: {total_pct}"


@pytest.mark.tier4
@pytest.mark.web
def test_flask_dashboard_ui_and_reviews_api(flask_client):
    """
    Test 4.6: Validate HTML dashboard rendering at / and review retrieval/filtering API at /api/reviews.
    """
    # 1. UI Root Route
    res_ui = flask_client.get("/")
    assert res_ui.status_code == 200
    html_content = res_ui.data.decode("utf-8")
    assert "Product Review Sentiment" in html_content or "Dashboard" in html_content

    # 2. Filterable Reviews REST API
    res_reviews = flask_client.get("/api/reviews?limit=5")
    assert res_reviews.status_code == 200
    reviews_data = res_reviews.get_json()
    assert isinstance(reviews_data, list)
    assert len(reviews_data) <= 5

    if len(reviews_data) > 0:
        item = reviews_data[0]
        assert "review_id" in item
        assert "product_name" in item
        assert "sentiment_label" in item or "sentiment" in item
