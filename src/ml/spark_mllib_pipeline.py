"""
Spark MLlib Sentiment Classification Pipeline.
Module: src.ml.spark_mllib_pipeline (Unit 5)

Implements a 6-stage classification pipeline supporting native PySpark MLlib
and high-fidelity zero-JVM Scikit-Learn local execution adapter.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.config import DEFAULT_SEED, MODELS_DIR, SENTIMENT_LABELS, is_spark_available
from src.data.data_cleaner import (
    ECOMMERCE_STOPWORDS,
    NEGATION_WORDS,
    STANDARD_STOPWORDS,
    clean_text,
)
from src.spark_session import SimulatedDataFrame, SparkSessionFactory

logger = logging.getLogger(__name__)

# Combined stopwords list preserving negations for sentiment discrimination
COMBINED_STOPWORDS = sorted(list((STANDARD_STOPWORDS | ECOMMERCE_STOPWORDS) - NEGATION_WORDS))


# ==============================================================================
# Local Vector & Simulated MLlib Components
# ==============================================================================

class LocalVector:
    """Wraps 1D numpy array with PySpark Vector interface (.size, .shape, .toArray())."""

    def __init__(self, data: Union[np.ndarray, List[float], Tuple[float, ...]]):
        self._data = np.asarray(data, dtype=float)

    @property
    def size(self) -> int:
        return int(self._data.size)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    def toArray(self) -> np.ndarray:
        return self._data

    def tolist(self) -> List[float]:
        return self._data.tolist()

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx: int) -> float:
        return float(self._data[idx])

    def __repr__(self) -> str:
        return f"LocalVector(size={self.size})"


class LocalStringIndexerModel:
    """Fitted string indexer model converting string labels to float indices."""

    def __init__(self, labels: List[str], input_col: str, output_col: str):
        self.labels = list(labels)
        self.input_col = input_col
        self.output_col = output_col
        self._mapping = {lbl: float(idx) for idx, lbl in enumerate(self.labels)}

    def transform(self, df: Any) -> SimulatedDataFrame:
        p_df = df.toPandas() if hasattr(df, "toPandas") else pd.DataFrame(df)
        p_df[self.output_col] = p_df[self.input_col].map(self._mapping).fillna(0.0).astype(float)
        return SimulatedDataFrame(p_df)


class LocalStringIndexer:
    """Estimator for indexing string labels into ordered numeric indices."""

    def __init__(self, input_col: str = "sentiment_label", output_col: str = "label"):
        self.input_col = input_col
        self.output_col = output_col
        self.labels: List[str] = []

    def fit(self, df: Any) -> LocalStringIndexerModel:
        p_df = df.toPandas() if hasattr(df, "toPandas") else pd.DataFrame(df)
        if self.input_col in p_df.columns:
            counts = p_df[self.input_col].value_counts()
            self.labels = list(counts.index)
        else:
            self.labels = ["positive", "neutral", "negative"]

        # Ensure all canonical labels are covered if fewer than 3 are present
        for def_lbl in ["positive", "neutral", "negative"]:
            if def_lbl not in self.labels:
                self.labels.append(def_lbl)

        return LocalStringIndexerModel(self.labels, self.input_col, self.output_col)


class LocalIndexToString:
    """Model converting numeric class index back to original string label."""

    def __init__(
        self,
        indexer: Optional[LocalStringIndexer] = None,
        labels: Optional[List[str]] = None,
        input_col: str = "label",
        output_col: str = "predicted_sentiment",
    ):
        self.indexer = indexer
        self.labels = labels
        self.input_col = input_col
        self.output_col = output_col

    def transform(self, df: Any) -> SimulatedDataFrame:
        p_df = df.toPandas() if hasattr(df, "toPandas") else pd.DataFrame(df)
        labels = self.labels or (self.indexer.labels if self.indexer and self.indexer.labels else ["positive", "neutral", "negative"])
        col_to_use = (
            self.input_col
            if self.input_col in p_df.columns
            else ("prediction" if "prediction" in p_df.columns else "label")
        )
        rev_mapping = {float(idx): lbl for idx, lbl in enumerate(labels)}
        p_df[self.output_col] = p_df[col_to_use].map(rev_mapping).fillna("neutral")
        return SimulatedDataFrame(p_df)


class LocalPipelineModel:
    """Fitted local classification pipeline model executing text transforms and predictions."""

    def __init__(
        self,
        vectorizer: Any,
        idf: Any,
        classifier: Any,
        labels: List[str],
        feature_type: str = "hashing_tf",
    ):
        self.vectorizer = vectorizer
        self.idf = idf
        self.classifier = classifier
        self.labels = list(labels)
        self.feature_type = feature_type

    def transform(self, df: Any) -> SimulatedDataFrame:
        p_df = df.toPandas() if hasattr(df, "toPandas") else pd.DataFrame(df)
        texts = p_df["review_text"].fillna("").astype(str).tolist() if "review_text" in p_df.columns else []

        processed_texts = []
        for t in texts:
            clean_t = clean_text(t)
            words = re.findall(r"[a-zA-Z]{2,}", clean_t.lower())
            words = [w for w in words if w not in COMBINED_STOPWORDS]
            processed_texts.append(" ".join(words))

        raw_X = self.vectorizer.transform(processed_texts)
        X_tfidf = self.idf.transform(raw_X)

        raw_probs = self.classifier.predict_proba(X_tfidf)
        num_classes = max(len(self.labels), 3)

        # Ensure probability matrix aligns with all class labels
        if raw_probs.shape[1] < num_classes:
            full_probs = np.zeros((raw_probs.shape[0], num_classes), dtype=float)
            for col_idx, class_idx in enumerate(self.classifier.classes_):
                full_probs[:, int(class_idx)] = raw_probs[:, col_idx]
            probs = full_probs
        else:
            probs = raw_probs

        # Ensure normalized simplex sum == 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums

        preds = np.argmax(probs, axis=1).astype(float)

        features_list = [LocalVector(X_tfidf[i].toarray().ravel()) for i in range(X_tfidf.shape[0])]
        prob_list = [LocalVector(probs[i]) for i in range(len(probs))]
        sentiment_list = [self.labels[int(p)] if int(p) < len(self.labels) else "neutral" for p in preds]

        p_df["features"] = features_list
        p_df["probability"] = prob_list
        p_df["prediction"] = preds
        p_df["predicted_sentiment"] = sentiment_list
        p_df["sentiment"] = sentiment_list

        return SimulatedDataFrame(p_df)


class LocalPipeline:
    """Estimator assembling feature extractor, IDF weighting, and Scikit-Learn classifier."""

    def __init__(self, feature_type: str = "hashing_tf", classifier_type: str = "logistic_regression", seed: int = DEFAULT_SEED):
        self.feature_type = feature_type
        self.classifier_type = classifier_type
        self.seed = seed

    def fit(self, df: Any) -> LocalPipelineModel:
        p_df = df.toPandas() if hasattr(df, "toPandas") else pd.DataFrame(df)
        texts = p_df["review_text"].fillna("").astype(str).tolist() if "review_text" in p_df.columns else []

        processed_texts = []
        for t in texts:
            clean_t = clean_text(t)
            words = re.findall(r"[a-zA-Z]{2,}", clean_t.lower())
            words = [w for w in words if w not in COMBINED_STOPWORDS]
            processed_texts.append(" ".join(words))

        counts = p_df["sentiment_label"].value_counts() if "sentiment_label" in p_df.columns else pd.Series(["positive", "neutral", "negative"])
        labels = list(counts.index)
        for def_lbl in ["positive", "neutral", "negative"]:
            if def_lbl not in labels:
                labels.append(def_lbl)

        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        y = np.array([label_to_idx.get(str(lbl).lower(), 0) for lbl in p_df["sentiment_label"]]) if "sentiment_label" in p_df.columns else np.zeros(len(texts), dtype=int)

        if self.feature_type == "count_vectorizer":
            vec = CountVectorizer(max_features=5000, min_df=1)
            raw_X = vec.fit_transform(processed_texts)
        else:
            vec = HashingVectorizer(n_features=16384, alternate_sign=False, norm=None)
            raw_X = vec.transform(processed_texts)

        idf = TfidfTransformer()
        X_tfidf = idf.fit_transform(raw_X)

        if self.classifier_type in ("naive_bayes", "nb"):
            clf = MultinomialNB(alpha=1.0)
        elif self.classifier_type in ("random_forest", "rf"):
            clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=self.seed)
        else:
            clf = LogisticRegression(C=10.0, max_iter=200, random_state=self.seed)

        clf.fit(X_tfidf, y)
        return LocalPipelineModel(vec, idf, clf, labels, self.feature_type)


# ==============================================================================
# SentimentMLlibPipeline Main Class
# ==============================================================================

class SentimentMLlibPipeline:
    """
    Unit 5 Apache Spark MLlib 6-Stage Sentiment Classification Pipeline.
    Supports native PySpark MLlib and embedded zero-JVM local execution adapter.
    """

    def __init__(
        self,
        feature_type: str = "hashing_tf",
        classifier_type: str = "logistic_regression",
        num_features: int = 16384,
        vocab_size: int = 5000,
        reg_param: float = 0.01,
        seed: int = DEFAULT_SEED,
    ):
        self.feature_type = feature_type
        self.classifier_type = classifier_type
        self.num_features = num_features
        self.vocab_size = vocab_size
        self.reg_param = reg_param
        self.seed = seed
        self._model: Optional[Any] = None
        self._is_spark = is_spark_available()

    @classmethod
    def get_label_indexer(
        cls,
        input_col: str = "sentiment_label",
        label_col: str = "label",
        output_col: str = "predicted_sentiment",
    ) -> Tuple[Any, Any]:
        """Returns tuple of (StringIndexer, IndexToString) for label roundtrip."""
        if is_spark_available():
            from pyspark.ml.feature import IndexToString, StringIndexer
            indexer = StringIndexer(inputCol=input_col, outputCol=label_col, stringOrderType="frequencyDesc")
            converter = IndexToString(inputCol=label_col, outputCol=output_col)
            return indexer, converter
        else:
            indexer = LocalStringIndexer(input_col=input_col, output_col=label_col)
            converter = LocalIndexToString(indexer=indexer, input_col=label_col, output_col=output_col)
            return indexer, converter

    def build_pipeline(
        self,
        feature_type: Optional[str] = None,
        classifier_type: Optional[str] = None,
    ) -> Any:
        """Assembles the 6-stage MLlib / Local classification pipeline."""
        f_type = feature_type or self.feature_type
        c_type = classifier_type or self.classifier_type

        if self._is_spark:
            from pyspark.ml import Pipeline
            from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
            from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, RegexTokenizer, StopWordsRemover, StringIndexer

            tok = RegexTokenizer(inputCol="review_text", outputCol="raw_tokens", pattern=r"\W+", toLowercase=True)
            remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens", stopWords=COMBINED_STOPWORDS)

            if f_type == "count_vectorizer":
                fe = CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features", vocabSize=self.vocab_size, minDF=1.0)
            else:
                fe = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=self.num_features)

            idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=1)
            indexer = StringIndexer(inputCol="sentiment_label", outputCol="label", stringOrderType="frequencyDesc")

            if c_type in ("naive_bayes", "nb"):
                clf = NaiveBayes(featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability")
            elif c_type in ("random_forest", "rf"):
                clf = RandomForestClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability", numTrees=20, maxDepth=5, seed=self.seed)
            else:
                clf = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability", maxIter=20, regParam=self.reg_param)

            return Pipeline(stages=[tok, remover, fe, idf, indexer, clf])
        else:
            return LocalPipeline(feature_type=f_type, classifier_type=c_type, seed=self.seed)

    def train(
        self,
        train_df: Any,
        feature_type: Optional[str] = None,
        classifier_type: Optional[str] = None,
    ) -> Any:
        """Fits the pipeline on the training DataFrame and caches the fitted model."""
        pipeline = self.build_pipeline(feature_type=feature_type, classifier_type=classifier_type)
        model = pipeline.fit(train_df)
        self._model = model
        return model

    def predict(self, model: Optional[Any], test_df: Any) -> Any:
        """Transforms test_df with model, returning predictions, probabilities, and features."""
        m = model if model is not None else self._model
        if m is None:
            self._init_default_model()
            m = self._model

        if hasattr(m, "transform"):
            return m.transform(test_df)
        raise ValueError("Provided model does not provide a transform() method.")

    def predict_single(self, text: str, model: Optional[Any] = None) -> Dict[str, Any]:
        """
        Executes single-text sentiment prediction for REST APIs and edge-case testing.
        Guarantees non-NaN confidence and bounded values across empty, single-token, and extreme inputs.
        """
        m = model if model is not None else self._model
        if m is None:
            self._init_default_model()
            m = self._model

        clean_t = clean_text(text)
        words = re.findall(r"[a-zA-Z]{2,}", clean_t.lower())
        filtered = [w for w in words if w not in COMBINED_STOPWORDS]
        proc = " ".join(filtered)

        if hasattr(m, "vectorizer") and hasattr(m, "classifier"):
            raw_X = m.vectorizer.transform([proc])
            X_tfidf = m.idf.transform(raw_X)
            raw_probs = m.classifier.predict_proba(X_tfidf)[0]

            num_classes = max(len(m.labels), 3)
            if len(raw_probs) < num_classes:
                full_probs = np.zeros(num_classes, dtype=float)
                for col_idx, class_idx in enumerate(m.classifier.classes_):
                    full_probs[int(class_idx)] = raw_probs[col_idx]
                probs = full_probs
            else:
                probs = raw_probs

            prob_sum = float(np.sum(probs))
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.ones(num_classes, dtype=float) / num_classes

            pred_idx = int(np.argmax(probs))
            conf = float(np.max(probs))

            if math.isnan(conf) or not (0.0 <= conf <= 1.0):
                conf = 0.333

            sentiment = m.labels[pred_idx] if pred_idx < len(m.labels) else "neutral"

            return {
                "text": text,
                "predicted_sentiment": sentiment,
                "sentiment": sentiment,
                "confidence": float(conf),
                "prediction": float(pred_idx),
                "probability": probs.tolist(),
            }
        else:
            # Fallback using DataFrame transform
            spark = SparkSessionFactory.get_or_create()
            single_df = spark.createDataFrame([{"review_text": text, "sentiment_label": "neutral"}])
            res_df = self.predict(m, single_df)
            first_row = res_df.first()
            pred_idx = float(first_row.get("prediction", 0.0))
            sentiment = str(first_row.get("predicted_sentiment", "neutral"))
            prob = first_row.get("probability", [0.333, 0.333, 0.334])
            prob_list = prob.toArray().tolist() if hasattr(prob, "toArray") else list(prob)
            conf = float(max(prob_list)) if prob_list else 0.50
            return {
                "text": text,
                "predicted_sentiment": sentiment,
                "sentiment": sentiment,
                "confidence": conf,
                "prediction": pred_idx,
                "probability": prob_list,
            }

    def evaluate(self, model: Optional[Any], test_df: Any) -> Dict[str, Any]:
        """Evaluates model performance against test_df and returns bounded metrics dictionary."""
        from src.ml.evaluator import evaluate_model
        m = model if model is not None else self._model
        if m is None:
            self._init_default_model()
            m = self._model
        return evaluate_model(m, test_df)

    def save_model(self, model: Any, path: Union[str, Path]) -> None:
        """Serializes trained model to disk with 100% reload parity."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._is_spark and hasattr(model, "write"):
            model.write().overwrite().save(str(p))
        else:
            joblib.dump(model, p / "model.joblib")

    def load_model(self, path: Union[str, Path]) -> Any:
        """Loads serialized model from disk."""
        p = Path(path)
        if (p / "model.joblib").exists():
            loaded = joblib.load(p / "model.joblib")
            self._model = loaded
            return loaded
        elif self._is_spark:
            from pyspark.ml import PipelineModel
            loaded = PipelineModel.load(str(p))
            self._model = loaded
            return loaded
        else:
            raise FileNotFoundError(f"No valid serialized model found at {path}")

    def train_test_split(self, df: Any, train_ratio: float = 0.8, seed: int = DEFAULT_SEED) -> Tuple[Any, Any]:
        """Splits DataFrame into deterministic train and test partitions."""
        if hasattr(df, "randomSplit") and self._is_spark:
            splits = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
            return splits[0], splits[1]
        p_df = df.toPandas() if hasattr(df, "toPandas") else pd.DataFrame(df)
        train_p = p_df.sample(frac=train_ratio, random_state=seed)
        test_p = p_df.drop(train_p.index)
        return SimulatedDataFrame(train_p.reset_index(drop=True)), SimulatedDataFrame(test_p.reset_index(drop=True))

    def _init_default_model(self) -> None:
        """Lazily trains a baseline model on canonical reviews to prevent None model exceptions."""
        canonical_samples = [
            {"review_text": "Exceptional sound quality, superb battery life, fantastic and highly recommend!", "sentiment_label": "positive"},
            {"review_text": "Great wonderful excellent product, love it so much, best purchase ever!", "sentiment_label": "positive"},
            {"review_text": "Amazing performance, fantastic build quality, highly pleased and satisfied!", "sentiment_label": "positive"},
            {"review_text": "Defective item, broke after one hour, terrible complete waste of money.", "sentiment_label": "negative"},
            {"review_text": "Awful garbage bad poor quality do not buy defect stopped working.", "sentiment_label": "negative"},
            {"review_text": "Horrible experience, completely broken, flimsy junk and useless.", "sentiment_label": "negative"},
            {"review_text": "Average product, neither good nor bad, standard quality.", "sentiment_label": "neutral"},
            {"review_text": "Acceptable item, ordinary features, neutral feelings overall.", "sentiment_label": "neutral"},
            {"review_text": "Fair standard item, acceptable for the price, gets job done.", "sentiment_label": "neutral"},
        ]
        spark = SparkSessionFactory.get_or_create()
        df = spark.createDataFrame(canonical_samples)
        self.train(df)
