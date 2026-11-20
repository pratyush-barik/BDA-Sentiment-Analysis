"""
Apache Spark MLlib Sentiment Classification & Evaluation Package (Unit 5).
Module: src.ml
"""

from src.ml.evaluator import (
    MLlibEvaluator,
    SentimentEvaluator,
    compute_metrics,
    evaluate_model,
    export_metrics_json,
    generate_confusion_matrix,
)
from src.ml.spark_mllib_pipeline import LocalVector, SentimentMLlibPipeline

__all__ = [
    "SentimentMLlibPipeline",
    "LocalVector",
    "SentimentEvaluator",
    "MLlibEvaluator",
    "compute_metrics",
    "generate_confusion_matrix",
    "evaluate_model",
    "export_metrics_json",
]
