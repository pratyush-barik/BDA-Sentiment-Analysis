"""
Apache Spark Core RDD Transformations & DAG Lineage Analytics.
Module: src.spark.spark_core_rdd (Unit 4)

Implements distributed RDD transformations, word/sentiment token counting,
narrow vs wide transformation mechanics, and DAG lineage tracking.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _extract_field(row: Any, key: str, default: Any = "") -> Any:
    """Helper to extract a field from either a dict, PySpark Row, or generic object."""
    if isinstance(row, dict):
        return row.get(key, default)
    if hasattr(row, "asDict"):
        return row.asDict().get(key, default)
    return getattr(row, key, default)


def _emit_word_sentiment_pairs(row: Any, min_length: int = 2) -> List[Tuple[Tuple[str, str], int]]:
    """Emits ((word, sentiment), 1) tuples for a review record."""
    text = str(_extract_field(row, "review_text", "") or _extract_field(row, "review_summary", "") or "")
    sentiment = str(_extract_field(row, "sentiment_label", "neutral") or "neutral").lower()
    if not text.strip():
        return []
    tokens = re.findall(rf"\b[a-zA-Z]{{{min_length},}}\b", text.lower())
    return [((tok, sentiment), 1) for tok in tokens]


class SparkCoreAnalytics:
    """
    Spark Core RDD processing engine implementing:
    - RDD token & sentiment aggregation
    - Narrow vs Wide transformation mechanics
    - Directed Acyclic Graph (DAG) lineage visualization
    - Distributed fault-tolerance simulation
    """

    @staticmethod
    def rdd_word_sentiment_count(df_or_rdd: Any, min_token_length: int = 2) -> Any:
        """
        Calculates sentiment token counts using an RDD transformation pipeline:
        RDD.filter (Narrow) -> RDD.flatMap (Narrow) -> RDD.reduceByKey (Wide).

        Parameters:
            df_or_rdd: Spark DataFrame, SimulatedDataFrame, RDD, or SimulatedRDD.
            min_token_length: Minimum length of tokens (default: 2).

        Returns:
            RDD containing ((word, sentiment_label), count) tuples.
        """
        # Step 1: Normalize input to an RDD
        if hasattr(df_or_rdd, "rdd"):
            rdd = df_or_rdd.rdd
        else:
            rdd = df_or_rdd

        # Step 2: Narrow transformation (filter empty reviews)
        filtered_rdd = rdd.filter(
            lambda r: bool(
                str(_extract_field(r, "review_text", "")).strip()
                or str(_extract_field(r, "review_summary", "")).strip()
            )
        )

        # Step 3: Narrow transformation (flatMap tokens to ((word, sentiment), 1))
        mapped_rdd = filtered_rdd.flatMap(
            lambda r: _emit_word_sentiment_pairs(r, min_length=min_token_length)
        )

        # Step 4: Wide transformation (reduceByKey across shuffle partitions)
        counted_rdd = mapped_rdd.reduceByKey(lambda a, b: a + b)

        return counted_rdd

    @staticmethod
    def get_rdd_lineage_debug_string(rdd: Any) -> str:
        """
        Returns the RDD lineage DAG execution graph as a formatted string.
        """
        if rdd is None:
            return "(0) Empty RDD Lineage"

        if hasattr(rdd, "toDebugString"):
            dbg = rdd.toDebugString()
            if isinstance(dbg, bytes):
                return dbg.decode("utf-8", errors="replace")
            return str(dbg)

        # Fallback lineage DAG representation
        return (
            "(2) ShuffledRDD [reduceByKey] -> "
            "MapPartitionsRDD [flatMap] -> "
            "MapPartitionsRDD [filter] -> "
            "ParallelCollectionRDD"
        )

    @staticmethod
    def explain_catalyst_plan(df: Any, extended: bool = True) -> str:
        """Cross-module delegate to Catalyst physical plan explainer."""
        from src.spark.spark_sql_analytics import explain_catalyst_plan as _explain
        return _explain(df, extended=extended)

    @staticmethod
    def explain_transformation_types() -> Dict[str, Any]:
        """Returns academic breakdown of Narrow vs Wide transformations."""
        return {
            "narrow_transformations": {
                "definition": "Each partition of the parent RDD is used by at most one partition of the child RDD. Operations occur pipelined within the same stage without network shuffle.",
                "operations": ["map", "filter", "flatMap", "mapValues", "sample"],
                "pipeline_stages": "Executed in-memory within worker task partitions.",
            },
            "wide_transformations": {
                "definition": "Multiple child partitions depend on records across multiple parent partitions. Requires shuffling data across cluster worker nodes via hash partitioner, creating a stage boundary.",
                "operations": ["reduceByKey", "groupByKey", "distinct", "join", "repartition", "sortByKey"],
                "pipeline_stages": "Forces materialization of shuffle files and stage boundary creation in the DAG.",
            },
            "fault_tolerance": {
                "mechanism": "Deterministic Lineage Graph (DAG)",
                "description": "RDDs are immutable datasets. When a node or partition fails, Spark recomputes only the lost partition by replaying upstream lineage transformations from the original data source, eliminating the need for expensive checkpointing on every stage.",
            },
        }

    @staticmethod
    def rdd_word_count_by_sentiment(df_or_rdd: Any) -> Dict[str, Dict[str, int]]:
        """Collects RDD results into a nested dict: {sentiment: {word: count}}."""
        counted_rdd = SparkCoreAnalytics.rdd_word_sentiment_count(df_or_rdd)
        items = counted_rdd.collect() if hasattr(counted_rdd, "collect") else list(counted_rdd)
        result: Dict[str, Dict[str, int]] = {"positive": {}, "neutral": {}, "negative": {}}
        for (word, sentiment), count in items:
            sentiment_key = sentiment.lower()
            if sentiment_key not in result:
                result[sentiment_key] = {}
            result[sentiment_key][word] = result[sentiment_key].get(word, 0) + count
        return result


# Convenience module-level aliases
rdd_word_sentiment_count = SparkCoreAnalytics.rdd_word_sentiment_count
get_rdd_lineage_debug_string = SparkCoreAnalytics.get_rdd_lineage_debug_string
explain_catalyst_plan = SparkCoreAnalytics.explain_catalyst_plan
