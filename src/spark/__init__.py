"""
Apache Spark Core & Spark SQL Analytics Package (Unit 4).
Module: src.spark
"""

from src.spark.spark_core_rdd import (
    SparkCoreAnalytics,
    get_rdd_lineage_debug_string,
    rdd_word_sentiment_count,
)
from src.spark.spark_sql_analytics import (
    SparkSQLAnalytics,
    execute_business_queries,
    execute_spark_sql_queries,
    explain_catalyst_plan,
    register_temp_view,
)

__all__ = [
    "SparkCoreAnalytics",
    "SparkSQLAnalytics",
    "rdd_word_sentiment_count",
    "get_rdd_lineage_debug_string",
    "register_temp_view",
    "execute_business_queries",
    "execute_spark_sql_queries",
    "explain_catalyst_plan",
]
