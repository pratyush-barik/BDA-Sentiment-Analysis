"""
MapReduce Processing Engine (Unit 2).
Big Data Product Review Sentiment Analysis System.
"""

from src.mapreduce.sentiment_mr import (
    MapReducePartitioner,
    MapReduceResult,
    MapReduceShuffleAndSort,
    ReviewTokenizerMapper,
    SentimentCombiner,
    SentimentReducer,
    run_mapreduce_job,
    simulate_partition_failure,
)

__all__ = [
    "ReviewTokenizerMapper",
    "SentimentCombiner",
    "MapReducePartitioner",
    "MapReduceShuffleAndSort",
    "SentimentReducer",
    "MapReduceResult",
    "run_mapreduce_job",
    "simulate_partition_failure",
]
