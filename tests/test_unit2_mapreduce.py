"""
tests/test_unit2_mapreduce.py
Unit tests for Unit 2 MapReduce Processing Engine.
Validates ReviewTokenizerMapper, SentimentCombiner, MapReducePartitioner,
MapReduceShuffleAndSort, SentimentReducer, run_mapreduce_job, and simulate_partition_failure.
"""

from typing import Any, Dict, List
import pytest

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


@pytest.fixture
def sample_test_records() -> List[Dict[str, Any]]:
    return [
        {
            "review_id": "REV_001",
            "review_summary": "Outstanding battery and sound quality",
            "review_text": "The battery life is exceptional. Crisp sound, definitely recommend!",
            "rating": 5.0,
            "sentiment_label": "positive",
        },
        {
            "review_id": "REV_002",
            "review_summary": "Defective screen and terrible service",
            "review_text": "Screen cracked on arrival. Customer service was not helpful at all.",
            "rating": 1.0,
            "sentiment_label": "negative",
        },
        {
            "review_id": "REV_003",
            "review_summary": "Average performance",
            "review_text": "It works as expected. Nothing special, but battery is okay.",
            "rating": 3.0,
            "sentiment_label": "neutral",
        },
    ]


def test_mapper_emission_format_and_types(sample_test_records):
    mapper = ReviewTokenizerMapper()
    emissions = list(mapper.map(sample_test_records[0]))

    assert len(emissions) > 0
    for key, val in emissions:
        assert isinstance(key, tuple)
        assert len(key) == 2
        word, sentiment = key
        assert isinstance(word, str)
        assert sentiment == "positive"
        assert val == 1


def test_mapper_batch_streaming(sample_test_records):
    mapper = ReviewTokenizerMapper()
    batch_emissions = list(mapper.map_batch(sample_test_records))
    assert len(batch_emissions) > 0

    sentiments = {sent for (word, sent), val in batch_emissions}
    assert "positive" in sentiments
    assert "negative" in sentiments
    assert "neutral" in sentiments


def test_mapper_sentiment_derivation_fallback():
    mapper = ReviewTokenizerMapper()

    # Explicit labels
    assert mapper._extract_sentiment({"sentiment_label": "POSITIVE"}) == "positive"
    assert mapper._extract_sentiment({"sentiment_label": "NEGATIVE"}) == "negative"
    assert mapper._extract_sentiment({"sentiment_label": "NEUTRAL"}) == "neutral"

    # Rating threshold fallback
    assert mapper._extract_sentiment({"rating": 5.0}) == "positive"
    assert mapper._extract_sentiment({"rating": 4.0}) == "positive"
    assert mapper._extract_sentiment({"rating": 3.5}) == "neutral"
    assert mapper._extract_sentiment({"rating": 3.0}) == "neutral"
    assert mapper._extract_sentiment({"rating": 2.0}) == "negative"
    assert mapper._extract_sentiment({"rating": 1.0}) == "negative"

    # Invalid / missing ratings
    assert mapper._extract_sentiment({"rating": "invalid"}) == "neutral"
    assert mapper._extract_sentiment({}) == "neutral"


def test_mapper_negation_preservation_and_noise_cleaning():
    mapper = ReviewTokenizerMapper()
    record = {
        "review_summary": "<h1>Warning!</h1>",
        "review_text": "I don't recommend this product. Check http://example.com for details! ⭐",
        "rating": 1.0,
    }
    emissions = list(mapper.map(record))
    words = [k[0] for k, _ in emissions]

    # Contraction expanded and negation preserved
    assert "not" in words
    assert "recommend" in words
    # Noise stripped
    assert "h1" not in words
    assert "http" not in words
    assert "example" not in words


def test_mapper_empty_and_malformed_records():
    mapper = ReviewTokenizerMapper()

    assert list(mapper.map({})) == []
    assert list(mapper.map(None)) == []  # type: ignore[arg-type]
    assert list(mapper.map("not a dict")) == []  # type: ignore[arg-type]
    assert list(mapper.map({"review_text": None, "review_summary": None})) == []
    assert list(mapper.map({"review_text": "   \n\t   "})) == []

    # Pre-existing tokens branch
    record_with_tokens = {
        "tokens": ["awesome", "display", "fast"],
        "sentiment_label": "positive",
    }
    emissions = list(mapper.map(record_with_tokens))
    assert len(emissions) == 3
    assert emissions[0] == (("awesome", "positive"), 1)


def test_combiner_token_conservation_and_compression():
    combiner = SentimentCombiner()
    raw_emissions = [
        (("battery", "positive"), 1),
        (("battery", "positive"), 1),
        (("battery", "positive"), 1),
        (("sound", "positive"), 1),
        (("defective", "negative"), 1),
        (("defective", "negative"), 1),
    ]

    combined = list(combiner.combine(raw_emissions))

    # Compression invariant
    assert len(combined) < len(raw_emissions)
    assert len(combined) == 3

    # Token conservation invariant
    sum_raw = sum(v for _, v in raw_emissions)
    sum_combined = sum(v for _, v in combined)
    assert sum_raw == sum_combined == 6

    # Specific counts
    combined_dict = dict(combined)
    assert combined_dict[("battery", "positive")] == 3
    assert combined_dict[("sound", "positive")] == 1
    assert combined_dict[("defective", "negative")] == 2


def test_combiner_empty_input():
    combiner = SentimentCombiner()
    assert list(combiner.combine([])) == []


def test_partitioner_deterministic_distribution():
    partitioner = MapReducePartitioner()

    # Determinism
    p1 = partitioner.get_partition(("battery", "positive"), num_partitions=4)
    p2 = partitioner.get_partition(("battery", "negative"), num_partitions=4)
    assert p1 == p2, "Same word should route to identical partition regardless of sentiment"
    assert 0 <= p1 < 4

    # Non-negative partition across any word
    for word in ["test", "alpha", "negative", "ultra", "xyz_1234"]:
        pid = partitioner.get_partition((word, "neutral"), num_partitions=3)
        assert 0 <= pid < 3

    # Error on non-positive partitions
    with pytest.raises(ValueError, match="num_partitions must be >= 1"):
        partitioner.get_partition(("battery", "positive"), num_partitions=0)


def test_shuffle_and_sort_grouping_and_order():
    shuffler = MapReduceShuffleAndSort(num_reducers=3)
    emissions = [
        (("zebra", "positive"), 2),
        (("apple", "negative"), 5),
        (("mango", "neutral"), 1),
        (("banana", "positive"), 3),
        (("apple", "positive"), 1),
    ]

    partitions = shuffler.shuffle_and_sort(emissions)
    assert len(partitions) == 3

    # Total items across partitions matches input
    total_items = sum(len(items) for items in partitions.values())
    assert total_items == len(emissions)

    # Within each partition, keys must be lexicographically sorted
    for pid, items in partitions.items():
        keys = [item[0] for item in items]
        assert keys == sorted(keys)


def test_reducer_global_aggregation():
    reducer = SentimentReducer()
    emissions = [
        (("battery", "positive"), 5),
        (("battery", "negative"), 2),
        (("sound", "positive"), 3),
        (("sound", "neutral"), 1),
    ]

    reduced = reducer.reduce(emissions)
    assert isinstance(reduced, dict)
    assert "battery" in reduced
    assert reduced["battery"] == {"positive": 5, "negative": 2}
    assert reduced["sound"] == {"positive": 3, "neutral": 1}

    # Empty emissions
    assert reducer.reduce([]) == {}


def test_reducer_polarity_calculation():
    reducer = SentimentReducer()

    # Pure positive
    assert reducer.compute_polarity({"positive": 10}) == 1.0
    # Pure negative
    assert reducer.compute_polarity({"negative": 10}) == -1.0
    # Neutral only
    assert reducer.compute_polarity({"neutral": 10}) == 0.0
    # Balanced
    assert reducer.compute_polarity({"positive": 5, "negative": 5}) == 0.0
    # Mixed: (8 - 2) / (8 + 2 + 0) = 0.6
    assert reducer.compute_polarity({"positive": 8, "negative": 2}) == 0.6
    # Empty
    assert reducer.compute_polarity({}) == 0.0


def test_reducer_auxiliary_metrics():
    reducer = SentimentReducer()
    counts = {
        "great": {"positive": 15, "neutral": 2},
        "bad": {"negative": 12, "neutral": 1},
        "okay": {"neutral": 10, "positive": 2, "negative": 2},
    }

    polarities = reducer.compute_all_polarity_scores(counts)
    assert polarities["great"] > 0
    assert polarities["bad"] < 0
    assert -1.0 <= polarities["okay"] <= 1.0

    frequencies = reducer.compute_word_frequencies(counts)
    assert frequencies["great"] == 17
    assert frequencies["bad"] == 13
    assert frequencies["okay"] == 14

    top_pos = reducer.get_top_sentiment_words(counts, "positive", top_n=2)
    assert len(top_pos) == 2
    assert top_pos[0] == ("great", 15)
    assert top_pos[1] == ("okay", 2)

    top_neg = reducer.get_top_sentiment_words(counts, "negative", top_n=2)
    assert top_neg[0] == ("bad", 12)


def test_run_mapreduce_job_pipeline(sample_test_records):
    # Run with combiner enabled
    result_with_comb = run_mapreduce_job(
        sample_test_records, num_mappers=2, num_reducers=2, use_combiner=True
    )
    assert isinstance(result_with_comb, MapReduceResult)
    assert result_with_comb["status"] == "SUCCESS"
    assert result_with_comb["total_records_processed"] == 3
    assert result_with_comb["total_emissions"] > 0
    assert result_with_comb["unique_words"] > 0
    assert result_with_comb["shuffle_bandwidth_reduction_pct"] >= 0.0

    # Key lookup transparency
    assert "battery" in result_with_comb
    assert isinstance(result_with_comb["battery"], dict)

    # Run with combiner disabled
    result_no_comb = run_mapreduce_job(
        sample_test_records, num_mappers=2, num_reducers=2, use_combiner=False
    )
    # Conservation of total tokens between with-combiner and without-combiner
    assert result_with_comb["total_emissions"] == result_no_comb["total_emissions"]
    assert result_no_comb["shuffle_bandwidth_reduction_pct"] == 0.0

    # Empty inputs
    empty_result = run_mapreduce_job([])
    assert empty_result["total_records_processed"] == 0
    assert empty_result["unique_words"] == 0
    assert empty_result["total_emissions"] == 0


def test_simulate_partition_failure_idempotence(sample_test_records):
    sim_result = simulate_partition_failure(
        sample_test_records, failed_mapper_idx=0, num_mappers=2
    )

    assert sim_result["status"] == "RECOVERED"
    assert sim_result["failed_partition"] == 0
    assert sim_result["recomputed_emissions_count"] > 0
    assert sim_result["idempotence_verified"] is True
    assert sim_result["token_conservation_preserved"] is True

    # Out of range partition fallback
    sim_fallback = simulate_partition_failure(sample_test_records, failed_mapper_idx=99)
    assert sim_fallback["idempotence_verified"] is True

    # Empty dataset simulation
    empty_sim = simulate_partition_failure([], failed_mapper_idx=0)
    assert empty_sim["idempotence_verified"] is True


def test_mapreduce_result_get_method(sample_test_records):
    """
    Validates MapReduceResult.get() lookup parity:
    Retrieves keyword sentiment frequency dictionary for keywords,
    outer metadata for dictionary keys, and default for missing keys.
    """
    result = run_mapreduce_job(sample_test_records, num_mappers=2, num_reducers=2)

    # 1. Direct keyword lookup via .get()
    assert "battery" in result
    battery_dict = result.get("battery")
    assert battery_dict is not None
    assert isinstance(battery_dict, dict)
    assert "positive" in battery_dict
    assert battery_dict["positive"] >= 1

    # 2. Outer metadata lookup via .get()
    assert result.get("status") == "SUCCESS"
    assert result.get("total_records_processed") == 3
    assert isinstance(result.get("word_sentiment_frequencies"), dict)

    # 3. Missing keyword lookup with default=None
    assert result.get("non_existent_token_xyz") is None

    # 4. Missing keyword lookup with custom default
    default_fallback = {"positive": 0, "negative": 0}
    assert result.get("non_existent_token_xyz", default_fallback) == default_fallback


def test_simulate_partition_failure_generator_input(sample_test_records):
    """
    Validates simulate_partition_failure when input records are provided
    as a generator or iterator (preventing TypeError and generator exhaustion).
    """
    # 1. Generator input
    gen_records = (dict(r) for r in sample_test_records)
    sim_result = simulate_partition_failure(gen_records, failed_mapper_idx=0, num_mappers=2)

    assert sim_result["status"] == "RECOVERED"
    assert sim_result["failed_partition"] == 0
    assert sim_result["recomputed_emissions_count"] > 0
    assert sim_result["idempotence_verified"] is True
    assert sim_result["token_conservation_preserved"] is True

    # 2. Iterator input
    iter_records = iter([dict(r) for r in sample_test_records])
    sim_result_iter = simulate_partition_failure(iter_records, failed_mapper_idx=1, num_mappers=2)
    assert sim_result_iter["status"] == "RECOVERED"
    assert sim_result_iter["idempotence_verified"] is True

