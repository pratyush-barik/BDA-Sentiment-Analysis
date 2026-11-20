"""
tests/test_tier3_document_store.py
Unit tests for Unit 3 Document Store Integration.
Validates IDocumentStore abstract contract, LocalJsonDocumentStore CRUD,
query operators, nested dot notation, aggregation pipeline accumulators,
secondary indexing acceleration, atomic persistence round-trip,
factory fallback, and concurrent thread safety.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.database.document_store import IDocumentStore, get_document_store
from src.database.local_store import LocalJsonDocumentStore
from src.database.mongo_store import MongoDocumentStore


@pytest.fixture
def sample_docs() -> List[Dict[str, Any]]:
    return [
        {
            "_id": "DOC_001",
            "review_id": "REV_001",
            "product_category": "Electronics",
            "product_name": "Noise-Cancelling Headphones",
            "rating": 5.0,
            "verified_purchase": True,
            "sentiment_label": "positive",
            "meta": {"tokens": ["crisp", "sound"], "score": 0.95},
        },
        {
            "_id": "DOC_002",
            "review_id": "REV_002",
            "product_category": "Electronics",
            "product_name": "Wireless Mouse",
            "rating": 4.0,
            "verified_purchase": True,
            "sentiment_label": "positive",
            "meta": {"tokens": ["ergonomic", "smooth"], "score": 0.85},
        },
        {
            "_id": "DOC_003",
            "review_id": "REV_003",
            "product_category": "Books",
            "product_name": "Data Architecture Guide",
            "rating": 5.0,
            "verified_purchase": False,
            "sentiment_label": "positive",
            "meta": {"tokens": ["informative", "deep"], "score": 0.92},
        },
        {
            "_id": "DOC_004",
            "review_id": "REV_004",
            "product_category": "Books",
            "product_name": "Python Cookbook",
            "rating": 2.0,
            "verified_purchase": True,
            "sentiment_label": "negative",
            "meta": {"tokens": ["outdated"], "score": 0.25},
        },
        {
            "_id": "DOC_005",
            "review_id": "REV_005",
            "product_category": "Home & Kitchen",
            "product_name": "Espresso Machine",
            "rating": 3.0,
            "verified_purchase": False,
            "sentiment_label": "neutral",
            "meta": {"tokens": ["decent", "coffee"], "score": 0.50},
        },
    ]


# ---------------------------------------------------------------------------
# 1. IDocumentStore Abstract Base Class Contract
# ---------------------------------------------------------------------------

def test_document_store_abstract_interface_contract():
    # Direct instantiation of ABC must fail
    with pytest.raises(TypeError):
        IDocumentStore()  # type: ignore[abstract]

    # Concrete store satisfies IDocumentStore
    store = LocalJsonDocumentStore(in_memory=True)
    assert isinstance(store, IDocumentStore)

    # Context manager test
    with store as active_store:
        assert active_store.is_connected
    assert not store.is_connected


# ---------------------------------------------------------------------------
# 2. CRUD Operations & ID Generation
# ---------------------------------------------------------------------------

def test_local_store_insert_one_and_many_with_id_generation():
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()

    # 1. Insert one with pre-existing _id
    doc_id1 = store.insert_one({"_id": "CUSTOM_1", "name": "Item 1"})
    assert doc_id1 == "CUSTOM_1"

    # 2. Insert one with auto-generated _id
    doc_id2 = store.insert_one({"name": "Item 2"})
    assert doc_id2.startswith("doc_")
    assert store.count_documents() == 2

    # 3. Duplicate _id rejection
    with pytest.raises(ValueError, match="already exists"):
        store.insert_one({"_id": "CUSTOM_1", "name": "Duplicate"})

    # 4. Insert many
    batch = [{"name": "Batch 1"}, {"name": "Batch 2"}, {"_id": "BATCH_3", "name": "Batch 3"}]
    ids = store.insert_many(batch)
    assert len(ids) == 3
    assert ids[2] == "BATCH_3"
    assert store.count_documents() == 5

    # 5. Empty insert_many
    assert store.insert_many([]) == []


# ---------------------------------------------------------------------------
# 3. Query Comparison Operators
# ---------------------------------------------------------------------------

def test_local_store_query_operators_comparison(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # $eq
    res_eq = store.find({"rating": {"$eq": 5.0}})
    assert len(res_eq) == 2
    assert all(d["rating"] == 5.0 for d in res_eq)

    # Literal equality
    res_lit = store.find({"product_category": "Books"})
    assert len(res_lit) == 2

    # $ne
    res_ne = store.find({"product_category": {"$ne": "Electronics"}})
    assert len(res_ne) == 3

    # $gte
    res_gte = store.find({"rating": {"$gte": 4.0}})
    assert len(res_gte) == 3
    assert all(d["rating"] >= 4.0 for d in res_gte)

    # $gt
    res_gt = store.find({"rating": {"$gt": 4.0}})
    assert len(res_gt) == 2

    # $lte
    res_lte = store.find({"rating": {"$lte": 3.0}})
    assert len(res_lte) == 2

    # $lt
    res_lt = store.find({"rating": {"$lt": 3.0}})
    assert len(res_lt) == 1
    assert res_lt[0]["_id"] == "DOC_004"


# ---------------------------------------------------------------------------
# 4. Query Membership, Exists & Regex Operators
# ---------------------------------------------------------------------------

def test_local_store_query_operators_membership_and_regex(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # $in on scalar
    res_in = store.find({"product_category": {"$in": ["Books", "Home & Kitchen"]}})
    assert len(res_in) == 3

    # $nin on scalar
    res_nin = store.find({"product_category": {"$nin": ["Books", "Home & Kitchen"]}})
    assert len(res_nin) == 2
    assert all(d["product_category"] == "Electronics" for d in res_nin)

    # $exists
    assert len(store.find({"verified_purchase": {"$exists": True}})) == 5
    assert len(store.find({"non_existent": {"$exists": True}})) == 0
    assert len(store.find({"non_existent": {"$exists": False}})) == 5

    # $regex
    res_regex = store.find({"product_name": {"$regex": r"Guide|Cookbook"}})
    assert len(res_regex) == 2

    # $regex with case insensitivity option
    res_regex_i = store.find({"product_name": {"$regex": "headphones", "$options": "i"}})
    assert len(res_regex_i) == 1


# ---------------------------------------------------------------------------
# 5. Nested Field Lookups & Logical Operators
# ---------------------------------------------------------------------------

def test_local_store_nested_field_lookups(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # Nested float comparison
    res_score = store.find({"meta.score": {"$gte": 0.90}})
    assert len(res_score) == 2

    # $in on nested list
    res_tok = store.find({"meta.tokens": {"$in": ["smooth"]}})
    assert len(res_tok) == 1
    assert res_tok[0]["_id"] == "DOC_002"

    # Non-existent nested path returns empty list safely
    assert len(store.find({"meta.missing.subfield": 123})) == 0

    # $and / $or logical operators
    res_and = store.find({
        "$and": [
            {"product_category": "Electronics"},
            {"rating": {"$gte": 5.0}},
        ]
    })
    assert len(res_and) == 1
    assert res_and[0]["_id"] == "DOC_001"

    res_or = store.find({
        "$or": [
            {"rating": 2.0},
            {"rating": 3.0},
        ]
    })
    assert len(res_or) == 2


# ---------------------------------------------------------------------------
# 6. Projection Engine
# ---------------------------------------------------------------------------

def test_local_store_projection_inclusion_and_exclusion(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # Inclusion projection (with default _id included)
    docs_inc = store.find({"_id": "DOC_001"}, projection={"product_name": 1, "rating": 1})
    assert len(docs_inc) == 1
    doc = docs_inc[0]
    assert set(doc.keys()) == {"_id", "product_name", "rating"}

    # Inclusion projection (with explicit _id: 0)
    docs_no_id = store.find({"_id": "DOC_001"}, projection={"product_name": 1, "_id": 0})
    assert set(docs_no_id[0].keys()) == {"product_name"}

    # Exclusion projection
    docs_exc = store.find({"_id": "DOC_001"}, projection={"meta": 0, "verified_purchase": 0})
    assert "meta" not in docs_exc[0]
    assert "verified_purchase" not in docs_exc[0]
    assert "product_name" in docs_exc[0]


# ---------------------------------------------------------------------------
# 7. Update & Delete Operations
# ---------------------------------------------------------------------------

def test_local_store_find_one_update_and_delete(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # find_one
    found = store.find_one({"_id": "DOC_001"})
    assert found is not None
    assert found["_id"] == "DOC_001"

    assert store.find_one({"_id": "NON_EXISTENT"}) is None

    # update_one ($set and $inc)
    updated = store.update_one(
        {"_id": "DOC_001"},
        {"$set": {"product_name": "Updated Headphones"}, "$inc": {"rating": -0.5}},
    )
    assert updated is True
    doc_after = store.find_one({"_id": "DOC_001"})
    assert doc_after["product_name"] == "Updated Headphones"
    assert doc_after["rating"] == 4.5

    # update_many
    modified_cnt = store.update_many(
        {"product_category": "Books"},
        {"$set": {"on_sale": True}},
    )
    assert modified_cnt == 2
    assert all(d.get("on_sale") for d in store.find({"product_category": "Books"}))

    # delete_one
    del_one = store.delete_one({"_id": "DOC_005"})
    assert del_one is True
    assert store.find_one({"_id": "DOC_005"}) is None
    assert store.count_documents() == 4

    # delete_many
    del_many_cnt = store.delete_many({"product_category": "Books"})
    assert del_many_cnt == 2
    assert store.count_documents() == 2


# ---------------------------------------------------------------------------
# 8. Aggregation Pipeline: All Accumulators
# ---------------------------------------------------------------------------

def test_local_store_aggregation_group_all_accumulators(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    pipeline = [
        {"$match": {"rating": {"$gte": 2.0}}},
        {
            "$group": {
                "_id": "$product_category",
                "total_count": {"$sum": 1},
                "total_rating_sum": {"$sum": "$rating"},
                "avg_rating": {"$avg": "$rating"},
                "min_rating": {"$min": "$rating"},
                "max_rating": {"$max": "$rating"},
                "pushed_ids": {"$push": "$_id"},
                "first_name": {"$first": "$product_name"},
                "last_name": {"$last": "$product_name"},
                "doc_count": {"$count": {}},
            }
        },
        {"$sort": {"total_count": -1}},
    ]

    agg_results = store.aggregate(pipeline)
    assert len(agg_results) == 3

    # Mathematical conservation of grouped document counts
    total_grouped = sum(g["total_count"] for g in agg_results)
    assert total_grouped == 5

    # Inspect electronics category
    elec = next(g for g in agg_results if g["_id"] == "Electronics")
    assert elec["total_count"] == 2
    assert elec["avg_rating"] == 4.5
    assert elec["min_rating"] == 4.0
    assert elec["max_rating"] == 5.0
    assert set(elec["pushed_ids"]) == {"DOC_001", "DOC_002"}
    assert elec["doc_count"] == 2


# ---------------------------------------------------------------------------
# 9. Aggregation Pipeline: Composite Group & Conditional Accumulators
# ---------------------------------------------------------------------------

def test_local_store_aggregation_composite_group_and_cond(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # 1. Composite Group Key
    pipeline_composite = [
        {
            "$group": {
                "_id": {
                    "category": "$product_category",
                    "sentiment": "$sentiment_label",
                },
                "count": {"$sum": 1},
            }
        }
    ]
    comp_results = store.aggregate(pipeline_composite)
    assert len(comp_results) > 0
    for group in comp_results:
        assert isinstance(group["_id"], dict)
        assert "category" in group["_id"]
        assert "sentiment" in group["_id"]

    # 2. Conditional $sum with $cond
    pipeline_cond = [
        {
            "$group": {
                "_id": "$product_category",
                "verified_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$verified_purchase", True]},
                            1,
                            0,
                        ]
                    }
                },
                "unverified_count": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$verified_purchase", False]},
                            1,
                            0,
                        ]
                    }
                },
            }
        }
    ]
    cond_results = store.aggregate(pipeline_cond)
    elec_cond = next(g for g in cond_results if g["_id"] == "Electronics")
    assert elec_cond["verified_count"] == 2
    assert elec_cond["unverified_count"] == 0

    books_cond = next(g for g in cond_results if g["_id"] == "Books")
    assert books_cond["verified_count"] == 1
    assert books_cond["unverified_count"] == 1


# ---------------------------------------------------------------------------
# 10. Aggregation Pipeline: Multi-Key Sort, Limit & Project
# ---------------------------------------------------------------------------

def test_local_store_aggregation_sort_limit_project(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    pipeline = [
        {"$sort": {"rating": -1, "product_name": 1}},
        {"$limit": 3},
        {
            "$project": {
                "name": "$product_name",
                "stars": "$rating",
                "_id": 0,
            }
        },
    ]

    results = store.aggregate(pipeline)
    assert len(results) == 3

    # Top two should both have rating 5.0, sorted alphabetically by name
    assert results[0]["stars"] == 5.0
    assert results[1]["stars"] == 5.0
    assert results[0]["name"] <= results[1]["name"]
    assert "_id" not in results[0]
    assert "name" in results[0]
    assert "stars" in results[0]


# ---------------------------------------------------------------------------
# 11. Secondary Indexing Simulation & Query Parity
# ---------------------------------------------------------------------------

def test_local_store_secondary_indexing_acceleration(sample_docs):
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # Baseline query without secondary index
    baseline_query = {"product_category": "Electronics", "rating": 5.0}
    baseline_results = store.find(baseline_query)
    assert len(baseline_results) == 1
    assert store.last_query_used_index is None

    # Create single-field index
    idx_single = store.create_index([("product_category", 1)])
    assert idx_single == "product_category_1"

    # Query using single-field index
    res_single = store.find({"product_category": "Books"})
    assert len(res_single) == 2
    assert store.last_query_used_index == "product_category_1"

    # Create compound index
    idx_comp = store.create_index([("product_category", 1), ("rating", -1)])
    assert idx_comp == "product_category_1_rating_-1"

    # Query matching compound index
    res_comp = store.find({"product_category": "Electronics", "rating": 5.0})
    assert len(res_comp) == 1
    assert store.last_query_used_index == "product_category_1_rating_-1"

    # Result Parity Assertion
    assert res_comp == baseline_results


# ---------------------------------------------------------------------------
# 12. Atomic File Persistence Round-Trip
# ---------------------------------------------------------------------------

def test_local_store_atomic_file_persistence_roundtrip(tmp_path: Path, sample_docs):
    store_file = tmp_path / "persist_test.json"

    # Session 1: Populate and write
    store1 = LocalJsonDocumentStore(storage_path=store_file)
    store1.connect()
    store1.create_index([("product_category", 1)])
    store1.insert_many(sample_docs)
    assert store1.count_documents() == len(sample_docs)
    store1.close()

    assert store_file.exists()

    # Session 2: Reload from disk in a fresh store instance
    store2 = LocalJsonDocumentStore(storage_path=store_file)
    store2.connect()
    assert store2.count_documents() == len(sample_docs)

    # Verify query and index retention
    books = store2.find({"product_category": "Books"})
    assert len(books) == 2
    assert store2.last_query_used_index == "product_category_1"
    store2.close()


# ---------------------------------------------------------------------------
# 13. Factory Auto-Detection & Fallback
# ---------------------------------------------------------------------------

def test_factory_auto_detection_and_fallback(tmp_path: Path):
    # 1. prefer_mongo=False directly yields LocalJsonDocumentStore
    store_local = get_document_store(
        prefer_mongo=False, storage_path=tmp_path / "local1.json"
    )
    assert isinstance(store_local, LocalJsonDocumentStore)

    # 2. prefer_mongo=True gracefully falls back to LocalJsonDocumentStore
    # when MongoDB daemon is not running (zero unhandled exceptions)
    store_fallback = get_document_store(
        prefer_mongo=True,
        storage_path=tmp_path / "fallback.json",
        server_selection_timeout_ms=100,
    )
    assert isinstance(store_fallback, (LocalJsonDocumentStore, MongoDocumentStore))
    # In test environment without mongod, fallback is guaranteed
    assert isinstance(store_fallback, LocalJsonDocumentStore)


# ---------------------------------------------------------------------------
# 14. Thread Safety & Concurrent Writes
# ---------------------------------------------------------------------------

def test_local_store_thread_safety():
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()

    num_threads = 5
    records_per_thread = 20
    errors: List[Exception] = []

    def worker(thread_idx: int):
        try:
            for i in range(records_per_thread):
                store.insert_one({
                    "thread_id": thread_idx,
                    "seq": i,
                    "rating": 4.0,
                })
        except Exception as ex:
            errors.append(ex)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread errors observed: {errors}"
    expected_total = num_threads * records_per_thread
    assert store.count_documents() == expected_total


# ---------------------------------------------------------------------------
# 15. Aggregation Pipeline: Exclusion Projection & Heterogeneous Sort
# ---------------------------------------------------------------------------

def test_local_store_aggregation_project_exclusion_mode(sample_docs):
    """
    Validates $project in aggregation pipelines for exclusion mode:
    - Exclusion of single field preserves all other fields.
    - Exclusion of field + _id excludes both while retaining remainder.
    - Exclusion of _id only excludes _id.
    - Nested field exclusion removes only specified subfield.
    """
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()
    store.insert_many(sample_docs)

    # 1. Exclusion of single field
    res_exc_meta = store.aggregate([{"$project": {"meta": 0}}])
    assert len(res_exc_meta) == len(sample_docs)
    for doc in res_exc_meta:
        assert "meta" not in doc
        assert "_id" in doc
        assert "product_name" in doc
        assert "rating" in doc
        assert len(doc) >= 5

    # 2. Exclusion of field and _id
    res_exc_both = store.aggregate([{"$project": {"meta": 0, "_id": 0}}])
    assert len(res_exc_both) == len(sample_docs)
    for doc in res_exc_both:
        assert "meta" not in doc
        assert "_id" not in doc
        assert "product_name" in doc
        assert "rating" in doc

    # 3. Exclusion of _id only
    res_exc_id_only = store.aggregate([{"$project": {"_id": 0}}])
    assert len(res_exc_id_only) == len(sample_docs)
    for doc in res_exc_id_only:
        assert "_id" not in doc
        assert "product_name" in doc
        assert "meta" in doc

    # 4. Nested field exclusion
    res_exc_nested = store.aggregate([{"$project": {"meta.score": 0}}])
    assert len(res_exc_nested) == len(sample_docs)
    for doc in res_exc_nested:
        assert "_id" in doc
        assert "tokens" in doc["meta"]
        assert "score" not in doc["meta"]


def test_local_store_aggregation_sort_heterogeneous_types():
    """
    Validates $sort on fields with heterogeneous types across documents:
    Ensures well-defined priority sorting without raising TypeError between
    incompatible types (None, numeric, string, objects).
    """
    store = LocalJsonDocumentStore(in_memory=True)
    store.connect()

    hetero_docs = [
        {"_id": "D1", "val": 100, "sec": 1},
        {"_id": "D2", "val": "zebra", "sec": 2},
        {"_id": "D3", "val": None, "sec": 3},
        {"_id": "D4", "missing_val": True, "sec": 4},
        {"_id": "D5", "val": 25.5, "sec": 5},
        {"_id": "D6", "val": True, "sec": 6},  # bool -> numeric 1.0
        {"_id": "D7", "val": "apple", "sec": 7},
        {"_id": "D8", "val": {"tag": "complex"}, "sec": 8},
    ]
    store.insert_many(hetero_docs)

    # 1. Ascending sort on heterogeneous field
    asc_results = store.aggregate([{"$sort": {"val": 1}}])
    assert len(asc_results) == 8
    asc_ids = [d["_id"] for d in asc_results]

    # Priority 0: D3 (None), D4 (missing)
    assert set(asc_ids[0:2]) == {"D3", "D4"}
    # Priority 1: D6 (True/1.0), D5 (25.5), D1 (100)
    assert asc_ids[2:5] == ["D6", "D5", "D1"]
    # Priority 2: D7 ("apple"), D2 ("zebra")
    assert asc_ids[5:7] == ["D7", "D2"]
    # Priority 3: D8 (dict)
    assert asc_ids[7] == "D8"

    # 2. Descending sort on heterogeneous field
    desc_results = store.aggregate([{"$sort": {"val": -1}}])
    assert len(desc_results) == 8
    desc_ids = [d["_id"] for d in desc_results]

    # Reversed: Priority 3 first, then Priority 2, then Priority 1, then Priority 0
    assert desc_ids[0] == "D8"
    assert desc_ids[1:3] == ["D2", "D7"]
    assert desc_ids[3:6] == ["D1", "D5", "D6"]
    assert set(desc_ids[6:8]) == {"D3", "D4"}

    # 3. Multi-key sort with heterogeneous types
    multi_results = store.aggregate([{"$sort": {"val": 1, "sec": -1}}])
    assert len(multi_results) == 8
    # For D3 and D4 (both Priority 0), sec=-1 determines order (D4 sec=4 > D3 sec=3)
    assert multi_results[0]["_id"] == "D4"
    assert multi_results[1]["_id"] == "D3"

