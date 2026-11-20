"""
MongoDB Ingestion, Indexing, and Database Extraction Pipeline.
Module: src.database.mongo_pipeline
Unit 4 / Big Data NoSQL Persistence & Database Extraction.

Demonstrates:
1. Ingestion of structured e-commerce product reviews into MongoDB.
2. Compound Indexing on category, rating, and sentiment.
3. Complex Query-based Extraction (filtering, projection, sorting).
4. MongoDB Aggregation Pipeline ($group, $sum, $avg, $cond).
5. ETL Extraction from MongoDB directly into Pandas and Spark DataFrames.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import pymongo
    from bson import ObjectId
except ImportError:
    pymongo = None
    ObjectId = None

try:
    import mongomock
except ImportError:
    mongomock = None

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MONGO_COLLECTION_NAME,
    MONGO_DB_NAME,
    MONGO_URI,
)

logger = logging.getLogger(__name__)


class MongoReviewPipeline:
    """
    Manages MongoDB lifecycle: Connection, Data Ingestion, Secondary Indexing,
    and Advanced Analytical Extraction.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI", MONGO_URI)
        self.db_name = db_name or MONGO_DB_NAME
        self.collection_name = collection_name or MONGO_COLLECTION_NAME
        self.client: Any = None
        self.db: Any = None
        self.collection: Any = None
        self.is_live: bool = False
        self._init_connection()

    def _init_connection(self) -> None:
        """Connects to live MongoDB instance or transparent in-memory mongomock fallback."""
        # 1. Attempt connection to live MongoDB server
        if pymongo is not None:
            try:
                live_client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=800)
                live_client.admin.command("ping")
                self.client = live_client
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                self.is_live = True
                logger.info(f"Connected to live MongoDB at {self.mongo_uri}")
                return
            except Exception as e:
                logger.debug(f"Live MongoDB unreachable at {self.mongo_uri}: {e}")

        # 2. Transparent fallback to mongomock for self-contained execution
        if mongomock is not None:
            mock_client = mongomock.MongoClient()
            self.client = mock_client
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.is_live = False
            logger.info("Using embedded MongoDB engine (mongomock) for database extraction demonstration.")
            return

        raise RuntimeError("Neither pymongo nor mongomock is available.")

    def store_reviews(
        self,
        data_source: Optional[Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Stores product reviews into MongoDB.
        Creates compound indexes and returns ingestion statistics.
        """
        records: List[Dict[str, Any]] = []

        if isinstance(data_source, list):
            records = data_source
        elif isinstance(data_source, pd.DataFrame):
            records = data_source.to_dict(orient="records")
        elif data_source and Path(data_source).exists():
            records = pd.read_csv(data_source).to_dict(orient="records")
        else:
            candidates = [
                DATA_PROCESSED_DIR / "cleaned_reviews.csv",
                DATA_RAW_DIR / "reviews_sample.csv",
                DATA_RAW_DIR / "reviews_medium.csv",
            ]
            for p in candidates:
                if p.exists():
                    records = pd.read_csv(p).to_dict(orient="records")
                    break

        if not records:
            from src.data.review_generator import generate_review_records
            records = generate_review_records(num_records=100, seed=42)

        if limit and limit > 0:
            records = records[:limit]

        # Clean records of any existing _id to avoid collision
        clean_records = []
        for r in records:
            d = dict(r)
            if "_id" in d:
                del d["_id"]
            clean_records.append(d)

        # Clear existing collection and insert batch
        self.collection.drop()
        insert_result = self.collection.insert_many(clean_records)

        # Create secondary compound indexes for performance
        try:
            self.collection.create_index([("product_category", 1), ("sentiment_label", 1)])
            self.collection.create_index([("rating", -1)])
            self.collection.create_index([("review_id", 1)], unique=True)
        except Exception as e:
            logger.debug(f"Index creation note: {e}")

        total_count = self.collection.count_documents({})
        sample_doc = self.collection.find_one({})
        if sample_doc and "_id" in sample_doc:
            sample_doc["_id"] = str(sample_doc["_id"])

        return {
            "status": "success",
            "database": self.db_name,
            "collection": self.collection_name,
            "engine": "Live MongoDB (daemon)" if self.is_live else "MongoDB In-Memory Engine (mongomock)",
            "inserted_count": len(insert_result.inserted_ids),
            "total_documents": total_count,
            "sample_document": sample_doc,
        }

    def extract_reviews(
        self,
        filter_query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort_by: Optional[List[Tuple[str, int]]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Extracts documents from MongoDB based on query filters and projections.
        Example: extract negative reviews with rating <= 2.0.
        """
        q = filter_query or {}
        proj = projection or {"_id": 1, "review_id": 1, "product_name": 1, "product_category": 1, "rating": 1, "sentiment_label": 1, "review_text": 1}
        sort_order = sort_by or [("rating", -1)]

        cursor = self.collection.find(q, proj).sort(sort_order).limit(limit)
        results = []
        for doc in cursor:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            results.append(doc)

        return results

    def extract_category_aggregates(self) -> List[Dict[str, Any]]:
        """
        Executes a multi-stage MongoDB Aggregation Pipeline ($group, $avg, $sum, $sort).
        Computes category-wise sentiment counts and average ratings.
        """
        pipeline = [
            {
                "$group": {
                    "_id": "$product_category",
                    "total_reviews": {"$sum": 1},
                    "avg_rating": {"$avg": "$rating"},
                    "positive_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$sentiment_label", "positive"]}, 1, 0]
                        }
                    },
                    "neutral_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$sentiment_label", "neutral"]}, 1, 0]
                        }
                    },
                    "negative_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$sentiment_label", "negative"]}, 1, 0]
                        }
                    },
                }
            },
            {"$sort": {"total_reviews": -1}},
        ]

        try:
            agg_results = list(self.collection.aggregate(pipeline))
        except Exception:
            # Fallback Python simulation for mock implementations lacking complex cond
            agg_results = []
            df = pd.DataFrame(list(self.collection.find({})))
            if not df.empty and "product_category" in df.columns:
                grouped = df.groupby("product_category")
                for cat, group in grouped:
                    pos = int((group["sentiment_label"] == "positive").sum())
                    neu = int((group["sentiment_label"] == "neutral").sum())
                    neg = int((group["sentiment_label"] == "negative").sum())
                    agg_results.append({
                        "_id": cat,
                        "total_reviews": len(group),
                        "avg_rating": round(float(group["rating"].mean()), 2),
                        "positive_count": pos,
                        "neutral_count": neu,
                        "negative_count": neg,
                    })
                agg_results.sort(key=lambda x: x["total_reviews"], reverse=True)

        for row in agg_results:
            if "avg_rating" in row and row["avg_rating"] is not None:
                row["avg_rating"] = round(float(row["avg_rating"]), 2)

        return agg_results

    def extract_to_dataframe(
        self,
        filter_query: Optional[Dict[str, Any]] = None,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Full ETL Database Extraction: Extracts MongoDB documents directly into
        a Pandas DataFrame and exports to CSV.
        """
        q = filter_query or {}
        cursor = self.collection.find(q)
        records = []
        for doc in cursor:
            d = dict(doc)
            if "_id" in d:
                d["_id"] = str(d["_id"])
            records.append(d)

        df = pd.DataFrame(records)
        if save_csv and not df.empty:
            out_file = DATA_PROCESSED_DIR / "mongo_extracted_reviews.csv"
            df.to_csv(out_file, index=False)
            logger.info(f"Exported {len(df)} MongoDB extracted records to {out_file}")

        return df

    def get_status(self) -> Dict[str, Any]:
        """Returns connection health and collection document counts."""
        count = self.collection.count_documents({}) if self.collection is not None else 0
        return {
            "connected": self.client is not None,
            "engine": "Live MongoDB Server" if self.is_live else "MongoDB In-Memory Engine (mongomock)",
            "uri": self.mongo_uri,
            "database": self.db_name,
            "collection": self.collection_name,
            "total_documents": count,
        }


def run_mongo_demo(sample_size: int = 100) -> Dict[str, Any]:
    """Demonstrates full MongoDB ingestion, extraction, and aggregation pipeline."""
    pipeline = MongoReviewPipeline()

    # Step 1: Store Data to MongoDB
    store_res = pipeline.store_reviews(limit=sample_size)

    # Step 2: Database Extraction - Filter Negative Reviews
    neg_reviews = pipeline.extract_reviews(
        filter_query={"sentiment_label": "negative"},
        limit=5,
    )

    # Step 3: Database Extraction - Aggregation Pipeline
    category_aggr = pipeline.extract_category_aggregates()

    # Step 4: ETL Extraction to DataFrame & CSV
    df = pipeline.extract_to_dataframe(save_csv=True)

    # Save aggregation report to JSON
    agg_file = DATA_PROCESSED_DIR / "mongo_category_aggregates.json"
    with open(agg_file, "w") as f:
        json.dump(category_aggr, f, indent=2)

    return {
        "storage": store_res,
        "extracted_negative_sample": neg_reviews,
        "category_aggregates": category_aggr,
        "extracted_dataframe_shape": df.shape,
    }


if __name__ == "__main__":
    demo = run_mongo_demo(sample_size=100)
    print("\n" + "=" * 70)
    print("MONGODB DATA INGESTION & EXTRACTION DEMO")
    print("=" * 70)
    print(f"Engine: {demo['storage']['engine']}")
    print(f"Stored Documents: {demo['storage']['total_documents']}")
    print(f"Sample Ingested Document ID: {demo['storage']['sample_document'].get('_id')}")
    print(f"Negative Reviews Extracted: {len(demo['extracted_negative_sample'])}")
    print(f"Categories Aggregated: {len(demo['category_aggregates'])}")
    print(f"Exported CSV: data/processed/mongo_extracted_reviews.csv")
    print("=" * 70 + "\n")
