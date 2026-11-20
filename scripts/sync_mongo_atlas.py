"""
MongoDB Atlas Cloud Synchronization & Distributed Storage Ingestion.
Script: scripts/sync_mongo_atlas.py
Enterprise Distributed Database Interconnection.

Features:
- Ingests review records directly to your cloud-hosted MongoDB Atlas cluster.
- Configures secondary & compound indexes for distributed query routing.
- Executes real-time MongoDB Aggregation Pipelines in the cloud.
- Extracts cloud data back into local analytical files (CSV, Parquet, JSON).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MONGO_COLLECTION_NAME,
    MONGO_DB_NAME,
    MONGO_URI,
)
from src.database.mongo_pipeline import MongoReviewPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MongoAtlasSync")


def sync_to_atlas(
    mongo_uri: Optional[str] = None,
    db_name: str = "bda_reviews_db",
    collection_name: str = "product_reviews",
    sample_size: int = 500,
) -> None:
    """Connects to MongoDB Atlas, ingests records, and runs database extraction."""
    target_uri = mongo_uri or os.getenv("MONGO_URI", MONGO_URI)

    print("\n" + "=" * 75)
    print("MONGODB ATLAS CLOUD SYNCHRONIZATION")
    print("=" * 75)
    print(f"Target Database:   {db_name}")
    print(f"Target Collection: {collection_name}")
    print(f"Sample Batch Size: {sample_size} records")
    print(f"Connecting to:     {target_uri.split('@')[-1] if '@' in target_uri else target_uri}")

    pipeline = MongoReviewPipeline(
        mongo_uri=target_uri,
        db_name=db_name,
        collection_name=collection_name,
    )

    print(f"Active Engine:     {pipeline.get_status()['engine']}")

    # 1. Store Data to MongoDB Atlas
    print("\n[Step 1/4] Ingesting reviews into cloud database...")
    store_result = pipeline.store_reviews(limit=sample_size)
    print(f"  --> Successfully stored {store_result['inserted_count']} documents with BSON ObjectIds!")
    print(f"  --> Sample Ingested Document ID: {store_result['sample_document'].get('_id')}")

    # 2. Extract Negative Sentiment Subset
    print("\n[Step 2/4] Query-Based Extraction (Critical Defect Subgraph)...")
    neg_reviews = pipeline.extract_reviews(
        filter_query={"sentiment_label": "negative"},
        limit=5,
    )
    print(f"  --> Extracted {len(neg_reviews)} negative review records matching query.")
    for doc in neg_reviews[:2]:
        print(f"      - [{doc.get('review_id')}] {doc.get('product_name')}: {doc.get('rating')} stars")

    # 3. Aggregation Pipeline
    print("\n[Step 3/4] Running Cloud Aggregation Pipeline ($group, $avg, $sum)...")
    aggregates = pipeline.extract_category_aggregates()
    print(f"  --> Aggregated {len(aggregates)} product categories:")
    for agg in aggregates:
        print(
            f"      - {agg.get('_id'):<22} | Total: {agg.get('total_reviews'):>3} | "
            f"Avg Rating: {agg.get('avg_rating')} | Pos: {agg.get('positive_count'):>3} | Neg: {agg.get('negative_count'):>2}"
        )

    # 4. Extract into local analytical artifacts
    print("\n[Step 4/4] Extracting cloud data into local DataFrames...")
    df = pipeline.extract_to_dataframe(save_csv=True)
    out_csv = DATA_PROCESSED_DIR / "mongo_extracted_reviews.csv"
    out_json = DATA_PROCESSED_DIR / "mongo_category_aggregates.json"
    with open(out_json, "w") as f:
        json.dump(aggregates, f, indent=2)

    print(f"  --> Saved extracted records to: {out_csv} ({len(df)} rows)")
    print(f"  --> Saved category aggregates to: {out_json}")

    print("\n" + "=" * 75)
    print("SUCCESS: Cloud MongoDB Synchronization Complete!")
    if pipeline.is_live:
        print("You can now view all stored documents and indexes directly in your")
        print("MongoDB Atlas Web Console at: https://cloud.mongodb.com/")
        print(f"Navigate to: Deployment -> Database -> Browse Collections -> {db_name}.{collection_name}")
    else:
        print("Note: Ingested to local/mock engine. To sync directly to your personal")
        print("MongoDB Atlas account, set your MONGO_URI in .env or pass --uri:")
        print("python scripts/sync_mongo_atlas.py --uri \"mongodb+srv://<user>:<password>@cluster0.xxx.mongodb.net/\"")
    print("=" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync BDA product reviews to MongoDB Atlas Cloud")
    parser.add_argument("--uri", type=str, default=None, help="MongoDB Atlas connection string")
    parser.add_argument("--db", type=str, default="bda_reviews_db", help="Database name")
    parser.add_argument("--collection", type=str, default="product_reviews", help="Collection name")
    parser.add_argument("--size", type=int, default=500, help="Number of records to sync")

    args = parser.parse_args()
    sync_to_atlas(
        mongo_uri=args.uri,
        db_name=args.db,
        collection_name=args.collection,
        sample_size=args.size,
    )


if __name__ == "__main__":
    main()
