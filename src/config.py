"""
Global Configuration & Constants for BDA Sentiment Analysis System.
Covers Paths, Schema Definitions, Dataset Tiers, and Syllabus Mapping (Units 1-6).
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Project Directories & Base Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.getenv("BDA_PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATA_RAW_DIR = RAW_DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATA_PROCESSED_DIR = PROCESSED_DATA_DIR
WAREHOUSE_DIR = DATA_DIR / "warehouse"
STORAGE_DIR = DATA_DIR / "storage"
HDFS_DATA_DIR = STORAGE_DIR / "hdfs"

REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALIZATIONS_DIR = REPORTS_DIR / "visualizations"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"

MODELS_DIR = PROJECT_ROOT / "models"
WEB_DIR = PROJECT_ROOT / "web"

# ---------------------------------------------------------------------------
# Dataset Tiers & Generation Constants
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42

TIER_A_COUNT = 100        # Tier A: Fast CI & Sample Reviews
TIER_B_COUNT = 5000       # Tier B: Standalone Pipeline Execution (3,000-5,000)
TIER_C_COUNT = 25000      # Tier C: Big Data Scaling Benchmarks

TIER_COUNTS: Dict[str, int] = {
    "a": TIER_A_COUNT,
    "b": TIER_B_COUNT,
    "c": TIER_C_COUNT,
    "sample": TIER_A_COUNT,
    "medium": TIER_B_COUNT,
    "benchmark": TIER_C_COUNT,
}

RAW_TIER_A_CSV = RAW_DATA_DIR / "reviews_tier_a.csv"
RAW_TIER_B_CSV = RAW_DATA_DIR / "reviews_tier_b.csv"
RAW_TIER_C_CSV = RAW_DATA_DIR / "reviews_tier_c.csv"
RAW_SAMPLE_CSV = RAW_DATA_DIR / "reviews_sample.csv"
RAW_MEDIUM_CSV = RAW_DATA_DIR / "reviews_medium.csv"

PROCESSED_TIER_A_CSV = PROCESSED_DATA_DIR / "cleaned_reviews_tier_a.csv"
PROCESSED_TIER_B_CSV = PROCESSED_DATA_DIR / "cleaned_reviews_tier_b.csv"
PROCESSED_TIER_C_CSV = PROCESSED_DATA_DIR / "cleaned_reviews_tier_c.csv"
PROCESSED_SAMPLE_CSV = PROCESSED_DATA_DIR / "cleaned_reviews.csv"

# ---------------------------------------------------------------------------
# Schema Definitions (16-Field Amazon Schema)
# ---------------------------------------------------------------------------
SCHEMA_FIELDS: List[str] = [
    "review_id",
    "reviewer_id",
    "reviewer_name",
    "product_id",
    "product_name",
    "product_category",
    "rating",
    "verified_purchase",
    "review_summary",
    "review_text",
    "review_date",
    "timestamp",
    "helpful_votes",
    "total_votes",
    "sentiment_label",
    "sentiment_numeric",
]

FIELD_TYPES: Dict[str, type] = {
    "review_id": str,
    "reviewer_id": str,
    "reviewer_name": str,
    "product_id": str,
    "product_name": str,
    "product_category": str,
    "rating": float,
    "verified_purchase": bool,
    "review_summary": str,
    "review_text": str,
    "review_date": str,
    "timestamp": int,
    "helpful_votes": int,
    "total_votes": int,
    "sentiment_label": str,
    "sentiment_numeric": float,
}

CATEGORIES: List[str] = [
    "Electronics",
    "Books",
    "Home & Kitchen",
    "Beauty & Personal Care",
    "Sports & Outdoors",
]
PRODUCT_CATEGORIES = CATEGORIES

SENTIMENT_MAP: Dict[float, tuple[str, float]] = {
    5.0: ("positive", 2.0),
    4.0: ("positive", 2.0),
    3.0: ("neutral", 1.0),
    2.0: ("negative", 0.0),
    1.0: ("negative", 0.0),
}

SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# HDFS Simulation (Unit 1)
# ---------------------------------------------------------------------------
HDFS_BLOCK_SIZE_MB = 128
HDFS_BLOCK_SIZE_BYTES = 128 * 1024 * 1024  # 134,217,728 bytes
HDFS_REPLICATION_FACTOR = 3

# ---------------------------------------------------------------------------
# Document Store (Unit 3)
# ---------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "bda_reviews_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "reviews")
LOCAL_DOCSTORE_PATH = DATA_DIR / "local_docstore.json"

# ---------------------------------------------------------------------------
# Spark & Distributed Engine (Unit 4 & 5)
# ---------------------------------------------------------------------------
SPARK_APP_NAME = "BDASentimentAnalysis"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# ---------------------------------------------------------------------------
# Web Dashboard (Unit 6 & R4)
# ---------------------------------------------------------------------------
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "5000"))

# ---------------------------------------------------------------------------
# Syllabus Topic & Course Outcome Traceability (Units 1-6, CO3/CO4)
# ---------------------------------------------------------------------------
SYLLABUS_MAPPING: Dict[str, Dict[str, Any]] = {
    "Unit 1": {
        "title": "Introduction to Big Data & HDFS Architecture",
        "module": "src.storage.hdfs_simulation",
        "concepts": [
            "HDFS Architecture",
            "128MB Block Partitioning",
            "Replication Factor",
            "NameNode/DataNode Simulation",
            "Big Data 5Vs Benchmark",
        ],
        "co": "CO3",
    },
    "Unit 2": {
        "title": "MapReduce Distributed Processing Paradigm",
        "module": "src.mapreduce.sentiment_mr",
        "concepts": [
            "ReviewTokenizerMapper",
            "SentimentCombiner",
            "SentimentReducer",
            "Key-Value Shuffling",
            "In-Memory Combiner Optimization",
        ],
        "co": "CO3",
    },
    "Unit 3": {
        "title": "NoSQL Document Stores & Aggregation Pipelines",
        "module": "src.database",
        "concepts": [
            "IDocumentStore Interface",
            "MongoDocumentStore",
            "LocalJsonDocumentStore Fallback",
            "BSON Schema Modeling",
            "Aggregation Pipelines ($match, $group, $sort)",
        ],
        "co": "CO3",
    },
    "Unit 4": {
        "title": "Apache Spark Core (RDDs) & Spark SQL",
        "module": "src.spark",
        "concepts": [
            "RDD DAG Lineage & toDebugString()",
            "Narrow vs Wide Transformations",
            "DataFrame Catalyst Optimizer explain()",
            "Temporary Views & Structured SQL",
        ],
        "co": "CO3",
    },
    "Unit 5": {
        "title": "Spark MLlib Sentiment Classification Pipeline",
        "module": "src.ml",
        "concepts": [
            "RegexTokenizer",
            "StopWordsRemover",
            "HashingTF & IDF",
            "Logistic Regression / Naive Bayes / Random Forest",
            "Multiclass Metrics & Confusion Matrix",
        ],
        "co": "CO4",
    },
    "Unit 6": {
        "title": "Graph Analytics & Interactive Visualization Dashboards",
        "module": "src.graph, src.visualization, web",
        "concepts": [
            "Bipartite Reviewer-Product Network",
            "Co-Review Projection",
            "PageRank & Degree Centrality",
            "8 Publication-Grade Visual Plots",
            "Flask Interactive Dashboard",
        ],
        "co": "CO4",
    },
}

# ---------------------------------------------------------------------------
# Environment Inspection Helpers
# ---------------------------------------------------------------------------
def is_java_available() -> bool:
    """Check if Java runtime is available on the system PATH."""
    return shutil.which("java") is not None

def is_spark_available() -> bool:
    """Check if PySpark is installed and Java is present."""
    if not is_java_available():
        return False
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False

def is_mongo_available() -> bool:
    """Check if PyMongo is installed and live server responds."""
    try:
        import pymongo
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=500)
        client.server_info()
        return True
    except Exception:
        return False

def ensure_directories() -> None:
    """Create all required project runtime directories."""
    for directory in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        WAREHOUSE_DIR,
        STORAGE_DIR,
        HDFS_DATA_DIR,
        REPORTS_DIR,
        VISUALIZATIONS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
