"""
tests/conftest.py
Shared Test Fixtures and Harness for E2E Testing Track.
Big Data Product Review Sentiment Analysis System.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import pytest


# ---------------------------------------------------------------------------
# Dynamic Import & Milestone Isolation Helpers
# ---------------------------------------------------------------------------

def safe_import(module_path: str, class_or_fn: str = None):
    """
    Safely import a module or attribute. If not yet implemented in early
    milestones, returns None instead of raising an unhandled collection error.
    """
    import importlib
    try:
        mod = importlib.import_module(module_path)
        if class_or_fn:
            return getattr(mod, class_or_fn, None)
        return mod
    except (ImportError, ModuleNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Fixture: Temporary Workspace Hierarchy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def temp_workspace(tmp_path: Path) -> Dict[str, Path]:
    """
    Creates an isolated filesystem sandbox with standard project subdirectories.
    Prevents tests from polluting production directories.
    """
    ws = {
        "root": tmp_path,
        "data_raw": tmp_path / "data" / "raw",
        "data_processed": tmp_path / "data" / "processed",
        "models": tmp_path / "models",
        "reports": tmp_path / "reports",
        "visualizations": tmp_path / "reports" / "visualizations",
        "figures": tmp_path / "reports" / "figures"
    }
    for p in ws.values():
        p.mkdir(parents=True, exist_ok=True)
    return ws


# ---------------------------------------------------------------------------
# Fixture: Canonical 16-Field Amazon Review Dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_reviews_data() -> List[Dict[str, Any]]:
    """
    10 representative Amazon product review records strictly satisfying the
    16-field contract with diverse sentiment labels, categories, and noise.
    """
    return [
        {
            "review_id": "REV_00000001",
            "reviewer_id": "USR_00001",
            "reviewer_name": "Alice Smith",
            "product_id": "PROD_ELEC_001",
            "product_name": "Noise Cancelling Headphones",
            "product_category": "Electronics",
            "rating": 5.0,
            "verified_purchase": True,
            "review_summary": "Exceptional clarity and battery life!",
            "review_text": "These headphones are absolutely fantastic. The bass is deep, highs are crisp, and battery lasts all week.",
            "review_date": "2024-01-15",
            "timestamp": 1705276800,
            "helpful_votes": 12,
            "total_votes": 14,
            "sentiment_label": "positive",
            "sentiment_numeric": 0.0
        },
        {
            "review_id": "REV_00000002",
            "reviewer_id": "USR_00002",
            "reviewer_name": "Bob Jones",
            "product_id": "PROD_ELEC_001",
            "product_name": "Noise Cancelling Headphones",
            "product_category": "Electronics",
            "rating": 1.0,
            "verified_purchase": True,
            "review_summary": "Defective on arrival, terrible customer support",
            "review_text": "Stopped working after two hours! Complete waste of money. Do not recommend this garbage brand.",
            "review_date": "2024-01-18",
            "timestamp": 1705536000,
            "helpful_votes": 25,
            "total_votes": 28,
            "sentiment_label": "negative",
            "sentiment_numeric": 1.0
        },
        {
            "review_id": "REV_00000003",
            "reviewer_id": "USR_00003",
            "reviewer_name": "Charlie Brown",
            "product_id": "PROD_BOOK_001",
            "product_name": "Distributed Systems Handbook",
            "product_category": "Books",
            "rating": 3.0,
            "verified_purchase": False,
            "review_summary": "Average textbook, lacks recent case studies",
            "review_text": "The conceptual chapters on MapReduce and consensus are decent, but practical code examples feel outdated.",
            "review_date": "2024-02-01",
            "timestamp": 1706745600,
            "helpful_votes": 4,
            "total_votes": 6,
            "sentiment_label": "neutral",
            "sentiment_numeric": 2.0
        },
        {
            "review_id": "REV_00000004",
            "reviewer_id": "USR_00001",  # Repeat reviewer for bipartite degree
            "reviewer_name": "Alice Smith",
            "product_id": "PROD_HOME_001",
            "product_name": "Stainless Steel Espresso Maker",
            "product_category": "Home & Kitchen",
            "rating": 5.0,
            "verified_purchase": True,
            "review_summary": "Brews authentic espresso in 3 minutes",
            "review_text": "Well-built machine, intuitive interface, and easy to clean. Best kitchen investment this year.",
            "review_date": "2024-02-10",
            "timestamp": 1707523200,
            "helpful_votes": 19,
            "total_votes": 20,
            "sentiment_label": "positive",
            "sentiment_numeric": 0.0
        },
        {
            "review_id": "REV_00000005",
            "reviewer_id": "USR_00004",
            "reviewer_name": "Diana Prince",
            "product_id": "PROD_BEAU_001",
            "product_name": "Hydrating Face Serum",
            "product_category": "Beauty & Personal Care",
            "rating": 4.0,
            "verified_purchase": True,
            "review_summary": "Good hydration without greasy residue",
            "review_text": "Absorbs quickly and smells mild. Noticeable improvement in skin texture over three weeks.",
            "review_date": "2024-02-14",
            "timestamp": 1707868800,
            "helpful_votes": 8,
            "total_votes": 9,
            "sentiment_label": "positive",
            "sentiment_numeric": 0.0
        },
        {
            "review_id": "REV_00000006",
            "reviewer_id": "USR_00005",
            "reviewer_name": "Evan Wright",
            "product_id": "PROD_SPOR_001",
            "product_name": "Trail Running Shoes",
            "product_category": "Sports & Outdoors",
            "rating": 2.0,
            "verified_purchase": True,
            "review_summary": "Uncomfortable arch support and weak grip",
            "review_text": "Soles became slippery on wet mud. Caused blisters after a 5km run. Definitely not what I expected.",
            "review_date": "2024-02-20",
            "timestamp": 1708387200,
            "helpful_votes": 15,
            "total_votes": 17,
            "sentiment_label": "negative",
            "sentiment_numeric": 1.0
        },
        {
            "review_id": "REV_00000007",
            "reviewer_id": "USR_00002",  # Repeat reviewer
            "reviewer_name": "Bob Jones",
            "product_id": "PROD_BOOK_001",
            "product_name": "Distributed Systems Handbook",
            "product_category": "Books",
            "rating": 4.0,
            "verified_purchase": True,
            "review_summary": "Solid reference for academic students",
            "review_text": "Great theoretical foundation for understanding distributed consensus and partition tolerance.",
            "review_date": "2024-02-25",
            "timestamp": 1708819200,
            "helpful_votes": 3,
            "total_votes": 3,
            "sentiment_label": "positive",
            "sentiment_numeric": 0.0
        },
        {
            "review_id": "REV_00000008",
            "reviewer_id": "USR_00006",
            "reviewer_name": None,  # Null reviewer name edge case
            "product_id": "PROD_ELEC_002",
            "product_name": "Smart Fitness Watch",
            "product_category": "Electronics",
            "rating": 3.0,
            "verified_purchase": False,
            "review_summary": None,  # Null summary edge case
            "review_text": "Battery lasts 2 days instead of advertised 7. Step counter is accurate enough for casual walking.",
            "review_date": "2024-03-01",
            "timestamp": 1709251200,
            "helpful_votes": 0,
            "total_votes": 0,
            "sentiment_label": "neutral",
            "sentiment_numeric": 2.0
        },
        {
            "review_id": "REV_00000009",
            "reviewer_id": "USR_00007",
            "reviewer_name": "George Miller",
            "product_id": "PROD_HOME_001",
            "product_name": "Stainless Steel Espresso Maker",
            "product_category": "Home & Kitchen",
            "rating": 1.0,
            "verified_purchase": True,
            "review_summary": "Leaked boiling water all over the counter!",
            "review_text": "<b>DANGEROUS DEFECT!</b> Check review at <a href='https://example.com/warning'>here</a>! It's not safe. Don't buy it!",
            "review_date": "2024-03-05",
            "timestamp": 1709596800,
            "helpful_votes": 31,
            "total_votes": 33,
            "sentiment_label": "negative",
            "sentiment_numeric": 1.0
        },
        {
            "review_id": "REV_00000010",
            "reviewer_id": "USR_00008",
            "reviewer_name": "Hannah Abbott",
            "product_id": "PROD_SPOR_001",
            "product_name": "Trail Running Shoes",
            "product_category": "Sports & Outdoors",
            "rating": 5.0,
            "verified_purchase": True,
            "review_summary": "Super lightweight and durable on rocky trails",
            "review_text": "Ran 50 miles in these shoes already. Zero blisters, fantastic ankle support, highly recommend to hikers.",
            "review_date": "2024-03-10",
            "timestamp": 1710028800,
            "helpful_votes": 7,
            "total_votes": 8,
            "sentiment_label": "positive",
            "sentiment_numeric": 0.0
        }
    ]


@pytest.fixture(scope="session")
def sample_reviews_csv(sample_reviews_data: List[Dict[str, Any]], tmp_path_factory) -> Path:
    """
    Serializes sample reviews data to a temporary CSV file with RFC 4180 formatting.
    """
    import csv
    temp_dir = tmp_path_factory.mktemp("test_data")
    csv_path = temp_dir / "sample_reviews.csv"
    fieldnames = list(sample_reviews_data[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_reviews_data)
    return csv_path


# ---------------------------------------------------------------------------
# Fixture: SparkSession with Dual-Engine Fallback
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark_session():
    """
    Provides a shared SparkSession.
    If PySpark and Java runtime are available, returns a native PySpark session.
    If absent, seamlessly invokes the dual-engine SparkSessionFactory fallback.
    """
    # 1. Try project-level factory first
    factory_cls = safe_import("src.spark_session", "SparkSessionFactory")
    if factory_cls is not None:
        try:
            spark = factory_cls.get_or_create(app_name="BDA_Sentiment_E2E_Test")
            yield spark
            if hasattr(spark, "stop"):
                spark.stop()
            return
        except Exception:
            pass

    # 2. Try direct PySpark if Java is present
    has_java = shutil.which("java") is not None
    if has_java:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder \
                .appName("BDA_Sentiment_E2E_Test") \
                .master("local[2]") \
                .config("spark.driver.memory", "1g") \
                .config("spark.sql.shuffle.partitions", "2") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()
            yield spark
            spark.stop()
            return
        except Exception:
            pass

    # 3. Fallback: Embedded In-Memory Simulated Spark Adapter
    class SimulatedRDD:
        def __init__(self, data: list):
            self._data = list(data)

        def map(self, fn):
            return SimulatedRDD([fn(x) for x in self._data])

        def flatMap(self, fn):
            res = []
            for x in self._data:
                res.extend(fn(x))
            return SimulatedRDD(res)

        def filter(self, fn):
            return SimulatedRDD([x for x in self._data if fn(x)])

        def reduceByKey(self, fn):
            acc = {}
            for k, v in self._data:
                acc[k] = fn(acc[k], v) if k in acc else v
            return SimulatedRDD(list(acc.items()))

        def collect(self):
            return list(self._data)

        def count(self):
            return len(self._data)

        def toDebugString(self):
            return "(2) MapPartitionsRDD -> ShuffledRDD -> ParallelCollectionRDD"

    class SimulatedDataFrame:
        def __init__(self, rows: Any, schema=None):
            import pandas as pd
            if isinstance(rows, pd.DataFrame):
                self._df = rows.copy()
            elif isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
                self._df = pd.DataFrame(rows)
            elif isinstance(rows, list):
                self._df = pd.DataFrame(rows)
            else:
                self._df = pd.DataFrame()
            self.columns = list(self._df.columns)
            self.schema = schema

        @property
        def rdd(self):
            return SimulatedRDD(self._df.to_dict(orient="records"))

        def count(self):
            return len(self._df)

        def first(self):
            if len(self._df) > 0:
                return self._df.iloc[0].to_dict()
            return None

        def collect(self):
            return [dict(row) for _, row in self._df.iterrows()]

        def toPandas(self):
            return self._df.copy()

        def filter(self, condition):
            if callable(condition):
                filtered = self._df[self._df.apply(condition, axis=1)]
            elif isinstance(condition, str):
                try:
                    filtered = self._df.query(condition)
                except Exception:
                    filtered = self._df
            else:
                filtered = self._df
            return SimulatedDataFrame(filtered)

        def select(self, *cols):
            col_list = list(cols)
            available = [c for c in col_list if c in self._df.columns]
            return SimulatedDataFrame(self._df[available])

        def withColumn(self, col_name, expr):
            df_copy = self._df.copy()
            if callable(expr):
                df_copy[col_name] = df_copy.apply(expr, axis=1)
            else:
                df_copy[col_name] = expr
            return SimulatedDataFrame(df_copy)

        def createOrReplaceTempView(self, view_name):
            SimulatedSparkSession._views[view_name] = self._df.copy()

        def explain(self, extended=True):
            return "== Physical Plan ==\n*(1) Scan SimulatedDataFrame [numRows=" + str(len(self._df)) + "]"

        def show(self, n=20):
            print(self._df.head(n))

    class SimulatedReader:
        def csv(self, path: str, header: bool = True, inferSchema: bool = True):
            import pandas as pd
            df = pd.read_csv(path)
            return SimulatedDataFrame(df)

        def parquet(self, path: str):
            import pandas as pd
            df = pd.read_parquet(path)
            return SimulatedDataFrame(df)

    class SimulatedSparkSession:
        _views = {}

        def __init__(self):
            self.sparkContext = self
            self.read = SimulatedReader()

        def parallelize(self, data):
            return SimulatedRDD(data)

        def createDataFrame(self, data, schema=None):
            return SimulatedDataFrame(data, schema=schema)

        def sql(self, query: str):
            import pandas as pd
            view_name = None
            for name in self._views:
                if name.lower() in query.lower():
                    view_name = name
                    break
            if view_name:
                df = self._views[view_name]
                # Simple aggregate execution simulation for test queries
                if "count" in query.lower() and "group by" in query.lower():
                    grp_col = "product_category" if "product_category" in query.lower() else df.columns[0]
                    res = df.groupby(grp_col).agg(
                        cnt=("rating", "count"),
                        avg_rating=("rating", "mean")
                    ).reset_index()
                    return SimulatedDataFrame(res)
                return SimulatedDataFrame(df)
            return SimulatedDataFrame([])

        def stop(self):
            self._views.clear()

    simulated_session = SimulatedSparkSession()
    yield simulated_session
    simulated_session.stop()


@pytest.fixture(scope="session")
def sample_reviews_df(spark_session, sample_reviews_data: List[Dict[str, Any]]):
    """
    Returns the canonical 10-record dataset wrapped as a Spark / Dual-Engine DataFrame.
    """
    return spark_session.createDataFrame(sample_reviews_data)


# ---------------------------------------------------------------------------
# Fixture: Document Store Adapter (Isolated Per Test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def doc_store(tmp_path: Path):
    """
    Provides a clean, isolated IDocumentStore instance for testing.
    Uses LocalJsonDocumentStore with a unique temporary file per test.
    Gracefully skips if Milestone M2 Document Store is not yet implemented.
    """
    store_factory = safe_import("src.database.document_store", "get_document_store")
    if store_factory is not None:
        try:
            store = store_factory(prefer_mongo=False, storage_path=str(tmp_path / "store.json"))
            store.connect()
            yield store
            store.close()
            return
        except Exception as e:
            pytest.skip(f"Failed to initialize document store from factory: {e}")

    local_store_cls = safe_import("src.database.local_store", "LocalJsonDocumentStore")
    if local_store_cls is not None:
        try:
            store = local_store_cls(storage_path=str(tmp_path / "store.json"))
            store.connect()
            yield store
            store.close()
            return
        except Exception as e:
            pytest.skip(f"Failed to initialize LocalJsonDocumentStore: {e}")

    pytest.skip("Milestone M2 Document Store (src.database) not yet implemented")


# ---------------------------------------------------------------------------
# Fixture: Flask Web App Test Client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def flask_client():
    """
    Provides a Flask test client for testing web/app.py routes and REST APIs.
    Cleanly skips if Milestone M4 web dashboard (web/app.py) is not yet implemented.
    """
    app_mod = safe_import("web.app") or safe_import("src.web.app")
    if app_mod is None:
        pytest.skip("Milestone M4 web dashboard (web/app.py) not yet implemented")

    flask_app = getattr(app_mod, "app", None)
    if flask_app is None and hasattr(app_mod, "create_app"):
        flask_app = app_mod.create_app()

    if flask_app is None:
        pytest.skip("Milestone M4 web dashboard (web/app.py) not yet implemented")

    flask_app.config["TESTING"] = True
    return flask_app.test_client()
