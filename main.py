"""
Master Pipeline Runner & CLI Orchestrator.
File: d:/Projects/BDA/main.py
Syllabus Mapping: Units 1 to 6 (End-to-End Big Data Architecture & Management)
Course Outcomes: CO3 (Big Data Processing) & CO4 (Machine Learning, Graph Analytics, Visualization)

Executes all pipeline stages:
1. Synthetic Data Generation (Amazon 16-field schema)
2. HDFS Distributed Storage & Block Replication Simulation
3. Text Cleaning & NLP Normalization
4. MapReduce Token Polarity Processing (Map, Shuffle-Sort, Reduce)
5. Document Store Ingestion (MongoDB / Local JSON Store)
6. Spark Core RDDs & Spark SQL Analytics
7. Spark MLlib Sentiment Classification & Model Evaluation
8. Graph Analytics (Bipartite Network, PageRank, Centrality)
9. Publication-Grade Visualization Artifact Generation
10. Web Dashboard Service Launch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    ensure_directories,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("BDA_Pipeline")


def run_stage_data_generation(sample_size: int = 100) -> Path:
    """Stage 1: Generate realistic Amazon product review dataset (Unit 1)."""
    logger.info(f"=== Stage 1: Data Generation (Size: {sample_size}) ===")
    from src.data.review_generator import generate_reviews
    out_path = DATA_RAW_DIR / "reviews_sample.csv"
    generate_reviews(num_records=sample_size, seed=42, output_format="csv", output_path=out_path)
    logger.info(f"Generated {sample_size} records at {out_path}")
    return out_path


def run_stage_hdfs_simulation() -> Any:
    """Stage 2: HDFS Distributed Storage Simulation (Unit 1 & 2)."""
    logger.info("=== Stage 2: HDFS Distributed Storage Simulation ===")
    from src.storage.hdfs_simulation import HDFSSimulator

    simulator = HDFSSimulator(block_size_mb=64, replication_factor=3)
    sample_csv = DATA_RAW_DIR / "reviews_sample.csv"
    if not sample_csv.exists():
        sample_csv = run_stage_data_generation(100)

    blocks = simulator.write_file(source_path=sample_csv, hdfs_path="/data/raw/reviews.csv")
    report = simulator.get_block_report()
    num_dns = len(report.get("datanodes", []))
    logger.info(
        f"HDFS Write Complete: {len(blocks)} blocks stored with 3x replication across {num_dns} DataNodes."
    )
    return report


def run_stage_cleaning(raw_csv_path: Path) -> Path:
    """Stage 3: Text Normalization and Cleaning (Unit 4 Spark/Pandas)."""
    logger.info("=== Stage 3: Data Cleaning & NLP Preprocessing ===")
    from src.data.data_cleaner import clean_dataframe
    import pandas as pd

    raw_df = pd.read_csv(raw_csv_path)
    cleaned_df = clean_dataframe(raw_df)
    out_csv = DATA_PROCESSED_DIR / "cleaned_reviews.csv"
    cleaned_df.to_csv(out_csv, index=False)
    out_parquet = DATA_PROCESSED_DIR / "cleaned_reviews.parquet"
    cleaned_df.to_parquet(out_parquet, index=False)
    logger.info(f"Cleaned {len(cleaned_df)} reviews saved to {out_csv} and {out_parquet}")
    return out_csv


def run_stage_mapreduce(csv_path: Path) -> Dict[str, Any]:
    """Stage 4: MapReduce Token Polarity Aggregation (Unit 2 & 3)."""
    logger.info("=== Stage 4: Hadoop MapReduce Processing ===")
    from src.mapreduce.sentiment_mr import run_mapreduce_job
    import pandas as pd

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    mr_results = run_mapreduce_job(records, num_reducers=2)
    logger.info(f"MapReduce Execution Finished: Extracted {len(mr_results)} unique sentiment tokens.")
    return mr_results


def run_stage_document_store(csv_path: Path) -> int:
    """Stage 5: Document Store Ingestion & Querying (Unit 4 MongoDB/LocalStore)."""
    logger.info("=== Stage 5: Document Store (MongoDB / LocalStore) ===")
    from src.database.document_store import get_document_store
    import pandas as pd

    store = get_document_store(prefer_mongo=False)
    store.connect()
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    store.insert_many(records)
    total_docs = store.count_documents({})
    logger.info(f"Document store populated with {total_docs} documents.")
    store.close()
    return total_docs


def run_stage_spark_analytics(csv_path: Path) -> Dict[str, Any]:
    """Stage 6: Apache Spark Core RDDs & Spark SQL Analytics (Unit 4)."""
    logger.info("=== Stage 6: Spark Core & Spark SQL Analytics ===")
    from src.spark_session import SparkSessionFactory
    from src.spark.spark_core_rdd import SparkCoreAnalytics
    from src.spark.spark_sql_analytics import execute_spark_sql_queries
    import pandas as pd

    spark = SparkSessionFactory.get_or_create("BDA_Master_Pipeline")
    df = pd.read_csv(csv_path)
    spark_df = spark.createDataFrame(df)
    core_analytics = SparkCoreAnalytics()
    rdd_counts = core_analytics.rdd_word_sentiment_count(spark_df)

    sql_results = execute_spark_sql_queries(spark_df)
    logger.info("Spark Core RDD transformations and Catalyst SQL queries completed successfully.")
    return {"rdd": rdd_counts, "sql": sql_results}


def run_stage_ml_pipeline(csv_path: Path) -> Dict[str, Any]:
    """Stage 7: Spark MLlib Sentiment Classification & Metrics Evaluation (Unit 5)."""
    logger.info("=== Stage 7: Spark MLlib Classification & Model Evaluation ===")
    from src.ml.spark_mllib_pipeline import SentimentMLlibPipeline
    from src.ml.evaluator import export_metrics_json
    import pandas as pd

    pipeline = SentimentMLlibPipeline()
    df = pd.read_csv(csv_path)

    # Train classifier models
    model = pipeline.train(df, classifier_type="logistic_regression")
    eval_metrics = pipeline.evaluate(model, df)

    # Export metrics.json
    export_metrics_json(eval_metrics, REPORTS_DIR / "metrics.json")
    pipeline.save_model(model, MODELS_DIR / "mllib_classifier")
    logger.info(f"MLlib Model evaluation complete. Metrics saved to {REPORTS_DIR / 'metrics.json'}")
    return eval_metrics


def run_stage_graph_analytics(csv_path: Path) -> Dict[str, Any]:
    """Stage 8: Graph Analytics, PageRank, and Centrality (Unit 5 & 6)."""
    logger.info("=== Stage 8: Graph Analytics & Bipartite Network Modeling ===")
    from src.graph.review_graph_analytics import ReviewGraphAnalytics
    import pandas as pd

    df = pd.read_csv(csv_path)
    analytics = ReviewGraphAnalytics()
    G = analytics.build_bipartite_graph(df)
    pagerank = analytics.compute_product_pagerank(G)
    centrality = analytics.compute_reviewer_centrality(G)
    neg_sub = analytics.extract_negative_sentiment_subgraph(G)

    logger.info(
        f"Graph Analytics Complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"{neg_sub.number_of_edges()} negative interactions identified."
    )
    return {"pagerank": pagerank, "centrality": centrality}


def run_stage_visualizations(csv_path: Path) -> List[Path]:
    """Stage 9: Publication-Quality Visual Artifact Generation (Unit 5 & 6)."""
    logger.info("=== Stage 9: Visual Artifacts & Dashboard Plots ===")
    from src.visualization.plot_generator import generate_all_plots

    plots = generate_all_plots(data_path=csv_path, output_dir=VISUALIZATIONS_DIR)
    logger.info(f"Generated {len(plots)} visualization plots in {VISUALIZATIONS_DIR}")
    return plots


def run_full_pipeline(sample_size: int = 100) -> None:
    """Executes the entire end-to-end Big Data Sentiment Analysis pipeline."""
    start_time = time.time()
    ensure_directories()

    # 1. Data Generation
    raw_csv = run_stage_data_generation(sample_size)

    # 2. HDFS Simulation
    run_stage_hdfs_simulation()

    # 3. Data Cleaning
    cleaned_csv = run_stage_cleaning(raw_csv)

    # 4. MapReduce
    run_stage_mapreduce(cleaned_csv)

    # 5. Document Store
    run_stage_document_store(cleaned_csv)

    # 6. Spark Analytics
    run_stage_spark_analytics(cleaned_csv)

    # 7. MLlib Machine Learning Pipeline
    run_stage_ml_pipeline(cleaned_csv)

    # 8. Graph Analytics
    run_stage_graph_analytics(cleaned_csv)

    # 9. Visualizations
    plots = run_stage_visualizations(cleaned_csv)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print("SUCCESS: Full Big Data Sentiment Analysis Pipeline Completed Successfully!")
    print(f"Elapsed Time: {elapsed}s | Generated Artifacts: {len(plots)} Plots | Metrics: reports/metrics.json")
    print("Mapped to Syllabus Units 1–6 and Course Outcomes CO3 & CO4.")
    print("=" * 70 + "\n")
    print("Done")


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="Big Data Analysis (BDA) - Product Review Sentiment Analysis Orchestrator"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "generate", "clean", "mapreduce", "spark", "ml", "graph", "visualize", "mongo", "web"],
        default="full",
        help="Pipeline execution mode (default: full)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of review records to generate or process (default: 100)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for Flask web dashboard (default: 5000)",
    )

    args = parser.parse_args()

    ensure_directories()
    sample_csv = DATA_PROCESSED_DIR / "cleaned_reviews.csv"
    if not sample_csv.exists():
        sample_csv = DATA_RAW_DIR / "reviews_sample.csv"

    if args.mode == "full":
        run_full_pipeline(sample_size=args.sample_size)
    elif args.mode == "generate":
        run_stage_data_generation(args.sample_size)
        print("Done")
    elif args.mode == "clean":
        raw_csv = DATA_RAW_DIR / "reviews_sample.csv"
        run_stage_cleaning(raw_csv)
        print("Done")
    elif args.mode == "mapreduce":
        run_stage_mapreduce(sample_csv)
        print("Done")
    elif args.mode == "spark":
        run_stage_spark_analytics(sample_csv)
        print("Done")
    elif args.mode == "ml":
        run_stage_ml_pipeline(sample_csv)
        print("Done")
    elif args.mode == "graph":
        run_stage_graph_analytics(sample_csv)
        print("Done")
    elif args.mode == "visualize":
        run_stage_visualizations(sample_csv)
        print("Done")
    elif args.mode == "mongo":
        from src.database.mongo_pipeline import run_mongo_demo
        run_mongo_demo(sample_size=args.sample_size)
        print("Done")
    elif args.mode == "web":
        from web.app import app
        print(f"Starting Web Dashboard at http://localhost:{args.port} ...")
        app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
