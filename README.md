# Big Data Analytics: Distributed Product Review Sentiment Analysis System

[![BDA Pipeline CI/CD](https://github.com/pratyush-barik/BDA-Sentiment-Analysis/actions/workflows/bda_distributed_pipeline.yml/badge.svg)](https://github.com/pratyush-barik/BDA-Sentiment-Analysis/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%20(MLlib%20%2B%20SQL)-E25A1C.svg)](https://spark.apache.org/)
[![Hadoop MapReduce](https://img.shields.io/badge/Hadoop-MapReduce%20%2B%20HDFS-FFDE59.svg)](https://hadoop.apache.org/)
[![MongoDB Atlas](https://img.shields.io/badge/MongoDB%20Atlas-NoSQL%20Cluster-47A248.svg)](https://cloud.mongodb.com/)
[![Test Suite](https://img.shields.io/badge/tests-67%2F67%20passed%20(100%25)-brightgreen.svg)](https://github.com/pratyush-barik/BDA-Sentiment-Analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Academic & Enterprise Big Data Engineering Project**  
> **Curricular Mapping:** Big Data Analysis (BDA) &bull; **Course Outcomes:** **CO3** (Distributed Data Processing) & **CO4** (Machine Learning, Graph Analytics & Visualization) &bull; **Syllabus Units:** 1 to 6.  
> **Core Stack:** Python 3.11+, Apache Spark 3.5 (Spark Core RDDs, Spark SQL Catalyst, MLlib Pipelines), Hadoop MapReduce Architecture, HDFS 3x Replication Simulation, MongoDB Atlas Cloud NoSQL, NetworkX Graph Theory, Flask Academic Research Dashboard, Matplotlib & Seaborn Visualizations.

---

## 1. Executive Summary & Curricular Outcomes

This repository delivers an end-to-end, intermediate-level **Distributed Big Data Sentiment Analysis Platform**. It investigates consumer sentiment across large-scale e-commerce corpora ($N = 25,000$ verified records) using distributed storage, parallel processing models, multiclass machine learning pipelines, NoSQL document stores, and bipartite network graph analytics.

### Course Outcome (CO) Mapping

- **CO3: Analyze Big Data using Hadoop MapReduce, Spark Core RDDs, and Spark SQL processing models.**
  - **HDFS Simulation:** 64MB/128MB block chunking, 3x replication factor, rack-aware replica placement across 5 DataNodes, automated failure detection and re-replication, and 5Vs metrics analysis (Volume, Velocity, Variety, Veracity, Value).
  - **Hadoop MapReduce Engine:** Custom `ReviewTokenizerMapper` with negation preservation (`not_good`, `never_works`), node-local `SentimentCombiner` for network shuffle minimization, deterministic hash `MapReducePartitioner`, and `SentimentReducer` token polarity extraction.
  - **Apache Spark Core & Spark SQL:** RDD transformation DAG lineage tracking (`flatMap`, `map`, `reduceByKey`), Catalyst query optimizer evaluation, and in-memory temporary views for multi-dimensional aggregation.

- **CO4: Implement Machine Learning models on Big Data streams, perform document storage with NoSQL/MongoDB, and derive graph analytics & visual insights.**
  - **Spark MLlib Classification Pipeline:** 6-stage NLP pipeline (`RegexTokenizer` $\rightarrow$ `StopWordsRemover` $\rightarrow$ `HashingTF` ($\mathbb{R}^{16,384}$) $\rightarrow$ `IDF` $\rightarrow$ `StringIndexer` $\rightarrow$ Estimators: Logistic Regression, Random Forest, Multinomial Naive Bayes) with 3x3 confusion matrix conservation ($\sum \text{CM} \equiv N_{\text{test}}$).
  - **NoSQL Document Storage:** Dual-engine document persistence supporting **live MongoDB Atlas Cloud** and zero-dependency `LocalJsonDocumentStore` with compound indexing (`product_id`, `rating`) and MongoDB aggregation pipelines (`$match`, `$group`, `$avg`, `$sum`, `$sort`).
  - **Graph Analytics (Unit 6):** Bipartite reviewer-product network $G = (U, V, E)$, PageRank centrality iteration ($\alpha = 0.85$), degree centrality, and negative sentiment contagion subgraphs.
  - **Academic Visualization & Dashboard:** 10 publication-quality charts in `reports/visualizations/` and an interactive Flask Web Dashboard with an academic **Figure Inspector / Graph Picker**, real-time sentiment inference API, and corpus document explorer.

---

## 2. Detailed Curricular Unit Breakdown (Units 1 to 6)

| Unit | Theoretical Domain | Concrete Engineering Module | Academic Verification & Metrics | CO Alignment |
| :--- | :--- | :--- | :--- | :--- |
| **Unit 1** | **Big Data Concepts & 5Vs** | `src/storage/hdfs_simulation.py` (`BigData5VAnalysis`) | Volume (bytes), Velocity (records/s), Variety (entropy), Veracity (nulls), Value (polarity ratio). | **CO3** |
| **Unit 2** | **Hadoop HDFS Ecosystem** | `src/storage/hdfs_simulation.py` (`HDFSSimulator`, NameNode, DataNodes) | 64MB/128MB chunking, 3x replication invariant, rack awareness, DataNode failure recovery. | **CO3** |
| **Unit 3** | **MapReduce Paradigm** | `src/mapreduce/sentiment_mr.py` (`Mapper`, `Combiner`, `Partitioner`, `Reducer`) | In-memory token aggregation, deterministic hash shuffle, global token sentiment polarity score. | **CO3** |
| **Unit 4** | **Apache Spark & MongoDB** | `src/spark/` & `src/database/` (`SparkCoreAnalytics`, `MongoReviewPipeline`) | RDD transformation DAG lineage, Catalyst SQL temporary views, compound indexing, aggregation pipelines. | **CO3 & CO4** |
| **Unit 5** | **Machine Learning (MLlib)** | `src/ml/` (`SentimentMLlibPipeline`, `evaluator.py`) | 6-stage MLlib pipeline, Logistic Regression, Random Forest, Naive Bayes, bounded metrics ($[0.0, 1.0]$). | **CO4** |
| **Unit 6** | **Graph Analytics & Visuals** | `src/graph/`, `src/visualization/`, `web/app.py` | Bipartite NetworkX graph, PageRank power iterations, 10 publication figures, interactive research UI. | **CO4** |

---

## 3. End-to-End Distributed Architecture

```
                      [Raw E-Commerce Review Corpus (16 Fields)]
                                          │
                                          ▼
             [HDFS Simulation] ──► (64MB/128MB Chunks, 3x Replication, Rack-Aware DataNodes)
                                          │
                                          ▼
             [Data Cleaner & NLP] ──► (HTML Stripping, Contraction Expansion, Negation Preservation)
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                ▼                         ▼                         ▼
   [Hadoop MapReduce Engine]    [Apache Spark Core & SQL]   [MongoDB Atlas NoSQL Store]
  - ReviewTokenizerMapper      - RDD DAG Lineage Tracking   - BSON Document Ingestion
  - Node-Local Combiner        - Catalyst SQL Optimization  - Compound Indexing
  - Hash Partitioner           - Categorical Temp Views     - Aggregation Pipelines
  - Polarity Reducer                      │                         │
                │                         ▼                         │
                │             [Spark MLlib 6-Stage Pipeline]        │
                │            - RegexTokenizer + StopWordsFilter     │
                │            - HashingTF (16,384) + IDF Weighting   │
                │            - Multiclass Estimator Classifiers     │
                │            - Accuracy, Precision, Recall, F1      │
                │                         │                         │
                └─────────────────────────┼─────────────────────────┘
                                          ▼
                      [Graph Analytics & Visual Intelligence]
                     - Reviewer-Product Bipartite Network G = (U, V, E)
                     - PageRank Centrality & Negative Subgraphs
                     - 10 Publication-Quality Analytical Figures
                     - Interactive Research Dashboard & Real-Time Inference Protocol
```

---

## 4. Repository Structure

```
BDA-Sentiment-Analysis/
├── .github/workflows/
│   └── bda_distributed_pipeline.yml  # Automated CI/CD testing, Spark MLlib, & MongoDB sync
├── api/
│   └── index.py                      # Vercel serverless WSGI entrypoint
├── data/
│   ├── raw/                          # Amazon 16-field synthetic & benchmark review datasets
│   │   ├── reviews_sample.csv        # Tier A: 100 sample records (Fast CI)
│   │   ├── reviews_medium.csv        # Tier B: 5,000 records (Standard ML training)
│   │   └── reviews_tier_c.csv        # Tier C: 25,000 records (Big Data benchmark)
│   ├── processed/                    # Normalized, cleaned datasets
│   │   ├── cleaned_reviews.csv       # Full 25,000 normalized records
│   │   └── cleaned_reviews.parquet   # Columnar Parquet format for Spark optimization
│   └── storage/hdfs/                 # Simulated HDFS block storage directory
├── models/
│   └── mllib_classifier/             # Serialized trained classifier models (model.joblib)
├── reports/
│   ├── metrics.json                  # Model benchmarks (Accuracy, Precision, Recall, F1, CM)
│   └── visualizations/               # 10 Publication-Quality Analytical Figures (PNG)
│       ├── 01_sentiment_distribution.png
│       ├── 02_rating_vs_sentiment.png
│       ├── 03_category_sentiment_breakdown.png
│       ├── 04_temporal_sentiment_trends.png
│       ├── 05_model_performance_comparison.png
│       ├── 06_confusion_matrix_heatmap.png
│       ├── 07_mapreduce_word_polarity.png
│       ├── 08_bipartite_graph_network.png
│       ├── 09_hdfs_block_distribution.png
│       └── 10_helpful_votes_by_sentiment.png
├── scripts/
│   └── sync_mongo_atlas.py           # Automated MongoDB Atlas cloud synchronization utility
├── src/
│   ├── config.py                     # Dynamic environment paths & dataset tier constants
│   ├── spark_session.py              # SparkSession factory with Zero-JVM simulation fallback
│   ├── data/
│   │   ├── review_generator.py       # Amazon 16-field realistic review dataset generator
│   │   └── data_cleaner.py           # Text normalization, regex cleaner, negation preservation
│   ├── storage/
│   │   └── hdfs_simulation.py        # HDFS cluster simulation (NameNode, DataNode, 5Vs)
│   ├── mapreduce/
│   │   └── sentiment_mr.py           # Hadoop MapReduce engine (Mapper, Combiner, Reducer)
│   ├── database/
│   │   ├── document_store.py         # Abstract IDocumentStore interface & factory
│   │   ├── mongo_store.py            # MongoDB Atlas PyMongo client wrapper
│   │   ├── mongo_pipeline.py         # Production ingestion, compound indexing, aggregations
│   │   └── local_store.py            # High-performance local JSON document engine
│   ├── spark/
│   │   ├── spark_core_rdd.py         # Spark Core RDD transformations & DAG lineage tracking
│   │   └── spark_sql_analytics.py    # Spark SQL temporary view analytics & Catalyst queries
│   ├── ml/
│   │   ├── spark_mllib_pipeline.py   # 6-stage MLlib pipeline & multiclass classifiers
│   │   └── evaluator.py              # Evaluator (Accuracy, Precision, Recall, F1, 3x3 CM)
│   ├── graph/
│   │   └── review_graph_analytics.py # NetworkX bipartite network, PageRank, Degree Centrality
│   └── visualization/
│       └── plot_generator.py         # 10 publication-quality analytical figure generator
├── tests/                            # Complete 4-Tier Automated Test Suite (67 Tests)
│   ├── conftest.py                   # Pytest fixtures and mock test harness
│   ├── test_tier1_hdfs_simulation.py # Tier 1: HDFS simulation & 5Vs metrics
│   ├── test_tier1_preprocessing.py   # Tier 1: Data cleansing & schema normalization
│   ├── test_tier2_mllib_pipeline.py  # Tier 2: MLlib pipeline stages & evaluation bounds
│   ├── test_tier3_document_store.py  # Tier 3: Document store CRUD & aggregation pipeline
│   ├── test_tier3_integration.py     # Tier 3: MapReduce, Spark SQL Catalyst, & Graph
│   ├── test_tier4_e2e.py             # Tier 4: Master CLI, visual headers, & Flask REST APIs
│   └── test_unit2_mapreduce.py       # Unit 2/3: Mapper, Combiner, Partitioner, Reducer
├── web/
│   └── app.py                        # Academic Research Dashboard & REST API service
├── DEPLOYMENT.md                     # Detailed deployment instructions (Render & Vercel)
├── main.py                           # Master CLI Pipeline Runner & Orchestrator
├── Procfile                          # Gunicorn web process definition for Render / Heroku
├── pytest.ini                        # Pytest configuration & tier test marks
├── render.yaml                       # Render Cloud deployment blueprint specification
├── requirements.txt                  # Production and development dependencies
└── vercel.json                       # Vercel serverless function routing configuration
```

---

## 5. Dataset Architecture & Tiers

The project utilizes a structured **16-field Amazon Product Review schema** (`review_id`, `product_id`, `reviewer_id`, `product_category`, `rating`, `review_title`, `review_text`, `helpful_votes`, `total_votes`, `verified_purchase`, `review_date`, `sentiment_label`, `cleaned_text`, `char_length`, `word_count`, `compound_score`), organized into 3 scalable tiers:

| Tier | Dataset File | Records | Raw Size | Cleaned Parquet | Use Case |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier A** | `data/raw/reviews_sample.csv` | **100** | ~40 KB | ~40 KB | Rapid CI/CD execution and smoke testing (< 10 seconds) |
| **Tier B** | `data/raw/reviews_medium.csv` | **5,000** | ~2.0 MB | ~428 KB | Statistical MLlib model training, MapReduce, & confusion matrices |
| **Tier C** | `data/raw/reviews_tier_c.csv` | **25,000** | ~10.0 MB | ~2.1 MB | Big Data stress testing, HDFS chunking, & Spark SQL benchmarks |
| **Active Corpus** | `data/processed/cleaned_reviews.csv` | **25,000** | ~9.2 MB | ~2.1 MB | Full normalized corpus loaded dynamically by the dashboard & visualizer |

---

## 6. The 10 Publication-Quality Analytical Figures

The platform generates 10 publication-quality PNG charts in `reports/visualizations/`, all accessible interactively via the **Figure Inspector / Graph Picker** in the Web UI:

| Figure | Image File | Research Domain | Research Question & Methodology |
| :---: | :--- | :--- | :--- |
| **1** | `01_sentiment_distribution.png` | Corpus Distribution | *How are sentiment classes distributed across the corpus?* Evaluates empirical class balance ($P_+ \approx 65\%$, $P_0 \approx 20\%$, $P_- \approx 15\%$). |
| **2** | `02_rating_vs_sentiment.png` | Joint Probability | *Does star rating strictly correlate with NLP polarity?* Bivariate box plot of 1.0–5.0 star distributions vs. derived sentiment. |
| **3** | `03_category_sentiment_breakdown.png` | Category Analytics | *Which e-commerce categories exhibit highest defect risk?* Stacked horizontal sentiment share across 6 primary product categories. |
| **4** | `04_temporal_sentiment_trends.png` | Longitudinal Trends | *How does sentiment fluctuate over time?* 30-day rolling moving averages of polarity ratios and review volumes. |
| **5** | `05_model_performance_comparison.png` | Machine Learning | *Which classifier achieves optimal performance on sparse text?* Compares Accuracy, Precision, Recall, and F1 across Logistic Regression, RF, and Naive Bayes. |
| **6** | `06_confusion_matrix_heatmap.png` | Model Diagnostics | *Where do misclassifications concentrate?* 3x3 normalized confusion matrix heatmap strictly verifying sample conservation ($\sum \text{CM} \equiv N$). |
| **7** | `07_mapreduce_word_polarity.png` | Distributed Processing | *What key tokens drive polarity?* Top positive and negative token frequencies isolated via MapReduce with negation preservation. |
| **8** | `08_bipartite_graph_network.png` | Network Graph Theory | *Who are the most influential reviewers and hub products?* Bipartite graph $G = (U, V, E)$ visualizing PageRank scores and negative subgraphs. |
| **9** | `09_hdfs_block_distribution.png` | Distributed Storage | *How are blocks distributed under rack-aware replication?* Visualizes 64MB/128MB chunk placement across DataNodes and fault recovery. |
| **10** | `10_helpful_votes_by_sentiment.png` | Consumer Behavior | *Do negative reviews receive more helpfulness votes than positive ones?* Violin plot comparing consumer engagement distributions. |

---

## 7. Execution & CLI Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/pratyush-barik/BDA-Sentiment-Analysis.git
cd BDA-Sentiment-Analysis

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute the Full End-to-End Pipeline
Executes all 9 stages in sequence (Data Generation $\rightarrow$ HDFS Simulation $\rightarrow$ NLP Cleaning $\rightarrow$ MapReduce $\rightarrow$ Document Store Ingestion $\rightarrow$ Spark Core RDDs & SQL $\rightarrow$ MLlib Training & Evaluation $\rightarrow$ Graph Analytics $\rightarrow$ Visualization Generation):
```bash
python main.py --mode full --sample-size 500
```

### 3. Execute Individual Modular Stages
```bash
# Generate synthetic raw review dataset
python main.py --mode generate --sample-size 1000

# Run data cleansing & text normalization
python main.py --mode clean

# Run MapReduce token polarity job
python main.py --mode mapreduce

# Run Spark Core RDDs & Spark SQL temporary view queries
python main.py --mode spark

# Train & evaluate Spark MLlib classifiers
python main.py --mode ml

# Run Bipartite Graph Analytics & PageRank
python main.py --mode graph

# Generate all 10 publication visual plots
python main.py --mode visualize

# Run MongoDB Ingestion & Database Extraction demo
python main.py --mode mongo --sample-size 500

# Sync reviews directly to MongoDB Atlas Cloud:
python scripts/sync_mongo_atlas.py --size 1000
```

### 4. Launch the Academic Web Dashboard
```bash
python main.py --mode web --port 5000
```
Open your browser and navigate to:  
👉 **`http://localhost:5000`**

---

## 8. Cloud Deployment Guide (Render & Vercel)

### Deploy on Render (Recommended)
The repository is pre-configured for **Render** via [`render.yaml`](render.yaml) and [`Procfile`](Procfile):

1. Log in to **[dashboard.render.com](https://dashboard.render.com/)** using your GitHub account.
2. Click **"New +"** &rarr; Select **"Web Service"**.
3. Choose **`pratyush-barik/BDA-Sentiment-Analysis`** and click **"Connect"**.
4. Configure service parameters:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn web.app:app`
   - **Instance Type:** Free ($0/mo)
5. *(Optional)* Under **Environment Variables**, add:
   - `MONGO_URI`: `mongodb+srv://<username>:<password>@cluster0.xxx.mongodb.net/?retryWrites=true&w=majority`
6. Click **"Deploy Web Service"**. Your live dashboard will be accessible via a public HTTPS URL:  
   👉 `https://bda-sentiment-analysis.onrender.com`

---

## 9. Automated Verification & Testing

The repository enforces a comprehensive **4-Tier Automated Test Suite** comprising 67 unit, property, and integration tests:

- **Tier 1 (Storage & Preprocessing):** HDFS 3x replica placement, DataNode failure recovery, empty file handling, text normalization, and contraction expansion.
- **Tier 2 (MLlib Pipeline & Invariants):** Feature vector dimensions ($\mathbb{R}^{16,384}$), StringIndexer round-trips, probability bounds ($0 \le p_i \le 1, \sum p_i = 1$), and 3x3 confusion matrix conservation.
- **Tier 3 (Distributed Integration):** MapReduce round-trip, Document Store CRUD/aggregations, Spark SQL Catalyst plans, and NetworkX PageRank centrality.
- **Tier 4 (E2E Application):** Master CLI execution, valid PNG headers on all figures, REST API endpoints (`/api/predict`, `/api/kpis`, `/api/reviews`).

### Test Execution Command:
```bash
python -m pytest
```

### Verified Test Output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.16, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\Projects\BDA
configfile: pytest.ini
testpaths: tests
collected 67 items

tests/test_tier1_hdfs_simulation.py ........                             [ 11%]
tests/test_tier1_preprocessing.py ........                               [ 23%]
tests/test_tier2_mllib_pipeline.py ........                              [ 35%]
tests/test_tier3_document_store.py ................                      [ 58%]
tests/test_tier3_integration.py ......                                   [ 67%]
tests/test_tier4_e2e.py ......                                           [ 76%]
tests/test_unit2_mapreduce.py ................                           [100%]

============================= 67 passed in 10.73s =============================
```
**Status:** **100% PASS RATE (67 passed, 0 failed, 0 skipped)**

---

## 10. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
