# Big Data Analysis (BDA): Product Review Sentiment Analysis System

> **College Project Submission**  
> **Course:** Big Data Analysis (BDA)  
> **Course Outcome (CO) Mapping:** **CO3** & **CO4**  
> **Technologies:** Python 3.13, Apache Spark (Spark Core RDDs, Spark SQL, Spark MLlib), Hadoop MapReduce Architecture, NoSQL / MongoDB Document Storage, NetworkX Graph Analytics, Flask Interactive Web Dashboard, Matplotlib / Seaborn Visualizations.

---

## 1. Executive Summary & Syllabus Mapping

This project implements an enterprise-grade, intermediate-level **Big Data Product Review Sentiment Analysis Pipeline** designed to demonstrate deep theoretical and applied understanding of distributed data architectures, machine learning workflows, NoSQL persistence, and graph-based relationship analytics.

### Course Outcome (CO) Alignment
- **CO3: Analyze Big Data using Hadoop MapReduce, Spark Core RDDs, and Spark SQL processing models.**
  - Satisfied by HDFS simulation with rack-aware 3x replication, multi-stage Hadoop MapReduce with in-memory combiners and deterministic partitioners, Spark Core RDD word-sentiment transformation lineage, and Catalyst Spark SQL temporary views.
- **CO4: Implement Machine Learning models on Big Data streams, perform document storage with NoSQL/MongoDB, and derive graph analytics & visual insights.**
  - Satisfied by a 6-stage Spark MLlib classification pipeline (Logistic Regression, Random Forest, Naive Bayes) with multiclass metrics, MongoDB / Local JSON Document Store persistence, NetworkX bipartite user-product graph analytics with PageRank and degree centrality, 10 publication-quality visual plots, and a real-time Flask Web Dashboard.

### Detailed Syllabus Unit Breakdown (Units 1 to 6)

| Unit | Syllabus Topic | Academic Concept | Concrete Project Implementation | CO Mapping |
| :--- | :--- | :--- | :--- | :--- |
| **Unit 1** | **Big Data Concepts** | 5 Vs (Volume, Velocity, Variety, Veracity, Value), distributed architectures, large-scale review analysis | `src/storage/hdfs_simulation.py` (`BigData5VAnalysis`), 16-field Amazon review schema generation (`src/data/review_generator.py`). | **CO3** |
| **Unit 2** | **Hadoop Ecosystem** | HDFS block distribution, replication factor, rack awareness, DataNode fault tolerance | `src/storage/hdfs_simulation.py` (`HDFSSimulator`, NameNode, DataNodes, BlockMetadata, cluster health recovery). | **CO3** |
| **Unit 3** | **MapReduce Paradigm** | Mapper, in-memory Combiner, Shuffle & Sort partitioner, Reducer token aggregation | `src/mapreduce/sentiment_mr.py` (`ReviewTokenizerMapper`, `SentimentCombiner`, `MapReducePartitioner`, `SentimentReducer`). | **CO3** |
| **Unit 4** | **Apache Spark & MongoDB** | Spark Core RDDs, DataFrames, Catalyst Spark SQL views, Document-oriented databases | `src/spark/spark_core_rdd.py`, `src/spark/spark_sql_analytics.py`, `src/database/mongo_store.py`, `src/database/local_store.py`. | **CO3 & CO4** |
| **Unit 5** | **Machine Learning (MLlib)** | Feature extraction (Tokenizer, StopWordsRemover, HashingTF, IDF), Classifiers, Metrics | `src/ml/spark_mllib_pipeline.py` (6-stage pipeline: Logistic Regression, Random Forest, Naive Bayes), `src/ml/evaluator.py`. | **CO4** |
| **Unit 6** | **Graph Analytics & Visualizations** | Bipartite graphs, PageRank, Degree Centrality, Kibana/Grafana dashboards | `src/graph/review_graph_analytics.py` (NetworkX bipartite network, negative subgraph), `src/visualization/plot_generator.py` (10 plots), `web/app.py`. | **CO4** |

---

## 2. System Architecture & End-to-End Pipeline

```
[Raw E-Commerce Reviews (16 Fields)]
                  │
                  ▼
  [Unit 1 & 2: HDFS Simulation] ──► (64MB/128MB Chunks, 3x Replication, Rack-Aware DataNodes)
                  │
                  ▼
   [Unit 4: Data Cleaner & NLP] ──► (HTML Stripping, Contraction Expansion, Negation Preservation)
                  │
                  ├────────────────────────────────────────┬────────────────────────────────────────┐
                  ▼                                        ▼                                        ▼
   [Unit 3: Hadoop MapReduce]                [Unit 4: Apache Spark Core & SQL]         [Unit 4: NoSQL Document Store]
  - Mapper (Token Polarity)                 - RDD Word-Sentiment Lineage               - MongoDB / Local JSON Store
  - In-Memory Combiner                      - Catalyst Query Optimization              - BSON/JSON Document CRUD
  - Shuffle & Sort Partitioner              - Category Aggrs & Temp Views              - Secondary Indices & Aggregation
  - Reducer (Global Aggregations)                          │                                        │
                  │                                        ▼                                        │
                  │                          [Unit 5: Spark MLlib Pipeline]                         │
                  │                         - Tokenizer + StopWordsRemover                          │
                  │                         - HashingTF + Inverse Doc Freq                          │
                  │                         - Multiclass Classifiers                                │
                  │                         - Accuracy, Precision, Recall, F1                       │
                  │                                        │                                        │
                  └────────────────────────────────────────┼────────────────────────────────────────┘
                                                           ▼
                                         [Unit 6: Graph Analytics & Visualizations]
                                        - Bipartite Network Graph (Reviewers <-> Products)
                                        - PageRank Scoring & Degree Centrality
                                        - 10 Static Publication-Grade PNG Visualizations
                                        - Real-Time Flask Web Dashboard with REST APIs
```

---

## 3. Directory Structure

```
BDA-Sentiment-Analysis/
├── data/
│   ├── raw/                       # Generated Amazon-format raw CSV datasets
│   │   ├── reviews_sample.csv     # 100 sample records (Tier A)
│   │   ├── reviews_medium.csv     # 5,000 records (Tier B)
│   │   └── reviews_tier_c.csv     # 25,000 records (Tier C)
│   ├── processed/                 # Cleaned and schema-conformed datasets
│   │   ├── cleaned_reviews.csv
│   │   └── cleaned_reviews.parquet
│   └── storage/hdfs/              # Simulated HDFS block storage directory
├── models/                        # Serialized MLlib classification models
│   └── mllib_classifier/
├── reports/
│   ├── metrics.json               # Model accuracy, precision, recall, F1, confusion matrix
│   └── visualizations/            # 10 publication-quality PNG charts (>5KB, PNG headers)
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
├── src/
│   ├── config.py                  # Global project configuration, paths, and constants
│   ├── spark_session.py           # SparkSession manager with Zero-JVM local fallback
│   ├── data/
│   │   ├── review_generator.py    # Realistic Amazon 16-field dataset generator
│   │   └── data_cleaner.py        # Text normalization, regex cleaning, stopwords
│   ├── storage/
│   │   └── hdfs_simulation.py     # HDFS cluster simulation (NameNode, DataNode, 5Vs)
│   ├── mapreduce/
│   │   └── sentiment_mr.py        # Hadoop MapReduce engine (Map, Shuffle, Reduce)
│   ├── database/
│   │   ├── document_store.py      # Abstract IDocumentStore interface & factory
│   │   ├── mongo_store.py         # MongoDB PyMongo connector
│   │   └── local_store.py         # High-performance local JSON document engine
│   ├── spark/
│   │   ├── spark_core_rdd.py      # Spark RDD transformations & DAG lineage tracking
│   │   └── spark_sql_analytics.py # Spark SQL aggregation queries & temp views
│   ├── ml/
│   │   ├── spark_mllib_pipeline.py# 6-stage MLlib pipeline & classifiers
│   │   └── evaluator.py           # Multiclass evaluator (Accuracy, Precision, Recall, F1)
│   ├── graph/
│   │   └── review_graph_analytics.py # NetworkX bipartite graphs, PageRank, Centrality
│   └── visualization/
│       └── plot_generator.py      # Matplotlib/Seaborn publication plot generator
├── tests/                         # Complete 4-Tier Automated Test Suite
│   ├── conftest.py                # Test fixtures and simulated test harness
│   ├── test_tier1_hdfs_simulation.py
│   ├── test_tier1_preprocessing.py
│   ├── test_tier2_mllib_pipeline.py
│   ├── test_tier3_document_store.py
│   ├── test_tier3_integration.py
│   ├── test_tier4_e2e.py
│   └── test_unit2_mapreduce.py
├── web/
│   └── app.py                     # Flask Web Dashboard & REST API service
├── main.py                        # Master CLI Pipeline Runner & Orchestrator
├── pytest.ini                     # Pytest configuration
└── requirements.txt               # Project dependencies
```

---

## 4. Key Implementation Highlights

### 1. Big Data 5Vs & HDFS Block Replication (Unit 1 & 2 / CO3)
- Evaluates the **5Vs of Big Data** (Volume: byte capacity, Velocity: ingestion throughput, Variety: schema entropy, Veracity: missing keys/noise, Value: actionable positive/negative business ratios).
- Simulates HDFS chunking into 64MB/128MB blocks with a **Replication Factor of 3** and rack-aware placement across multiple DataNodes. Demonstrates automated re-replication upon node failure.

### 2. Hadoop MapReduce Token Sentiment Extraction (Unit 2 & 3 / CO3)
- Implements custom `ReviewTokenizerMapper` which tokenizes normalized text while preserving critical negation contexts (e.g. `not_good`, `never_works`).
- `SentimentCombiner` performs node-local pre-aggregation to minimize network shuffle overhead.
- Deterministic hash `MapReducePartitioner` distributes intermediate key-value pairs across reducers.
- `SentimentReducer` aggregates global token polarity scores across partitions.

### 3. Apache Spark Core RDDs & Spark SQL Analytics (Unit 4 / CO3 & CO4)
- Leverages Spark Core RDD transformations (`map`, `flatMap`, `reduceByKey`, `filter`) to compute word frequencies by sentiment class with transparent DAG execution lineage.
- Registers in-memory temporary views to execute Catalyst-optimized Spark SQL queries analyzing rating distributions, verified purchase correlations, and category volumes.

### 4. NoSQL / MongoDB Document Persistence (Unit 4 / CO4)
- Implements unified `IDocumentStore` abstract interface supporting live **MongoDB** instances as well as a zero-dependency **Local JSON Document Store**.
- Features secondary indexing acceleration, nested field projection, and MongoDB aggregation pipelines.

### 5. Spark MLlib Classification Pipeline (Unit 5 / CO4)
- Implements a 6-stage ML pipeline:
  1. `RegexTokenizer` / custom tokenizer
  2. `StopWordsRemover` (ecommerce-filtered, negation-preserved)
  3. `HashingTF` / `CountVectorizer`
  4. `IDF` (Inverse Document Frequency weighting)
  5. `StringIndexer` for multiclass sentiment labels (`positive`, `neutral`, `negative`)
  6. Classifiers: **Logistic Regression**, **Random Forest**, and **Multinomial Naive Bayes**.
- Evaluator outputs strict mathematical invariants: Accuracy, Precision, Recall, F1-score, and a 3x3 Confusion Matrix exported to `reports/metrics.json`.

### 6. Graph Analytics & Bipartite Interaction Modeling (Unit 6 / CO4)
- Constructs a bipartite network $G = (U, V, E)$ connecting Reviewers ($U$) with Products ($V$).
- Computes **PageRank** scores over product nodes to identify highly influential or frequently reviewed items.
- Calculates **Degree Centrality** to pinpoint power reviewers in the community.
- Extracts **Negative Sentiment Subgraphs** to trace product defect contagion across categories.

### 7. Interactive Visual Dashboard & Publication Charts (Unit 6 / CO4)
- Produces **10 static publication-grade PNG visualizations** in `reports/visualizations/` spanning distributions, temporal trends, ML confusion matrices, and network layouts.
- Hosts an interactive **Flask Web Dashboard** with real-time sentiment inference (`POST /api/predict`), dataset KPIs (`GET /api/kpis`), and filterable review tables (`GET /api/reviews`).

---

## 5. Execution Guide

### 1. Run the Complete End-to-End Pipeline
Executes all 9 stages in sequential order with automated metrics and visualization generation:
```bash
python main.py --mode full --sample-size 100
```

### 2. Run Individual Modular Stages
```bash
# Generate synthetic raw review dataset
python main.py --mode generate --sample-size 500

# Run data cleansing & text normalization
python main.py --mode clean

# Run MapReduce token polarity job
python main.py --mode mapreduce

# Run Spark Core & Spark SQL analytical queries
python main.py --mode spark

# Train & evaluate Spark MLlib classifiers
python main.py --mode ml

# Run Graph Analytics & PageRank
python main.py --mode graph

# Generate all 10 publication visual plots
python main.py --mode visualize

# Run MongoDB Ingestion & Database Extraction demo
python main.py --mode mongo --sample-size 500

# Sync reviews directly to your MongoDB Atlas cloud account:
python scripts/sync_mongo_atlas.py --uri "mongodb+srv://<user>:<password>@cluster0.xxx.mongodb.net/"
```

### 3. Connect to MongoDB Atlas (Cloud)
To view all database documents, compound indexes, and aggregation pipelines in real-time on your **MongoDB Atlas** account:
1. Copy your MongoDB Atlas connection string from [cloud.mongodb.com](https://cloud.mongodb.com/).
2. Add it to your `.env` file (copied from `.env.example`):
   ```ini
   MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxx.mongodb.net/?retryWrites=true&w=majority
   MONGO_DB_NAME=bda_reviews_db
   MONGO_COLLECTION_NAME=product_reviews
   ```
3. Run the automated synchronization:
   ```bash
   python scripts/sync_mongo_atlas.py --size 1000
   ```
4. Open [cloud.mongodb.com](https://cloud.mongodb.com/) &rarr; Click **"Database"** &rarr; **"Browse Collections"** &rarr; view `bda_reviews_db.product_reviews` live!

### 4. Automated GitHub Actions CI/CD Interconnection
The repository includes an automated workflow in `.github/workflows/bda_distributed_pipeline.yml`:
- Set your `MONGO_URI` secret in your GitHub repository: **Settings &rarr; Secrets and variables &rarr; Actions &rarr; New repository secret** (`MONGO_URI`).
- On every push, GitHub Actions automatically executes the distributed Spark pipeline, performs MapReduce analysis, runs all 67 tests, and syncs the review data into your cloud MongoDB Atlas database!

### 5. Launch the Interactive Web Dashboard
```bash
python main.py --mode web --port 5000
```
Open your browser and navigate to:  
👉 `http://localhost:5000`

---

## 6. Automated Verification Test Suite

The project includes a comprehensive 4-tier automated test suite covering all units and edge cases:
- **Tier 1**: HDFS distributed storage simulation & text cleansing normalization.
- **Tier 2**: Spark MLlib feature stages, classification training, evaluation bounds, and persistence.
- **Tier 3**: Document Store CRUD/aggregations, MapReduce pipeline, Spark SQL Catalyst queries, and Graph PageRank.
- **Tier 4**: Real-world application scenarios, E2E CLI execution, visualization PNG headers, and Flask REST APIs.

### Test Execution Command:
```bash
python -m pytest
```

### Test Results:
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

============================= 67 passed in 8.12s ==============================
```
**Status:** **100% PASS RATE (67 passed, 0 failed, 0 skipped)**
