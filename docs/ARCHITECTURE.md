# System Architecture & Technical Specifications

> **Project:** Distributed Product Review Sentiment Analysis System  
> **Course / Academic Mapping:** Big Data Analysis (BDA) &bull; Course Outcomes **CO3** & **CO4** &bull; Syllabus Units 1 to 6  
> **Repository:** [https://github.com/pratyush-barik/BDA-Sentiment-Analysis](https://github.com/pratyush-barik/BDA-Sentiment-Analysis)

---

## 1. Executive Architectural Blueprint

The platform implements an end-to-end distributed Big Data processing, machine learning, and visual analytics architecture designed to process large-scale e-commerce consumer review corpora ($N = 25,000$ verified records).

```
                             [Raw E-Commerce Review Corpus (16 Fields)]
                                                 │
                                                 ▼
                [Unit 1 & 2: HDFS Cluster Storage Simulation]
                - Block Size: 64MB / 128MB Chunk Allocation
                - Invariant: 3x Replication Factor Across 5 DataNodes
                - Rack Awareness: 2 Replicas on Local Rack, 1 on Remote Rack
                - Fault Recovery: Automated Re-Replication upon Node Failure
                                                 │
                                                 ▼
                [Unit 4: Data Cleansing & NLP Normalization]
                - Regex HTML/URL Stripping & Character De-Noising
                - Lexicon Contraction Expansion (e.g. "don't" -> "do not")
                - Negation Preservation Context (e.g. "not_good", "never_works")
                - E-Commerce Filtered Stopword Elimination
                                                 │
                ┌────────────────────────────────┼────────────────────────────────┐
                ▼                                ▼                                ▼
   [Unit 3: Hadoop MapReduce]      [Unit 4: Apache Spark Core & SQL]   [Unit 4: NoSQL Document Store]
   - ReviewTokenizerMapper         - Spark Core RDD DAG Lineage        - MongoDB Atlas Cloud Cluster
   - In-Memory Combiner            - Transformations: map/reduceByKey  - LocalJsonDocumentStore Fallback
   - HashPartitioner [0..R-1]      - Catalyst Query Optimization       - Compound Indices (product, rating)
   - Polarity Reducer Output       - Temporary Views & Aggregations    - Aggregation ($group, $avg, $sum)
                │                                │                                │
                │                                ▼                                │
                │                 [Unit 5: Spark MLlib Pipeline]                  │
                │                 Stage 1: RegexTokenizer                         │
                │                 Stage 2: StopWordsRemover                       │
                │                 Stage 3: HashingTF (d = 16,384)                 │
                │                 Stage 4: Inverse Doc Frequency (IDF)            │
                │                 Stage 5: Multiclass StringIndexer               │
                │                 Stage 6: Estimators (LR, RF, NB)                │
                │                 Evaluation: Accuracy, Precision, Recall, F1     │
                │                 Conservation Law: sum(sum(CM)) == N_test        │
                │                                │                                │
                └────────────────────────────────┼────────────────────────────────┘
                                                 ▼
                               [Unit 6: Graph Analytics & UI]
                               - Bipartite Network G = (U, V, E)
                               - PageRank Centrality Iteration (alpha = 0.85)
                               - Negative Subgraph Defect Contagion
                               - 10 Publication-Quality Analytical Figures
                               - Academic Research Dashboard with 10-Figure Picker
                               - Dynamic Sample Size Controller (n <= 25,000)
```

---

## 2. Unit-by-Unit Theoretical & Mathematical Foundations

### Unit 1: Big Data Concepts & 5Vs Characterization
- **Volume ($V_1$):** Corpus size scales across three distinct tiers:
  - **Tier A (Sample):** $100$ records ($\sim 40\text{ KB}$) for instantaneous unit/CI testing.
  - **Tier B (Medium):** $5,000$ records ($\sim 2.0\text{ MB}$) for statistical ML training.
  - **Tier C (Benchmark):** $25,000$ records ($\sim 10.0\text{ MB}$) for Big Data stress testing.
- **Velocity ($V_2$):** Evaluated in ingestion throughput records/second ($R_{\text{ingest}} > 2,500\text{ docs/sec}$).
- **Variety ($V_3$):** Schema entropy across 16 heterogeneous attributes comprising numeric ratings, timestamps, boolean purchase verifications, free-form natural language text, and categorical labels.
- **Veracity ($V_4$):** Data cleansing measures noise ratio, missing fields, and URL/HTML artifacts.
- **Value ($V_5$):** Extracted business intelligence: customer satisfaction index ($\mu_{\text{rating}}$), negative defect ratios, and power reviewer discovery.

---

### Unit 2: Hadoop HDFS Architecture & Replica Placement Invariants
- **Block Allocation:** Simulates chunking large corpora into $64\text{ MB}$ or $128\text{ MB}$ blocks:
  $$\text{Block Count } B = \left\lceil \frac{\text{File Size}}{\text{Block Size}} \right\rceil$$
- **Replication Factor Invariant:** Every block $b_i$ maintains exactly $R = 3$ active replicas:
  $$\forall b_i \in B, \quad |\text{Replicas}(b_i)| \equiv 3$$
- **Rack-Aware Placement Topology:**
  - Replica 1: Placed on a local DataNode $DN_1$ in Rack 1.
  - Replica 2: Placed on a different DataNode $DN_2$ in the same Rack 1.
  - Replica 3: Placed on a DataNode $DN_3$ in a remote Rack 2.
  This topology minimizes cross-rack network traffic while surviving complete rack failure.
- **DataNode Failure Recovery:** Automated NameNode heartbeat monitoring detects dead DataNodes and initiates under-replicated block transfers to healthy nodes, maintaining 100% data availability.

---

### Unit 3: Hadoop MapReduce Processing Paradigm
- **Mapper (`ReviewTokenizerMapper`):** Tokenizes cleaned review text, expanding and attaching negation contexts (e.g. `not_bad` $\rightarrow$ positive context, `never_recommend` $\rightarrow$ negative context).
  $$\text{Map}: (\text{review\_id}, \text{record}) \rightarrow \left[ (w_1, (s, r)), (w_2, (s, r)), \dots \right]$$
- **Combiner (`SentimentCombiner`):** Node-local aggregation executes before the shuffle phase to compress network traffic:
  $$\text{Combine}: (w, [(s_1, r_1), (s_2, r_2), \dots]) \rightarrow (w, (\sum s_i, \sum r_i, \text{count}))$$
- **Partitioner (`MapReducePartitioner`):** Deterministic uniform distribution across $R$ reducers:
  $$\text{Partition}(w) = (\text{hash}(w) \ \& \ 0\text{x7FFFFFFF}) \pmod R$$
- **Reducer (`SentimentReducer`):** Computes global word frequencies, sentiment polarity bias, and mean associated ratings across the entire distributed cluster:
  $$\text{Polarity Score}(w) = \frac{\text{PosFreq}(w) - \text{NegFreq}(w)}{\text{TotalFreq}(w)}$$

---

### Unit 4: Apache Spark Core & NoSQL Persistence

#### 1. Apache Spark Core (RDDs & Spark SQL)
- **DAG Lineage:** Maintains lazy transformation lineages (`flatMap` $\rightarrow$ `map` $\rightarrow$ `filter` $\rightarrow$ `reduceByKey`), enabling deterministic partition recomputation upon worker node loss.
- **Spark SQL Catalyst Optimizer:** Registers in-memory temporary views (`reviews_view`) to execute relational analytics with tree transformation optimizations:
  ```sql
  SELECT product_category, 
         COUNT(*) AS review_count, 
         AVG(rating) AS mean_rating,
         SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pos_share
  FROM reviews_view
  GROUP BY product_category
  ORDER BY review_count DESC;
  ```

#### 2. NoSQL Document Storage (MongoDB Atlas & LocalStore)
- **Compound Indexing:** Provisions secondary indices for accelerated multi-field filtering:
  - `{"product_id": 1, "rating": -1}` (Product rating ranking index)
  - `{"sentiment_label": 1}` (Polarity slice index)
- **Aggregation Pipeline:** Executes server-side pipeline stages:
  ```json
  [
    {"$match": {"rating": {"$gte": 1.0}}},
    {"$group": {
      "_id": "$product_category",
      "avg_rating": {"$avg": "$rating"},
      "total_reviews": {"$sum": 1}
    }},
    {"$sort": {"total_reviews": -1}}
  ]
  ```
- **High-Fidelity Fallback:** `LocalJsonDocumentStore` implements identical CRUD and aggregation pipelines using pure in-memory indexing, ensuring 100% operation even when offline without database daemons.

---

### Unit 5: Scalable Machine Learning with Apache Spark MLlib

The 6-stage MLlib pipeline maps high-dimensional text to categorical sentiment classes:

$$\text{Pipeline} = \mathcal{T}_{\text{regex}} \circ \mathcal{S}_{\text{stopwords}} \circ \mathcal{H}_{\text{tf}} \circ \mathcal{I}_{\text{idf}} \circ \mathcal{L}_{\text{indexer}} \circ \mathcal{C}_{\text{estimator}}$$

1. **`RegexTokenizer`:** Lowercases and tokenizes sentences using regex pattern `\W+`.
2. **`StopWordsRemover`:** Removes general e-commerce filler words while strictly preserving critical sentiment negations (`no`, `not`, `never`, `hardly`).
3. **`HashingTF`:** Maps arbitrary token sequences into a fixed high-dimensional sparse vector space $\mathbb{R}^{d}$ with $d = 16,384$:
   $$h(w) = \text{MurmurHash3}(w) \pmod d$$
4. **`IDF` (Inverse Document Frequency):** Weights term importance inversely against corpus frequency:
   $$\text{IDF}(w, D) = \ln \left( \frac{|D| + 1}{\text{DF}(w, D) + 1} \right)$$
5. **`StringIndexer`:** Maps multiclass string labels (`positive`, `neutral`, `negative`) into ordered float indices $[0.0, 1.0, 2.0]$.
6. **Classification Estimators:**
   - **Multinomial Logistic Regression:** Evaluates softmax probabilities:
     $$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
   - **Random Forest Classifier:** Ensembles 20 decision trees with sub-feature bagging.
   - **Multinomial Naive Bayes:** Evaluates conditional feature likelihoods under Laplace smoothing.

#### Mathematical Evaluation Invariants
- **Probability Simplex Constraint:**
  $$\forall i \in \{0, 1, 2\}, \quad 0.0 \le P(y_i) \le 1.0 \quad \text{and} \quad \sum_{i=0}^2 P(y_i) = 1.0 \pm 10^{-4}$$
- **Confusion Matrix Conservation Law:**
  $$\sum_{i=1}^3 \sum_{j=1}^3 \text{CM}_{i,j} \equiv |D_{\text{test}}|$$

---

### Unit 6: Bipartite Graph Analytics & Visual Analytics

#### 1. Bipartite Network Topology ($G = (U, V, E)$)
- **Vertex Partitions:** Reviewers $U = \{u_1, u_2, \dots\}$ and Products $V = \{v_1, v_2, \dots\}$.
- **Edges:** An edge $e = (u_i, v_j) \in E$ denotes a verified review transaction.
- **PageRank Power Iteration:** Identifies central, highly connected catalog products:
  $$\mathbf{p}^{(k+1)} = \frac{1 - \alpha}{|V|} \mathbf{1} + \alpha \mathbf{M} \mathbf{p}^{(k)}$$
  where damping factor $\alpha = 0.85$ and $\mathbf{M}$ is the stochastic transition matrix.
- **Degree Centrality:** Measures the connectivity degree of power reviewers:
  $$C_D(u_i) = \frac{\deg(u_i)}{|V|}$$
- **Negative Sentiment Subgraphs:** Extracts subgraphs where rating $\le 2.0$, isolating contagious defect vectors across shared vendor categories.

#### 2. The 10 Publication-Grade Figures
1. `01_sentiment_distribution.png`: Multiclass class frequencies and percentage ratios.
2. `02_rating_vs_sentiment.png`: Bivariate box plot of star rating distributions by sentiment.
3. `03_category_sentiment_breakdown.png`: Stacked horizontal sentiment share across 6 domains.
4. `04_temporal_sentiment_trends.png`: 30-day rolling moving averages of polarity volume.
5. `05_model_performance_comparison.png`: Accuracy, Precision, Recall, and F1 across 3 classifiers.
6. `06_confusion_matrix_heatmap.png`: 3x3 normalized confusion matrix heatmap.
7. `07_mapreduce_word_polarity.png`: Top positive and negative tokens extracted via MapReduce.
8. `08_bipartite_graph_network.png`: Bipartite reviewer-product network graph with PageRank sizing.
9. `09_hdfs_block_distribution.png`: Simulated HDFS chunking and 3x replica placement across DataNodes.
10. `10_helpful_votes_by_sentiment.png`: Violin distribution of helpfulness votes by polarity class.

---

## 3. Data Schema & Schema Evolution

The raw and processed datasets strictly conform to the **16-Field Amazon Review Schema**:

| Column Name | Data Type | Description | Sample Value |
| :--- | :--- | :--- | :--- |
| `review_id` | String | Unique UUID identifier | `REV-9382104` |
| `product_id` | String | Unique product catalog ID | `PROD-ELEC-401` |
| `reviewer_id` | String | Unique consumer identifier | `USR-8192` |
| `reviewer_name` | String | Display name of reviewer | `Dr. Evelyn Vance` |
| `product_category` | String | Primary retail domain | `Electronics` |
| `product_name` | String | Descriptive item title | `Noise-Canceling Wireless Headphones` |
| `rating` | Float | Star score in $[1.0, 5.0]$ | `5.0` |
| `review_title` | String | Headline summary | `Superb acoustic fidelity` |
| `review_text` | String | Full free-form review prose | `Exceptional build quality and battery longevity...` |
| `helpful_votes` | Integer | Affirmative community votes | `42` |
| `total_votes` | Integer | Total community votes | `45` |
| `verified_purchase`| Boolean | Confirmed retail purchase | `True` |
| `review_date` | String | ISO 8601 timestamp | `2024-03-15` |
| `sentiment_label` | String | Canonical ground truth label | `positive` |
| `cleaned_text` | String | Normalized, regex-cleaned text | `exceptional build quality battery longevity` |
| `compound_score` | Float | Lexicon polarity score $[-1.0, 1.0]$ | `0.884` |

---

## 4. Dual-Engine Execution Architecture (JVM & Zero-JVM)

To guarantee portability across cloud containers (Render, Vercel, Docker) where native Java/JVM may not be available, the system incorporates an autonomous **Dual-Engine Abstraction Layer**:

```
                       User Request / Pipeline Execution
                                       │
                                       ▼
                         [SparkSessionFactory Factory]
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
      [Native Apache Spark 3.5]               [Zero-JVM Simulation Engine]
      - Active if Java & PySpark loaded       - Active if Java is absent
      - Native PySpark RDDs & DataFrames      - High-Fidelity Python Emulators
      - JVM Spark MLlib Pipelines             - Scikit-Learn Sparse TF-IDF Adapter
      - Full Catalyst SQL Optimizer           - In-Memory Relational SQL Simulator
```

- **Seamless API Parity:** `SimulatedDataFrame`, `SimulatedRDD`, and `SimulatedSparkSession` expose identical method signatures (`.createDataFrame()`, `.select()`, `.filter()`, `.groupBy()`, `.count()`, `.explain()`), ensuring 100% of pipeline stages run without code alterations.

---

## 5. Web Dashboard Architecture & Dynamic Sample Size Controller

The Flask application (`web/app.py`) provides an interactive academic research dashboard and real-time inference microservice:

### Dynamic Sample Size Controller ($n \le N$)
- Users can adjust the active analysis sample size $n$ anywhere from $50$ up to $25,000$ via:
  1. **Continuous Range Slider (`#sample-range`):** Smooth drag selection with live numeric feedback.
  2. **Direct Number Input (`#sample-number-input`):** Type any specific sample size value below $25,000$.
  3. **Preset Jump Buttons:** Instant one-click selection of standard tiers (`100`, `1k`, `5k`, `10k`, `25k`).
- **Reactive Updates:** Adjusting $n$ immediately updates the top header badge, dynamically queries `/api/kpis?sample_size=n`, recalculates mean ratings and class percentages for that specific slice, and updates the Corpus Document Store table.

### Interactive 10-Figure Inspector (Graph Picker)
- Replaces static multi-graph grids with a focused, publication-grade viewer.
- Includes domain filtering pills (*Corpus Distributions*, *Storage & MapReduce*, *MLlib Benchmark*, *Graph Network*), forward/backward navigation controls, formal academic captions, mathematical methodologies, and empirical findings.

---

## 6. Cloud Deployment & CI/CD Topology

- **GitHub Actions (`bda_distributed_pipeline.yml`):** Automatically executes on every push to an isolated Ubuntu 22.04 runner:
  1. Checks out repository.
  2. Sets up Python 3.11.
  3. Runs all 67 pytest automated tests.
  4. Executes the full distributed pipeline.
  5. Syncs data to MongoDB Atlas Cloud using encrypted `MONGO_URI` secret.
  6. Generates and uploads the `bda-visualizations-and-metrics` artifact (957 KB).
- **Render (`render.yaml` & `Procfile`):** Deploys the web service with multi-worker Gunicorn server.
- **Vercel (`vercel.json` & `api/index.py`):** Enables serverless WSGI execution on edge functions.
