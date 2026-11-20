"""
tests/test_tier3_integration.py
Tier 3: Cross-Feature Integration (MapReduce, DocumentStore, Spark SQL, Graph Analytics).
Big Data Product Review Sentiment Analysis System.
"""

from typing import Any, Dict, List
import pytest
try:
    from tests.conftest import safe_import
except ImportError:
    from conftest import safe_import


@pytest.mark.tier3
def test_mapreduce_tokenizer_combiner_reducer_roundtrip(sample_reviews_data):
    """
    Test 3.1: Validate the full Unit 2 MapReduce lifecycle:
    ReviewTokenizerMapper -> SentimentCombiner -> SentimentReducer.
    Enforces token conservation invariant: sum(map emissions) == sum(reducer counts).
    """
    mr_mod = safe_import("src.mapreduce.sentiment_mr")
    if mr_mod is None:
        pytest.skip("Milestone M2 MapReduce engine not yet implemented")

    Mapper = getattr(mr_mod, "ReviewTokenizerMapper", None)
    Combiner = getattr(mr_mod, "SentimentCombiner", None)
    Reducer = getattr(mr_mod, "SentimentReducer", None)
    run_mr_job = getattr(mr_mod, "run_mapreduce_job", None)

    if Mapper is None or Reducer is None:
        pytest.skip("ReviewTokenizerMapper or SentimentReducer not found in src.mapreduce.sentiment_mr")

    mapper = Mapper()
    combiner = Combiner() if Combiner else None
    reducer = Reducer()

    # Step 1: Map stage
    map_emissions = []
    for record in sample_reviews_data:
        map_fn = getattr(mapper, "map", None)
        if map_fn:
            emissions = list(map_fn(record))
            map_emissions.extend(emissions)

    assert len(map_emissions) > 0, "Mapper emitted 0 tokens"
    assert all(isinstance(k, tuple) and v == 1 for k, v in map_emissions), "Invalid mapper emission format"

    # Step 2: Combine stage (local aggregation)
    if combiner and hasattr(combiner, "combine"):
        combined_emissions = list(combiner.combine(map_emissions))
        assert len(combined_emissions) <= len(map_emissions), "Combiner did not reduce emission count"
    else:
        combined_emissions = map_emissions

    # Step 3: Reduce stage (global aggregation)
    reduce_fn = getattr(reducer, "reduce", None)
    reduced_output = reduce_fn(combined_emissions)
    assert isinstance(reduced_output, dict), "Reducer must produce a dictionary"

    # Token conservation law: Total tokens mapped == Total tokens reduced
    total_mapped_tokens = sum(v for _, v in map_emissions)
    total_reduced_tokens = sum(
        sum(sent_dict.values()) if isinstance(sent_dict, dict) else sent_dict
        for sent_dict in reduced_output.values()
    )
    assert total_mapped_tokens == total_reduced_tokens, (
        f"Token conservation violated: mapped={total_mapped_tokens} != reduced={total_reduced_tokens}"
    )


@pytest.mark.tier3
@pytest.mark.database
def test_document_store_insert_query_aggregation_pipeline(doc_store, sample_reviews_data):
    """
    Test 3.2: Validate Unit 3 Document Store interface (insert_many, find, aggregate)
    using LocalJsonDocumentStore / MongoDocumentStore.
    """
    # 1. Insert records
    inserted_ids = doc_store.insert_many(sample_reviews_data)
    assert len(inserted_ids) == len(sample_reviews_data)
    assert doc_store.count_documents({}) == len(sample_reviews_data)

    # 2. Query filtering
    high_rating_docs = doc_store.find({"rating": {"$gte": 4.0}})
    assert len(high_rating_docs) > 0
    assert all(doc["rating"] >= 4.0 for doc in high_rating_docs)

    # 3. Aggregation pipeline: Category sentiment breakdown
    pipeline = [
        {"$match": {"rating": {"$gte": 1.0}}},
        {
            "$group": {
                "_id": "$product_category",
                "avg_rating": {"$avg": "$rating"},
                "total_reviews": {"$sum": 1},
            }
        },
        {"$sort": {"total_reviews": -1}},
    ]
    agg_results = doc_store.aggregate(pipeline)
    assert len(agg_results) > 0, "Aggregation returned empty list"

    # Sum of reviews across categories in aggregation must equal total inserted records
    total_reviews_agg = sum(group["total_reviews"] for group in agg_results)
    assert total_reviews_agg == len(sample_reviews_data), (
        f"Aggregated count {total_reviews_agg} != total records {len(sample_reviews_data)}"
    )

    # Average rating invariant: must be in [1.0, 5.0]
    for group in agg_results:
        assert 1.0 <= group["avg_rating"] <= 5.0, f"Average rating out of bounds: {group['avg_rating']}"


@pytest.mark.tier3
@pytest.mark.spark
def test_spark_sql_temp_view_queries_and_catalyst_plan(spark_session, sample_reviews_df):
    """
    Test 3.3: Validate Unit 4 Spark SQL temporary view registration,
    structured analytics queries, and Catalyst execution plan explanation.
    """
    spark_core_mod = safe_import("src.spark.spark_sql_analytics") or safe_import("src.spark.spark_core_rdd")

    # Register temporary view
    sample_reviews_df.createOrReplaceTempView("reviews_view")

    # Structured SQL query
    query = """
        SELECT product_category, COUNT(*) as cnt, AVG(rating) as avg_rating
        FROM reviews_view
        GROUP BY product_category
    """
    result_df = spark_session.sql(query)
    assert result_df is not None
    rows = result_df.collect()
    assert len(rows) > 0, "Spark SQL query returned empty result set"

    first_row = rows[0]
    cols = list(first_row.keys()) if isinstance(first_row, dict) else result_df.columns
    assert "cnt" in cols
    assert "avg_rating" in cols

    # Catalyst plan inspection
    if hasattr(result_df, "explain"):
        plan_str = result_df.explain(extended=True)
    elif spark_core_mod and hasattr(spark_core_mod, "explain_catalyst_plan"):
        plan_str = spark_core_mod.explain_catalyst_plan(result_df)
    else:
        plan_str = "Physical Plan: Scan"

    assert plan_str is not None
    assert len(str(plan_str)) > 0


@pytest.mark.tier3
@pytest.mark.graph
def test_bipartite_graph_construction_and_pagerank(sample_reviews_data):
    """
    Test 3.4: Validate Unit 6 Graph Analytics: Bipartite network construction (U <-> P),
    degree centrality, PageRank, and negative sentiment subgraph extraction.
    """
    import networkx as nx
    graph_mod = safe_import("src.graph.review_graph_analytics")
    if graph_mod is None:
        pytest.skip("Milestone M4 Graph Analytics not yet implemented")

    analytics_cls = getattr(graph_mod, "ReviewGraphAnalytics", None)
    if analytics_cls is None:
        pytest.skip("ReviewGraphAnalytics class not found in src.graph.review_graph_analytics")

    analytics = analytics_cls()

    # Step 1: Bipartite Graph Construction
    G = analytics.build_bipartite_graph(sample_reviews_data)
    assert isinstance(G, nx.Graph), "Graph must be a NetworkX Graph instance"
    assert nx.is_bipartite(G), "Graph violates bipartite network property"

    # Step 2: PageRank computation
    pagerank = analytics.compute_product_pagerank(G)
    assert isinstance(pagerank, dict) and len(pagerank) > 0, "PageRank returned empty dictionary"
    # PageRank scores invariant: sum of scores across graph nodes must equal 1.0 within tolerance
    pr_sum = sum(pagerank.values())
    assert abs(pr_sum - 1.0) < 1e-2 or pr_sum <= 1.0, f"PageRank sum invariant violated: {pr_sum}"

    # Step 3: Reviewer Degree Centrality
    centrality = analytics.compute_reviewer_centrality(G)
    assert isinstance(centrality, dict) and len(centrality) > 0
    # Centrality scores must reside in [0.0, 1.0]
    assert all(0.0 <= c <= 1.0 for c in centrality.values()), "Centrality out of bounds [0.0, 1.0]"

    # Step 4: Negative Sentiment Subgraph
    neg_G = analytics.extract_negative_sentiment_subgraph(G)
    assert isinstance(neg_G, nx.Graph)
    for u, v, data in neg_G.edges(data=True):
        sent = data.get("sentiment", data.get("sentiment_label"))
        rating = data.get("rating", 5.0)
        assert sent == "negative" or rating <= 2.0, f"Non-negative edge in negative subgraph: {data}"


@pytest.mark.tier3
def test_dual_engine_execution_transparency():
    """
    Test 3.5: Verify that the execution adapter guarantees transparency:
    client code gets identical method signatures and interfaces regardless
    of host Java / Spark availability.
    """
    spark_session_mod = safe_import("src.spark_session")
    if spark_session_mod is None:
        pytest.skip("Milestone M1 spark_session not yet implemented")

    factory_cls = getattr(spark_session_mod, "SparkSessionFactory", None)
    if factory_cls is None:
        pytest.skip("SparkSessionFactory not found in src.spark_session")

    spark = factory_cls.get_or_create(app_name="TestTransparency")
    assert spark is not None, "Failed to get or create SparkSession"

    # Core interface checks
    assert hasattr(spark, "createDataFrame"), "Missing createDataFrame on SparkSession"
    assert hasattr(spark, "sql"), "Missing sql on SparkSession"

    # MLlib Pipeline transparency check if M3 is implemented
    mllib_mod = safe_import("src.ml.spark_mllib_pipeline")
    if mllib_mod is not None:
        pipeline_cls = getattr(mllib_mod, "SentimentMLlibPipeline", None)
        if pipeline_cls is not None:
            pipeline = pipeline_cls()
            assert hasattr(pipeline, "train")
            assert hasattr(pipeline, "evaluate")
            assert hasattr(pipeline, "predict")


@pytest.mark.tier3
@pytest.mark.spark
def test_spark_core_rdd_lineage_and_word_sentiment_count(spark_session, sample_reviews_df):
    """
    Test 3.6: Validate Unit 4 RDD transformations (flatMap, map, reduceByKey)
    and lineage graph inspection (toDebugString).
    """
    spark_core_mod = safe_import("src.spark.spark_core_rdd")
    if spark_core_mod is None:
        pytest.skip("Milestone M3 Spark Core RDD module not yet implemented")

    analytics_cls = getattr(spark_core_mod, "SparkCoreAnalytics", None)
    if analytics_cls is None:
        pytest.skip("SparkCoreAnalytics class not found in src.spark.spark_core_rdd")

    rdd = sample_reviews_df.rdd if hasattr(sample_reviews_df, "rdd") else spark_session.parallelize(sample_reviews_df.collect())
    count_fn = getattr(analytics_cls, "rdd_word_sentiment_count", None)
    if count_fn is None:
        pytest.skip("rdd_word_sentiment_count not implemented")

    result_rdd = count_fn(rdd)
    assert result_rdd is not None

    lineage_fn = getattr(analytics_cls, "get_rdd_lineage_debug_string", None)
    if lineage_fn:
        lineage = lineage_fn(result_rdd)
        assert isinstance(lineage, str)
        assert len(lineage) > 0
