"""
Graph Analytics Module for Product Reviews.
Module: src.graph.review_graph_analytics
Syllabus Mapping: Unit 5 & 6 (Graph Analytics, Neo4j/Network Modeling, Centrality, PageRank)
CO Mapping: CO4 (Graph Analytics & Advanced Analytics)

Constructs bipartite reviewer-product networks, computes PageRank and Degree Centrality,
and extracts sentiment-driven subgraphs (e.g. negative sentiment cascades).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class ReviewGraphAnalytics:
    """
    Analytics engine for modeling product-reviewer interactions as a bipartite graph.
    Supports degree centrality, PageRank, and sentiment-based subgraph filtering.
    """

    def __init__(self) -> None:
        self.graph: Optional[nx.Graph] = None
        self.reviewer_nodes: set = set()
        self.product_nodes: set = set()

    def build_bipartite_graph(
        self,
        reviews_data: Union[List[Dict[str, Any]], pd.DataFrame, Any],
    ) -> nx.Graph:
        """
        Constructs an undirected bipartite graph G = (U, V, E) where:
        - Set U (bipartite=0): Reviewer nodes
        - Set V (bipartite=1): Product nodes
        - Set E: Review edges with attributes (rating, sentiment, review_id, helpful_votes)
        """
        G = nx.Graph()
        self.reviewer_nodes.clear()
        self.product_nodes.clear()

        # Normalize input to list of dicts
        records: List[Dict[str, Any]] = []
        if isinstance(reviews_data, pd.DataFrame):
            records = reviews_data.to_dict(orient="records")
        elif isinstance(reviews_data, list):
            records = reviews_data
        elif hasattr(reviews_data, "collect"):  # Spark or SimulatedDataFrame
            try:
                records = [row.asDict() if hasattr(row, "asDict") else dict(row) for row in reviews_data.collect()]
            except Exception:
                records = []
        elif hasattr(reviews_data, "to_dict"):
            records = reviews_data.to_dict(orient="records")

        for row in records:
            rev_id = str(row.get("reviewer_id") or f"USR_{row.get('review_id', 'unknown')}")
            # Ensure unique node namespace between reviewers and products
            u_node = f"user:{rev_id}"
            prod_id = str(row.get("product_id") or "PROD_UNKNOWN")
            p_node = f"prod:{prod_id}"

            sentiment = str(row.get("sentiment_label") or row.get("sentiment") or "neutral").lower()
            rating = float(row.get("rating", 3.0))
            helpful_votes = int(row.get("helpful_votes", 0))

            if u_node not in G:
                G.add_node(
                    u_node,
                    bipartite=0,
                    node_type="reviewer",
                    reviewer_id=rev_id,
                    name=row.get("reviewer_name", rev_id),
                )
                self.reviewer_nodes.add(u_node)

            if p_node not in G:
                G.add_node(
                    p_node,
                    bipartite=1,
                    node_type="product",
                    product_id=prod_id,
                    product_name=row.get("product_name", prod_id),
                    product_category=row.get("product_category", "General"),
                )
                self.product_nodes.add(p_node)

            # Add or update edge
            G.add_edge(
                u_node,
                p_node,
                review_id=str(row.get("review_id", "")),
                rating=rating,
                sentiment=sentiment,
                sentiment_label=sentiment,
                helpful_votes=helpful_votes,
                weight=max(0.1, rating / 5.0),
            )

        self.graph = G
        logger.info(
            f"Built bipartite graph: {len(self.reviewer_nodes)} reviewers, "
            f"{len(self.product_nodes)} products, {G.number_of_edges()} edges"
        )
        return G

    def compute_product_pagerank(
        self,
        G: Optional[nx.Graph] = None,
        alpha: float = 0.85,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """
        Computes PageRank over the graph.
        Returns a dictionary mapping node IDs to their PageRank scores.
        Sum of returned scores across all nodes is 1.0 (satisfies invariant).
        """
        target_g = G if G is not None else self.graph
        if target_g is None or target_g.number_of_nodes() == 0:
            return {}

        pr: Dict[str, float] = nx.pagerank(target_g, alpha=alpha, max_iter=max_iter, weight="weight")
        return pr

    def compute_reviewer_centrality(
        self,
        G: Optional[nx.Graph] = None,
    ) -> Dict[str, float]:
        """
        Computes degree centrality for reviewers in the bipartite graph.
        Returns dict mapping reviewer node IDs to centrality scores in [0.0, 1.0].
        """
        target_g = G if G is not None else self.graph
        if target_g is None or target_g.number_of_nodes() == 0:
            return {}

        centrality: Dict[str, float] = nx.degree_centrality(target_g)
        return centrality

    def extract_negative_sentiment_subgraph(
        self,
        G: Optional[nx.Graph] = None,
    ) -> nx.Graph:
        """
        Extracts subgraph containing only negative review interactions
        (sentiment == 'negative' or rating <= 2.0).
        Useful for analyzing defect contagion or critical product issues.
        """
        target_g = G if G is not None else self.graph
        if target_g is None:
            return nx.Graph()

        neg_edges = []
        for u, v, data in target_g.edges(data=True):
            sent = data.get("sentiment", data.get("sentiment_label", "")).lower()
            rating = float(data.get("rating", 5.0))
            if sent == "negative" or rating <= 2.0:
                neg_edges.append((u, v, data))

        neg_g = nx.Graph()
        for u, v, data in neg_edges:
            if u not in neg_g:
                neg_g.add_node(u, **target_g.nodes[u])
            if v not in neg_g:
                neg_g.add_node(v, **target_g.nodes[v])
            neg_g.add_edge(u, v, **data)

        return neg_g

    def get_summary_metrics(self, G: Optional[nx.Graph] = None) -> Dict[str, Any]:
        """Returns key topological metrics of the bipartite graph."""
        target_g = G if G is not None else self.graph
        if target_g is None or target_g.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "num_reviewers": 0,
                "num_products": 0,
                "density": 0.0,
                "is_bipartite": True,
            }

        rev_count = sum(1 for _, d in target_g.nodes(data=True) if d.get("bipartite") == 0)
        prod_count = sum(1 for _, d in target_g.nodes(data=True) if d.get("bipartite") == 1)

        return {
            "num_nodes": target_g.number_of_nodes(),
            "num_edges": target_g.number_of_edges(),
            "num_reviewers": rev_count,
            "num_products": prod_count,
            "density": float(nx.density(target_g)),
            "is_bipartite": bool(nx.is_bipartite(target_g)),
        }
