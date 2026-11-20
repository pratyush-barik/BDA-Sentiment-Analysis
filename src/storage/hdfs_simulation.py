"""
Hadoop Distributed File System (HDFS) Simulation & Big Data 5Vs Analysis Engine.
Module: src.storage.hdfs_simulation
Aligned with Course Syllabus Unit 1 (CO3/CO4).

Provides:
- NameNode metadata tracking (Inodes, Block Maps, Generation Stamps).
- DataNode virtual replica storage with Rack Awareness and Checksum validation.
- Fault tolerance simulation: DataNode failure injection and self-healing replication.
- Big Data 5Vs Analytical Engine (Volume, Velocity, Variety, Veracity, Value).
- HDFS vs RDBMS comparative benchmark.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================================
# 1. HDFS Metadata & Node Abstractions
# ============================================================================

@dataclass
class BlockMetadata:
    """Represents metadata for a discrete HDFS block chunk."""
    block_id: str
    block_index: int
    file_path: str
    block_size_bytes: int
    actual_size_bytes: int
    checksum: str
    datanodes: List[str] = field(default_factory=list)
    generation_stamp: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class INode:
    """Represents a file or directory inode in the NameNode namespace."""
    path: str
    is_directory: bool
    file_size_bytes: int
    block_size_bytes: int
    replication_factor: int
    blocks: List[BlockMetadata] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    permissions: str = "rw-r--r--"
    owner: str = "bda_student"

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["blocks"] = [b.to_dict() for b in self.blocks]
        return d


class DataNode:
    """Simulates an HDFS worker node storing block replica payloads."""

    def __init__(
        self,
        node_id: str,
        rack_id: str = "/rack_1",
        capacity_bytes: int = 10 * 1024 * 1024 * 1024,  # 10 GB
        storage_path: Optional[Path] = None,
    ):
        self.node_id = node_id
        self.rack_id = rack_id
        self.capacity_bytes = capacity_bytes
        self.storage_path = Path(storage_path) if storage_path else None
        self.status = "ACTIVE"  # "ACTIVE" | "DEAD" | "DECOMMISSIONED"
        self._in_memory_blocks: Dict[str, bytes] = {}
        self._checksums: Dict[str, str] = {}
        self.last_heartbeat = time.time()

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def is_alive(self) -> bool:
        return self.status == "ACTIVE"

    def heartbeat(self) -> None:
        if self.is_alive:
            self.last_heartbeat = time.time()

    def write_block(self, block_id: str, data: bytes, checksum: str) -> bool:
        if not self.is_alive:
            raise IOError(f"DataNode {self.node_id} is DEAD: Cannot accept write.")
        if self.get_used_bytes() + len(data) > self.capacity_bytes:
            raise IOError(f"DataNode {self.node_id} disk full: Insufficient capacity.")

        self._in_memory_blocks[block_id] = data
        self._checksums[block_id] = checksum

        if self.storage_path:
            block_file = self.storage_path / f"{block_id}.dat"
            meta_file = self.storage_path / f"{block_id}.meta"
            block_file.write_bytes(data)
            meta_file.write_text(checksum, encoding="utf-8")

        return True

    def read_block(self, block_id: str) -> bytes:
        if not self.is_alive:
            raise IOError(f"DataNode {self.node_id} is DEAD: Cannot read block {block_id}.")
        if block_id not in self._in_memory_blocks:
            if self.storage_path:
                block_file = self.storage_path / f"{block_id}.dat"
                if block_file.exists():
                    return block_file.read_bytes()
            raise KeyError(f"Block {block_id} not found on DataNode {self.node_id}.")

        return self._in_memory_blocks[block_id]

    def verify_block_checksum(self, block_id: str) -> bool:
        data = self.read_block(block_id)
        computed = hashlib.sha256(data).hexdigest()
        expected = self._checksums.get(block_id)
        return computed == expected

    def delete_block(self, block_id: str) -> bool:
        self._in_memory_blocks.pop(block_id, None)
        self._checksums.pop(block_id, None)
        if self.storage_path:
            (self.storage_path / f"{block_id}.dat").unlink(missing_ok=True)
            (self.storage_path / f"{block_id}.meta").unlink(missing_ok=True)
        return True

    def has_block(self, block_id: str) -> bool:
        return block_id in self._in_memory_blocks

    def get_used_bytes(self) -> int:
        return sum(len(b) for b in self._in_memory_blocks.values())

    def get_available_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.get_used_bytes())

    def get_block_list(self) -> List[str]:
        return list(self._in_memory_blocks.keys())


class NameNode:
    """Simulates the HDFS NameNode managing filesystem namespace and block mappings."""

    def __init__(self):
        self.inodes: Dict[str, INode] = {}
        self.block_locations: Dict[str, Set[str]] = {}  # block_id -> Set[node_id]
        self.block_metadata_map: Dict[str, BlockMetadata] = {}
        self.datanodes: Dict[str, DataNode] = {}
        self._block_sequence: int = 1000
        self.safe_mode: bool = False

    def register_datanode(self, datanode: DataNode) -> None:
        self.datanodes[datanode.node_id] = datanode

    def allocate_block_id(self) -> str:
        self._block_sequence += 1
        return f"blk_{self._block_sequence:08d}"

    def select_datanodes_for_replication(
        self,
        replication_factor: int,
        exclude_nodes: Optional[Set[str]] = None,
    ) -> List[DataNode]:
        """
        Implements Hadoop Rack-Aware Replica Placement Policy:
        - Replica 1: Placed on a DataNode on Rack 1.
        - Replica 2: Placed on a DataNode on Rack 2 (different rack).
        - Replica 3: Placed on another DataNode on Rack 2.
        """
        exclude = exclude_nodes or set()
        active_nodes = [dn for dn in self.datanodes.values() if dn.is_alive and dn.node_id not in exclude]

        if len(active_nodes) < replication_factor:
            return active_nodes

        # Group by rack
        by_rack: Dict[str, List[DataNode]] = {}
        for dn in active_nodes:
            by_rack.setdefault(dn.rack_id, []).append(dn)

        racks = list(by_rack.keys())
        selected: List[DataNode] = []

        if len(racks) >= 2 and replication_factor >= 3:
            rack_1, rack_2 = racks[0], racks[1]
            # Replica 1: Rack 1, Node 1
            selected.append(by_rack[rack_1][0])
            # Replica 2: Rack 2, Node 1
            selected.append(by_rack[rack_2][0])
            # Replica 3: Rack 2, Node 2 (or Rack 1, Node 2)
            if len(by_rack[rack_2]) > 1:
                selected.append(by_rack[rack_2][1])
            elif len(by_rack[rack_1]) > 1:
                selected.append(by_rack[rack_1][1])
            else:
                remaining = [dn for dn in active_nodes if dn not in selected]
                if remaining:
                    selected.append(remaining[0])
        else:
            selected = active_nodes[:replication_factor]

        return selected


# ============================================================================
# 2. Main HDFSSimulator Class
# ============================================================================

class HDFSSimulator:
    """
    High-fidelity simulator of the Hadoop Distributed File System (HDFS).
    Complies with Interface Contract 2 in PROJECT.md.
    """

    def __init__(
        self,
        block_size_mb: float = 128.0,
        block_size_bytes: Optional[int] = None,
        replication_factor: int = 3,
        num_datanodes: int = 5,
        storage_dir: Optional[Union[str, Path]] = None,
    ):
        if block_size_bytes is not None:
            self.block_size_bytes = int(block_size_bytes)
        else:
            self.block_size_bytes = int(block_size_mb * 1024 * 1024)

        self.replication_factor = max(1, replication_factor)
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.namenode = NameNode()

        # Initialize DataNodes distributed across 2 virtual racks
        self._init_cluster(num_datanodes)

    def _init_cluster(self, num_datanodes: int) -> None:
        for i in range(1, num_datanodes + 1):
            node_id = f"dn_{i:02d}"
            rack_id = "/rack_1" if i <= 2 else "/rack_2"
            dn_storage = self.storage_dir / node_id if self.storage_dir else None
            dn = DataNode(node_id=node_id, rack_id=rack_id, storage_path=dn_storage)
            self.namenode.register_datanode(dn)

    # ------------------------------------------------------------------------
    # Core Operations: write_file, read_file, get_block_report
    # ------------------------------------------------------------------------

    def write_file(self, source_path: Union[str, Path], hdfs_path: str) -> List[BlockMetadata]:
        """
        Partitions local source file into HDFS block chunks and replicates them across DataNodes.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Local source file does not exist: {source_path}")

        file_bytes = source.read_bytes()
        return self.write_bytes(file_bytes, hdfs_path)

    def write_bytes(self, data: bytes, hdfs_path: str) -> List[BlockMetadata]:
        """Writes in-memory byte buffer into simulated HDFS blocks."""
        hdfs_path = self._normalize_path(hdfs_path)
        total_size = len(data)

        # Handle empty 0-byte file edge case
        if total_size == 0:
            inode = INode(
                path=hdfs_path,
                is_directory=False,
                file_size_bytes=0,
                block_size_bytes=self.block_size_bytes,
                replication_factor=self.replication_factor,
                blocks=[],
            )
            self.namenode.inodes[hdfs_path] = inode
            return []

        num_blocks = math.ceil(total_size / self.block_size_bytes)
        blocks_metadata: List[BlockMetadata] = []

        for blk_idx in range(num_blocks):
            start = blk_idx * self.block_size_bytes
            end = min(start + self.block_size_bytes, total_size)
            chunk_data = data[start:end]
            actual_size = len(chunk_data)

            block_id = self.namenode.allocate_block_id()
            checksum = hashlib.sha256(chunk_data).hexdigest()

            # Select replica DataNodes according to rack awareness policy
            target_nodes = self.namenode.select_datanodes_for_replication(self.replication_factor)
            if not target_nodes:
                raise IOError("HDFS write failed: No active DataNodes available in cluster.")

            assigned_node_ids = []
            for dn in target_nodes:
                dn.write_block(block_id, chunk_data, checksum)
                assigned_node_ids.append(dn.node_id)

            blk_meta = BlockMetadata(
                block_id=block_id,
                block_index=blk_idx,
                file_path=hdfs_path,
                block_size_bytes=self.block_size_bytes,
                actual_size_bytes=actual_size,
                checksum=checksum,
                datanodes=assigned_node_ids,
            )

            blocks_metadata.append(blk_meta)
            self.namenode.block_locations[block_id] = set(assigned_node_ids)
            self.namenode.block_metadata_map[block_id] = blk_meta

        inode = INode(
            path=hdfs_path,
            is_directory=False,
            file_size_bytes=total_size,
            block_size_bytes=self.block_size_bytes,
            replication_factor=self.replication_factor,
            blocks=blocks_metadata,
        )
        self.namenode.inodes[hdfs_path] = inode
        return blocks_metadata

    def read_file(self, hdfs_path: str) -> bytes:
        """
        Reads all blocks of an HDFS file, verifying checksums and falling back
        to surviving replicas if individual DataNodes fail.
        """
        hdfs_path = self._normalize_path(hdfs_path)
        if hdfs_path not in self.namenode.inodes:
            raise FileNotFoundError(f"File not found in HDFS: {hdfs_path}")

        inode = self.namenode.inodes[hdfs_path]
        if not inode.blocks:
            return b""

        reassembled_chunks: List[bytes] = []

        for blk in inode.blocks:
            block_id = blk.block_id
            expected_checksum = blk.checksum
            surviving_nodes = self.namenode.block_locations.get(block_id, set())

            chunk_read = None
            for node_id in surviving_nodes:
                dn = self.namenode.datanodes.get(node_id)
                if dn and dn.is_alive:
                    try:
                        payload = dn.read_block(block_id)
                        actual_checksum = hashlib.sha256(payload).hexdigest()
                        if actual_checksum == expected_checksum:
                            chunk_read = payload
                            break
                    except Exception:
                        continue

            if chunk_read is None:
                raise IOError(
                    f"HDFS Read Corrupted: Block {block_id} of file {hdfs_path} "
                    f"is unavailable across all replica DataNodes {surviving_nodes}."
                )

            reassembled_chunks.append(chunk_read)

        return b"".join(reassembled_chunks)

    def read_file_to_local(self, hdfs_path: str, target_path: Union[str, Path]) -> Path:
        """Helper to stream HDFS file directly to local filesystem."""
        data = self.read_file(hdfs_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def get_block_report(self) -> Dict[str, Any]:
        """Produces a comprehensive HDFS cluster status and block distribution report."""
        total_files = len(self.namenode.inodes)
        total_blocks = len(self.namenode.block_metadata_map)
        active_nodes = sum(1 for dn in self.namenode.datanodes.values() if dn.is_alive)
        dead_nodes = sum(1 for dn in self.namenode.datanodes.values() if not dn.is_alive)

        total_capacity = sum(dn.capacity_bytes for dn in self.namenode.datanodes.values())
        total_used = sum(dn.get_used_bytes() for dn in self.namenode.datanodes.values() if dn.is_alive)

        # Health assessment
        under_replicated_count = 0
        corrupt_or_missing_count = 0

        for blk_id, locs in self.namenode.block_locations.items():
            alive_replicas = [nid for nid in locs if self.namenode.datanodes[nid].is_alive]
            if len(alive_replicas) == 0:
                corrupt_or_missing_count += 1
            elif len(alive_replicas) < self.replication_factor:
                under_replicated_count += 1

        cluster_health = "HEALTHY"
        if corrupt_or_missing_count > 0:
            cluster_health = "CRITICAL"
        elif under_replicated_count > 0 or dead_nodes > 0:
            cluster_health = "DEGRADED"

        datanode_reports = []
        for dn in self.namenode.datanodes.values():
            datanode_reports.append({
                "node_id": dn.node_id,
                "rack_id": dn.rack_id,
                "status": dn.status,
                "block_count": len(dn.get_block_list()),
                "used_bytes": dn.get_used_bytes(),
                "capacity_bytes": dn.capacity_bytes,
                "utilization_pct": round((dn.get_used_bytes() / dn.capacity_bytes) * 100, 2),
            })

        return {
            "cluster_summary": {
                "total_files": total_files,
                "total_blocks": total_blocks,
                "configured_block_size_bytes": self.block_size_bytes,
                "configured_block_size_mb": round(self.block_size_bytes / (1024 * 1024), 2),
                "replication_factor": self.replication_factor,
                "active_datanodes": active_nodes,
                "dead_datanodes": dead_nodes,
                "total_capacity_bytes": total_capacity,
                "total_used_bytes": total_used,
                "cluster_health": cluster_health,
                "under_replicated_blocks": under_replicated_count,
                "missing_or_corrupt_blocks": corrupt_or_missing_count,
            },
            "datanodes": datanode_reports,
            "files": [inode.to_dict() for inode in self.namenode.inodes.values()],
        }

    # ------------------------------------------------------------------------
    # Fault Tolerance & Self-Healing
    # ------------------------------------------------------------------------

    def simulate_datanode_failure(self, node_id: str) -> Dict[str, Any]:
        """Simulates node crash/heartbeat timeout by marking DataNode DEAD."""
        if node_id not in self.namenode.datanodes:
            raise KeyError(f"DataNode {node_id} does not exist.")

        dn = self.namenode.datanodes[node_id]
        dn.status = "DEAD"

        affected_blocks = [
            blk_id for blk_id, locs in self.namenode.block_locations.items()
            if node_id in locs
        ]

        return {
            "failed_node": node_id,
            "status": "DEAD",
            "affected_blocks_count": len(affected_blocks),
            "affected_blocks": affected_blocks,
        }

    def recover_datanode(self, node_id: str) -> bool:
        """Restores a failed DataNode back to ACTIVE state."""
        if node_id not in self.namenode.datanodes:
            return False
        self.namenode.datanodes[node_id].status = "ACTIVE"
        return True

    def rebalance_or_recover(self) -> Dict[str, Any]:
        """
        Scans NameNode block location table, detects under-replicated blocks,
        and replicates them to available healthy DataNodes to restore target replication factor.
        """
        recovered_blocks = []
        unrecoverable_blocks = []

        for blk_id, blk_meta in self.namenode.block_metadata_map.items():
            current_locs = self.namenode.block_locations.get(blk_id, set())
            active_locs = {nid for nid in current_locs if self.namenode.datanodes[nid].is_alive}

            if len(active_locs) == 0:
                unrecoverable_blocks.append(blk_id)
                continue

            needed_replicas = self.replication_factor - len(active_locs)
            if needed_replicas > 0:
                source_nid = next(iter(active_locs))
                source_dn = self.namenode.datanodes[source_nid]
                payload = source_dn.read_block(blk_id)

                candidate_dns = [
                    dn for dn in self.namenode.datanodes.values()
                    if dn.is_alive and dn.node_id not in current_locs
                ]

                new_nodes = candidate_dns[:needed_replicas]
                for target_dn in new_nodes:
                    target_dn.write_block(blk_id, payload, blk_meta.checksum)
                    self.namenode.block_locations[blk_id].add(target_dn.node_id)
                    blk_meta.datanodes.append(target_dn.node_id)

                blk_meta.generation_stamp += 1
                recovered_blocks.append({
                    "block_id": blk_id,
                    "replicated_from": source_nid,
                    "replicated_to": [dn.node_id for dn in new_nodes],
                    "restored_replication": len(self.namenode.block_locations[blk_id]),
                })

        return {
            "recovery_timestamp": time.time(),
            "under_replicated_found": len(recovered_blocks),
            "recovered_blocks": recovered_blocks,
            "unrecoverable_blocks": unrecoverable_blocks,
            "cluster_status": "HEALTHY" if not unrecoverable_blocks else "CRITICAL",
        }

    # ------------------------------------------------------------------------
    # Filesystem Helpers
    # ------------------------------------------------------------------------

    def list_files(self, hdfs_dir: str = "/") -> List[Dict[str, Any]]:
        hdfs_dir = self._normalize_path(hdfs_dir)
        results = []
        for path, inode in self.namenode.inodes.items():
            if path.startswith(hdfs_dir):
                results.append({
                    "path": inode.path,
                    "size_bytes": inode.file_size_bytes,
                    "blocks": len(inode.blocks),
                    "replication": inode.replication_factor,
                    "modified_at": inode.modified_at,
                    "permissions": inode.permissions,
                })
        return results

    def delete_file(self, hdfs_path: str) -> bool:
        hdfs_path = self._normalize_path(hdfs_path)
        if hdfs_path not in self.namenode.inodes:
            return False

        inode = self.namenode.inodes.pop(hdfs_path)
        for blk in inode.blocks:
            blk_id = blk.block_id
            locs = self.namenode.block_locations.pop(blk_id, set())
            for nid in locs:
                dn = self.namenode.datanodes.get(nid)
                if dn:
                    dn.delete_block(blk_id)
            self.namenode.block_metadata_map.pop(blk_id, None)

        return True

    def exists(self, hdfs_path: str) -> bool:
        return self._normalize_path(hdfs_path) in self.namenode.inodes

    @staticmethod
    def _normalize_path(path: str) -> str:
        p = path.strip().replace("\\", "/")
        if not p.startswith("/"):
            p = "/" + p
        return p


# ============================================================================
# 3. Big Data 5Vs Analytical Engine (Unit 1 Syllabus)
# ============================================================================

@dataclass
class VolumeMetrics:
    total_bytes: int
    total_records: int
    total_columns: int
    memory_footprint_bytes: int
    avg_record_size_bytes: float
    total_words: int
    vocabulary_size: int
    projected_1m_bytes: int
    projected_1m_hdfs_blocks_128mb: int
    projected_10m_bytes: int
    projected_10m_hdfs_blocks_128mb: int


@dataclass
class VelocityMetrics:
    ingestion_duration_seconds: float
    records_per_second: float
    mb_per_second: float
    min_date: str
    max_date: str
    time_span_days: int
    avg_reviews_per_day: float
    spark_streaming_1s_microbatch_capacity: int


@dataclass
class VarietyMetrics:
    data_modalities: List[str]
    column_types: Dict[str, str]
    category_cardinality: int
    category_distribution: Dict[str, int]
    rating_distribution: Dict[str, int]
    verified_purchase_ratio: float
    avg_review_character_length: float
    max_review_character_length: int
    lexical_diversity_ttr: float  # Type-Token Ratio


@dataclass
class VeracityMetrics:
    overall_completeness_pct: float
    null_rates_per_column: Dict[str, float]
    duplicate_record_count: int
    duplicate_review_id_count: int
    rating_boundary_valid_pct: float
    vote_logic_valid_pct: float  # helpful_votes <= total_votes
    sentiment_consistency_rate_pct: float  # rating vs sentiment_label
    contradictory_anomalies_count: int
    html_tag_presence_count: int
    url_presence_count: int


@dataclass
class ValueMetrics:
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    net_sentiment_polarity_index: float  # % Pos - % Neg (NPS style)
    average_customer_satisfaction_rating: float
    category_performance: Dict[str, Dict[str, Any]]
    high_impact_reviews_count: int  # helpful_votes >= 10
    verified_vs_unverified_nps_delta: float


@dataclass
class BigData5VReport:
    timestamp: float
    dataset_name: str
    volume: VolumeMetrics
    velocity: VelocityMetrics
    variety: VarietyMetrics
    veracity: VeracityMetrics
    value: ValueMetrics

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_markdown(self) -> str:
        """Renders an academic submission table formatted for Chapter 2 of PROJECT_REPORT.md."""
        v = self.volume
        vel = self.velocity
        var = self.variety
        ver = self.veracity
        val = self.value

        return f"""### Big Data 5Vs Empirical Benchmark Report
| Dimension | Key Metric Indicator | Empirical Value | Academic / Business Significance |
|---|---|---|---|
| **Volume** | Total Raw Records | {v.total_records:,} rows | Sample volume baseline for batch processing |
| | Storage Size (Raw Bytes) | {v.total_bytes:,} bytes ({v.total_bytes / (1024*1024):.2f} MB) | In-memory vs disk chunking requirements |
| | Average Record Size | {v.avg_record_size_bytes:.1f} bytes | Enables scaling projections to 1M/10M scale |
| | Scaling: 1M Reviews HDFS Chunks | {v.projected_1m_hdfs_blocks_128mb} blocks (128MB) | Proves necessity of distributed block partitioning |
| **Velocity** | Ingestion Throughput | {vel.records_per_second:,.1f} records/sec | Measures single-node parsing capacity |
| | Bandwidth Throughput | {vel.mb_per_second:.2f} MB/sec | Ingestion bandwidth for ETL pipelines |
| | Streaming 1s Microbatch Window | {vel.spark_streaming_1s_microbatch_capacity:,} records | Spark Structured Streaming throughput limit |
| **Variety** | Data Modalities | 5 Modalities | Unstructured text, continuous/discrete numbers, categorical, temporal, boolean |
| | Lexical Diversity (TTR) | {var.lexical_diversity_ttr:.4f} | Lexical richness of consumer reviews |
| | Verified Purchase Ratio | {var.verified_purchase_ratio * 100:.1f}% | Trust diversity in reviewer population |
| **Veracity** | Dataset Completeness | {ver.overall_completeness_pct:.2f}% | Overall cell data hygiene |
| | Vote Logic Integrity | {ver.vote_logic_valid_pct:.2f}% | Verifies helpful_votes <= total_votes |
| | Sentiment Consistency Rate | {ver.sentiment_consistency_rate_pct:.2f}% | Semantic harmony between rating and sentiment label |
| | Contradictory Anomalies | {ver.contradictory_anomalies_count} | Outliers flagged for data cleaner sanitization |
| **Value** | Net Sentiment Polarity (NSPI) | {val.net_sentiment_polarity_index:+.2f}% | Executive customer satisfaction index |
| | Average Rating (CSAT) | {val.average_customer_satisfaction_rating:.2f} / 5.0 | Brand health across entire product catalog |
| | High-Impact Reviews | {val.high_impact_reviews_count:,} reviews | Decisive reviews driving purchase behavior |
"""


class BigData5VAnalysis:
    """
    Engine executing comprehensive empirical profiling of product review datasets
    across the 5 V's of Big Data (Volume, Velocity, Variety, Veracity, Value).
    """

    HTML_REGEX = re.compile(r"<[^>]+>")
    URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
    WORD_REGEX = re.compile(r"\b[a-zA-Z]{2,}\b")

    def __init__(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], str, Path],
        dataset_name: str = "AmazonProductReviews",
    ):
        self.dataset_name = dataset_name
        self.ingestion_start_time = time.time()

        if isinstance(data, (str, Path)):
            path = Path(data)
            self.raw_file_bytes = path.stat().st_size if path.exists() else 0
            if path.suffix.lower() == ".csv":
                self.df = pd.read_csv(path)
            elif path.suffix.lower() in [".json", ".jsonl"]:
                self.df = pd.read_json(path, lines=True)
            elif path.suffix.lower() == ".parquet":
                self.df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        elif isinstance(data, list):
            self.df = pd.DataFrame(data)
            self.raw_file_bytes = int(self.df.memory_usage(deep=True).sum())
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.raw_file_bytes = int(self.df.memory_usage(deep=True).sum())
        else:
            raise TypeError("Expected filepath, List[Dict], or DataFrame.")

        self.ingestion_duration = max(0.0001, time.time() - self.ingestion_start_time)

    def analyze_volume(self) -> VolumeMetrics:
        total_records = len(self.df)
        total_cols = len(self.df.columns)
        mem_bytes = int(self.df.memory_usage(deep=True).sum()) if total_records > 0 else 0
        file_bytes = self.raw_file_bytes or mem_bytes

        avg_record_bytes = (file_bytes / total_records) if total_records > 0 else 0.0

        all_words: List[str] = []
        if "review_text" in self.df.columns and total_records > 0:
            for text in self.df["review_text"].dropna():
                all_words.extend(self.WORD_REGEX.findall(str(text).lower()))

        total_words = len(all_words)
        vocab_size = len(set(all_words))

        proj_1m_bytes = int(avg_record_bytes * 1_000_000)
        proj_1m_blocks = max(1, math.ceil(proj_1m_bytes / (128 * 1024 * 1024)))

        proj_10m_bytes = int(avg_record_bytes * 10_000_000)
        proj_10m_blocks = max(1, math.ceil(proj_10m_bytes / (128 * 1024 * 1024)))

        return VolumeMetrics(
            total_bytes=file_bytes,
            total_records=total_records,
            total_columns=total_cols,
            memory_footprint_bytes=mem_bytes,
            avg_record_size_bytes=round(avg_record_bytes, 2),
            total_words=total_words,
            vocabulary_size=vocab_size,
            projected_1m_bytes=proj_1m_bytes,
            projected_1m_hdfs_blocks_128mb=proj_1m_blocks,
            projected_10m_bytes=proj_10m_bytes,
            projected_10m_hdfs_blocks_128mb=proj_10m_blocks,
        )

    def analyze_velocity(self, duration_override: Optional[float] = None) -> VelocityMetrics:
        duration = duration_override or self.ingestion_duration
        duration = max(0.0001, duration)
        total_records = len(self.df)

        rps = total_records / duration
        mb_ps = (self.raw_file_bytes / (1024 * 1024)) / duration

        min_date, max_date = "N/A", "N/A"
        span_days = 1
        avg_per_day = float(total_records)

        if "review_date" in self.df.columns and total_records > 0:
            try:
                dates = pd.to_datetime(self.df["review_date"].dropna())
                if not dates.empty:
                    min_dt = dates.min()
                    max_dt = dates.max()
                    min_date = min_dt.strftime("%Y-%m-%d")
                    max_date = max_dt.strftime("%Y-%m-%d")
                    span_days = max(1, (max_dt - min_dt).days + 1)
                    avg_per_day = total_records / span_days
            except Exception:
                pass

        return VelocityMetrics(
            ingestion_duration_seconds=round(duration, 4),
            records_per_second=round(rps, 2),
            mb_per_second=round(mb_ps, 2),
            min_date=min_date,
            max_date=max_date,
            time_span_days=span_days,
            avg_reviews_per_day=round(avg_per_day, 2),
            spark_streaming_1s_microbatch_capacity=int(rps),
        )

    def analyze_variety(self) -> VarietyMetrics:
        total_records = len(self.df)
        col_types = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

        modalities = [
            "Unstructured Text (review_text, review_summary)",
            "Continuous/Discrete Numerical (rating, helpful_votes, total_votes)",
            "Categorical Discrete (product_category, sentiment_label)",
            "Temporal (review_date, timestamp)",
            "Boolean (verified_purchase)",
            "Relational Identifiers (review_id, reviewer_id, product_id)",
        ]

        cat_dist = {}
        if "product_category" in self.df.columns:
            cat_dist = self.df["product_category"].value_counts().to_dict()

        rating_dist = {}
        if "rating" in self.df.columns:
            rating_dist = {str(k): int(v) for k, v in self.df["rating"].value_counts().to_dict().items()}

        verified_ratio = 0.0
        if "verified_purchase" in self.df.columns and total_records > 0:
            verified_ratio = float(self.df["verified_purchase"].mean())

        avg_char_len = 0.0
        max_char_len = 0
        ttr = 0.0
        if "review_text" in self.df.columns and total_records > 0:
            lengths = self.df["review_text"].dropna().astype(str).str.len()
            avg_char_len = float(lengths.mean()) if not lengths.empty else 0.0
            max_char_len = int(lengths.max()) if not lengths.empty else 0

            all_tokens = []
            for t in self.df["review_text"].dropna():
                all_tokens.extend(self.WORD_REGEX.findall(str(t).lower()))
            if all_tokens:
                ttr = len(set(all_tokens)) / len(all_tokens)

        return VarietyMetrics(
            data_modalities=modalities,
            column_types=col_types,
            category_cardinality=len(cat_dist),
            category_distribution=cat_dist,
            rating_distribution=rating_dist,
            verified_purchase_ratio=round(verified_ratio, 4),
            avg_review_character_length=round(avg_char_len, 2),
            max_review_character_length=max_char_len,
            lexical_diversity_ttr=round(ttr, 4),
        )

    def analyze_veracity(self) -> VeracityMetrics:
        total_records = len(self.df)
        total_cells = total_records * len(self.df.columns)
        null_cells = int(self.df.isna().sum().sum())
        completeness = ((total_cells - null_cells) / total_cells * 100.0) if total_cells > 0 else 100.0

        null_rates = {}
        for col in self.df.columns:
            null_count = int(self.df[col].isna().sum())
            null_rates[col] = round((null_count / total_records) * 100.0, 2) if total_records > 0 else 0.0

        dup_rows = int(self.df.duplicated().sum())
        dup_rev_id = int(self.df["review_id"].duplicated().sum()) if "review_id" in self.df.columns else 0

        rating_valid_pct = 100.0
        if "rating" in self.df.columns and total_records > 0:
            valid_mask = self.df["rating"].between(1.0, 5.0)
            rating_valid_pct = float(valid_mask.mean() * 100.0)

        vote_valid_pct = 100.0
        if "helpful_votes" in self.df.columns and "total_votes" in self.df.columns and total_records > 0:
            vote_mask = self.df["helpful_votes"] <= self.df["total_votes"]
            vote_valid_pct = float(vote_mask.mean() * 100.0)

        consistency_pct = 100.0
        contradictory_count = 0
        if "rating" in self.df.columns and "sentiment_label" in self.df.columns and total_records > 0:
            def check_alignment(row):
                r = row["rating"]
                s = str(row["sentiment_label"]).lower()
                if r >= 4.0 and s == "negative":
                    return False
                if r <= 2.0 and s == "positive":
                    return False
                return True

            alignment_series = self.df.apply(check_alignment, axis=1)
            consistency_pct = float(alignment_series.mean() * 100.0)
            contradictory_count = int((~alignment_series).sum())

        html_count = 0
        url_count = 0
        if "review_text" in self.df.columns:
            text_series = self.df["review_text"].dropna().astype(str)
            html_count = int(text_series.str.contains(r"<[^>]+>", regex=True).sum())
            url_count = int(text_series.str.contains(r"https?://|www\.", regex=True).sum())

        return VeracityMetrics(
            overall_completeness_pct=round(completeness, 2),
            null_rates_per_column=null_rates,
            duplicate_record_count=dup_rows,
            duplicate_review_id_count=dup_rev_id,
            rating_boundary_valid_pct=round(rating_valid_pct, 2),
            vote_logic_valid_pct=round(vote_valid_pct, 2),
            sentiment_consistency_rate_pct=round(consistency_pct, 2),
            contradictory_anomalies_count=contradictory_count,
            html_tag_presence_count=html_count,
            url_presence_count=url_count,
        )

    def analyze_value(self) -> ValueMetrics:
        total_records = len(self.df)

        sentiment_dist = {}
        sentiment_pcts = {}
        nspi = 0.0

        if "sentiment_label" in self.df.columns and total_records > 0:
            s_counts = self.df["sentiment_label"].str.lower().value_counts().to_dict()
            sentiment_dist = {str(k): int(v) for k, v in s_counts.items()}
            pos = sentiment_dist.get("positive", 0)
            neg = sentiment_dist.get("negative", 0)
            neu = sentiment_dist.get("neutral", 0)
            sentiment_pcts = {
                "positive": round((pos / total_records) * 100.0, 2),
                "neutral": round((neu / total_records) * 100.0, 2),
                "negative": round((neg / total_records) * 100.0, 2),
            }
            nspi = round(((pos - neg) / total_records) * 100.0, 2)

        avg_rating = 0.0
        if "rating" in self.df.columns and total_records > 0:
            avg_rating = float(self.df["rating"].mean())

        cat_perf = {}
        if "product_category" in self.df.columns and "rating" in self.df.columns:
            for cat, group in self.df.groupby("product_category"):
                c_total = len(group)
                c_avg_rating = float(group["rating"].mean())
                c_pos = int((group["sentiment_label"].str.lower() == "positive").sum()) if "sentiment_label" in group else 0
                c_neg = int((group["sentiment_label"].str.lower() == "negative").sum()) if "sentiment_label" in group else 0
                c_nspi = round(((c_pos - c_neg) / c_total) * 100.0, 2) if c_total > 0 else 0.0
                cat_perf[str(cat)] = {
                    "total_reviews": c_total,
                    "avg_rating": round(c_avg_rating, 2),
                    "pos_pct": round((c_pos / c_total) * 100.0, 2) if c_total > 0 else 0.0,
                    "neg_pct": round((c_neg / c_total) * 100.0, 2) if c_total > 0 else 0.0,
                    "nspi": c_nspi,
                }

        high_impact_count = 0
        if "helpful_votes" in self.df.columns:
            high_impact_count = int((self.df["helpful_votes"] >= 10).sum())

        delta_nps = 0.0
        if "verified_purchase" in self.df.columns and "sentiment_label" in self.df.columns:
            v_group = self.df[self.df["verified_purchase"] == True]
            u_group = self.df[self.df["verified_purchase"] == False]

            def calc_nps(sub_df):
                if sub_df.empty:
                    return 0.0
                p = (sub_df["sentiment_label"].str.lower() == "positive").sum()
                n = (sub_df["sentiment_label"].str.lower() == "negative").sum()
                return ((p - n) / len(sub_df)) * 100.0

            delta_nps = round(calc_nps(v_group) - calc_nps(u_group), 2)

        return ValueMetrics(
            sentiment_distribution=sentiment_dist,
            sentiment_percentages=sentiment_pcts,
            net_sentiment_polarity_index=nspi,
            average_customer_satisfaction_rating=round(avg_rating, 2),
            category_performance=cat_perf,
            high_impact_reviews_count=high_impact_count,
            verified_vs_unverified_nps_delta=delta_nps,
        )

    def run_full_analysis(self) -> BigData5VReport:
        """Executes full 5Vs analysis pipeline and returns unified report."""
        vol = self.analyze_volume()
        vel = self.analyze_velocity()
        var = self.analyze_variety()
        ver = self.analyze_veracity()
        val = self.analyze_value()

        return BigData5VReport(
            timestamp=time.time(),
            dataset_name=self.dataset_name,
            volume=vol,
            velocity=vel,
            variety=var,
            veracity=ver,
            value=val,
        )

    def to_markdown(self) -> str:
        """Convenience method returning the formatted 5Vs markdown table."""
        return self.run_full_analysis().to_markdown()

    def export_json(self, output_path: Union[str, Path]) -> Path:
        report = self.run_full_analysis()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        return out


# ============================================================================
# 4. HDFS vs RDBMS Comparative Benchmark (Unit 1 Syllabus)
# ============================================================================

def compare_hdfs_vs_rdbms(dataset_size_bytes: int, num_records: int) -> Dict[str, Any]:
    """
    Computes an academic comparison matrix between HDFS distributed storage
    and traditional RDBMS for product review analytics.
    """
    block_size = 128 * 1024 * 1024  # 128 MB
    num_blocks = max(1, math.ceil(dataset_size_bytes / block_size))
    replication = 3
    total_hdfs_storage = dataset_size_bytes * replication

    size_mb = dataset_size_bytes / (1024 * 1024)
    rdbms_scan_time_sec = size_mb / 150.0
    hdfs_parallel_scan_time_sec = size_mb / (100.0 * min(4, max(1, num_blocks)))

    return {
        "dataset_metrics": {
            "size_bytes": dataset_size_bytes,
            "size_mb": round(size_mb, 2),
            "record_count": num_records,
        },
        "hdfs_characteristics": {
            "block_size": "128 MB fixed chunking",
            "blocks_allocated": num_blocks,
            "replication_factor": replication,
            "raw_storage_with_replication_mb": round(total_hdfs_storage / (1024 * 1024), 2),
            "scaling_model": "Horizontal (Scale-Out across commodity nodes)",
            "data_access_model": "Write Once, Read Many (WORM), streaming batch throughput",
            "fault_tolerance": "Automated 3x block replication and heartbeat recovery",
            "estimated_parallel_scan_sec": round(hdfs_parallel_scan_time_sec, 4),
        },
        "rdbms_characteristics": {
            "storage_model": "Single monolithic disk volume with B-Tree indices",
            "scaling_model": "Vertical (Scale-Up: upgrade CPU/RAM)",
            "data_access_model": "ACID transactional row updates with table-level / page locks",
            "fault_tolerance": "Active-Passive standby or RAID redundancy",
            "index_overhead": "20% - 40% additional disk space for secondary indices",
            "estimated_single_disk_scan_sec": round(rdbms_scan_time_sec, 4),
        },
        "academic_verdict": (
            "HDFS eliminates RDBMS row-lock overhead and disk I/O bottlenecks for multi-gigabyte "
            "review sentiment batch analytics by partitioning data across 128MB sequential blocks "
            "supporting parallel MapReduce / Spark workers."
        ),
    }
