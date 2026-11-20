"""
tests/test_tier1_hdfs_simulation.py
===================================
Unit 1 Testing:
- Simulated HDFS Storage Architecture (128MB chunks, 1KB test blocks, checksums)
- Rack-Aware replica placement (/rack_1 and /rack_2)
- Fault tolerance, DataNode failure simulation, and NameNode self-healing recovery
- Big Data 5Vs Analytical Engine (Volume, Velocity, Variety, Veracity, Value)
- HDFS vs RDBMS Comparative Benchmark
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from src.storage.hdfs_simulation import (
    BigData5VAnalysis,
    BigData5VReport,
    BlockMetadata,
    DataNode,
    HDFSSimulator,
    NameNode,
    compare_hdfs_vs_rdbms,
)
from src.data.review_generator import generate_review_records


@pytest.mark.tier1
def test_hdfs_write_and_read_roundtrip(tmp_path: Path):
    """Test writing multi-block payload and reading it back bit-for-bit."""
    payload = b"HDFS_Distributed_Storage_Simulation_Unit1_" * 150  # 6,450 bytes
    block_size = 1024  # 1 KB blocks for unit testing
    hdfs = HDFSSimulator(block_size_bytes=block_size, replication_factor=3, storage_dir=tmp_path / "hdfs_storage")

    hdfs_path = "/user/bda/dataset/reviews_stream.dat"
    blocks = hdfs.write_bytes(payload, hdfs_path)

    # 6,300 / 1,024 = 7 blocks (6 full + 1 partial 156-byte block)
    assert len(blocks) == 7
    assert blocks[0].actual_size_bytes == 1024
    assert blocks[-1].actual_size_bytes == 156
    assert sum(b.actual_size_bytes for b in blocks) == len(payload)

    # Each block should have exactly 3 replicas
    for blk in blocks:
        assert len(blk.datanodes) == 3
        assert len(blk.checksum) == 64  # SHA-256 hex digest

    # Read back and assert identical payload
    read_data = hdfs.read_file(hdfs_path)
    assert read_data == payload


@pytest.mark.tier1
def test_hdfs_small_file_single_block():
    """Test file smaller than block size allocates exactly 1 block with no padding."""
    payload = b"Small file payload"
    hdfs = HDFSSimulator(block_size_bytes=1024, replication_factor=3)
    blocks = hdfs.write_bytes(payload, "/small.txt")
    assert len(blocks) == 1
    assert blocks[0].actual_size_bytes == len(payload)
    assert hdfs.read_file("/small.txt") == payload


@pytest.mark.tier1
def test_hdfs_zero_byte_empty_file():
    """Test 0-byte file edge case creates empty metadata and returns b''."""
    hdfs = HDFSSimulator(block_size_bytes=1024, replication_factor=3)
    blocks = hdfs.write_bytes(b"", "/empty.txt")
    assert blocks == []
    assert hdfs.read_file("/empty.txt") == b""
    assert hdfs.exists("/empty.txt")


@pytest.mark.tier1
def test_hdfs_rack_awareness_replica_placement():
    """Test that replica placement spans both /rack_1 and /rack_2."""
    hdfs = HDFSSimulator(block_size_bytes=1024, replication_factor=3)
    blocks = hdfs.write_bytes(b"Rack awareness test" * 200, "/rack_test.txt")

    for blk in blocks:
        assigned_nodes = [hdfs.namenode.datanodes[nid] for nid in blk.datanodes]
        racks = {dn.rack_id for dn in assigned_nodes}
        assert "/rack_1" in racks, "Replica 1 must be on /rack_1"
        assert "/rack_2" in racks, "Replicas 2 & 3 must be on /rack_2"


@pytest.mark.tier1
def test_hdfs_datanode_failure_and_recovery():
    """Test resilience against node failure and NameNode self-healing rebalancing."""
    data = b"SelfHealingResiliencePayload" * 120
    hdfs = HDFSSimulator(block_size_bytes=512, replication_factor=3)
    hdfs_file = "/resilience_test.bin"
    hdfs.write_bytes(data, hdfs_file)

    # Simulate failure of dn_01
    fail_report = hdfs.simulate_datanode_failure("dn_01")
    assert fail_report["status"] == "DEAD"
    assert fail_report["affected_blocks_count"] > 0

    # Cluster health is degraded
    report = hdfs.get_block_report()
    assert report["cluster_summary"]["cluster_health"] == "DEGRADED"

    # Reading file must still succeed through surviving replicas
    read_data = hdfs.read_file(hdfs_file)
    assert read_data == data

    # Self-healing recovery
    recovery = hdfs.rebalance_or_recover()
    assert recovery["under_replicated_found"] > 0
    assert recovery["cluster_status"] == "HEALTHY"

    # Post-recovery read
    assert hdfs.read_file(hdfs_file) == data


@pytest.mark.tier1
def test_hdfs_cluster_block_report():
    """Test block report structure and accurate accounting of capacity and utilization."""
    hdfs = HDFSSimulator(block_size_bytes=1024, replication_factor=3)
    hdfs.write_bytes(b"Sample payload" * 50, "/sample.txt")
    report = hdfs.get_block_report()

    assert "cluster_summary" in report
    assert "datanodes" in report
    assert "files" in report

    summary = report["cluster_summary"]
    assert summary["total_files"] == 1
    assert summary["active_datanodes"] == 5
    assert summary["dead_datanodes"] == 0
    assert summary["cluster_health"] == "HEALTHY"
    assert len(report["datanodes"]) == 5


@pytest.mark.tier1
def test_bigdata_5vs_analysis_metrics(tmp_path: Path):
    """Test quantitative empirical metrics across all 5 dimensions of Big Data."""
    records = generate_review_records(num_records=100, seed=42)
    analyzer = BigData5VAnalysis(records, dataset_name="AmazonReviews_Unit1")

    # Volume
    vol = analyzer.analyze_volume()
    assert vol.total_records == 100
    assert vol.total_columns == 16
    assert vol.avg_record_size_bytes > 0
    assert vol.projected_1m_bytes > 0
    assert vol.projected_1m_hdfs_blocks_128mb >= 1

    # Velocity
    vel = analyzer.analyze_velocity()
    assert vel.records_per_second > 0
    assert vel.spark_streaming_1s_microbatch_capacity > 0

    # Variety
    var = analyzer.analyze_variety()
    assert var.category_cardinality == 5
    assert len(var.data_modalities) >= 5
    assert var.verified_purchase_ratio > 0.5
    assert 0.0 <= var.lexical_diversity_ttr <= 1.0

    # Veracity
    ver = analyzer.analyze_veracity()
    assert ver.overall_completeness_pct == 100.0
    assert ver.duplicate_record_count == 0
    assert ver.rating_boundary_valid_pct == 100.0
    assert ver.vote_logic_valid_pct == 100.0
    assert ver.sentiment_consistency_rate_pct == 100.0

    # Value
    val = analyzer.analyze_value()
    assert -100.0 <= val.net_sentiment_polarity_index <= 100.0
    assert 1.0 <= val.average_customer_satisfaction_rating <= 5.0
    assert len(val.category_performance) == 5

    # Full report & Markdown export
    report = analyzer.run_full_analysis()
    md = report.to_markdown()
    assert "Volume" in md
    assert "Velocity" in md
    assert "Variety" in md
    assert "Veracity" in md
    assert "Value" in md

    # JSON export
    json_path = tmp_path / "5vs_report.json"
    exported = analyzer.export_json(json_path)
    assert exported.exists()
    with open(exported, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["dataset_name"] == "AmazonReviews_Unit1"
    assert loaded["volume"]["total_records"] == 100


@pytest.mark.tier1
def test_compare_hdfs_vs_rdbms():
    """Test comparative benchmark function between HDFS and RDBMS storage."""
    benchmark = compare_hdfs_vs_rdbms(dataset_size_bytes=500 * 1024 * 1024, num_records=1200000)
    assert "dataset_metrics" in benchmark
    assert "hdfs_characteristics" in benchmark
    assert "rdbms_characteristics" in benchmark
    assert "academic_verdict" in benchmark
    assert benchmark["hdfs_characteristics"]["replication_factor"] == 3
    assert "128 MB" in benchmark["hdfs_characteristics"]["block_size"]
    assert "Horizontal" in benchmark["hdfs_characteristics"]["scaling_model"]
    assert "Vertical" in benchmark["rdbms_characteristics"]["scaling_model"]
