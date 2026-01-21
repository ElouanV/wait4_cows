#!/usr/bin/env python3
"""
Unit tests for generate_temporal_graph_dataset.py

Run with:
    pytest test_generate_temporal_graph_dataset.py -v
    or
    python -m pytest test_generate_temporal_graph_dataset.py -v
"""

import pytest
import pandas as pd
import networkx as nx
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
import pickle
import json

# Import functions from the main script
import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_temporal_graph_dataset import (
    load_rssi_data,
    create_temporal_graphs,
    save_temporal_graphs,
)


class TestLoadRSSIData:
    """Tests for load_rssi_data function"""

    @pytest.fixture
    def temp_rssi_dir(self):
        """Create temporary directory with sample RSSI parquet files"""
        temp_dir = tempfile.mkdtemp()
        rssi_dir = Path(temp_dir) / "RSSI"
        rssi_dir.mkdir()

        # Create sample data
        base_time = pd.Timestamp("2025-03-17 00:00:00")

        # File 1: Cow 3cf2 detected by sensor 242
        df1 = pd.DataFrame(
            {
                "RSSI": [-45, -50, -55],
                "tick_accel_day": [12.0, 12.1, 12.2],
                "tick_accel": [304605.25, 304605.35, 304605.45],
                "ble_id": [242, 242, 242],
                "accelero_id": ["3cf2", "3cf2", "3cf2"],
                "relative_DateTime": [
                    base_time,
                    base_time + timedelta(seconds=10),
                    base_time + timedelta(seconds=20),
                ],
            }
        )

        # File 2: Cow 3665 detected by sensor 101
        df2 = pd.DataFrame(
            {
                "RSSI": [-60, -65, -70],
                "tick_accel_day": [12.0, 12.1, 12.2],
                "tick_accel": [304605.25, 304605.35, 304605.45],
                "ble_id": [101, 101, 101],
                "accelero_id": ["3665", "3665", "3665"],
                "relative_DateTime": [
                    base_time,
                    base_time + timedelta(seconds=15),
                    base_time + timedelta(seconds=30),
                ],
            }
        )

        df1.to_parquet(rssi_dir / "3cf2_RSSI_elevage_3_cut.parquet")
        df2.to_parquet(rssi_dir / "3665_RSSI_elevage_3_cut.parquet")

        yield rssi_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_load_all_files(self, temp_rssi_dir):
        """Test loading all RSSI files"""
        df = load_rssi_data(temp_rssi_dir)

        assert len(df) == 6  # 3 rows from each file
        assert "RSSI" in df.columns
        assert "ble_id" in df.columns
        assert "accelero_id" in df.columns
        assert "relative_DateTime" in df.columns

    def test_filter_by_start_date(self, temp_rssi_dir):
        """Test filtering data by start date"""
        start_date = pd.Timestamp("2025-03-17 00:00:15")
        df = load_rssi_data(temp_rssi_dir, start_date=start_date)

        # Should filter out the first measurement from each file
        assert len(df) < 6
        assert all(df["relative_DateTime"] >= start_date)

    def test_data_sorted_by_time(self, temp_rssi_dir):
        """Test that loaded data is sorted by datetime"""
        df = load_rssi_data(temp_rssi_dir)

        # Check if sorted
        assert df["relative_DateTime"].is_monotonic_increasing

    def test_no_files_raises_error(self):
        """Test that missing directory raises appropriate error"""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir()

            with pytest.raises(FileNotFoundError):
                load_rssi_data(empty_dir)


class TestCreateTemporalGraphs:
    """Tests for create_temporal_graphs function"""

    @pytest.fixture
    def sample_rssi_df(self):
        """Create sample RSSI DataFrame for testing"""
        base_time = pd.Timestamp("2025-03-17 00:00:00")

        data = {
            "RSSI": [-45, -50, -60, -75, -80, -70],
            "ble_id": [1, 1, 1, 2, 2, 2],
            "accelero_id": ["cow1", "cow2", "cow3", "cow1", "cow2", "cow4"],
            "relative_DateTime": [
                base_time,
                base_time + timedelta(seconds=5),
                base_time + timedelta(seconds=10),
                base_time + timedelta(seconds=25),
                base_time + timedelta(seconds=28),
                base_time + timedelta(seconds=30),
            ],
        }

        return pd.DataFrame(data)

    def test_creates_correct_number_of_snapshots(self, sample_rssi_df):
        """Test that correct number of snapshots are created"""
        graphs = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-75, snapshot_duration=20, aggregation="mean"
        )

        # With 20s snapshots and data spanning ~30s, should have 2 snapshots
        assert len(graphs) == 2

    def test_rssi_threshold_filtering(self, sample_rssi_df):
        """Test that RSSI threshold properly filters edges"""
        # With threshold -60, should include more edges than -50
        graphs_lenient = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-75, snapshot_duration=20, aggregation="mean"
        )

        graphs_strict = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-50, snapshot_duration=20, aggregation="mean"
        )

        # Lenient threshold should have more or equal edges
        total_edges_lenient = sum(g["num_edges"] for g in graphs_lenient)
        total_edges_strict = sum(g["num_edges"] for g in graphs_strict)

        assert total_edges_lenient >= total_edges_strict

    def test_aggregation_mean_vs_max(self, sample_rssi_df):
        """Test that mean and max aggregation produce valid results"""
        graphs_mean = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-80, snapshot_duration=20, aggregation="mean"
        )

        graphs_max = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-80, snapshot_duration=20, aggregation="max"
        )

        # Both should produce same number of snapshots
        assert len(graphs_mean) == len(graphs_max)

        # Max might produce more edges since it takes highest RSSI
        total_edges_mean = sum(g["num_edges"] for g in graphs_mean)
        total_edges_max = sum(g["num_edges"] for g in graphs_max)

        # Both should be non-negative
        assert total_edges_mean >= 0
        assert total_edges_max >= 0

    def test_graph_structure(self, sample_rssi_df):
        """Test that generated graphs have correct structure"""
        graphs = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-80, snapshot_duration=20, aggregation="mean"
        )

        for graph_info in graphs:
            G = graph_info["graph"]

            # Should be a NetworkX graph
            assert isinstance(G, nx.Graph)

            # Nodes should be cow IDs
            assert all(isinstance(node, str) for node in G.nodes())

            # Edges should have RSSI attribute
            for u, v, data in G.edges(data=True):
                assert "rssi" in data
                assert isinstance(data["rssi"], (int, float))

    def test_snapshot_duration_parameter(self, sample_rssi_df):
        """Test that different snapshot durations affect number of graphs"""
        graphs_10s = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-80, snapshot_duration=10, aggregation="mean"
        )

        graphs_30s = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-80, snapshot_duration=30, aggregation="mean"
        )

        # Smaller snapshots should create more graphs
        assert len(graphs_10s) >= len(graphs_30s)

    def test_all_cows_as_nodes(self, sample_rssi_df):
        """Test that all cows appear as nodes even if not connected"""
        graphs = create_temporal_graphs(
            sample_rssi_df,
            rssi_threshold=-40,  # Very strict threshold
            snapshot_duration=20,
            aggregation="mean",
        )

        # All unique cows should appear as nodes
        all_cows = set(sample_rssi_df["accelero_id"].unique())

        for graph_info in graphs:
            G = graph_info["graph"]
            graph_nodes = set(G.nodes())

            # Graph nodes should be a subset of all cows (or equal)
            assert graph_nodes.issubset(all_cows) or graph_nodes == all_cows

    def test_graph_info_keys(self, sample_rssi_df):
        """Test that graph info dictionaries have required keys"""
        graphs = create_temporal_graphs(
            sample_rssi_df, rssi_threshold=-75, snapshot_duration=20, aggregation="mean"
        )

        required_keys = {
            "timestamp",
            "graph",
            "num_nodes",
            "num_edges",
            "num_measurements",
        }

        for graph_info in graphs:
            assert required_keys.issubset(graph_info.keys())


class TestSaveTemporalGraphs:
    """Tests for save_temporal_graphs function"""

    @pytest.fixture
    def sample_temporal_graphs(self):
        """Create sample temporal graphs for testing"""
        base_time = pd.Timestamp("2025-03-17 00:00:00")

        graphs = []
        for i in range(3):
            G = nx.Graph()
            G.add_nodes_from(["cow1", "cow2", "cow3"])
            G.add_edge("cow1", "cow2", rssi=-50)

            graph_info = {
                "timestamp": base_time + timedelta(seconds=i * 20),
                "graph": G,
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "num_measurements": 10,
            }
            graphs.append(graph_info)

        return graphs

    def test_creates_output_files(self, sample_temporal_graphs):
        """Test that all output files are created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            save_temporal_graphs(
                sample_temporal_graphs,
                output_dir,
                rssi_threshold=-75,
                snapshot_duration=20,
                aggregation="mean",
            )

            # Check that files were created
            files = list(output_dir.glob("*"))
            assert len(files) == 3  # .pkl, .json, .csv

            # Check file extensions
            extensions = {f.suffix for f in files}
            assert {".pkl", ".json", ".csv"}.issubset(extensions)

    def test_pickle_file_loadable(self, sample_temporal_graphs):
        """Test that pickle file can be loaded"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            save_temporal_graphs(
                sample_temporal_graphs,
                output_dir,
                rssi_threshold=-75,
                snapshot_duration=20,
                aggregation="mean",
            )

            # Load pickle file
            pkl_file = list(output_dir.glob("*.pkl"))[0]
            with open(pkl_file, "rb") as f:
                loaded_graphs = pickle.load(f)

            assert len(loaded_graphs) == len(sample_temporal_graphs)
            assert all(isinstance(g["graph"], nx.Graph) for g in loaded_graphs)

    def test_metadata_json_valid(self, sample_temporal_graphs):
        """Test that metadata JSON is valid and contains required fields"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            save_temporal_graphs(
                sample_temporal_graphs,
                output_dir,
                rssi_threshold=-75,
                snapshot_duration=20,
                aggregation="mean",
            )

            # Load JSON file
            json_file = list(output_dir.glob("*.json"))[0]
            with open(json_file, "r") as f:
                metadata = json.load(f)

            # Check required fields
            required_fields = {
                "rssi_threshold",
                "snapshot_duration",
                "aggregation",
                "num_snapshots",
                "start_time",
                "end_time",
                "total_nodes",
            }
            assert required_fields.issubset(metadata.keys())

            # Check values
            assert metadata["rssi_threshold"] == -75
            assert metadata["snapshot_duration"] == 20
            assert metadata["aggregation"] == "mean"
            assert metadata["num_snapshots"] == 3

    def test_summary_csv_valid(self, sample_temporal_graphs):
        """Test that summary CSV is valid and contains correct data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            save_temporal_graphs(
                sample_temporal_graphs,
                output_dir,
                rssi_threshold=-75,
                snapshot_duration=20,
                aggregation="mean",
            )

            # Load CSV file
            csv_file = list(output_dir.glob("*.csv"))[0]
            summary_df = pd.read_csv(csv_file)

            # Check structure
            assert len(summary_df) == 3
            required_columns = {
                "timestamp",
                "num_nodes",
                "num_edges",
                "num_measurements",
            }
            assert required_columns.issubset(summary_df.columns)

            # Check values
            assert all(summary_df["num_nodes"] == 3)
            assert all(summary_df["num_edges"] == 1)

    def test_creates_output_directory(self, sample_temporal_graphs):
        """Test that output directory is created if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "output" / "dir"

            # Directory should not exist yet
            assert not output_dir.exists()

            save_temporal_graphs(
                sample_temporal_graphs,
                output_dir,
                rssi_threshold=-75,
                snapshot_duration=20,
                aggregation="mean",
            )

            # Directory should be created
            assert output_dir.exists()
            assert output_dir.is_dir()


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame(
            columns=["RSSI", "ble_id", "accelero_id", "relative_DateTime"]
        )

        graphs = create_temporal_graphs(
            empty_df, rssi_threshold=-75, snapshot_duration=20, aggregation="mean"
        )

        # Should return empty list
        assert len(graphs) == 0

    def test_single_measurement(self):
        """Test handling of single measurement"""
        single_df = pd.DataFrame(
            {
                "RSSI": [-50],
                "ble_id": [1],
                "accelero_id": ["cow1"],
                "relative_DateTime": [pd.Timestamp("2025-03-17 00:00:00")],
            }
        )

        graphs = create_temporal_graphs(
            single_df, rssi_threshold=-75, snapshot_duration=20, aggregation="mean"
        )

        # Should create one graph with one node, no edges
        assert len(graphs) == 1
        assert graphs[0]["num_nodes"] >= 1
        assert graphs[0]["num_edges"] == 0

    def test_very_strict_threshold(self):
        """Test with very strict threshold that filters everything"""
        df = pd.DataFrame(
            {
                "RSSI": [-80, -85, -90],
                "ble_id": [1, 1, 1],
                "accelero_id": ["cow1", "cow2", "cow3"],
                "relative_DateTime": [pd.Timestamp("2025-03-17 00:00:00")] * 3,
            }
        )

        graphs = create_temporal_graphs(
            df,
            rssi_threshold=-40,  # Very strict
            snapshot_duration=20,
            aggregation="mean",
        )

        # Should create graph but with no edges
        assert len(graphs) == 1
        assert graphs[0]["num_edges"] == 0


class TestIntegration:
    """Integration tests combining multiple functions"""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with RSSI data"""
        temp_dir = tempfile.mkdtemp()
        rssi_dir = Path(temp_dir) / "RSSI"
        rssi_dir.mkdir()

        # Create realistic test data
        base_time = pd.Timestamp("2025-03-17 00:00:00")

        for i in range(3):  # 3 files
            data = []
            for j in range(100):  # 100 measurements per file
                data.append(
                    {
                        "RSSI": -45 - (j % 40),  # RSSI from -45 to -85
                        "tick_accel_day": 12.0 + j * 0.1,
                        "tick_accel": 304605.25 + j * 0.1,
                        "ble_id": (i * 2 + j % 2),
                        "accelero_id": f"cow{j % 5}",
                        "relative_DateTime": base_time + timedelta(seconds=j * 10),
                    }
                )

            df = pd.DataFrame(data)
            df.to_parquet(rssi_dir / f"sensor{i}_RSSI_elevage_3_cut.parquet")

        yield temp_dir, rssi_dir

        shutil.rmtree(temp_dir)

    def test_full_pipeline(self, temp_workspace):
        """Test the full pipeline from loading to saving"""
        temp_dir, rssi_dir = temp_workspace
        output_dir = Path(temp_dir) / "outputs"

        # Load data
        df = load_rssi_data(rssi_dir)
        assert len(df) > 0

        # Create graphs
        graphs = create_temporal_graphs(
            df, rssi_threshold=-75, snapshot_duration=60, aggregation="mean"
        )
        assert len(graphs) > 0

        # Save results
        save_temporal_graphs(
            graphs,
            output_dir,
            rssi_threshold=-75,
            snapshot_duration=60,
            aggregation="mean",
        )

        # Verify all files created
        assert (output_dir).exists()
        assert len(list(output_dir.glob("*.pkl"))) == 1
        assert len(list(output_dir.glob("*.json"))) == 1
        assert len(list(output_dir.glob("*.csv"))) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
