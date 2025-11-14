#!/usr/bin/env python3
"""
Temporal Graph Dataset Generator

This script creates temporal graph datasets from RSSI sensor data where:
- Nodes represent cows (identified by accelero_id)
- Edges connect two cows when RSSI signal is above a threshold
- Graphs are generated at regular time intervals (snapshots)

OUTPUT FILES:
-------------
The script generates 3 files in the output directory:

1. **{base_name}.pkl** - Pickle file containing temporal graphs
   - Python list of dictionaries, one per time snapshot
   - Each dictionary contains:
     * 'timestamp': pd.Timestamp - Time of this snapshot
     * 'graph': nx.Graph - NetworkX graph object with cow nodes and proximity edges
     * 'num_nodes': int - Number of cows (nodes) in the graph
     * 'num_edges': int - Number of proximity connections (edges)
     * 'num_measurements': int - Number of RSSI measurements in this time window
   - Graph structure:
     * Nodes: Cow IDs (accelero_id) as strings (e.g., '3cf2', '3665')
     * Edges: Connections between cows with 'rssi' attribute (float)
   - Load with: pickle.load(open('filename.pkl', 'rb'))

2. **{base_name}_metadata.json** - Configuration and summary metadata
   - Contains:
     * rssi_threshold: RSSI threshold used (dB)
     * snapshot_duration: Duration of each snapshot (seconds)
     * aggregation: Aggregation method used ('mean' or 'max')
     * num_snapshots: Total number of temporal graphs created
     * start_time: First timestamp in dataset (ISO format)
     * end_time: Last timestamp in dataset (ISO format)
     * total_nodes: Total number of cows in the dataset
     * created_at: When the dataset was generated (ISO format)
   - Load with: json.load(open('filename_metadata.json', 'r'))

3. **{base_name}_summary.csv** - Per-snapshot statistics table
   - CSV with columns:
     * timestamp: Time of each snapshot
     * num_nodes: Number of cows in that snapshot's graph
     * num_edges: Number of edges in that snapshot's graph
     * num_measurements: Number of RSSI readings in that time window
   - Useful for time-series analysis and visualization
   - Load with: pd.read_csv('filename_summary.csv')

FILENAME FORMAT:
----------------
temporal_graphs_rssi{threshold}_snap{duration}s_{aggregation}_{timestamp}.*

Example:
temporal_graphs_rssi-75_snap20s_mean_20251103_143022.pkl
temporal_graphs_rssi-75_snap20s_mean_20251103_143022_metadata.json
temporal_graphs_rssi-75_snap20s_mean_20251103_143022_summary.csv

USAGE EXAMPLES:
---------------
Basic usage (20s snapshots, -75 dB threshold, mean aggregation):
    python generate_temporal_graph_dataset.py

Custom parameters:
    python generate_temporal_graph_dataset.py \\
        --rssi_threshold -70 \\
        --snapshot_duration 30 \\
        --aggregation max \\
        --start_date "2025-03-17 06:00:00"

Filter noisy data:
    python generate_temporal_graph_dataset.py \\
        --start_date "2025-03-17 08:00:00"

LOADING THE OUTPUT:
-------------------
import pickle
import pandas as pd
import json

# Load temporal graphs
with open('temporal_graphs_rssi-75_snap20s_mean_20251103_143022.pkl', 'rb') as f:
    temporal_graphs = pickle.load(f)

# Access first graph
first_snapshot = temporal_graphs[0]
print(f"Time: {first_snapshot['timestamp']}")
print(f"Nodes: {first_snapshot['num_nodes']}, Edges: {first_snapshot['num_edges']}")

# Access NetworkX graph
G = first_snapshot['graph']
print(f"Cows: {list(G.nodes())}")
for cow1, cow2, data in G.edges(data=True):
    print(f"  {cow1} <-> {cow2}, RSSI: {data['rssi']:.1f} dB")

# Load metadata
with open('temporal_graphs_rssi-75_snap20s_mean_20251103_143022_metadata.json', 'r') as f:
    metadata = json.load(f)
print(f"Dataset covers: {metadata['start_time']} to {metadata['end_time']}")

# Load summary statistics
summary_df = pd.read_csv('temporal_graphs_rssi-75_snap20s_mean_20251103_143022_summary.csv')
print(summary_df.describe())
"""

import argparse
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime
import pickle
import json
from typing import List, Dict, Literal
import warnings

warnings.filterwarnings("ignore")


def load_rssi_data(
    rssi_dir: Path, start_date: datetime = None, end_date: datetime = None
) -> pd.DataFrame:
    """
    Load all RSSI parquet files from the directory.

    The filename contains the receiver cow ID, and the dataframe contains:
    - accelero_id/ble_id: the emitter cow ID
    - RSSI: signal strength received

    Args:
        rssi_dir: Path to directory containing RSSI parquet files
        start_date: Optional datetime to filter data after this date
        end_date: Optional datetime to filter data before this date

    Returns:
        Combined DataFrame with columns: receiver_id, emitter_id, RSSI, relative_DateTime
    """
    print(f"Loading RSSI data from {rssi_dir}")

    # Get all parquet files
    parquet_files = sorted(rssi_dir.glob("*_RSSI_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No RSSI parquet files found in {rssi_dir}")

    print(f"Found {len(parquet_files)} parquet files")
    dfs = []
    for file in parquet_files:
        # Extract receiver cow ID from filename (e.g., "365d_RSSI_..." -> "365d")
        receiver_id = file.name.split("_")[0]

        df = pd.read_parquet(file)
        df["receiver_id"] = receiver_id
        df["emitter_id"] = df["accelero_id"]
        df = df[["receiver_id", "emitter_id", "RSSI", "relative_DateTime"]]
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter date
    if start_date is not None:
        combined_df = combined_df[combined_df["relative_DateTime"] >= start_date]
    if end_date is not None:
        combined_df = combined_df[combined_df["relative_DateTime"] <= end_date]

    # Sort by datetime
    combined_df = combined_df.sort_values("relative_DateTime").reset_index(drop=True)

    return combined_df


def create_temporal_graphs(
    rssi_df: pd.DataFrame,
    rssi_threshold: float,
    snapshot_duration: int,
    aggregation: Literal["mean", "max"],
) -> List[Dict]:
    """
    Create temporal graphs from RSSI data.

    Each row represents: receiver_id detected emitter_id with RSSI strength.
    If RSSI >= threshold, create an edge between receiver and emitter.

    Args:
        rssi_df: DataFrame with columns: receiver_id, emitter_id, RSSI, relative_DateTime
        rssi_threshold: Minimum RSSI value to create an edge
        snapshot_duration: Duration of each snapshot in seconds
        aggregation: Method to aggregate RSSI ('mean' or 'max')

    Returns:
        List of dictionaries containing graph info for each snapshot
    """
    print("Creating temporal graphs")
    print(f"RSSI threshold: {rssi_threshold} dB")
    print(f"Snapshot duration: {snapshot_duration} seconds")

    if len(rssi_df) == 0:
        print("Warning: Empty DataFrame, returning empty list")
        return []

    # Create time bins
    rssi_df["time_bin"] = rssi_df["relative_DateTime"].dt.floor(f"{snapshot_duration}s")

    # Get unique timestamps
    unique_times = sorted(rssi_df["time_bin"].unique())

    # Get all unique cows (union of receivers and emitters)
    all_cows = set(rssi_df["receiver_id"].unique()) | set(
        rssi_df["emitter_id"].unique()
    )

    temporal_graphs = []

    for i, timestamp in enumerate(unique_times, 1):
        window_df = rssi_df[rssi_df["time_bin"] == timestamp]

        # Aggregate RSSI for each receiver-emitter pair
        if aggregation == "mean":
            agg_df = (
                window_df.groupby(["receiver_id", "emitter_id"])["RSSI"]
                .mean()
                .reset_index()
            )
        else:  # (elif  max)
            agg_df = (
                window_df.groupby(["receiver_id", "emitter_id"])["RSSI"]
                .max()
                .reset_index()
            )

        agg_df = agg_df[agg_df["RSSI"] >= rssi_threshold]

        G = nx.Graph()
        G.add_nodes_from(all_cows)

        # Create edges: if receiver detects emitter with RSSI >= threshold
        # Add edge between receiver and emitter
        for _, row in agg_df.iterrows():
            receiver = row["receiver_id"]
            emitter = row["emitter_id"]
            rssi = row["RSSI"]

            # Skip self-loops (cow detecting itself)
            if receiver == emitter:
                continue

            # If edge already exists (bidirectional detection), keep max RSSI
            if G.has_edge(receiver, emitter):
                current_rssi = G[receiver][emitter]["rssi"]
                if rssi > current_rssi:
                    G[receiver][emitter]["rssi"] = rssi
            else:
                G.add_edge(receiver, emitter, rssi=rssi)

        # Store graph info
        graph_info = {
            "timestamp": timestamp,
            "graph": G,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_measurements": len(window_df),
        }

        temporal_graphs.append(graph_info)

        if i % 500 == 0:
            print(f"Processed {i}/{len(unique_times)} snapshots")

    return temporal_graphs


def save_temporal_graphs(
    temporal_graphs: List[Dict],
    output_dir: Path,
    rssi_threshold: float,
    snapshot_duration: int,
    aggregation: str,
):
    """
    Save temporal graphs and metadata to disk.

    Args:
        temporal_graphs: List of graph dictionaries
        output_dir: Directory to save outputs
        rssi_threshold: RSSI threshold used
        snapshot_duration: Snapshot duration used
        aggregation: Aggregation method used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold_str = str(rssi_threshold).replace(".", "_").replace("-", "neg")
    base_name = (
        f"temporal_graphs_rssi{threshold_str}_"
        f"snap{snapshot_duration}s_{aggregation}_{timestamp_str}"
    )

    graphs_file = output_dir / f"{base_name}.pkl"
    print(f"Saving graphs to {graphs_file}")
    with open(graphs_file, "wb") as f:
        pickle.dump(temporal_graphs, f)

    metadata = {
        "rssi_threshold": rssi_threshold,
        "snapshot_duration": snapshot_duration,
        "aggregation": aggregation,
        "num_snapshots": len(temporal_graphs),
        "start_time": str(temporal_graphs[0]["timestamp"]),
        "end_time": str(temporal_graphs[-1]["timestamp"]),
        "total_nodes": temporal_graphs[0]["num_nodes"],
        "created_at": datetime.now().isoformat(),
    }

    metadata_file = output_dir / f"{base_name}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, indent=2, fp=f)

    # Save summary statistics as CSV
    summary_data = []
    for graph_info in temporal_graphs:
        summary_data.append(
            {
                "timestamp": graph_info["timestamp"],
                "num_nodes": graph_info["num_nodes"],
                "num_edges": graph_info["num_edges"],
                "num_measurements": graph_info["num_measurements"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"{base_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nDataset created: {len(temporal_graphs)} snapshots")
    print(f"Output files:")
    print(f"  - {graphs_file.name}")
    print(f"  - {metadata_file.name}")
    print(f"  - {summary_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate temporal graph dataset from RSSI data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--rssi-threshold",
        type=float,
        default=-75,
        help="Minimum RSSI value (in dB) to create an edge",
    )

    parser.add_argument(
        "--snapshot-time",
        type=int,
        default=20,
        help="Duration of each snapshot in seconds",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Method to aggregate RSSI values",
    )

    parser.add_argument(
        "--start-after-hours", type=float, default=0, help="Skip first N hours of data"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help='End date (YYYY-MM-DD or "YYYY-MM-DD HH:MM:SS")',
    )

    parser.add_argument(
        "--rssi-dir",
        type=str,
        default="data/RSSI",
        help="Directory containing RSSI parquet files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/temporal_graphs",
        help="Directory to save output files",
    )

    args = parser.parse_args()

    # Convert start to datetime
    start_date = None
    if args.start_after_hours > 0:
        rssi_dir_temp = Path(args.rssi_dir)
        if rssi_dir_temp.exists():
            # Load first file to get start time
            first_file = sorted(rssi_dir_temp.glob("*_RSSI_*.parquet"))[0]
            df_temp = pd.read_parquet(first_file)
            min_time = df_temp["relative_DateTime"].min()
            start_date = min_time + pd.Timedelta(hours=args.start_after_hours)

    # Parse end date if provided
    end_date = None
    if args.end_date:
        try:
            end_date = pd.to_datetime(args.end_date)
        except Exception as e:
            print(f"Error parsing end date: {e}")
            return

    # Convert paths
    rssi_dir = Path(args.rssi_dir)
    output_dir = Path(args.output_dir)

    if not rssi_dir.exists():
        print(f"RSSI directory not found: {rssi_dir}")
        return

    # Load data
    rssi_df = load_rssi_data(rssi_dir, start_date, end_date)

    # Create temporal graphs
    temporal_graphs = create_temporal_graphs(
        rssi_df, args.rssi_threshold, args.snapshot_time, args.aggregation
    )

    # Save results
    save_temporal_graphs(
        temporal_graphs,
        output_dir,
        args.rssi_threshold,
        args.snapshot_time,
        args.aggregation,
    )


if __name__ == "__main__":
    main()
