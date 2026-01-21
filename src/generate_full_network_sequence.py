#!/usr/bin/env python3
"""
Generate full network temporal graph sequence (not centered on any POI).

This script creates a sequence of temporal network graphs with all cows,
using a specified RSSI threshold for proximity detection.

Usage:
    python src/generate_full_network_sequence.py \
        --rssi-threshold -68.0 \
        --snapshot-time 30 \
        --output-dir network_sequence \
        --start-after-hours 12
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from generate_temporal_graph_dataset import (
    load_rssi_data,
    create_temporal_graphs,
)


def save_network_sequence(
    temporal_graphs: List[Dict],
    output_dir: Path,
    rssi_threshold: float,
    snapshot_time: int,
):
    """
    Save the temporal graph sequence and metadata.
    
    Args:
        temporal_graphs: List of temporal graph dictionaries
        output_dir: Directory to save results
        rssi_threshold: RSSI threshold used
        snapshot_time: Snapshot duration in seconds
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"network_sequence_rssi{int(rssi_threshold)}_{timestamp}"
    
    # Calculate statistics
    total_snapshots = len(temporal_graphs)
    non_empty = sum(1 for g in temporal_graphs if g['graph'].number_of_nodes() > 0)
    all_nodes = set()
    total_edges = 0
    
    for graph_info in temporal_graphs:
        G = graph_info['graph']
        all_nodes.update(G.nodes())
        total_edges += G.number_of_edges()
    
    avg_edges = total_edges / total_snapshots if total_snapshots > 0 else 0
    
    metadata = {
        'rssi_threshold': rssi_threshold,
        'snapshot_duration_seconds': snapshot_time,
        'total_snapshots': total_snapshots,
        'snapshots_with_activity': non_empty,
        'unique_nodes': len(all_nodes),
        'total_edges_across_all_snapshots': total_edges,
        'avg_edges_per_snapshot': round(avg_edges, 2),
        'node_list': sorted(all_nodes),
        'generation_date': timestamp,
    }
    
    # Save temporal graphs
    pkl_file = output_dir / f"{base_filename}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(temporal_graphs, f)
    print(f"✅ Saved temporal graphs: {pkl_file.name}")
    
    # Save metadata
    meta_file = output_dir / f"{base_filename}_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {meta_file.name}")
    
    # Save summary statistics
    summary_file = output_dir / f"{base_filename}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NETWORK SEQUENCE SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"RSSI Threshold: {rssi_threshold} dB\n")
        f.write(f"Snapshot Duration: {snapshot_time} seconds\n\n")
        f.write(f"Total Snapshots: {total_snapshots:,}\n")
        f.write(f"Snapshots with Activity: {non_empty:,} ({100*non_empty/total_snapshots:.1f}%)\n")
        f.write(f"Unique Nodes (Cows): {len(all_nodes)}\n")
        f.write(f"Average Edges per Snapshot: {avg_edges:.2f}\n\n")
        f.write(f"Generated: {timestamp}\n")
    print(f"✅ Saved summary: {summary_file.name}")
    
    return base_filename


def main():
    parser = argparse.ArgumentParser(
        description="Generate full network temporal graph sequence"
    )
    parser.add_argument(
        '--rssi-threshold',
        type=float,
        default=-68.0,
        help='RSSI threshold in dB for proximity detection (default: -68.0)'
    )
    parser.add_argument(
        '--snapshot-time',
        type=int,
        default=30,
        help='Duration of each snapshot in seconds (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='network_sequence',
        help='Output directory for results (default: network_sequence)'
    )
    parser.add_argument(
        '--start-after-hours',
        type=float,
        default=12.0,
        help='Skip first N hours of data (default: 12.0)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL NETWORK TEMPORAL GRAPH SEQUENCE GENERATION")
    print("="*70)
    print()
    
    if args.start_after_hours > 0:
        print(f"⏭️  Skipping first {args.start_after_hours} hours of data")
        from datetime import timedelta
        rssi_dir = Path('data/RSSI')
        # Quick load to get start date
        temp_df = load_rssi_data(rssi_dir)
        start_date = temp_df['relative_DateTime'].min() + timedelta(hours=args.start_after_hours)
        print(f"   Start date: {start_date}\n")
        del temp_df
    else:
        start_date = None
    
    # Step 1: Load RSSI data
    print("📊 Step 1: Loading RSSI data...")
    rssi_dir = Path('data/RSSI')
    rssi_data = load_rssi_data(rssi_dir, start_date=start_date)
    print(f"   Loaded {len(rssi_data):,} RSSI measurements\n")
    
    # Step 2: Generate temporal graphs
    print("📈 Step 2: Generating temporal graphs...")
    print(f"RSSI threshold: {args.rssi_threshold} dB")
    print(f"Snapshot duration: {args.snapshot_time} seconds")
    
    temporal_graphs = create_temporal_graphs(
        rssi_data,
        args.rssi_threshold,
        args.snapshot_time,
        aggregation="max"
    )
    print(f"   Created {len(temporal_graphs):,} temporal graph snapshots\n")
    
    # Step 3: Save results
    print("💾 Step 3: Saving results...")
    output_dir = Path(args.output_dir)
    base_filename = save_network_sequence(
        temporal_graphs,
        output_dir,
        args.rssi_threshold,
        args.snapshot_time
    )
    
    print()
    print("="*70)
    print("✅ GENERATION COMPLETE!")
    print("="*70)
    print()
    print(f"📁 Results saved to: {args.output_dir}/")
    print(f"   Base filename: {base_filename}")
    print()
    print("💡 Use this data for:")
    print("   - Network visualization and animation")
    print("   - Community detection analysis")
    print("   - Global network statistics")
    print("   - Temporal pattern mining")


if __name__ == "__main__":
    main()
