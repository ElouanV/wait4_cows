#!/usr/bin/env python3
"""
Point of Interest (POI) Proximity Analysis Pipeline

This script performs comprehensive proximity analysis for a specific point of interest (POI)
such as brush, water spot, or lactation machine. It runs the complete pipeline:
1. Generate temporal graphs with specified RSSI threshold
2. Extract POI proximity subgraphs
3. Perform pattern mining (frequent itemsets, edge patterns, triangles)
4. Train GNN for proximity prediction
5. Generate visualizations and statistics

Usage:
    python src/poi_proximity_analysis.py \
        --poi-id 366b \
        --poi-name brush \
        --rssi-threshold -70.0 \
        --snapshot-time 30 \
        --output-dir brush_experiment
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from generate_temporal_graph_dataset import (
    load_rssi_data,
    create_temporal_graphs,
)


def extract_poi_subgraph(graph: nx.Graph, poi_id: str) -> nx.Graph:
    """
    Extract subgraph containing only the POI and its neighbors.
    
    Args:
        graph: NetworkX graph
        poi_id: ID of the point of interest node
        
    Returns:
        Induced subgraph with POI and its neighbors, or None if POI not in graph
    """
    if poi_id not in graph:
        return None
    
    # Get POI neighbors
    neighbors = list(graph.neighbors(poi_id))
    
    if len(neighbors) == 0:
        return None
    
    # Create subgraph with POI and neighbors
    subgraph_nodes = [poi_id] + neighbors
    subgraph = graph.subgraph(subgraph_nodes).copy()
    
    return subgraph


def extract_poi_proximity_graphs(
    temporal_graphs: List[Dict],
    poi_id: str
) -> Tuple[List[Dict], Dict]:
    """
    Extract proximity subgraphs for POI from temporal graphs.
    
    Args:
        temporal_graphs: List of graph dictionaries
        poi_id: ID of the point of interest
        
    Returns:
        Tuple of (proximity_graphs, statistics)
    """
    proximity_graphs = []
    stats = {
        'total_snapshots': len(temporal_graphs),
        'snapshots_with_poi': 0,
        'snapshots_with_neighbors': 0,
        'total_neighbor_appearances': 0,
        'unique_neighbors': set()
    }
    
    for graph_info in tqdm(temporal_graphs, desc="Extracting POI subgraphs"):
        G = graph_info['graph']
        
        if poi_id in G:
            stats['snapshots_with_poi'] += 1
            
            # Extract subgraph
            subgraph = extract_poi_subgraph(G, poi_id)
            
            if subgraph is not None and subgraph.number_of_nodes() > 1:
                stats['snapshots_with_neighbors'] += 1
                neighbors = [n for n in subgraph.nodes() if n != poi_id]
                stats['total_neighbor_appearances'] += len(neighbors)
                stats['unique_neighbors'].update(neighbors)
                
                proximity_graphs.append({
                    'timestamp': graph_info['timestamp'],
                    'graph': subgraph,
                    'num_nodes': subgraph.number_of_nodes(),
                    'num_edges': subgraph.number_of_edges(),
                    'neighbors': neighbors
                })
    
    stats['unique_neighbors'] = list(stats['unique_neighbors'])
    
    return proximity_graphs, stats


def mine_frequent_itemsets(
    proximity_graphs: List[Dict],
    poi_id: str,
    min_support: float = 0.05
) -> pd.DataFrame:
    """
    Mine frequent itemsets using Apriori algorithm.
    
    Args:
        proximity_graphs: List of proximity graph dictionaries
        poi_id: ID of the point of interest
        min_support: Minimum support threshold
        
    Returns:
        DataFrame with frequent itemsets
    """
    # Create transactions (each snapshot = one transaction)
    transactions = []
    for graph_info in proximity_graphs:
        # Get cows near POI (excluding POI itself)
        neighbors = [n for n in graph_info['graph'].nodes() if n != poi_id]
        if len(neighbors) > 0:
            transactions.append(neighbors)
    
    if len(transactions) == 0:
        return pd.DataFrame()
    
    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Mine frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    
    return frequent_itemsets


def analyze_edge_patterns(proximity_graphs: List[Dict], poi_id: str) -> pd.DataFrame:
    """
    Analyze frequent edge patterns (pairs of cows near POI together).
    
    Args:
        proximity_graphs: List of proximity graph dictionaries
        poi_id: ID of the point of interest
        
    Returns:
        DataFrame with edge pattern frequencies
    """
    edge_counts = {}
    
    for graph_info in proximity_graphs:
        G = graph_info['graph']
        
        # Get all edges (excluding those directly to POI)
        for u, v in G.edges():
            if u != poi_id and v != poi_id:
                edge = tuple(sorted([u, v]))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    if not edge_counts:
        return pd.DataFrame()
    
    # Convert to DataFrame
    edge_data = [
        {
            'cow1': edge[0],
            'cow2': edge[1],
            'count': count,
            'frequency': count / len(proximity_graphs)
        }
        for edge, count in edge_counts.items()
    ]
    
    df = pd.DataFrame(edge_data)
    df = df.sort_values('count', ascending=False)
    
    return df


def analyze_triangle_patterns(proximity_graphs: List[Dict], poi_id: str) -> pd.DataFrame:
    """
    Analyze triangle patterns (3-cow groups near POI).
    
    Args:
        proximity_graphs: List of proximity graph dictionaries  
        poi_id: ID of the point of interest
        
    Returns:
        DataFrame with triangle pattern frequencies
    """
    triangle_counts = {}
    
    for graph_info in proximity_graphs:
        G = graph_info['graph']
        
        # Find triangles
        triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
        
        for triangle in triangles:
            # Only count if POI is part of triangle
            if poi_id in triangle:
                others = tuple(sorted([n for n in triangle if n != poi_id]))
                triangle_counts[others] = triangle_counts.get(others, 0) + 1
    
    if not triangle_counts:
        return pd.DataFrame()
    
    # Convert to DataFrame
    triangle_data = [
        {
            'cow1': triangle[0],
            'cow2': triangle[1],
            'count': count,
            'frequency': count / len(proximity_graphs)
        }
        for triangle, count in triangle_counts.items()
    ]
    
    df = pd.DataFrame(triangle_data)
    df = df.sort_values('count', ascending=False)
    
    return df


def calculate_time_near_poi(
    proximity_graphs: List[Dict],
    poi_id: str,
    snapshot_duration: int
) -> pd.DataFrame:
    """
    Calculate total time each cow spends near the POI.
    
    Args:
        proximity_graphs: List of proximity graph dictionaries
        poi_id: ID of the point of interest
        snapshot_duration: Duration of each snapshot in seconds
        
    Returns:
        DataFrame with time statistics per cow
    """
    cow_appearances = {}
    
    for graph_info in proximity_graphs:
        neighbors = [n for n in graph_info['graph'].nodes() if n != poi_id]
        for cow in neighbors:
            if cow not in cow_appearances:
                cow_appearances[cow] = []
            cow_appearances[cow].append(graph_info['timestamp'])
    
    # Calculate statistics
    stats_data = []
    for cow, timestamps in cow_appearances.items():
        total_snapshots = len(timestamps)
        total_time_seconds = total_snapshots * snapshot_duration
        total_time_minutes = total_time_seconds / 60
        
        stats_data.append({
            'cow_id': cow,
            'num_appearances': total_snapshots,
            'total_time_seconds': total_time_seconds,
            'total_time_minutes': total_time_minutes,
            'frequency': total_snapshots / len(proximity_graphs)
        })
    
    df = pd.DataFrame(stats_data)
    df = df.sort_values('num_appearances', ascending=False)
    
    return df


def save_poi_analysis_results(
    proximity_graphs: List[Dict],
    stats: Dict,
    frequent_itemsets: pd.DataFrame,
    edge_patterns: pd.DataFrame,
    triangle_patterns: pd.DataFrame,
    time_stats: pd.DataFrame,
    output_dir: Path,
    poi_name: str,
    poi_id: str,
    rssi_threshold: float,
    snapshot_duration: int
):
    """Save all analysis results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{poi_name}_proximity_rssi{int(rssi_threshold)}_{timestamp_str}"
    
    # Save proximity graphs
    graphs_file = output_dir / f"{base_name}.pkl"
    with open(graphs_file, 'wb') as f:
        pickle.dump(proximity_graphs, f)
    print(f"✅ Saved proximity graphs: {graphs_file.name}")
    
    # Save metadata
    metadata = {
        'poi_name': poi_name,
        'poi_id': poi_id,
        'rssi_threshold': rssi_threshold,
        'snapshot_duration': snapshot_duration,
        'statistics': {k: v for k, v in stats.items() if k != 'unique_neighbors'},
        'unique_neighbors': stats['unique_neighbors'],
        'num_proximity_graphs': len(proximity_graphs),
        'created_at': datetime.now().isoformat()
    }
    
    metadata_file = output_dir / f"{base_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {metadata_file.name}")
    
    # Save time statistics
    if not time_stats.empty:
        time_file = output_dir / f"{base_name}_time_stats.csv"
        time_stats.to_csv(time_file, index=False)
        print(f"✅ Saved time statistics: {time_file.name}")
    
    # Save frequent itemsets
    if not frequent_itemsets.empty:
        itemsets_file = output_dir / f"{base_name}_frequent_itemsets.csv"
        # Convert frozenset to string for CSV
        freq_copy = frequent_itemsets.copy()
        freq_copy['itemsets'] = freq_copy['itemsets'].apply(lambda x: ','.join(sorted(x)))
        freq_copy.to_csv(itemsets_file, index=False)
        print(f"✅ Saved frequent itemsets: {itemsets_file.name}")
    
    # Save edge patterns
    if not edge_patterns.empty:
        edges_file = output_dir / f"{base_name}_edge_patterns.csv"
        edge_patterns.to_csv(edges_file, index=False)
        print(f"✅ Saved edge patterns: {edges_file.name}")
    
    # Save triangle patterns
    if not triangle_patterns.empty:
        triangles_file = output_dir / f"{base_name}_triangle_patterns.csv"
        triangle_patterns.to_csv(triangles_file, index=False)
        print(f"✅ Saved triangle patterns: {triangles_file.name}")
    
    # Save summary CSV
    summary_data = []
    for graph_info in proximity_graphs:
        summary_data.append({
            'timestamp': graph_info['timestamp'],
            'num_nodes': graph_info['num_nodes'],
            'num_edges': graph_info['num_edges'],
            'num_neighbors': len(graph_info['neighbors'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"{base_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Saved summary: {summary_file.name}")
    
    return base_name


def main():
    parser = argparse.ArgumentParser(
        description="Perform comprehensive POI proximity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--poi-id",
        type=str,
        required=True,
        help="ID of the point of interest (e.g., 366b for brush)"
    )
    
    parser.add_argument(
        "--poi-name",
        type=str,
        required=True,
        help="Name of the POI (e.g., brush, water_spot, lactation)"
    )
    
    parser.add_argument(
        "--rssi-threshold",
        type=float,
        default=-70.0,
        help="RSSI threshold in dB"
    )
    
    parser.add_argument(
        "--snapshot-time",
        type=int,
        default=30,
        help="Snapshot duration in seconds"
    )
    
    parser.add_argument(
        "--rssi-dir",
        type=str,
        default="data/RSSI",
        help="Directory containing RSSI parquet files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="Minimum support for frequent itemset mining"
    )
    
    parser.add_argument(
        "--start-after-hours",
        type=float,
        default=0,
        help="Skip first N hours of data"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"POI PROXIMITY ANALYSIS: {args.poi_name.upper()} (ID: {args.poi_id})")
    print(f"{'='*70}\n")
    
    # Convert paths
    rssi_dir = Path(args.rssi_dir)
    output_dir = Path(args.output_dir)
    
    if not rssi_dir.exists():
        print(f"❌ RSSI directory not found: {rssi_dir}")
        return
    
    # Calculate start date if needed
    start_date = None
    if args.start_after_hours > 0:
        first_file = sorted(rssi_dir.glob("*_RSSI_*.parquet"))[0]
        df_temp = pd.read_parquet(first_file)
        min_time = df_temp["relative_DateTime"].min()
        start_date = min_time + pd.Timedelta(hours=args.start_after_hours)
        print(f"⏭️  Skipping first {args.start_after_hours} hours of data")
        print(f"   Start date: {start_date}\n")
    
    # Step 1: Load RSSI data
    print("📊 Step 1: Loading RSSI data...")
    rssi_df = load_rssi_data(rssi_dir, start_date=start_date)
    print(f"   Loaded {len(rssi_df):,} RSSI measurements\n")
    
    # Step 2: Generate temporal graphs
    print("📈 Step 2: Generating temporal graphs...")
    temporal_graphs = create_temporal_graphs(
        rssi_df,
        args.rssi_threshold,
        args.snapshot_time,
        aggregation="max"
    )
    print(f"   Created {len(temporal_graphs):,} temporal graph snapshots\n")
    
    # Step 3: Extract POI proximity subgraphs
    print(f"🎯 Step 3: Extracting {args.poi_name} proximity subgraphs...")
    proximity_graphs, stats = extract_poi_proximity_graphs(temporal_graphs, args.poi_id)
    print(f"   Total snapshots: {stats['total_snapshots']:,}")
    print(f"   Snapshots with POI: {stats['snapshots_with_poi']:,}")
    print(f"   Snapshots with neighbors: {stats['snapshots_with_neighbors']:,}")
    print(f"   Unique neighbors: {len(stats['unique_neighbors'])}\n")
    
    if len(proximity_graphs) == 0:
        print(f"❌ No proximity graphs found for POI {args.poi_id}")
        return
    
    # Step 4: Calculate time near POI
    print("⏱️  Step 4: Calculating time each cow spends near POI...")
    time_stats = calculate_time_near_poi(proximity_graphs, args.poi_id, args.snapshot_time)
    print(f"   Top 5 most frequent visitors:")
    for idx, row in time_stats.head(5).iterrows():
        print(f"      {row['cow_id']}: {row['total_time_minutes']:.1f} min ({row['num_appearances']} snapshots)")
    print()
    
    # Step 5: Mine frequent itemsets
    print("⛏️  Step 5: Mining frequent itemsets...")
    frequent_itemsets = mine_frequent_itemsets(proximity_graphs, args.poi_id, args.min_support)
    if not frequent_itemsets.empty:
        print(f"   Found {len(frequent_itemsets)} frequent itemsets")
        print(f"   Itemsets by size: {frequent_itemsets['length'].value_counts().to_dict()}\n")
    else:
        print(f"   No frequent itemsets found with min_support={args.min_support}\n")
    
    # Step 6: Analyze edge patterns
    print("🔗 Step 6: Analyzing edge patterns...")
    edge_patterns = analyze_edge_patterns(proximity_graphs, args.poi_id)
    if not edge_patterns.empty:
        print(f"   Found {len(edge_patterns)} edge patterns")
        print(f"   Top pattern: {edge_patterns.iloc[0]['cow1']} <-> {edge_patterns.iloc[0]['cow2']} ({edge_patterns.iloc[0]['count']} times)\n")
    else:
        print(f"   No edge patterns found\n")
    
    # Step 7: Analyze triangle patterns
    print("📐 Step 7: Analyzing triangle patterns...")
    triangle_patterns = analyze_triangle_patterns(proximity_graphs, args.poi_id)
    if not triangle_patterns.empty:
        print(f"   Found {len(triangle_patterns)} triangle patterns\n")
    else:
        print(f"   No triangle patterns found\n")
    
    # Step 8: Save results
    print("💾 Step 8: Saving results...")
    base_name = save_poi_analysis_results(
        proximity_graphs,
        stats,
        frequent_itemsets,
        edge_patterns,
        triangle_patterns,
        time_stats,
        output_dir,
        args.poi_name,
        args.poi_id,
        args.rssi_threshold,
        args.snapshot_time
    )
    
    print(f"\n{'='*70}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   Base filename: {base_name}")
    print(f"\n💡 Next step: Train GNN model with:")
    print(f"   python src/gnn_brush_prediction.py \\")
    print(f"       --pkl {output_dir}/{base_name}.pkl \\")
    print(f"       --brush-id {args.poi_id} \\")
    print(f"       --out-dir {output_dir}/gnn_model\n")


if __name__ == "__main__":
    main()
