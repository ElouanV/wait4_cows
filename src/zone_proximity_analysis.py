#!/usr/bin/env python3
"""
Zone Proximity Analysis Pipeline

Analyzes proximity to a zone defined by multiple POI sensors.
A cow is considered "near the zone" if it's connected to ANY of the POI sensors.

This is useful for:
- Water spot (multiple sensors around the same location)
- Any distributed resource with multiple detection points

Usage:
    python src/zone_proximity_analysis.py \
        --poi-ids 3cf7 3662 3cf4 \
        --zone-name water_spot_zone \
        --rssi-threshold -70.0 \
        --snapshot-time 30 \
        --output-dir water_spot_experiment
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def extract_zone_subgraph(graph: nx.Graph, poi_ids: List[str]) -> nx.Graph:
    """
    Extract subgraph containing all POIs and their neighbors.
    
    Args:
        graph: NetworkX graph
        poi_ids: List of POI IDs that define the zone
        
    Returns:
        Induced subgraph with POIs and all their neighbors
    """
    # Collect all POIs present in graph and their neighbors
    zone_nodes = set()
    neighbors_by_poi = {}
    
    for poi_id in poi_ids:
        if poi_id in graph:
            zone_nodes.add(poi_id)
            neighbors = set(graph.neighbors(poi_id))
            neighbors_by_poi[poi_id] = neighbors
            zone_nodes.update(neighbors)
    
    if len(zone_nodes) <= len([p for p in poi_ids if p in graph]):
        # Only POIs, no neighbors
        return None
    
    # Create subgraph
    subgraph = graph.subgraph(zone_nodes).copy()
    
    # Add metadata about which POIs each cow is connected to
    for node in subgraph.nodes():
        if node not in poi_ids:
            connected_pois = [poi for poi, neighbors in neighbors_by_poi.items() if node in neighbors]
            subgraph.nodes[node]['connected_pois'] = connected_pois
            subgraph.nodes[node]['num_poi_connections'] = len(connected_pois)
    
    return subgraph


def extract_zone_proximity_graphs(
    temporal_graphs: List[Dict],
    poi_ids: List[str]
) -> Tuple[List[Dict], Dict]:
    """
    Extract proximity subgraphs for a zone defined by multiple POIs.
    
    Args:
        temporal_graphs: List of graph dictionaries
        poi_ids: List of POI IDs defining the zone
        
    Returns:
        Tuple of (proximity_graphs, statistics)
    """
    proximity_graphs = []
    stats = {
        'total_snapshots': len(temporal_graphs),
        'snapshots_with_any_poi': 0,
        'snapshots_with_neighbors': 0,
        'total_neighbor_appearances': 0,
        'unique_neighbors': set(),
        'pois_present_count': {poi: 0 for poi in poi_ids},
        'neighbor_poi_connections': {},  # How many POIs each neighbor connects to
    }
    
    for graph_info in tqdm(temporal_graphs, desc="Extracting zone subgraphs"):
        G = graph_info['graph']
        
        # Check which POIs are present
        pois_present = [poi for poi in poi_ids if poi in G]
        
        if len(pois_present) > 0:
            stats['snapshots_with_any_poi'] += 1
            for poi in pois_present:
                stats['pois_present_count'][poi] += 1
            
            # Extract zone subgraph
            subgraph = extract_zone_subgraph(G, poi_ids)
            
            if subgraph is not None:
                stats['snapshots_with_neighbors'] += 1
                
                # Get all neighbors (excluding POIs)
                neighbors = [n for n in subgraph.nodes() if n not in poi_ids]
                
                if len(neighbors) > 0:
                    stats['total_neighbor_appearances'] += len(neighbors)
                    stats['unique_neighbors'].update(neighbors)
                    
                    # Track POI connection counts
                    for neighbor in neighbors:
                        connected_pois = subgraph.nodes[neighbor]['connected_pois']
                        num_connections = len(connected_pois)
                        
                        if neighbor not in stats['neighbor_poi_connections']:
                            stats['neighbor_poi_connections'][neighbor] = {
                                'total_appearances': 0,
                                'connection_counts': {}
                            }
                        
                        stats['neighbor_poi_connections'][neighbor]['total_appearances'] += 1
                        
                        key = f"{num_connections}_pois"
                        stats['neighbor_poi_connections'][neighbor]['connection_counts'][key] = \
                            stats['neighbor_poi_connections'][neighbor]['connection_counts'].get(key, 0) + 1
                    
                    proximity_graphs.append({
                        'timestamp': graph_info['timestamp'],
                        'graph': subgraph,
                        'num_nodes': subgraph.number_of_nodes(),
                        'num_edges': subgraph.number_of_edges(),
                        'neighbors': neighbors,
                        'pois_present': pois_present,
                        'num_pois_present': len(pois_present)
                    })
    
    stats['unique_neighbors'] = list(stats['unique_neighbors'])
    
    return proximity_graphs, stats


def calculate_time_near_zone(
    proximity_graphs: List[Dict],
    poi_ids: List[str],
    snapshot_duration: int
) -> pd.DataFrame:
    """
    Calculate total time each cow spends near the zone.
    
    Args:
        proximity_graphs: List of proximity graph dictionaries
        poi_ids: List of POI IDs
        snapshot_duration: Duration of each snapshot in seconds
        
    Returns:
        DataFrame with time statistics per cow
    """
    cow_stats = {}
    
    for graph_info in proximity_graphs:
        neighbors = graph_info['neighbors']
        subgraph = graph_info['graph']
        
        for cow in neighbors:
            if cow not in cow_stats:
                cow_stats[cow] = {
                    'timestamps': [],
                    'connected_pois_per_snapshot': [],
                    'max_rssi_per_snapshot': []
                }
            
            cow_stats[cow]['timestamps'].append(graph_info['timestamp'])
            
            # Get which POIs this cow is connected to in this snapshot
            connected_pois = subgraph.nodes[cow]['connected_pois']
            cow_stats[cow]['connected_pois_per_snapshot'].append(connected_pois)
            
            # Get max RSSI to any POI
            max_rssi = -100
            for poi in connected_pois:
                if subgraph.has_edge(cow, poi):
                    edge_data = subgraph.get_edge_data(cow, poi)
                    rssi = edge_data.get('rssi', -100) if edge_data else -100
                    max_rssi = max(max_rssi, rssi)
            cow_stats[cow]['max_rssi_per_snapshot'].append(max_rssi)
    
    # Calculate statistics
    stats_data = []
    for cow, data in cow_stats.items():
        total_snapshots = len(data['timestamps'])
        total_time_seconds = total_snapshots * snapshot_duration
        total_time_minutes = total_time_seconds / 60
        
        # Count how often cow connects to 1, 2, or 3 POIs
        poi_connection_distribution = {}
        for connected_pois in data['connected_pois_per_snapshot']:
            num = len(connected_pois)
            poi_connection_distribution[num] = poi_connection_distribution.get(num, 0) + 1
        
        # Average max RSSI
        avg_max_rssi = np.mean(data['max_rssi_per_snapshot'])
        
        stats_data.append({
            'cow_id': cow,
            'num_appearances': total_snapshots,
            'total_time_seconds': total_time_seconds,
            'total_time_minutes': total_time_minutes,
            'frequency': total_snapshots / len(proximity_graphs),
            'avg_max_rssi': avg_max_rssi,
            'connections_1_poi': poi_connection_distribution.get(1, 0),
            'connections_2_pois': poi_connection_distribution.get(2, 0),
            'connections_3_pois': poi_connection_distribution.get(3, 0),
        })
    
    df = pd.DataFrame(stats_data)
    df = df.sort_values('num_appearances', ascending=False)
    
    return df


def mine_frequent_itemsets(
    proximity_graphs: List[Dict],
    poi_ids: List[str],
    min_support: float = 0.05
) -> pd.DataFrame:
    """Mine frequent itemsets from zone proximity graphs."""
    transactions = []
    for graph_info in proximity_graphs:
        neighbors = [n for n in graph_info['neighbors'] if n not in poi_ids]
        if len(neighbors) > 0:
            transactions.append(neighbors)
    
    if len(transactions) == 0:
        return pd.DataFrame()
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) > 0:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
    
    return frequent_itemsets


def save_zone_analysis_results(
    proximity_graphs: List[Dict],
    stats: Dict,
    frequent_itemsets: pd.DataFrame,
    time_stats: pd.DataFrame,
    output_dir: Path,
    zone_name: str,
    poi_ids: List[str],
    rssi_threshold: float,
    snapshot_duration: int
):
    """Save all zone analysis results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{zone_name}_rssi{int(rssi_threshold)}_{timestamp_str}"
    
    # Save proximity graphs
    graphs_file = output_dir / f"{base_name}.pkl"
    with open(graphs_file, 'wb') as f:
        pickle.dump(proximity_graphs, f)
    print(f"✅ Saved proximity graphs: {graphs_file.name}")
    
    # Save metadata
    metadata = {
        'zone_name': zone_name,
        'poi_ids': poi_ids,
        'rssi_threshold': rssi_threshold,
        'snapshot_duration': snapshot_duration,
        'statistics': {
            'total_snapshots': stats['total_snapshots'],
            'snapshots_with_any_poi': stats['snapshots_with_any_poi'],
            'snapshots_with_neighbors': stats['snapshots_with_neighbors'],
            'total_neighbor_appearances': stats['total_neighbor_appearances'],
            'num_unique_neighbors': len(stats['unique_neighbors']),
            'pois_present_count': stats['pois_present_count']
        },
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
        freq_copy = frequent_itemsets.copy()
        freq_copy['itemsets'] = freq_copy['itemsets'].apply(lambda x: ','.join(sorted(x)))
        freq_copy.to_csv(itemsets_file, index=False)
        print(f"✅ Saved frequent itemsets: {itemsets_file.name}")
    
    # Save POI connection analysis
    if stats['neighbor_poi_connections']:
        poi_conn_data = []
        for cow, data in stats['neighbor_poi_connections'].items():
            poi_conn_data.append({
                'cow_id': cow,
                'total_appearances': data['total_appearances'],
                **data['connection_counts']
            })
        poi_conn_df = pd.DataFrame(poi_conn_data)
        poi_conn_file = output_dir / f"{base_name}_poi_connections.csv"
        poi_conn_df.to_csv(poi_conn_file, index=False)
        print(f"✅ Saved POI connection analysis: {poi_conn_file.name}")
    
    # Save summary CSV
    summary_data = []
    for graph_info in proximity_graphs:
        summary_data.append({
            'timestamp': graph_info['timestamp'],
            'num_nodes': graph_info['num_nodes'],
            'num_edges': graph_info['num_edges'],
            'num_neighbors': len(graph_info['neighbors']),
            'num_pois_present': graph_info['num_pois_present']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"{base_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Saved summary: {summary_file.name}")
    
    return base_name


def main():
    parser = argparse.ArgumentParser(
        description="Perform zone proximity analysis for multiple POI sensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--poi-ids",
        type=str,
        nargs='+',
        required=True,
        help="List of POI IDs defining the zone (e.g., 3cf7 3662 3cf4)"
    )
    
    parser.add_argument(
        "--zone-name",
        type=str,
        required=True,
        help="Name of the zone (e.g., water_spot_zone)"
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
    print(f"ZONE PROXIMITY ANALYSIS: {args.zone_name.upper()}")
    print(f"{'='*70}")
    print(f"POI IDs: {', '.join(args.poi_ids)}")
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
    
    # Step 3: Extract zone proximity subgraphs
    print(f"🎯 Step 3: Extracting zone proximity subgraphs...")
    proximity_graphs, stats = extract_zone_proximity_graphs(temporal_graphs, args.poi_ids)
    print(f"   Total snapshots: {stats['total_snapshots']:,}")
    print(f"   Snapshots with any POI: {stats['snapshots_with_any_poi']:,}")
    print(f"   Snapshots with neighbors: {stats['snapshots_with_neighbors']:,}")
    print(f"   Unique neighbors: {len(stats['unique_neighbors'])}")
    print(f"\n   POI presence:")
    for poi, count in stats['pois_present_count'].items():
        print(f"      {poi}: {count:,} snapshots")
    print()
    
    if len(proximity_graphs) == 0:
        print(f"❌ No proximity graphs found for zone")
        return
    
    # Step 4: Calculate time near zone
    print("⏱️  Step 4: Calculating time each cow spends near zone...")
    time_stats = calculate_time_near_zone(proximity_graphs, args.poi_ids, args.snapshot_time)
    print(f"   Top 5 most frequent zone visitors:")
    for idx, row in time_stats.head(5).iterrows():
        conn_info = f"1poi:{row['connections_1_poi']} 2pois:{row['connections_2_pois']} 3pois:{row['connections_3_pois']}"
        print(f"      {row['cow_id']}: {row['total_time_minutes']:.1f} min ({row['num_appearances']} snapshots, {conn_info})")
    print()
    
    # Step 5: Mine frequent itemsets
    print("⛏️  Step 5: Mining frequent itemsets...")
    frequent_itemsets = mine_frequent_itemsets(proximity_graphs, args.poi_ids, args.min_support)
    if not frequent_itemsets.empty:
        print(f"   Found {len(frequent_itemsets)} frequent itemsets")
        print(f"   Itemsets by size: {frequent_itemsets['length'].value_counts().to_dict()}\n")
    else:
        print(f"   No frequent itemsets found with min_support={args.min_support}\n")
    
    # Step 6: Save results
    print("💾 Step 6: Saving results...")
    base_name = save_zone_analysis_results(
        proximity_graphs,
        stats,
        frequent_itemsets,
        time_stats,
        output_dir,
        args.zone_name,
        args.poi_ids,
        args.rssi_threshold,
        args.snapshot_time
    )
    
    print(f"\n{'='*70}")
    print(f"✅ ZONE ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   Base filename: {base_name}")
    print(f"\n💡 Next step: Train GNN model or compare with individual POI analyses\n")


if __name__ == "__main__":
    main()
