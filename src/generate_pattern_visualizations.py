#!/usr/bin/env python3
"""
Generate pattern mining visualizations as images for presentations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for clean presentation plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)


def plot_frequent_cows(poi_name, results_dir, output_dir):
    """Create bar chart of most frequent cows."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find itemsets file
    itemset_files = list(results_path.glob(f"{poi_name}_*_frequent_itemsets.csv"))
    if not itemset_files:
        print(f"No itemsets found for {poi_name}")
        return
    
    itemsets = pd.read_csv(itemset_files[0])
    
    # Filter single cows
    single_cows = itemsets[itemsets['length'] == 1].copy()
    single_cows['support_pct'] = single_cows['support'] * 100
    single_cows = single_cows.head(10)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(single_cows['itemsets'], single_cows['support_pct'], 
                    color='steelblue', edgecolor='navy', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, single_cows['support_pct'])):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cow ID', fontsize=12, fontweight='bold')
    ax.set_title(f'Most Frequent Cows at {poi_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / f'{poi_name}_frequent_cows.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f'✓ Created: {output_file}')


def plot_cow_pairs(poi_name, results_dir, output_dir):
    """Create bar chart of most frequent cow pairs."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find edge patterns file
    edge_files = list(results_path.glob(f"{poi_name}_*_edge_patterns.csv"))
    if not edge_files:
        print(f"No edge patterns found for {poi_name}")
        return
    
    edges = pd.read_csv(edge_files[0])
    edges = edges.head(10).copy()
    edges['pair_label'] = edges['cow1'] + ' + ' + edges['cow2']
    edges['frequency_pct'] = edges['frequency'] * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    bars = ax.barh(edges['pair_label'], edges['count'], 
                    color='coral', edgecolor='darkred', linewidth=1.5)
    
    # Add value labels
    for bar, count, freq in zip(bars, edges['count'], edges['frequency_pct']):
        ax.text(count + max(edges['count'])*0.02, bar.get_y() + bar.get_height()/2, 
                f'{count} ({freq:.2f}%)', va='center', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Number of Co-occurrences', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cow Pair', fontsize=12, fontweight='bold')
    ax.set_title(f'Most Frequent Cow Pairs at {poi_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / f'{poi_name}_cow_pairs.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f'✓ Created: {output_file}')


def plot_network_graph(poi_name, results_dir, output_dir):
    """Create network visualization of cow connections."""
    import networkx as nx
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find edge patterns file
    edge_files = list(results_path.glob(f"{poi_name}_*_edge_patterns.csv"))
    if not edge_files:
        return
    
    edges = pd.read_csv(edge_files[0])
    
    # Keep only strong connections (top 20)
    edges = edges.head(20)
    
    # Create network
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row['cow1'], row['cow2'], weight=row['count'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges with varying thickness
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    edge_widths = [3 * w / max_weight for w in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    
    # Draw nodes
    node_sizes = [500 * G.degree(node) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                           node_color='lightblue', edgecolors='navy', linewidths=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    ax.set_title(f'Social Network at {poi_name.replace("_", " ").title()}\n' +
                 '(Node size = connections, Edge thickness = co-occurrence frequency)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    output_file = output_path / f'{poi_name}_network.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f'✓ Created: {output_file}')


def plot_comparison_chart(output_dir):
    """Create comparison chart across all POIs."""
    output_path = Path(output_dir)
    
    # Collect data from all POIs
    poi_data = []
    
    for poi, directory in [('Brush', 'brush_experiment'),
                           ('Lactation', 'lactation_experiment'),
                           ('Water Spot', 'water_spot_experiment')]:
        
        # Get metadata
        meta_files = list(Path(directory).glob('*_metadata.json'))
        if meta_files:
            import json
            with open(meta_files[0]) as f:
                meta = json.load(f)
            
            stats = meta.get('statistics', {})
            poi_data.append({
                'POI': poi,
                'Total Snapshots': stats.get('snapshots_with_neighbors', 0),
                'Unique Visitors': len(meta.get('unique_neighbors', []))
            })
    
    if not poi_data:
        return
    
    df = pd.DataFrame(poi_data)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Snapshots with activity
    ax1.bar(df['POI'], df['Total Snapshots'], color=['steelblue', 'coral', 'lightgreen'],
            edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Active Snapshots', fontsize=12, fontweight='bold')
    ax1.set_title('Activity Level by POI', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (poi, val) in enumerate(zip(df['POI'], df['Total Snapshots'])):
        ax1.text(i, val + max(df['Total Snapshots'])*0.02, 
                f'{val:,}', ha='center', fontweight='bold')
    
    # Plot 2: Unique visitors
    ax2.bar(df['POI'], df['Unique Visitors'], color=['steelblue', 'coral', 'lightgreen'],
            edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Unique Cows', fontsize=12, fontweight='bold')
    ax2.set_title('Unique Visitors by POI', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (poi, val) in enumerate(zip(df['POI'], df['Unique Visitors'])):
        ax2.text(i, val + max(df['Unique Visitors'])*0.02, 
                f'{val}', ha='center', fontweight='bold')
    
    plt.suptitle('POI Comparison Summary', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_path / 'poi_comparison.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f'✓ Created: {output_file}')


def main():
    """Generate all visualizations."""
    
    output_dir = 'presentation_figures'
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENERATING PATTERN MINING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # Brush
    print("Brush visualizations...")
    plot_frequent_cows('brush_proximity', 'brush_experiment', output_dir)
    plot_cow_pairs('brush_proximity', 'brush_experiment', output_dir)
    plot_network_graph('brush_proximity', 'brush_experiment', output_dir)
    print()
    
    # Lactation
    print("Lactation visualizations...")
    plot_frequent_cows('lactation_proximity', 'lactation_experiment', output_dir)
    plot_cow_pairs('lactation_proximity', 'lactation_experiment', output_dir)
    plot_network_graph('lactation_proximity', 'lactation_experiment', output_dir)
    print()
    
    # Water spot
    print("Water spot visualizations...")
    plot_frequent_cows('water_spot_proximity', 'water_spot_experiment', output_dir)
    plot_cow_pairs('water_spot_proximity', 'water_spot_experiment', output_dir)
    plot_network_graph('water_spot_proximity', 'water_spot_experiment', output_dir)
    print()
    
    # Comparison
    print("Comparison chart...")
    plot_comparison_chart(output_dir)
    print()
    
    print(f"{'='*70}")
    print("ALL VISUALIZATIONS GENERATED!")
    print(f"{'='*70}")
    print(f"\nAll images saved to: {output_dir}/")
    print("\nFiles created:")
    print("  - *_frequent_cows.png (bar charts)")
    print("  - *_cow_pairs.png (pair frequency charts)")
    print("  - *_network.png (network graphs)")
    print("  - poi_comparison.png (summary comparison)")
    print("\nReady for PowerPoint/Keynote/Beamer! 🎉")


if __name__ == "__main__":
    main()
