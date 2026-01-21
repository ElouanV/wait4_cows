"""
Compute statistics across all network snapshots.
"""

import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_network_sequence(filepath):
    """Load network sequence data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def compute_snapshot_statistics(sequences):
    """Compute statistics for each snapshot."""
    stats = {
        'num_nodes': [],
        'num_edges': [],
        'density': [],
        'avg_degree': [],
        'max_degree': [],
        'min_degree': []
    }
    
    print("Computing statistics for all snapshots...")
    for snapshot in tqdm(sequences):
        G = snapshot['graph']
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0
        
        if num_nodes > 0:
            degrees = [G.degree(node) for node in G.nodes()]
            avg_degree = np.mean(degrees)
            max_degree = max(degrees) if degrees else 0
            min_degree = min(degrees) if degrees else 0
        else:
            avg_degree = max_degree = min_degree = 0
        
        stats['num_nodes'].append(num_nodes)
        stats['num_edges'].append(num_edges)
        stats['density'].append(density)
        stats['avg_degree'].append(avg_degree)
        stats['max_degree'].append(max_degree)
        stats['min_degree'].append(min_degree)
    
    return stats

def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SNAPSHOT STATISTICS SUMMARY")
    print("="*60)
    
    print(f"\nNumber of Snapshots: {len(stats['num_nodes'])}")
    
    print(f"\nNodes per snapshot:")
    print(f"  Mean: {np.mean(stats['num_nodes']):.2f}")
    print(f"  Std:  {np.std(stats['num_nodes']):.2f}")
    print(f"  Min:  {np.min(stats['num_nodes'])}")
    print(f"  Max:  {np.max(stats['num_nodes'])}")
    
    print(f"\nEdges per snapshot:")
    print(f"  Mean: {np.mean(stats['num_edges']):.2f}")
    print(f"  Std:  {np.std(stats['num_edges']):.2f}")
    print(f"  Min:  {np.min(stats['num_edges'])}")
    print(f"  Max:  {np.max(stats['num_edges'])}")
    
    print(f"\nDensity:")
    print(f"  Mean: {np.mean(stats['density']):.4f}")
    print(f"  Std:  {np.std(stats['density']):.4f}")
    print(f"  Min:  {np.min(stats['density']):.4f}")
    print(f"  Max:  {np.max(stats['density']):.4f}")
    
    print(f"\nAverage Degree:")
    print(f"  Mean: {np.mean(stats['avg_degree']):.2f}")
    print(f"  Std:  {np.std(stats['avg_degree']):.2f}")
    print(f"  Min:  {np.min(stats['avg_degree']):.2f}")
    print(f"  Max:  {np.max(stats['avg_degree']):.2f}")
    
    print(f"\nMax Degree per snapshot:")
    print(f"  Mean: {np.mean(stats['max_degree']):.2f}")
    print(f"  Std:  {np.std(stats['max_degree']):.2f}")
    print(f"  Min:  {np.min(stats['max_degree'])}")
    print(f"  Max:  {np.max(stats['max_degree'])}")
    
    print("\n" + "="*60)

def plot_statistics(stats, output_path):
    """Plot statistics distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Number of edges
    ax = axes[0, 0]
    ax.hist(stats['num_edges'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(stats['num_edges']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(stats["num_edges"]):.1f}')
    ax.set_xlabel('Number of Edges', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Edges per Snapshot', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Density
    ax = axes[0, 1]
    ax.hist(stats['density'], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(stats['density']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(stats["density"]):.4f}')
    ax.set_xlabel('Density', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Network Density', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Average degree
    ax = axes[0, 2]
    ax.hist(stats['avg_degree'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(stats['avg_degree']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(stats["avg_degree"]):.2f}')
    ax.set_xlabel('Average Degree', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Average Degree per Snapshot', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Time series - edges
    ax = axes[1, 0]
    ax.plot(stats['num_edges'], linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Snapshot Index', fontsize=11)
    ax.set_ylabel('Number of Edges', fontsize=11)
    ax.set_title('Edges Over Time', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Time series - density
    ax = axes[1, 1]
    ax.plot(stats['density'], linewidth=0.5, alpha=0.7, color='green')
    ax.set_xlabel('Snapshot Index', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Density Over Time', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Time series - avg degree
    ax = axes[1, 2]
    ax.plot(stats['avg_degree'], linewidth=0.5, alpha=0.7, color='orange')
    ax.set_xlabel('Snapshot Index', fontsize=11)
    ax.set_ylabel('Average Degree', fontsize=11)
    ax.set_title('Average Degree Over Time', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Network Snapshot Statistics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nPlots saved to {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze snapshot statistics')
    parser.add_argument('--input', required=True, help='Input network_sequence pickle file')
    parser.add_argument('--output', default='presentation_plots_final/snapshot_statistics.png',
                       help='Output plot path')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    data = load_network_sequence(args.input)
    
    stats = compute_snapshot_statistics(data)
    
    print_summary(stats)
    
    plot_statistics(stats, args.output)

if __name__ == '__main__':
    main()
