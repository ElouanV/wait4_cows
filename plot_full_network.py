"""
Plot the full temporal proximity network from network_sequence data.
Shows the aggregated network structure used for CowBERT/CowLSTM training.
"""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_network_sequence(filepath):
    """Load network sequence data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def aggregate_network(sequences):
    """
    Create an aggregated network from all temporal snapshots.
    Edge weights represent co-occurrence frequency.
    """
    G = nx.Graph()
    edge_counter = Counter()
    
    # Count all edges across all snapshots
    for seq_data in sequences:
        graph = seq_data['graph']
        edges = graph.edges()
        for edge in edges:
            cow1, cow2 = sorted([edge[0], edge[1]])  # Normalize edge direction
            edge_counter[(cow1, cow2)] += 1
    
    # Add edges with weights
    for (cow1, cow2), count in edge_counter.items():
        G.add_edge(cow1, cow2, weight=count)
    
    return G, edge_counter

def plot_network(G, edge_counter, output_path, top_k=None):
    """Plot the full network with layout."""
    
    print(f"\nNetwork Statistics:")
    print(f"  Nodes (cows): {G.number_of_nodes()}")
    print(f"  Edges (connections): {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    
    # If top_k, filter to strongest edges
    if top_k:
        top_edges = sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
        G_filtered = nx.Graph()
        for (cow1, cow2), count in top_edges:
            G_filtered.add_edge(cow1, cow2, weight=count)
        G = G_filtered
        print(f"  Filtered to top {top_k} edges")
    
    # Calculate layout
    print("Computing layout...")
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Get edge weights for visualization
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    min_weight = min(weights)
    
    # Normalize weights for visualization
    edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) for w in weights]
    edge_alphas = [0.3 + 0.7 * (w - min_weight) / (max_weight - min_weight) for w in weights]
    
    # Node sizes based on degree
    node_sizes = [300 + 20 * G.degree(node) for node in G.nodes()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges with varying thickness and transparency
    for (u, v), width, alpha in zip(edges, edge_widths, edge_alphas):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, 'gray', linewidth=width, alpha=alpha, zorder=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                          edgecolors='darkblue', linewidths=2, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    # Title and info
    title = f"Temporal Proximity Network\n{G.number_of_nodes()} cows, {G.number_of_edges()} connections"
    if top_k:
        title += f" (top {top_k} strongest)"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add statistics text box
    degree_dist = [G.degree(node) for node in G.nodes()]
    stats_text = (f"Avg degree: {np.mean(degree_dist):.1f}\n"
                 f"Max degree: {max(degree_dist)}\n"
                 f"Min edge weight: {min_weight}\n"
                 f"Max edge weight: {max_weight}")
    
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved to {output_path}")
    
    plt.show()
    plt.close()

def plot_degree_distribution(G, output_path):
    """Plot degree distribution."""
    degrees = [G.degree(node) for node in G.nodes()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(degrees, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Number of Cows', fontsize=12)
    ax1.set_title('Degree Distribution', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Cumulative distribution
    sorted_degrees = sorted(degrees, reverse=True)
    ax2.plot(range(len(sorted_degrees)), sorted_degrees, 'b-', linewidth=2)
    ax2.set_xlabel('Cow Rank', fontsize=12)
    ax2.set_ylabel('Degree', fontsize=12)
    ax2.set_title('Degree by Rank', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved degree distribution to {output_path}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot full temporal network')
    parser.add_argument('--input', required=True, help='Input network_sequence pickle file')
    parser.add_argument('--output', default='presentation_plots_final/full_network.png',
                       help='Output plot path')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Filter to top K strongest edges (default: all edges)')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    data = load_network_sequence(args.input)
    
    print(f"Total sequences: {len(data)}")
    
    print("\nBuilding aggregated network...")
    G, edge_counter = aggregate_network(data)
    
    # Plot main network
    plot_network(G, edge_counter, args.output, top_k=args.top_k)
    
    # Plot degree distribution
    degree_output = args.output.replace('.png', '_degree_dist.png')
    plot_degree_distribution(G, degree_output)
    
    # Print top connections
    print("\nTop 20 strongest connections:")
    top_connections = sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, ((cow1, cow2), count) in enumerate(top_connections, 1):
        print(f"  {i:2d}. {cow1} -- {cow2}: {count} co-occurrences")

if __name__ == '__main__':
    main()
