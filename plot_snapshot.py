"""
Plot a single snapshot from the temporal network sequence.
Shows the proximity network at one specific point in time.
"""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_network_sequence(filepath):
    """Load network sequence data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_snapshot(snapshot_data, output_path, snapshot_idx=None):
    """Plot a single network snapshot."""
    
    G = snapshot_data['graph']
    timestamp = snapshot_data['timestamp']
    num_nodes = snapshot_data['num_nodes']
    num_edges = snapshot_data['num_edges']
    
    # Convert timestamp to readable format
    if hasattr(timestamp, 'timestamp'):
        # pandas Timestamp
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Unix timestamp
        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\nSnapshot Information:")
    print(f"  Timestamp: {time_str}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    
    # Get edge data (RSSI values if available)
    edge_data = nx.get_edge_attributes(G, 'rssi')
    if edge_data:
        edge_weights = list(edge_data.values())
        # Normalize RSSI values for visualization (RSSI is negative, closer to 0 = stronger)
        max_rssi = max(edge_weights)
        min_rssi = min(edge_weights)
        
        # Transform RSSI to positive weights for layout (stronger signal = higher weight)
        # RSSI closer to 0 = stronger, so we invert: weight = max_rssi - rssi
        for u, v, data in G.edges(data=True):
            if 'rssi' in data:
                # Transform to positive weight (higher = stronger connection = closer in layout)
                data['layout_weight'] = max_rssi - data['rssi'] + 1
    else:
        max_rssi = min_rssi = None
    
    # Calculate layout with weighted edges (stronger connections pull nodes closer)
    pos = nx.spring_layout(G, weight='layout_weight', k=1.5, iterations=100, seed=42)
    
    if edge_data:
        edge_widths = [1 + 3 * (rssi - min_rssi) / (max_rssi - min_rssi) for rssi in edge_weights]
        edge_alphas = [0.3 + 0.7 * (rssi - min_rssi) / (max_rssi - min_rssi) for rssi in edge_weights]
        print(f"  RSSI range: [{min_rssi:.1f}, {max_rssi:.1f}]")
    else:
        edge_widths = [2.0] * G.number_of_edges()
        edge_alphas = [0.6] * G.number_of_edges()
    
    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [300 + 50 * degrees[node] for node in G.nodes()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges
    edges = list(G.edges())
    for (u, v), width, alpha in zip(edges, edge_widths, edge_alphas):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, 'gray', linewidth=width, alpha=alpha, zorder=1)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightcoral',
                          edgecolors='darkred', linewidths=2, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    
    # Title
    title = f"Proximity Network Snapshot\n{time_str}"
    if snapshot_idx is not None:
        title += f" (Snapshot #{snapshot_idx})"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add statistics text box
    degree_values = list(degrees.values())
    stats_text = (f"Nodes: {num_nodes}\n"
                 f"Edges: {num_edges}\n"
                 f"Density: {nx.density(G):.3f}\n"
                 f"Avg degree: {np.mean(degree_values):.1f}")
    
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved to {output_path}")
    
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot a single network snapshot')
    parser.add_argument('--input', required=True, help='Input network_sequence pickle file')
    parser.add_argument('--output', default='presentation_plots_final/snapshot_example.png',
                       help='Output plot path')
    parser.add_argument('--index', type=int, default=1000,
                       help='Snapshot index to plot (default: 1000)')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    data = load_network_sequence(args.input)
    
    print(f"Total snapshots: {len(data)}")
    
    if args.index >= len(data):
        print(f"Warning: Index {args.index} out of range, using last snapshot")
        args.index = len(data) - 1
    
    print(f"\nPlotting snapshot {args.index}...")
    plot_snapshot(data[args.index], args.output, snapshot_idx=args.index)

if __name__ == '__main__':
    main()
