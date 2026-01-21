#!/usr/bin/env python3
"""
Generate network animation GIFs for cow proximity data.
Creates temporal animations showing how the proximity network evolves over time.
"""

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta


def load_proximity_graphs(pkl_file):
    """Load proximity graphs from pickle file."""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def create_network_animation(graphs_data, output_path, poi_id, 
                             max_frames=200, fps=5, layout_seed=42):
    """
    Create an animated GIF showing network evolution.
    
    Parameters:
    -----------
    graphs_data : list
        List of dicts with 'graph', 'timestamp', 'neighbors'
    output_path : str
        Output file path for the GIF
    poi_id : str
        ID of the Point of Interest sensor
    max_frames : int
        Maximum number of frames to include
    fps : int
        Frames per second for the animation
    layout_seed : int
        Random seed for consistent layout
    """
    
    # Sample frames if too many
    if len(graphs_data) > max_frames:
        step = len(graphs_data) // max_frames
        sampled_data = graphs_data[::step]
    else:
        sampled_data = graphs_data
    
    print(f"Creating animation with {len(sampled_data)} frames...")
    
    # Get all unique nodes across all graphs for consistent layout
    all_nodes = set()
    for data in sampled_data:
        all_nodes.update(data['graph'].nodes())
    all_nodes.add(poi_id)  # Ensure POI is included
    
    # Create a master graph with all nodes for layout calculation
    master_graph = nx.Graph()
    master_graph.add_nodes_from(all_nodes)
    
    # Add edges from all graphs to inform layout
    for data in sampled_data:
        master_graph.add_edges_from(data['graph'].edges())
    
    # Calculate fixed layout
    pos = nx.spring_layout(master_graph, k=2, iterations=50, seed=layout_seed)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    def init():
        """Initialize animation."""
        ax.clear()
        return []
    
    def update(frame_idx):
        """Update function for animation."""
        ax.clear()
        
        data = sampled_data[frame_idx]
        G = data['graph']
        timestamp = data['timestamp']
        neighbors = data['neighbors']
        
        # Format timestamp
        ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Draw edges
        if G.number_of_edges() > 0:
            # Get edge weights (RSSI values)
            edge_weights = [G[u][v].get('rssi', -70) for u, v in G.edges()]
            # Normalize weights for visualization (stronger signal = thicker line)
            max_rssi = -50
            min_rssi = -100
            normalized_weights = [(w - min_rssi) / (max_rssi - min_rssi) * 3 + 0.5 
                                 for w in edge_weights]
            
            nx.draw_networkx_edges(G, pos, ax=ax, width=normalized_weights, 
                                  alpha=0.4, edge_color='gray')
        
        # Separate POI from other nodes
        poi_nodes = [poi_id]
        other_nodes = [n for n in G.nodes() if n != poi_id]
        
        # Draw POI node (large, red)
        if poi_id in pos:
            nx.draw_networkx_nodes(G, pos, nodelist=poi_nodes, ax=ax,
                                  node_color='red', node_size=1000, 
                                  edgecolors='darkred', linewidths=3, 
                                  label='POI Sensor')
        
        # Draw neighbor nodes (medium, blue)
        if other_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, ax=ax,
                                  node_color='lightblue', node_size=500,
                                  edgecolors='navy', linewidths=2,
                                  label='Cows in proximity')
        
        # Draw labels
        labels_dict = {poi_id: poi_id}
        labels_dict.update({n: n for n in other_nodes})
        nx.draw_networkx_labels(G, pos, labels_dict, ax=ax, 
                               font_size=9, font_weight='bold')
        
        # Title with statistics
        title = f'Cow Proximity Network - {ts_str}\n'
        title += f'Active Connections: {G.number_of_edges()} | '
        title += f'Cows in Range: {len(neighbors)} | '
        title += f'Frame {frame_idx + 1}/{len(sampled_data)}'
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, init_func=init, 
                        frames=len(sampled_data), 
                        interval=1000//fps, blit=True)
    
    # Save as GIF
    print(f"Saving animation to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=80)
    plt.close()
    
    print(f"✓ Animation saved successfully!")
    print(f"  Frames: {len(sampled_data)}")
    print(f"  Duration: {len(sampled_data)/fps:.1f} seconds")
    print(f"  FPS: {fps}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate network animation GIF from proximity data'
    )
    parser.add_argument('--pkl', required=True,
                       help='Path to proximity graphs pickle file')
    parser.add_argument('--poi-id', required=True,
                       help='ID of the Point of Interest sensor')
    parser.add_argument('--output', required=True,
                       help='Output GIF file path')
    parser.add_argument('--max-frames', type=int, default=200,
                       help='Maximum number of frames (default: 200)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for layout (default: 42)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("NETWORK ANIMATION GENERATOR")
    print(f"{'='*70}\n")
    
    print(f"Loading data from: {args.pkl}")
    graphs_data = load_proximity_graphs(args.pkl)
    print(f"Loaded {len(graphs_data)} temporal snapshots\n")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate animation
    create_network_animation(
        graphs_data=graphs_data,
        output_path=str(output_path),
        poi_id=args.poi_id,
        max_frames=args.max_frames,
        fps=args.fps,
        layout_seed=args.seed
    )
    
    print(f"\n{'='*70}")
    print("ANIMATION COMPLETE!")
    print(f"{'='*70}\n")
    print(f"GIF saved to: {args.output}")
    print(f"\nYou can now view the animation in your browser or")
    print(f"include it in presentations/documents.")


if __name__ == "__main__":
    main()
