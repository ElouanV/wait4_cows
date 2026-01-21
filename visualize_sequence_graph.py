#!/usr/bin/env python3
"""
Visualize temporal graph sequences for next cow prediction.
Creates both static plots and animated GIFs showing cow proximity networks.
"""

import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_data():
    """Load dataset and metadata."""
    dataset_dir = Path('next_cow_data')
    
    with open(dataset_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    with open(dataset_dir / 'train_sequences.pkl', 'rb') as f:
        sequences = pickle.load(f)
    
    return sequences, metadata

def create_graph_from_sequence(sequence, metadata):
    """Create a graph showing the proximity relationships in a sequence."""
    G = nx.DiGraph()
    
    source_cow = sequence['source_cow']
    input_seq = sequence['input_sequence']
    target = sequence['target']
    
    # Add source cow as central node
    G.add_node(source_cow, node_type='source')
    
    # Add edges showing temporal progression of closest cows
    for i, closest_cow in enumerate(input_seq):
        G.add_node(closest_cow, node_type='neighbor', time_step=i)
        G.add_edge(source_cow, closest_cow, weight=len(input_seq)-i, time_step=i)
    
    # Add target
    G.add_node(target, node_type='target')
    G.add_edge(source_cow, target, weight=1, time_step=len(input_seq), predicted=True)
    
    return G

def plot_sequence_graph(sequence, metadata, output_path='presentation_plots_final/sequence_graph_example.png'):
    """Plot a single sequence as a network graph."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    G = create_graph_from_sequence(sequence, metadata)
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Separate nodes by type
    source_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'source']
    neighbor_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'neighbor']
    target_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, 
                          node_color='red', node_size=800, 
                          label='Source Cow', ax=ax, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=neighbor_nodes, 
                          node_color='lightblue', node_size=500, 
                          label='Closest Neighbors', ax=ax, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, 
                          node_color='green', node_size=800, 
                          label='Target (Next)', ax=ax, alpha=0.9)
    
    # Draw edges with varying thickness based on time
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('predicted', False)]
    predicted_edge = [(u, v) for u, v, d in G.edges(data=True) if d.get('predicted', False)]
    
    # Draw regular edges
    edge_weights = [G[u][v]['weight'] for u, v in regular_edges]
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, 
                          width=[w/3 for w in edge_weights],
                          alpha=0.5, edge_color='gray',
                          arrowsize=20, ax=ax)
    
    # Draw predicted edge
    nx.draw_networkx_edges(G, pos, edgelist=predicted_edge, 
                          width=3, alpha=0.9, edge_color='green',
                          style='dashed', arrowsize=25, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Add title and info
    source_cow = sequence['source_cow']
    target_cow = sequence['target']
    ax.set_title(f'Next Cow Prediction: Temporal Proximity Network\n'
                f'Source: {source_cow} → Target: {target_cow}',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add sequence info
    info_text = f"Sequence: {' → '.join(sequence['input_sequence'])} → {target_cow}"
    ax.text(0.5, 0.02, info_text, transform=ax.transAxes,
           ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def create_temporal_gif(sequence, metadata, output_path='presentation_plots_final/sequence_evolution.gif'):
    """Create animated GIF showing temporal evolution of the sequence."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    source_cow = sequence['source_cow']
    input_seq = sequence['input_sequence']
    target = sequence['target']
    
    def update(frame):
        ax.clear()
        
        # Create graph up to current frame
        G = nx.DiGraph()
        G.add_node(source_cow, node_type='source')
        
        # Add nodes and edges up to current frame
        for i in range(min(frame + 1, len(input_seq))):
            closest_cow = input_seq[i]
            G.add_node(closest_cow, node_type='neighbor')
            G.add_edge(source_cow, closest_cow, time_step=i)
        
        # Add target in last frame
        if frame >= len(input_seq):
            G.add_node(target, node_type='target')
            G.add_edge(source_cow, target, predicted=True)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw
        source_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'source']
        neighbor_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'neighbor']
        target_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']
        
        nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, 
                              node_color='red', node_size=800, ax=ax, alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=neighbor_nodes, 
                              node_color='lightblue', node_size=500, ax=ax, alpha=0.7)
        if target_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, 
                                  node_color='green', node_size=800, ax=ax, alpha=0.9)
        
        # Edges
        regular_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('predicted', False)]
        predicted_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('predicted', False)]
        
        if regular_edges:
            nx.draw_networkx_edges(G, pos, edgelist=regular_edges, 
                                  width=2, alpha=0.6, edge_color='gray',
                                  arrowsize=20, ax=ax)
        if predicted_edges:
            nx.draw_networkx_edges(G, pos, edgelist=predicted_edges, 
                                  width=3, alpha=0.9, edge_color='green',
                                  style='dashed', arrowsize=25, ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Title
        if frame < len(input_seq):
            title = f'Time Step {frame + 1}/{len(input_seq)}: Cow {source_cow} closest to {input_seq[frame]}'
        else:
            title = f'Prediction: Next closest cow is {target}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(input_seq)+2, 
                                  interval=800, repeat=True)
    
    anim.save(output_path, writer='pillow', fps=1, dpi=150)
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    print("=" * 70)
    print("SEQUENCE GRAPH VISUALIZATION")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path('presentation_plots_final')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    sequences, metadata = load_data()
    print(f"   Loaded {len(sequences)} sequences")
    print(f"   Vocabulary: {metadata['num_cows']} cows")
    
    # Pick an interesting example (with some variety in the sequence)
    print("\n🔍 Finding interesting example...")
    example = None
    for seq in sequences:
        if len(set(seq['input_sequence'])) >= 5:  # At least 5 different cows
            example = seq
            break
    
    if example is None:
        example = sequences[100]  # Fallback
    
    print(f"   Source cow: {example['source_cow']}")
    print(f"   Sequence: {' → '.join(example['input_sequence'])}")
    print(f"   Target: {example['target']}")
    print(f"   Unique cows in sequence: {len(set(example['input_sequence']))}")
    
    # Create static plot
    print("\n📊 Creating static network plot...")
    plot_sequence_graph(example, metadata, output_dir / 'sequence_graph_example.png')
    
    # Create animated GIF
    print("\n🎬 Creating animated GIF...")
    create_temporal_gif(example, metadata, output_dir / 'sequence_evolution.gif')
    
    # Create a few more examples
    print("\n📊 Creating additional examples...")
    for i, seq in enumerate(sequences[200:205], 1):
        plot_sequence_graph(seq, metadata, 
                          output_dir / f'sequence_example_{i}.png')
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files in {output_dir}/:")
    print("  - sequence_graph_example.png/pdf (main example)")
    print("  - sequence_evolution.gif (animated)")
    print("  - sequence_example_1-5.png (additional examples)")

if __name__ == '__main__':
    main()
