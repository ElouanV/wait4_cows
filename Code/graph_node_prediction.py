"""
Graph-based Node Prediction for Cow Identification
Each graph snapshot is treated independently (no temporal dimension).
Task: Mask a cow node and predict which cow it is based on graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import networkx as nx


class GraphSnapshotDataset(Dataset):
    """
    Dataset of individual graph snapshots.
    Each sample: a graph with one node masked, predict the masked node's identity.
    """
    
    def __init__(self, graph_snapshots, cow_to_idx, min_degree=1):
        """
        Args:
            graph_snapshots: List of networkx graphs
            cow_to_idx: Dict mapping cow ID to integer index
            min_degree: Minimum degree for a node to be maskable (default: 1, must have at least 1 neighbor)
        """
        self.samples = []
        self.cow_to_idx = cow_to_idx
        self.idx_to_cow = {v: k for k, v in cow_to_idx.items()}
        self.num_classes = len(cow_to_idx)
        
        print(f"Creating dataset from {len(graph_snapshots)} graph snapshots...")
        
        for graph_idx, G in enumerate(tqdm(graph_snapshots, desc="Processing graphs")):
            # Get all cow nodes (exclude POIs if present)
            cow_nodes = [n for n in G.nodes() if n in cow_to_idx]
            
            # For each cow in the graph, create a sample where that cow is masked
            for masked_cow in cow_nodes:
                # Check if node has enough neighbors
                neighbors = list(G.neighbors(masked_cow))
                if len(neighbors) >= min_degree:
                    self.samples.append({
                        'graph_idx': graph_idx,
                        'graph': G,
                        'masked_cow': masked_cow,
                        'label': cow_to_idx[masked_cow]
                    })
        
        print(f"Created {len(self.samples)} samples from {len(graph_snapshots)} graphs")
        print(f"Average samples per graph: {len(self.samples) / len(graph_snapshots):.1f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        G = sample['graph']
        masked_cow = sample['masked_cow']
        label = sample['label']
        
        # Build PyTorch Geometric Data object
        # Nodes: all cows in the graph
        nodes = [n for n in G.nodes() if n in self.cow_to_idx]
        node_to_local_idx = {n: i for i, n in enumerate(nodes)}
        
        # Node features: one-hot encoding of cow ID, with masked cow set to zeros
        num_nodes = len(nodes)
        x = torch.zeros(num_nodes, self.num_classes)
        
        for i, node in enumerate(nodes):
            if node != masked_cow:
                x[i, self.cow_to_idx[node]] = 1.0
            # else: leave as zeros (masked)
        
        # Store which node is masked (for prediction)
        masked_node_idx = node_to_local_idx[masked_cow]
        
        # Build edge index and edge attributes (RSSI values)
        edge_index = []
        edge_attr = []
        
        for u, v, data in G.edges(data=True):
            if u in node_to_local_idx and v in node_to_local_idx:
                u_idx = node_to_local_idx[u]
                v_idx = node_to_local_idx[v]
                rssi = data.get('rssi', -50)  # Default RSSI if not present
                
                # Add both directions (undirected graph)
                edge_index.append([u_idx, v_idx])
                edge_attr.append([rssi])
                edge_index.append([v_idx, u_idx])
                edge_attr.append([rssi])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Normalize RSSI to [0, 1] range (assuming RSSI in [-100, -30])
        edge_attr = (edge_attr + 100) / 70.0
        edge_attr = torch.clamp(edge_attr, 0, 1)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
            masked_node_idx=torch.tensor([masked_node_idx], dtype=torch.long)
        )
        
        return data


class GraphNodePredictor(nn.Module):
    """
    GNN model to predict masked node identity from graph structure.
    """
    
    def __init__(self, num_classes, hidden_dim=64, num_layers=3, gnn_type='gcn'):
        super().__init__()
        
        self.num_classes = num_classes
        self.gnn_type = gnn_type
        
        # Input: one-hot encoding of cow IDs
        input_dim = num_classes
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=False))
        
        # Prediction head
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        masked_node_idx = data.masked_node_idx
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Extract embeddings of masked nodes
        # For batched data, need to adjust indices
        batch_offset = torch.zeros_like(batch)
        for i in range(1, batch.max().item() + 1):
            batch_offset[batch == i] = (batch == (i - 1)).sum()
        
        # Get the absolute indices of masked nodes in the batched graph
        global_masked_idx = masked_node_idx.squeeze() + batch_offset[masked_node_idx.squeeze()]
        
        # Get embeddings of masked nodes
        masked_embeddings = x[global_masked_idx]
        
        # Predict class
        logits = self.fc(masked_embeddings)
        
        return logits


def load_network_sequences(data_dir, pattern="network_sequence_rssi-68_*.pkl"):
    """Load all graph snapshots from pickle files."""
    data_path = Path(data_dir) / "network_sequence"
    files = sorted(data_path.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern {pattern} in {data_path}")
    
    all_graphs = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            # Handle dictionary format with 'graph' key
            if isinstance(data, dict) and 'graph' in data:
                all_graphs.append(data['graph'])
            elif isinstance(data, list):
                # Handle list of dictionaries
                for item in data:
                    if isinstance(item, dict) and 'graph' in item:
                        all_graphs.append(item['graph'])
                    else:
                        all_graphs.append(item)
            else:
                all_graphs.append(data)
    
    print(f"Loaded {len(all_graphs)} graph snapshots from {len(files)} files")
    return all_graphs


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y.squeeze()).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            pred = logits.argmax(dim=1)
            
            correct += (pred == batch.y.squeeze()).sum().item()
            total += batch.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
    
    accuracy = correct / total
    return accuracy, all_preds, all_labels


def plot_training_curves(train_losses, train_accs, val_accs, best_epoch, output_dir):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.axvline(best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.scatter([best_epoch], [val_accs[best_epoch - 1]], color='red', s=100, zorder=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")
    plt.close()


def compute_per_cow_accuracy(all_preds, all_labels, idx_to_cow):
    """Compute accuracy per cow."""
    per_cow_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, label in zip(all_preds, all_labels):
        cow_id = idx_to_cow[label]
        per_cow_stats[cow_id]['total'] += 1
        if pred == label:
            per_cow_stats[cow_id]['correct'] += 1
    
    # Compute accuracy
    results = []
    for cow_id, stats in per_cow_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        results.append({
            'cow_id': cow_id,
            'correct': stats['correct'],
            'total': stats['total'],
            'accuracy': accuracy
        })
    
    return sorted(results, key=lambda x: x['accuracy'], reverse=True)


def plot_sample_graphs(dataset, idx_to_cow, output_dir, num_samples=6):
    """Visualize sample graphs from the dataset."""
    print(f"\nPlotting {num_samples} sample graphs...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx, ax in zip(indices, axes):
        sample = dataset.samples[idx]
        G = sample['graph']
        masked_cow = sample['masked_cow']
        
        # Create a subgraph with only cow nodes
        cow_nodes = [n for n in G.nodes() if n in dataset.cow_to_idx]
        subG = G.subgraph(cow_nodes).copy()
        
        # Node colors: masked node is red, others are blue
        node_colors = ['red' if n == masked_cow else 'lightblue' for n in subG.nodes()]
        
        # Node sizes based on degree
        node_sizes = [300 + 100 * subG.degree(n) for n in subG.nodes()]
        
        # Draw graph
        pos = nx.spring_layout(subG, seed=42, k=0.5)
        nx.draw_networkx_nodes(subG, pos, node_color=node_colors, 
                               node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(subG, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(subG, pos, font_size=6, ax=ax)
        
        # Add edge labels (RSSI values)
        edge_labels = {(u, v): f"{d.get('rssi', 0):.0f}" 
                      for u, v, d in subG.edges(data=True)}
        nx.draw_networkx_edge_labels(subG, pos, edge_labels, font_size=5, ax=ax)
        
        ax.set_title(f"Masked: {masked_cow} (red)\nNodes: {len(subG.nodes())}, Edges: {len(subG.edges())}", 
                    fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_graphs.png', dpi=200, bbox_inches='tight')
    print(f"Saved sample graphs to {output_dir / 'sample_graphs.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Graph-based Node Prediction for Cow Identification')
    parser.add_argument('--data-dir', type=str, default='/home/elouan/wait4_data',
                        help='Directory containing network sequence pickle files')
    parser.add_argument('--output-dir', type=str, default='graph_prediction_out',
                        help='Output directory for results')
    parser.add_argument('--gnn-type', type=str, default='gcn', choices=['gcn', 'gat'],
                        help='Type of GNN to use')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--min-degree', type=int, default=1,
                        help='Minimum degree for maskable nodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load graph snapshots
    print("\n" + "="*70)
    print("GRAPH-BASED NODE PREDICTION FOR COW IDENTIFICATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  GNN Type: {args.gnn_type.upper()}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Num Layers: {args.num_layers}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Min Degree: {args.min_degree}")
    print()
    
    graphs = load_network_sequences(args.data_dir)
    
    # Build vocabulary (cow IDs)
    all_cow_ids = set()
    for G in graphs:
        all_cow_ids.update(G.nodes())
    
    cow_to_idx = {cow: idx for idx, cow in enumerate(sorted(all_cow_ids))}
    idx_to_cow = {idx: cow for cow, idx in cow_to_idx.items()}
    num_classes = len(cow_to_idx)
    
    print(f"Number of unique cows: {num_classes}")
    print(f"Cow IDs: {sorted(all_cow_ids)[:10]}{'...' if len(all_cow_ids) > 10 else ''}")
    print()
    
    # Split graphs into train/val/test
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=args.seed)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.2, random_state=args.seed)
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_graphs)} graphs")
    print(f"  Val: {len(val_graphs)} graphs")
    print(f"  Test: {len(test_graphs)} graphs")
    print()
    
    # Create datasets
    train_dataset = GraphSnapshotDataset(train_graphs, cow_to_idx, min_degree=args.min_degree)
    val_dataset = GraphSnapshotDataset(val_graphs, cow_to_idx, min_degree=args.min_degree)
    test_dataset = GraphSnapshotDataset(test_graphs, cow_to_idx, min_degree=args.min_degree)
    
    print(f"\nSample counts:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print()
    
    # Plot sample graphs
    plot_sample_graphs(train_dataset, idx_to_cow, output_dir, num_samples=6)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=lambda x: Batch.from_data_list(x))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=lambda x: Batch.from_data_list(x))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=lambda x: Batch.from_data_list(x))
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    model = GraphNodePredictor(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type
    ).to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("="*70)
    print("TRAINING")
    print("="*70)
    
    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} ⭐ New best!")
        else:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    print()
    print("="*70)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print("="*70)
    print()
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_accs, best_epoch, output_dir)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc:.2%})")
    print()
    
    # Per-cow accuracy
    print("Computing per-cow accuracy...")
    per_cow_results = compute_per_cow_accuracy(test_preds, test_labels, idx_to_cow)
    
    # Save per-cow results
    import pandas as pd
    df = pd.DataFrame(per_cow_results)
    df.to_csv(output_dir / 'per_cow_accuracy.csv', index=False)
    print(f"Saved per-cow accuracy to {output_dir / 'per_cow_accuracy.csv'}")
    
    # Print top and bottom performers
    print("\nTop 5 most predictable cows:")
    for i, result in enumerate(per_cow_results[:5], 1):
        print(f"  {i}. {result['cow_id']}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    
    print("\nBottom 5 least predictable cows:")
    for i, result in enumerate(per_cow_results[-5:], 1):
        print(f"  {i}. {result['cow_id']}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'cow_to_idx': cow_to_idx,
        'idx_to_cow': idx_to_cow,
        'args': args,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }, output_dir / 'best_model.pt')
    
    print(f"\nSaved model to {output_dir / 'best_model.pt'}")
    
    # Save summary
    with open(output_dir / 'results_summary.txt', 'w') as f:
        f.write("GRAPH-BASED NODE PREDICTION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  GNN Type: {args.gnn_type.upper()}\n")
        f.write(f"  Hidden Dim: {args.hidden_dim}\n")
        f.write(f"  Num Layers: {args.num_layers}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Min Degree: {args.min_degree}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc:.2%})\n")
        f.write(f"  Test Accuracy: {test_acc:.4f} ({test_acc:.2%})\n\n")
        f.write(f"Dataset:\n")
        f.write(f"  Train: {len(train_dataset)} samples from {len(train_graphs)} graphs\n")
        f.write(f"  Val: {len(val_dataset)} samples from {len(val_graphs)} graphs\n")
        f.write(f"  Test: {len(test_dataset)} samples from {len(test_graphs)} graphs\n")
        f.write(f"  Num Classes: {num_classes}\n")
    
    print(f"Saved results summary to {output_dir / 'results_summary.txt'}")
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == '__main__':
    main()
