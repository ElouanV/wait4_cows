#!/usr/bin/env python3
"""
improved_gnn_predictor.py

Enhanced GNN for POI proximity prediction with multiple improvements:
1. RSSI edge features (instead of binary connectivity)
2. Rich node features (degree, avg RSSI, POI distance)
3. Graph Attention Networks (GAT) for better neighbor weighting
4. Class imbalance handling
5. Top-K evaluation metrics

Usage:
    python src/improved_gnn_predictor.py \
      --pkl brush_experiment/brush_proximity_rssi-70_20251119_122738.pkl \
      --poi-id 366b \
      --out-dir brush_experiment/improved_gnn_model
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

try:
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError as e:
    raise ImportError(
        "This script requires PyTorch Geometric.\n"
        f"Original error: {e}"
    )


def load_pickle_graphs(pkl_path: Path) -> List[Dict]:
    """Load proximity graphs from pickle file."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    
    if isinstance(obj, list):
        # Check if it's list of dicts with 'graph' key
        if len(obj) > 0 and isinstance(obj[0], dict) and "graph" in obj[0]:
            return obj
        else:
            raise ValueError("Unexpected pickle format")
    else:
        raise ValueError("Pickle must contain a list of graph dictionaries")


class ImprovedProximityDataset(InMemoryDataset):
    """
    Dataset with enhanced features:
    - RSSI edge features
    - Rich node features (degree, avg RSSI, distance to POI)
    - Top-k labeling option
    """
    
    def __init__(
        self,
        graph_dicts: List[Dict],
        poi_id: str,
        cow2idx: Dict[str, int] = None,
        use_edge_features: bool = True,
        use_rich_node_features: bool = True,
        top_k_labels: int = 1,
        transform=None,
    ):
        super().__init__(None, transform)
        self.raw_graph_dicts = graph_dicts
        self.poi_id = str(poi_id)
        self.use_edge_features = use_edge_features
        self.use_rich_node_features = use_rich_node_features
        self.top_k_labels = top_k_labels
        self.cow2idx = cow2idx or self._build_cow_mapping()
        self.data_list = self._build_data_list()
    
    def _build_cow_mapping(self) -> Dict[str, int]:
        """Create mapping from cow ID to integer index."""
        cows = set()
        for graph_dict in self.raw_graph_dicts:
            G = graph_dict['graph']
            for n in G.nodes():
                cows.add(str(n))
        cows = sorted(cows)
        return {cow: i for i, cow in enumerate(cows)}
    
    def _get_node_features(self, G: nx.Graph, node: str) -> List[float]:
        """
        Extract rich node features.
        
        Features:
        1. Degree (normalized)
        2. Average RSSI to neighbors
        3. Max RSSI to neighbors
        4. RSSI to POI (if connected, else -100)
        5. Is POI (binary)
        """
        features = []
        
        # Degree (normalized by max possible)
        degree = G.degree(node)
        max_degree = len(G.nodes()) - 1
        features.append(degree / max_degree if max_degree > 0 else 0)
        
        # RSSI statistics to neighbors
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            rssi_values = []
            for neighbor in neighbors:
                edge_data = G.get_edge_data(node, neighbor)
                rssi = edge_data.get('rssi', -80) if edge_data else -80
                rssi_values.append(rssi)
            
            avg_rssi = np.mean(rssi_values)
            max_rssi = np.max(rssi_values)
        else:
            avg_rssi = -100
            max_rssi = -100
        
        # Normalize RSSI values (-100 to -50 range → 0 to 1)
        features.append((avg_rssi + 100) / 50)
        features.append((max_rssi + 100) / 50)
        
        # RSSI to POI
        if G.has_edge(node, self.poi_id):
            edge_data = G.get_edge_data(node, self.poi_id)
            rssi_to_poi = edge_data.get('rssi', -80) if edge_data else -80
        else:
            rssi_to_poi = -100
        features.append((rssi_to_poi + 100) / 50)
        
        # Is POI (binary indicator)
        features.append(1.0 if node == self.poi_id else 0.0)
        
        return features
    
    def _label_top_k_nodes(self, G: nx.Graph) -> Tuple[List[float], bool]:
        """
        Label top-k nodes by RSSI to POI.
        
        Returns:
            labels: Float labels (1.0 for top-1, 0.7 for top-2, etc.)
            has_label: Whether valid labels exist
        """
        nodes = list(G.nodes())
        
        if self.poi_id not in G:
            return [0.0] * len(nodes), False
        
        # Get all neighbors of POI with their RSSI
        neighbor_rssi = []
        for neighbor in G.neighbors(self.poi_id):
            if neighbor == self.poi_id:
                continue
            edge_data = G.get_edge_data(self.poi_id, neighbor)
            rssi = edge_data.get('rssi', -100) if edge_data else -100
            neighbor_rssi.append((neighbor, rssi))
        
        if len(neighbor_rssi) == 0:
            return [0.0] * len(nodes), False
        
        # Sort by RSSI (descending - higher is better)
        neighbor_rssi.sort(key=lambda x: x[1], reverse=True)
        
        # Create labels
        labels = [0.0] * len(nodes)
        k = min(self.top_k_labels, len(neighbor_rssi))
        
        for rank, (neighbor, rssi) in enumerate(neighbor_rssi[:k]):
            idx = nodes.index(neighbor)
            # Assign decreasing weights: 1.0, 0.7, 0.5, 0.3, ...
            weight = 1.0 - (rank * 0.3)
            weight = max(0.1, weight)  # Minimum 0.1
            labels[idx] = weight
        
        return labels, True
    
    def _build_data_list(self) -> List[Data]:
        """Build list of PyG Data objects."""
        data_list = []
        
        for graph_dict in self.raw_graph_dicts:
            G = graph_dict['graph']
            nodes = list(G.nodes())
            
            if len(nodes) == 0:
                continue
            
            # Node features
            if self.use_rich_node_features:
                # Rich features: [degree, avg_rssi, max_rssi, rssi_to_poi, is_poi]
                x_list = []
                for node in nodes:
                    features = self._get_node_features(G, node)
                    x_list.append(features)
                x = torch.tensor(x_list, dtype=torch.float32)
            else:
                # One-hot encoding (original)
                x = torch.zeros((len(nodes), len(self.cow2idx)), dtype=torch.float32)
                for i, node in enumerate(nodes):
                    key = str(node)
                    if key in self.cow2idx:
                        x[i, self.cow2idx[key]] = 1.0
            
            # Edge index and edge attributes
            edge_index = []
            edge_attr = []
            
            for u, v in G.edges():
                ui = nodes.index(u)
                vi = nodes.index(v)
                
                # Add both directions
                edge_index.append([ui, vi])
                edge_index.append([vi, ui])
                
                if self.use_edge_features:
                    # Get RSSI value
                    edge_data = G.get_edge_data(u, v)
                    rssi = edge_data.get('rssi', -80) if edge_data else -80
                    # Normalize RSSI (-100 to -50 → 0 to 1)
                    rssi_normalized = (rssi + 100) / 50
                    edge_attr.append([rssi_normalized])
                    edge_attr.append([rssi_normalized])  # Same for both directions
            
            if len(edge_index) == 0:
                continue
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            if self.use_edge_features:
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            else:
                edge_attr = None
            
            # Labels
            labels, has_label = self._label_top_k_nodes(G)
            if not has_label:
                continue
            
            y = torch.tensor(labels, dtype=torch.float32)
            
            # Create Data object
            if edge_attr is not None:
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                data = Data(x=x, edge_index=edge_index, y=y)
            
            data_list.append(data)
        
        return data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


class ImprovedGNN(nn.Module):
    """
    Improved GNN with:
    - Graph Attention Networks (GAT) for better neighbor weighting
    - Edge feature support
    - Deeper architecture
    """
    
    def __init__(
        self,
        in_channels: int,
        edge_dim: int = None,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First GAT layer
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim
            )
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
        
        # Last layer (single head for output)
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=1,
                dropout=dropout,
                edge_dim=edge_dim
            )
        )
        
        # Output layer
        self.lin = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # Not last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final output
        x = self.lin(x).squeeze(-1)
        return x


def train(model, loader, optim, device, pos_weight=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_nodes = 0
    
    for data in loader:
        data = data.to(device)
        optim.zero_grad()
        
        # Forward pass
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            logits = model(data.x, data.edge_index, data.edge_attr)
        else:
            logits = model(data.x, data.edge_index)
        
        # Loss with class weighting
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        loss = criterion(logits, data.y)
        loss.backward()
        optim.step()
        
        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
    
    return total_loss / total_nodes if total_nodes > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model with multiple metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            logits = model(data.x, data.edge_index, data.edge_attr)
        else:
            logits = model(data.x, data.edge_index)
        
        loss = F.binary_cross_entropy_with_logits(logits, data.y)
        total_loss += loss.item() * data.num_nodes
        
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_preds.append((probs > 0.5).astype(int))
        all_targets.append(data.y.cpu().numpy())
    
    total_nodes = sum([len(a) for a in all_targets])
    avg_loss = total_loss / total_nodes if total_nodes > 0 else 0.0
    
    # Top-1 accuracy: predict node with highest score per graph
    correct_top1 = 0
    correct_top3 = 0
    total_graphs = 0
    
    for probs, targets in zip(all_probs, all_targets):
        if len(probs) == 0:
            continue
        
        # Top-1
        pred_idx = int(np.argmax(probs))
        if targets[pred_idx] > 0:
            correct_top1 += 1
        
        # Top-3
        top3_idx = np.argsort(probs)[-3:]
        if any(targets[idx] > 0 for idx in top3_idx):
            correct_top3 += 1
        
        total_graphs += 1
    
    top1_acc = correct_top1 / total_graphs if total_graphs > 0 else 0.0
    top3_acc = correct_top3 / total_graphs if total_graphs > 0 else 0.0
    
    # Calculate precision/recall for positive class
    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)
    all_probs_flat = np.concatenate(all_probs)
    
    # Binary targets for P/R calculation
    binary_targets = (all_targets_flat > 0).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_targets,
        all_preds_flat,
        average='binary',
        zero_division=0
    )
    
    # AUC if we have both classes
    if len(np.unique(binary_targets)) > 1:
        auc = roc_auc_score(binary_targets, all_probs_flat)
    else:
        auc = 0.0
    
    metrics = {
        'loss': avg_loss,
        'top1_acc': top1_acc,
        'top3_acc': top3_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train improved GNN for POI proximity prediction"
    )
    parser.add_argument(
        "--pkl", type=str, required=True, help="Path to proximity graphs pickle"
    )
    parser.add_argument(
        "--poi-id", type=str, required=True, help="POI node ID"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--top-k-labels", type=int, default=1)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    pkl_path = Path(args.pkl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"IMPROVED GNN TRAINING")
    print(f"{'='*70}\n")
    print(f"POI ID: {args.poi_id}")
    print(f"Pickle: {pkl_path.name}")
    print(f"Output: {out_dir}\n")
    
    # Load data
    print("Loading graphs...")
    graph_dicts = load_pickle_graphs(pkl_path)
    print(f"Loaded {len(graph_dicts)} proximity graph snapshots\n")
    
    # Create dataset with improvements
    print("Building dataset with enhanced features...")
    dataset = ImprovedProximityDataset(
        graph_dicts,
        poi_id=args.poi_id,
        use_edge_features=True,
        use_rich_node_features=True,
        top_k_labels=args.top_k_labels
    )
    
    print(f"Dataset contains {len(dataset)} graphs with valid labels")
    print(f"Using rich node features (5D)")
    print(f"Using RSSI edge features")
    print(f"Top-{args.top_k_labels} labeling\n")
    
    if len(dataset) == 0:
        raise RuntimeError("No usable graphs after filtering")
    
    # Save metadata
    metadata = {
        'poi_id': args.poi_id,
        'num_graphs': len(dataset),
        'num_cows': len(dataset.cow2idx),
        'top_k_labels': args.top_k_labels,
        'use_edge_features': True,
        'use_rich_node_features': True
    }
    
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    with open(out_dir / "cow2idx.json", "w") as f:
        json.dump(dataset.cow2idx, f, indent=2)
    
    # TEMPORAL SPLIT (NO SHUFFLING - maintains chronological order)
    # Train on PAST data, validate on MIDDLE data, test on FUTURE data
    n = len(dataset)
    train_n = int(0.7 * n)   # First 70% (earliest timestamps)
    val_n = int(0.85 * n)    # Next 15% (middle timestamps)
    # Remaining 15% for future testing
    
    train_list = dataset[:train_n]        # Past data
    val_list = dataset[train_n:val_n]     # Middle data
    test_list = dataset[val_n:]           # Future data
    
    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)
    
    print("TEMPORAL SPLIT (no shuffling - prevents data leakage):")
    print(f"  Train set: {len(train_list)} graphs (samples 0-{train_n-1}, earliest 70%)")
    print(f"  Val set:   {len(val_list)} graphs (samples {train_n}-{val_n-1}, middle 15%)")
    print(f"  Test set:  {len(test_list)} graphs (samples {val_n}-{n-1}, latest 15%)")
    print(f"✅ No temporal leakage: train on past, validate on middle, test on future\n")
    
    # Create model
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    
    # Get feature dimensions from first sample
    sample = dataset[0]
    in_channels = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else None
    
    print(f"Model architecture:")
    print(f"  Input features: {in_channels}D")
    print(f"  Edge features: {edge_dim}D" if edge_dim else "  No edge features")
    print(f"  Hidden: {args.hidden}D")
    print(f"  Layers: {args.num_layers}")
    print(f"  Attention heads: {args.heads}")
    print(f"  Dropout: {args.dropout}\n")
    
    model = ImprovedGNN(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Positive weight for class imbalance
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    
    print("Training...\n")
    best_top1_acc = 0.0
    best_f1 = 0.0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optim, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch:03d} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val loss: {val_metrics['loss']:.4f} | "
              f"Top-1: {val_metrics['top1_acc']:.3f} | "
              f"Top-3: {val_metrics['top3_acc']:.3f} | "
              f"F1: {val_metrics['f1']:.3f} | "
              f"AUC: {val_metrics['auc']:.3f}")
        
        # Save best model (by top-1 accuracy)
        if val_metrics['top1_acc'] > best_top1_acc:
            best_top1_acc = val_metrics['top1_acc']
            torch.save({
                'model_state': model.state_dict(),
                'cow2idx': dataset.cow2idx,
                'val_metrics': val_metrics,
                'epoch': epoch
            }, out_dir / "best_model_top1.pth")
        
        # Also save best by F1
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'model_state': model.state_dict(),
                'cow2idx': dataset.cow2idx,
                'val_metrics': val_metrics,
                'epoch': epoch
            }, out_dir / "best_model_f1.pth")
    
    # Save training history
    with open(out_dir / "training_history.json", "w") as f:
        # Convert numpy types to Python types for JSON
        history_json = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_metrics': [
                {k: float(v) for k, v in m.items()}
                for m in history['val_metrics']
            ]
        }
        json.dump(history_json, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"Best Top-1 Accuracy (validation): {best_top1_acc:.3f}")
    print(f"Best F1 Score (validation): {best_f1:.3f}")
    
    # Evaluate on TEST set (unseen future data)
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION ON TEST SET (Future Data)")
    print(f"{'='*70}")
    
    # Load best model
    checkpoint = torch.load(out_dir / "best_model_f1.pth")
    model.load_state_dict(checkpoint['model_state'])
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"Test Set Performance (on unseen future data):")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {test_metrics['top1_acc']:.3f}")
    print(f"  Top-3 Accuracy: {test_metrics['top3_acc']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall: {test_metrics['recall']:.3f}")
    print(f"  F1 Score: {test_metrics['f1']:.3f}")
    print(f"  AUC: {test_metrics['auc']:.3f}")
    
    # Save test results
    with open(out_dir / "test_results.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    
    print(f"\nModels saved to: {out_dir}/")
    print(f"  - best_model_top1.pth")
    print(f"  - best_model_f1.pth")
    print(f"  - test_results.json\n")


if __name__ == "__main__":
    main()
