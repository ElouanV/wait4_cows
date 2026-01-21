#!/usr/bin/env python3
"""
gnn_brush_predictor.py

Loads the brush-focused NetworkX graph sequence (pickle), builds a PyTorch Geometric dataset
with node one-hot features (cow ID), labels each node as 1 when it is the brushed cow (highest
RSSI edge to the brush) and 0 otherwise, trains a small GCN for node-level binary classification,
and saves the trained model and cow-id mapping.

Notes:
- The model does NOT use RSSI values as node or edge features (per your request). RSSI is only
  used to determine the label (which node is being brushed).
- The script attempts to be tolerant to a few common pickle layouts used in notebooks.

Usage (example):
    python Code/gnn_brush_predictor.py \
      --pkl ../outputs/brush_proximity/brush_proximity_graphs_20251112_111121.pkl \
      --brush-id 366b --epochs 30 --batch-size 16 --out-dir ../outputs/brush_proximity/gnn_model

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

# Try to import PyG, but allow the script to fail with a clear error if not installed
try:
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise ImportError(
        "This script requires PyTorch Geometric (torch_geometric).\n"
        "Install it following instructions at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html\n"
        f"Original import error: {e}"
    )


def load_pickle_graphs(pkl_path: Path) -> Tuple[List[nx.Graph], Dict]:
    """Load a pickle and return (list_of_networkx_graphs, optional_metadata_dict).

    The pickle may contain several formats; we handle common variants.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    graphs = []
    metadata = {}

    # Common patterns
    if isinstance(obj, dict):
        # { 'graphs': [...], 'metadata': {...} }
        if "graphs" in obj:
            graphs = obj["graphs"]
            metadata = obj.get("metadata", {})
        else:
            # maybe a mapping of timestamp->graph
            possible_graphs = [
                v for v in obj.values() if isinstance(v, (nx.Graph, nx.DiGraph))
            ]
            if possible_graphs:
                graphs = possible_graphs
                metadata = {
                    k: v
                    for k, v in obj.items()
                    if not isinstance(v, (nx.Graph, nx.DiGraph))
                }
            else:
                raise ValueError(
                    "Pickle is dict-like but no 'graphs' key or Graph values found"
                )
    elif isinstance(obj, list):
        # Could be list of graphs or list of dicts with 'graph' key
        if len(obj) == 0:
            raise ValueError("Pickle contains empty list")
        if isinstance(obj[0], (nx.Graph, nx.DiGraph)):
            graphs = obj
        elif isinstance(obj[0], dict) and "graph" in obj[0]:
            graphs = [elt["graph"] for elt in obj]
            # collect other fields into metadata list
            metadata = {
                "entries": [
                    {k: v for k, v in elt.items() if k != "graph"} for elt in obj
                ]
            }
        else:
            raise ValueError(
                "Unrecognized list format inside pickle; expected list of graphs or dicts with 'graph' key"
            )
    else:
        raise ValueError(
            "Unrecognized pickle content type: must be dict or list containing graphs"
        )

    # Ensure all elements are networkx graphs
    graphs = [g for g in graphs if isinstance(g, (nx.Graph, nx.DiGraph))]
    if not graphs:
        raise ValueError("No networkx graphs found in pickle file")

    return graphs, metadata


class BrushProximityDataset(InMemoryDataset):
    def __init__(
        self,
        graphs: List[nx.Graph],
        brush_id: str,
        cow2idx: Dict[str, int] = None,
        transform=None,
    ):
        super().__init__(None, transform)
        self.raw_graphs = graphs
        self.brush_id = str(brush_id)
        self.cow2idx = cow2idx or self._build_cow_mapping(graphs)
        self.data_list = self._build_data_list()

    def _build_cow_mapping(self, graphs: List[nx.Graph]) -> Dict[str, int]:
        cows = set()
        for G in graphs:
            for n in G.nodes():
                cows.add(str(n))
        cows = sorted(cows)
        return {cow: i for i, cow in enumerate(cows)}

    def _label_brushed_node(self, G: nx.Graph) -> Tuple[List[float], bool]:
        """Return per-node binary labels (1.0 for brushed cow) and a boolean indicating whether label exists.

        Definition: cow being brushed = neighbor of brush with highest RSSI on the edge to brush.
        RSSI is expected to be stored in edge attribute 'rssi' or 'weight'.
        """
        nodes = list(G.nodes())
        if self.brush_id not in nodes:
            return [0.0] * len(nodes), False

        brush_neighbors = list(G.neighbors(self.brush_id))
        if not brush_neighbors:
            return [0.0] * len(nodes), False

        best_node = None
        best_rssi = -np.inf
        for nb in brush_neighbors:
            # edge keys in networkx may store attributes
            try:
                ed = G.get_edge_data(self.brush_id, nb)
                # If multigraph, pick first edge
                if isinstance(ed, dict) and "rssi" in ed:
                    val = ed["rssi"]
                elif isinstance(ed, dict) and "weight" in ed:
                    val = ed["weight"]
                else:
                    # if ed is nested (multigraph) try further
                    if isinstance(ed, dict) and any(
                        isinstance(v, dict) for v in ed.values()
                    ):
                        # multigraph data
                        for sub in ed.values():
                            if isinstance(sub, dict) and (
                                "rssi" in sub or "weight" in sub
                            ):
                                val = sub.get("rssi", sub.get("weight", None))
                                break
                            else:
                                val = None
                    else:
                        val = None
                if val is None:
                    continue
                # ensure numeric
                try:
                    valf = float(val)
                except Exception:
                    continue
                # note: higher RSSI (less negative) is closer, so use valf directly
                if valf > best_rssi:
                    best_rssi = valf
                    best_node = nb
            except Exception:
                continue

        labels = [0.0] * len(nodes)
        if best_node is not None:
            idx = nodes.index(best_node)
            labels[idx] = 1.0
            return labels, True
        else:
            return labels, False

    def _build_data_list(self) -> List[Data]:
        data_list = []
        for G in self.raw_graphs:
            nodes = list(G.nodes())
            # node feature: one-hot of cow id over global mapping
            x = torch.zeros((len(nodes), len(self.cow2idx)), dtype=torch.float32)
            for i, n in enumerate(nodes):
                key = str(n)
                if key in self.cow2idx:
                    x[i, self.cow2idx[key]] = 1.0

            # edge_index
            edge_index = []
            for u, v in G.edges():
                ui = nodes.index(u)
                vi = nodes.index(v)
                edge_index.append([ui, vi])
                edge_index.append([vi, ui])
            if len(edge_index) == 0:
                # skip isolated graphs
                continue
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # labels: binary per node (float)
            labels, has_label = self._label_brushed_node(G)
            if not has_label:
                # skip graphs without a valid brushed node
                continue
            y = torch.tensor(labels, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class GCNNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        out = self.lin(x).squeeze(-1)  # per-node logits
        return out


def train(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    for data in loader:
        data = data.to(device)
        optim.zero_grad()
        logits = model(data.x, data.edge_index)
        # y is float 0/1
        loss = F.binary_cross_entropy_with_logits(logits, data.y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
    return total_loss / total_nodes if total_nodes > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, data.y)
        total_loss += loss.item() * data.num_nodes
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(data.y.cpu().numpy())

    total_nodes = sum([len(a) for a in all_targets])
    avg_loss = total_loss / total_nodes

    # compute simple accuracy of predicting the single brushed node per graph
    # for each graph, select node with highest predicted score and check if target==1
    correct = 0
    total_graphs = 0
    for preds, targets in zip(all_preds, all_targets):
        if len(preds) == 0:
            continue
        pred_idx = int(np.argmax(preds))
        if int(targets[pred_idx]) == 1:
            correct += 1
        total_graphs += 1
    acc = correct / total_graphs if total_graphs > 0 else 0.0

    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl", type=str, required=True, help="Path to brush proximity graphs pickle"
    )
    parser.add_argument(
        "--brush-id", type=str, default="366b", help="Brush node ID string"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="./gnn_out")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graphs from:", pkl_path)
    graphs, metadata = load_pickle_graphs(pkl_path)
    print(f"Loaded {len(graphs)} raw graphs")

    dataset = BrushProximityDataset(graphs, brush_id=args.brush_id)
    print(
        f"Filtered dataset contains {len(dataset)} graphs (snapshots) with valid brushed labels"
    )
    if len(dataset) == 0:
        raise RuntimeError("No usable graphs found after filtering for brush neighbors")

    # save cow mapping
    cow2idx = dataset.cow2idx
    with open(out_dir / "cow2idx.json", "w") as f:
        json.dump(cow2idx, f, indent=2)
    print(f"Saved cow2idx mapping with {len(cow2idx)} unique nodes")

    # split
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_n = int(0.8 * n)
    train_idx = idx[:train_n].tolist()
    val_idx = idx[train_n:].tolist()

    train_list = [dataset[i] for i in train_idx]
    val_list = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=args.batch_size, shuffle=False)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    model = GCNNodeClassifier(in_channels=len(cow2idx), hidden_channels=args.hidden).to(
        device
    )
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_path = out_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optim, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "cow2idx": cow2idx}, best_path
            )

    print(
        f"Training finished. Best val acc: {best_val_acc:.4f}. Model saved to {best_path}"
    )


if __name__ == "__main__":
    main()
