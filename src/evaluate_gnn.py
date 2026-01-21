#!/usr/bin/env python3
"""
evaluate_gnn.py

Loads a trained GNN model and evaluates it on the brush proximity dataset,
generating detailed metrics and visualizations.

Usage:
    python Code/evaluate_gnn.py \
      --model outputs/brush_proximity/gnn_model/best_model.pth \
      --pkl outputs/brush_proximity/brush_proximity_graphs_20251112_111121.pkl \
      --brush-id 366b \
      --out-dir outputs/brush_proximity/gnn_eval
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


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
        out = self.lin(x).squeeze(-1)
        return out


def load_pickle_graphs(pkl_path: Path) -> List[nx.Graph]:
    """Load NetworkX graphs from pickle file."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    graphs = []
    if isinstance(obj, dict):
        if "graphs" in obj:
            graphs = obj["graphs"]
        else:
            possible_graphs = [
                v for v in obj.values() if isinstance(v, (nx.Graph, nx.DiGraph))
            ]
            if possible_graphs:
                graphs = possible_graphs
    elif isinstance(obj, list):
        if len(obj) > 0:
            if isinstance(obj[0], (nx.Graph, nx.DiGraph)):
                graphs = obj
            elif isinstance(obj[0], dict) and "graph" in obj[0]:
                graphs = [elt["graph"] for elt in obj]

    graphs = [g for g in graphs if isinstance(g, (nx.Graph, nx.DiGraph))]
    return graphs


def label_brushed_node(G: nx.Graph, brush_id: str) -> Tuple[List[float], int]:
    """Return (labels, brushed_node_idx) or (labels, -1) if no valid label."""
    nodes = list(G.nodes())
    if brush_id not in nodes:
        return [0.0] * len(nodes), -1

    brush_neighbors = list(G.neighbors(brush_id))
    if not brush_neighbors:
        return [0.0] * len(nodes), -1

    best_node = None
    best_rssi = -np.inf
    for nb in brush_neighbors:
        try:
            ed = G.get_edge_data(brush_id, nb)
            if isinstance(ed, dict) and "rssi" in ed:
                val = ed["rssi"]
            elif isinstance(ed, dict) and "weight" in ed:
                val = ed["weight"]
            else:
                if isinstance(ed, dict) and any(
                    isinstance(v, dict) for v in ed.values()
                ):
                    for sub in ed.values():
                        if isinstance(sub, dict) and ("rssi" in sub or "weight" in sub):
                            val = sub.get("rssi", sub.get("weight", None))
                            break
                    else:
                        val = None
                else:
                    val = None
            if val is None:
                continue
            try:
                valf = float(val)
            except Exception:
                continue
            if valf > best_rssi:
                best_rssi = valf
                best_node = nb
        except Exception:
            continue

    labels = [0.0] * len(nodes)
    if best_node is not None:
        idx = nodes.index(best_node)
        labels[idx] = 1.0
        return labels, idx
    else:
        return labels, -1


def build_pyg_data(
    G: nx.Graph, cow2idx: Dict[str, int], brush_id: str
) -> Tuple[Data, int]:
    """Build PyG Data object and return (data, true_brushed_idx)."""
    nodes = list(G.nodes())
    x = torch.zeros((len(nodes), len(cow2idx)), dtype=torch.float32)
    for i, n in enumerate(nodes):
        key = str(n)
        if key in cow2idx:
            x[i, cow2idx[key]] = 1.0

    edge_index = []
    for u, v in G.edges():
        ui = nodes.index(u)
        vi = nodes.index(v)
        edge_index.append([ui, vi])
        edge_index.append([vi, ui])

    if len(edge_index) == 0:
        return None, -1

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    labels, true_idx = label_brushed_node(G, brush_id)
    if true_idx == -1:
        return None, -1

    y = torch.tensor(labels, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data, true_idx


def evaluate_model(model, graphs, cow2idx, brush_id, device):
    """Evaluate model and return detailed metrics."""
    model.eval()
    results = {
        "correct": 0,
        "total": 0,
        "top_k_acc": {1: 0, 2: 0, 3: 0},
        "predictions": [],
        "true_labels": [],
        "predicted_cows": [],
        "true_cows": [],
        "confidence_scores": [],
    }

    idx2cow = {v: k for k, v in cow2idx.items()}

    with torch.no_grad():
        for G in graphs:
            data, true_idx = build_pyg_data(G, cow2idx, brush_id)
            if data is None:
                continue

            data = data.to(device)
            logits = model(data.x, data.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Get predictions
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            # Top-k accuracy
            top_k_indices = np.argsort(probs)[::-1]
            for k in [1, 2, 3]:
                if true_idx in top_k_indices[:k]:
                    results["top_k_acc"][k] += 1

            # Record results
            results["total"] += 1
            if pred_idx == true_idx:
                results["correct"] += 1

            nodes = list(G.nodes())
            true_cow = str(nodes[true_idx])
            pred_cow = str(nodes[pred_idx])

            results["predictions"].append(pred_idx)
            results["true_labels"].append(true_idx)
            results["predicted_cows"].append(pred_cow)
            results["true_cows"].append(true_cow)
            results["confidence_scores"].append(confidence)

    # Compute accuracies
    results["accuracy"] = results["correct"] / results["total"]
    for k in [1, 2, 3]:
        results["top_k_acc"][k] /= results["total"]

    return results


def plot_results(results, out_dir):
    """Generate evaluation visualizations."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion matrix of top predicted cows
    from collections import Counter

    true_counter = Counter(results["true_cows"])
    pred_counter = Counter(results["predicted_cows"])

    top_cows = sorted(true_counter.keys(), key=lambda x: true_counter[x], reverse=True)[
        :10
    ]

    confusion = np.zeros((len(top_cows), len(top_cows)))
    cow_to_idx = {cow: i for i, cow in enumerate(top_cows)}

    for true_cow, pred_cow in zip(results["true_cows"], results["predicted_cows"]):
        if true_cow in cow_to_idx and pred_cow in cow_to_idx:
            confusion[cow_to_idx[true_cow], cow_to_idx[pred_cow]] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        confusion,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=top_cows,
        yticklabels=top_cows,
        ax=ax,
    )
    ax.set_xlabel("Predicted Cow", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Cow", fontsize=12, fontweight="bold")
    ax.set_title(
        "Confusion Matrix: Top 10 Cows Being Brushed", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # 2. Confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    correct_conf = [
        results["confidence_scores"][i]
        for i in range(len(results["confidence_scores"]))
        if results["predictions"][i] == results["true_labels"][i]
    ]
    incorrect_conf = [
        results["confidence_scores"][i]
        for i in range(len(results["confidence_scores"]))
        if results["predictions"][i] != results["true_labels"][i]
    ]

    axes[0].hist(correct_conf, bins=30, alpha=0.7, color="green", label="Correct")
    axes[0].hist(incorrect_conf, bins=30, alpha=0.7, color="red", label="Incorrect")
    axes[0].set_xlabel("Confidence Score", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Frequency", fontsize=11, fontweight="bold")
    axes[0].set_title(
        "Prediction Confidence Distribution", fontsize=12, fontweight="bold"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 3. Top-k accuracy
    k_vals = [1, 2, 3]
    accs = [results["top_k_acc"][k] for k in k_vals]
    axes[1].bar(k_vals, accs, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("k", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Top-k Accuracy", fontsize=11, fontweight="bold")
    axes[1].set_title("Top-k Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xticks(k_vals)
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)

    for i, (k, acc) in enumerate(zip(k_vals, accs)):
        axes[1].text(k, acc + 0.02, f"{acc:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "performance_metrics.png", dpi=150)
    plt.close()

    print(f"\n✅ Visualizations saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to saved model .pth file"
    )
    parser.add_argument(
        "--pkl", type=str, required=True, help="Path to brush proximity graphs pickle"
    )
    parser.add_argument("--brush-id", type=str, default="366b", help="Brush node ID")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./gnn_eval",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location="cpu")
    cow2idx = checkpoint["cow2idx"]

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    model = GCNNodeClassifier(
        in_channels=len(cow2idx), hidden_channels=128, dropout=0.5
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model loaded. Device: {device}")

    # Load graphs
    print(f"Loading graphs from {args.pkl}...")
    graphs = load_pickle_graphs(Path(args.pkl))
    print(f"Loaded {len(graphs)} graphs")

    # Evaluate
    print("\n🔍 Evaluating model...")
    results = evaluate_model(model, graphs, cow2idx, args.brush_id, device)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total graphs evaluated: {results['total']}")
    print(
        f"Accuracy (Top-1): {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    print(f"Top-2 Accuracy: {results['top_k_acc'][2]:.4f}")
    print(f"Top-3 Accuracy: {results['top_k_acc'][3]:.4f}")
    print(f"Mean Confidence: {np.mean(results['confidence_scores']):.4f}")
    print(f"Median Confidence: {np.median(results['confidence_scores']):.4f}")
    print("=" * 70)

    # Top predicted cows
    from collections import Counter

    pred_counter = Counter(results["predicted_cows"])
    true_counter = Counter(results["true_cows"])

    print("\n🏆 Top 10 Most Frequently Brushed Cows (Ground Truth):")
    for cow, count in true_counter.most_common(10):
        pct = 100 * count / results["total"]
        print(f"  {cow}: {count} times ({pct:.1f}%)")

    print("\n🤖 Top 10 Most Frequently Predicted Cows:")
    for cow, count in pred_counter.most_common(10):
        pct = 100 * count / results["total"]
        print(f"  {cow}: {count} times ({pct:.1f}%)")

    # Save metrics
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": float(results["accuracy"]),
        "top_2_accuracy": float(results["top_k_acc"][2]),
        "top_3_accuracy": float(results["top_k_acc"][3]),
        "total_graphs": results["total"],
        "correct_predictions": results["correct"],
        "mean_confidence": float(np.mean(results["confidence_scores"])),
        "median_confidence": float(np.median(results["confidence_scores"])),
        "true_cow_distribution": dict(true_counter.most_common(20)),
        "predicted_cow_distribution": dict(pred_counter.most_common(20)),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Metrics saved to {out_dir / 'metrics.json'}")

    # Generate plots
    plot_results(results, out_dir)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
