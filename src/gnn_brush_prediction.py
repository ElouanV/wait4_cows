"""
GNN for Predicting Which Cow is Being Brushed

This script implements a Graph Neural Network to predict which cow is
closest to the brush (i.e., being brushed) based on the network structure
alone, without using RSSI signals.

The cow being brushed is defined as the one with the highest RSSI signal
to the brush (366b).
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
BRUSH_ID = "366b"
DATA_DIR = Path("../outputs/brush_proximity")
OUTPUT_DIR = Path("../outputs/gnn_brush_prediction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GNN for Brush Usage Prediction")
print("=" * 80)


# ============================================================================
# 1. Load Dataset
# ============================================================================
print("\n📂 Loading brush proximity dataset...")

# Find the most recent brush proximity graphs
pkl_files = sorted(DATA_DIR.glob("brush_proximity_graphs_*.pkl"))
if not pkl_files:
    raise FileNotFoundError("No brush proximity graph files found!")

graphs_file = pkl_files[-1]
print(f"   Loading: {graphs_file.name}")

with open(graphs_file, "rb") as f:
    data = pickle.load(f)
    brush_graphs = data["graphs"]
    metadata = data["metadata"]

print(f"✅ Loaded {len(brush_graphs)} snapshots")
print(f"   Time range: {metadata['start_time']} to {metadata['end_time']}")


# ============================================================================
# 2. Identify All Unique Cows (for one-hot encoding)
# ============================================================================
print("\n🐄 Building cow ID vocabulary...")

all_cows = set()
for snapshot in brush_graphs:
    graph = snapshot["graph"]
    all_cows.update([node for node in graph.nodes() if node != BRUSH_ID])

# Sort for consistent ordering
cow_list = sorted(list(all_cows))
cow_to_idx = {cow: idx for idx, cow in enumerate(cow_list)}
idx_to_cow = {idx: cow for cow, idx in cow_to_idx.items()}
num_cows = len(cow_list)

print(f"✅ Found {num_cows} unique cows (excluding brush)")
cow_preview = ", ".join(cow_list[:10])
suffix = "..." if num_cows > 10 else ""
print(f"   Cow IDs: {cow_preview}{suffix}")


# ============================================================================
# 3. Prepare Graph Data with Labels
# ============================================================================
print("\n🏗️  Preparing graph dataset with labels...")

dataset_graphs = []
labels = []
skipped = 0

for snapshot in brush_graphs:
    graph = snapshot["graph"]
    timestamp = snapshot["timestamp"]

    # Skip if brush not in graph or no neighbors
    if BRUSH_ID not in graph.nodes():
        skipped += 1
        continue

    # Get brush neighbors and their RSSI values
    brush_neighbors = list(graph.neighbors(BRUSH_ID))
    if len(brush_neighbors) == 0:
        skipped += 1
        continue

    # Find cow with highest RSSI to brush (closest = being brushed)
    max_rssi = -float("inf")
    brushed_cow = None

    for neighbor in brush_neighbors:
        edge_data = graph[BRUSH_ID][neighbor]
        rssi = edge_data.get("rssi", -100)
        if rssi > max_rssi:
            max_rssi = rssi
            brushed_cow = neighbor

    if brushed_cow is None:
        skipped += 1
        continue

    # Store graph and label
    dataset_graphs.append(
        {
            "graph": graph,
            "timestamp": timestamp,
            "brushed_cow": brushed_cow,
            "max_rssi": max_rssi,
        }
    )
    labels.append(cow_to_idx[brushed_cow])

print(f"✅ Prepared {len(dataset_graphs)} valid snapshots")
print(f"   Skipped {skipped} snapshots (no brush or no neighbors)")

# Analyze label distribution
label_counts = Counter(labels)
print("\n📊 Label distribution (top 10 most brushed cows):")
for idx, count in label_counts.most_common(10):
    cow_id = idx_to_cow[idx]
    percentage = (count / len(labels)) * 100
    print(f"   {cow_id}: {count} times ({percentage:.2f}%)")


# ===========================================================================
# 4. Convert NetworkX Graphs to PyTorch Geometric Data
# ===========================================================================
print("\n🔄 Converting to PyTorch Geometric format...")


def networkx_to_pyg(nx_graph, cow_to_idx, brush_id="366b"):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Node features: One-hot encoding of cow IDs (excluding RSSI).
    Edges: Connections between nodes (no edge attributes).
    """
    # Get all nodes except brush
    nodes = [n for n in nx_graph.nodes() if n != brush_id]
    if not nodes:
        return None

    # Create node mapping
    node_to_local_idx = {node: idx for idx, node in enumerate(nodes)}

    # Create node features: one-hot encoding
    x = torch.zeros((len(nodes), len(cow_to_idx)))
    for node, local_idx in node_to_local_idx.items():
        if node in cow_to_idx:
            x[local_idx, cow_to_idx[node]] = 1.0

    # Create edge index (undirected graph)
    edge_list = []
    for u, v in nx_graph.edges():
        # Skip edges involving the brush
        if u == brush_id or v == brush_id:
            continue
        if u in node_to_local_idx and v in node_to_local_idx:
            u_idx = node_to_local_idx[u]
            v_idx = node_to_local_idx[v]
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])  # Add reverse edge

    if len(edge_list) == 0:
        # Graph with no edges (isolated nodes)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


pyg_data_list = []
valid_labels = []

for i, (graph_data, label) in enumerate(zip(dataset_graphs, labels)):
    pyg_data = networkx_to_pyg(graph_data["graph"], cow_to_idx, BRUSH_ID)
    if pyg_data is not None:
        pyg_data_list.append(pyg_data)
        valid_labels.append(label)

print(f"✅ Converted {len(pyg_data_list)} graphs to PyTorch Geometric format")


# ============================================================================
# 5. Split Dataset
# ============================================================================
print("\n✂️  Splitting dataset (70% train, 15% val, 15% test)...")

# First split: 70% train, 30% temp
train_idx, temp_idx = train_test_split(
    range(len(pyg_data_list)), test_size=0.3, random_state=42, stratify=valid_labels
)

# Second split: 15% val, 15% test from the 30% temp
temp_labels = [valid_labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
)

train_data = [pyg_data_list[i] for i in train_idx]
val_data = [pyg_data_list[i] for i in val_idx]
test_data = [pyg_data_list[i] for i in test_idx]

train_labels = torch.tensor([valid_labels[i] for i in train_idx], dtype=torch.long)
val_labels = torch.tensor([valid_labels[i] for i in val_idx], dtype=torch.long)
test_labels = torch.tensor([valid_labels[i] for i in test_idx], dtype=torch.long)

print(f"✅ Train: {len(train_data)} samples")
print(f"✅ Val:   {len(val_data)} samples")
print(f"✅ Test:  {len(test_data)} samples")


# ============================================================================
# 6. Define GNN Model
# ============================================================================
print("\n🧠 Defining GNN architecture...")


class BrushPredictionGNN(nn.Module):
    """
    Graph Convolutional Network for predicting which cow is being brushed.

    Architecture:
    - 2 GCN layers for message passing
    - Global mean pooling to aggregate node features
    - 2 fully connected layers for classification
    """

    def __init__(self, num_node_features, hidden_dim, num_classes, dropout=0.3):
        super(BrushPredictionGNN, self).__init__()

        # GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # GCN layers with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling (if batch is provided, otherwise use mean)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        return x


# Model parameters
num_node_features = num_cows  # One-hot encoding dimension
hidden_dim = 128
num_classes = num_cows  # Predict which cow is being brushed

model = BrushPredictionGNN(
    num_node_features=num_node_features,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    dropout=0.3,
)

print(f"✅ Model created:")
print(f"   Input features: {num_node_features}")
print(f"   Hidden dimension: {hidden_dim}")
print(f"   Output classes: {num_classes}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")


# ============================================================================
# 7. Training Setup
# ============================================================================
print("\n⚙️  Setting up training...")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"   Device: {device}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True
)


# Training function
def train_epoch(model, data_list, labels, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, label in zip(data_list, labels):
        data = data.to(device)
        label = label.unsqueeze(0).to(device)

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += 1

    return total_loss / total, correct / total


# Evaluation function
def evaluate(model, data_list, labels, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for data, label in zip(data_list, labels):
            data = data.to(device)
            label = label.unsqueeze(0).to(device)

            out = model(data.x, data.edge_index)
            loss = criterion(out, label)

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += 1
            predictions.append(pred.cpu().item())

    return total_loss / total, correct / total, predictions


# ============================================================================
# 8. Train Model
# ============================================================================
print("\n🚀 Training model...")

num_epochs = 100
best_val_acc = 0
patience = 20
patience_counter = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(
        model, train_data, train_labels, optimizer, criterion, device
    )

    # Validate
    val_loss, val_acc, _ = evaluate(model, val_data, val_labels, criterion, device)

    # Update scheduler
    scheduler.step(val_loss)

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ Training completed!")
print(f"   Best validation accuracy: {best_val_acc:.4f}")


# ============================================================================
# 9. Evaluate on Test Set
# ============================================================================
print("\n📊 Evaluating on test set...")

# Load best model
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))

test_loss, test_acc, test_preds = evaluate(
    model, test_data, test_labels, criterion, device
)

print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"   Test Loss: {test_loss:.4f}")

# Detailed classification report
test_labels_np = test_labels.numpy()
print("\n📋 Classification Report (Top 10 classes by frequency):")
print("=" * 80)

# Get top 10 most common classes in test set
top_classes_idx = [idx for idx, _ in Counter(test_labels_np).most_common(10)]
top_class_names = [idx_to_cow[idx] for idx in top_classes_idx]

# Filter predictions and labels for top classes
mask = np.isin(test_labels_np, top_classes_idx)
filtered_labels = test_labels_np[mask]
filtered_preds = np.array([test_preds[i] for i in range(len(test_preds)) if mask[i]])

print(
    classification_report(
        filtered_labels,
        filtered_preds,
        labels=top_classes_idx,
        target_names=top_class_names,
        zero_division=0,
    )
)


# ============================================================================
# 10. Visualizations
# ============================================================================
print("\n📈 Generating visualizations...")

# Plot 1: Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, label="Train Loss", linewidth=2)
axes[0].plot(val_losses, label="Val Loss", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Loss", fontsize=12, fontweight="bold")
axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(train_accs, label="Train Accuracy", linewidth=2)
axes[1].plot(val_accs, label="Val Accuracy", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Accuracy", fontsize=12, fontweight="bold")
axes[1].set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Confusion matrix (for top classes)
cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes_idx)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=top_class_names,
    yticklabels=top_class_names,
    cbar_kws={"label": "Count"},
)
plt.xlabel("Predicted Cow", fontsize=12, fontweight="bold")
plt.ylabel("True Cow", fontsize=12, fontweight="bold")
plt.title("Confusion Matrix (Top 10 Most Brushed Cows)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"✅ Visualizations saved to {OUTPUT_DIR}")


# ============================================================================
# 11. Save Results
# ============================================================================
print("\n💾 Saving results...")

results = {
    "model_architecture": {
        "num_node_features": num_node_features,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
    },
    "training": {
        "num_epochs": len(train_losses),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
    },
    "data": {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "num_cows": num_cows,
    },
    "cow_mapping": {"cow_to_idx": cow_to_idx, "idx_to_cow": idx_to_cow},
}

with open(OUTPUT_DIR / "training_results.pkl", "wb") as f:
    pickle.dump(results, f)

# Save metrics as CSV
metrics_df = pd.DataFrame(
    {
        "epoch": range(len(train_losses)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    }
)
metrics_df.to_csv(OUTPUT_DIR / "training_metrics.csv", index=False)

print(f"✅ Results saved to {OUTPUT_DIR}")

print("\n" + "=" * 80)
print("🎉 GNN Training Complete!")
print("=" * 80)
print(f"📊 Final Test Accuracy: {test_acc:.4f}")
print(f"💾 Model saved to: {OUTPUT_DIR / 'best_model.pt'}")
print(f"📈 Visualizations saved to: {OUTPUT_DIR}")
print("=" * 80)
