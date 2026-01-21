GNN brush predictor

This folder contains `gnn_brush_predictor.py`, a script that trains a small Graph Convolutional
Network (GCN) to predict which cow is currently being brushed given the brush-proximity
NetworkX graphs.

Quick notes
- Input: a pickle file containing the brush-focused NetworkX graphs. Example file in this
  repository: `../outputs/brush_proximity/brush_proximity_graphs_20251112_111121.pkl`.
- The script determines the ground-truth brushed cow per snapshot by selecting the brush's
  neighbor with the highest RSSI value on the brush→cow edge. RSSI is only used to compute
  labels — it is NOT used as features in the GNN.
- Node features: one-hot encoding of cow IDs (global mapping saved to `cow2idx.json`).
- Task: node-level binary classification (per-node: brushed or not). During eval we compute a
  per-graph accuracy by checking whether the top-scoring node is the true brushed node.

Run example

```bash
python Code/gnn_brush_predictor.py \
  --pkl outputs/brush_proximity/brush_proximity_graphs_20251112_111121.pkl \
  --brush-id 366b \
  --epochs 30 \
  --batch-size 16 \
  --out-dir outputs/brush_proximity/gnn_out
```

Dependencies
- PyTorch (matching your CUDA / CPU version)
- PyTorch Geometric (requires matching torch; see https://pytorch-geometric.readthedocs.io/)

Installation hint (CPU-only example)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
# For proper torch-geometric wheel selection, follow the official installation guide
```

If you want, I can try running a quick smoke test in this environment (it may require installing
PyG and CUDA-aware wheels) — say the word and I'll attempt it and report results.
