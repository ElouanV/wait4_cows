#!/usr/bin/env python3
"""
Graph-BERT: Graph-based Masked Cow Identification

This module implements a Graph Neural Network (using Transformer with Graph Masking)
to learn cow embeddings from static proximity graphs.

Approach:
1. Treat each timestamp's proximity graph as a sample.
2. Nodes are cows. Edges are proximity contacts.
3. Masking: Randomly mask the identity of a node (replace with [MASK] token).
4. Objective: Predict the true identity of the masked node based on its neighbors.
5. Architecture: Transformer Encoder where Attention is restricted to graph neighbors.

Usage:
    python src/graph_bert.py \
        --input network_sequence/network_sequence_rssi-68_*.pkl \
        --output-dir graph_bert_models \
        --epochs 50
"""

import argparse
import pickle
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple, Set

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Configuration ---
EXCLUDED_SENSORS = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
MASK_TOKEN_ID = -1  # Will be set dynamically based on vocab size
PAD_TOKEN_ID = -2   # Will be set dynamically

class CowGraphDataset(Dataset):
    def __init__(self, pickle_path: str, split: str = 'train', split_ratio: float = 0.8, window_size: int = 1):
        """
        Args:
            pickle_path: Path to the network sequence pickle file.
            split: 'train' or 'val'.
            split_ratio: Ratio of data to use for training (temporal split).
            window_size: Number of consecutive snapshots to aggregate.
        """
        self.window_size = window_size
        print(f"Loading data from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Sort by timestamp to ensure temporal split
        self.data.sort(key=lambda x: x['timestamp'])
        
        # Build Vocabulary
        self.vocab = self._build_vocab()
        self.cow_to_idx = {cow: i for i, cow in enumerate(self.vocab)}
        self.idx_to_cow = {i: cow for i, cow in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        self.mask_token_idx = self.vocab_size
        self.pad_token_idx = self.vocab_size + 1
        
        # Split data
        split_idx = int(len(self.data) * split_ratio)
        if split == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
            
        print(f"Dataset ({split}): {len(self.data)} snapshots (window_size={window_size}).")
        print(f"Vocabulary size: {self.vocab_size} cows.")

    def _build_vocab(self) -> List[str]:
        vocab = set()
        for entry in self.data:
            G = entry['graph']
            for node in G.nodes():
                if node not in EXCLUDED_SENSORS:
                    vocab.add(node)
        return sorted(list(vocab))

    def __len__(self):
        return max(0, len(self.data) - self.window_size + 1)

    def __getitem__(self, idx):
        # Aggregate graphs in the window
        window_entries = self.data[idx : idx + self.window_size]
        
        # We use the union of nodes in the window? 
        # Or just the nodes present in the middle?
        # Let's use the union of all active nodes in the window.
        
        active_nodes_set = set()
        edges = set()
        
        for entry in window_entries:
            G = entry['graph']
            current_nodes = [n for n in G.nodes() if n not in EXCLUDED_SENSORS]
            active_nodes_set.update(current_nodes)
            
            for u, v in G.edges():
                if u not in EXCLUDED_SENSORS and v not in EXCLUDED_SENSORS:
                    # Undirected edges
                    if u > v: u, v = v, u
                    edges.add((u, v))
        
        active_nodes = sorted(list(active_nodes_set))
        
        # If graph is empty or has too few nodes, return None (handled in collate)
        if not active_nodes:
            return None
            
        # Map nodes to indices
        node_indices = [self.cow_to_idx[n] for n in active_nodes]
        
        # Create Adjacency Matrix (local to this subgraph)
        num_nodes = len(active_nodes)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        
        # Map global cow ID to local index 0..num_nodes-1
        node_to_local = {n: i for i, n in enumerate(active_nodes)}
        
        for u, v in edges:
            if u in node_to_local and v in node_to_local:
                i, j = node_to_local[u], node_to_local[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        
        # Add self-loops
        adj.fill_diagonal_(1.0)
        
        return {
            'node_ids': torch.tensor(node_indices, dtype=torch.long),
            'adj': adj,
            'num_nodes': num_nodes
        }

def collate_fn(batch):
    """
    Custom collate to handle variable graph sizes.
    We will pad everything to the max graph size in the batch.
    """
    # Filter None
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    max_nodes = max(b['num_nodes'] for b in batch)
    
    # Prepare batch tensors
    batch_size = len(batch)
    # Input IDs: (Batch, Max_Nodes)
    padded_ids = torch.full((batch_size, max_nodes), fill_value=-1, dtype=torch.long) # -1 placeholder
    # Adjacency: (Batch, Max_Nodes, Max_Nodes)
    padded_adj = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.float)
    # Attention Mask: (Batch, Max_Nodes, Max_Nodes) - True where we should NOT attend
    # In PyTorch Transformer: mask is additive (0 for keep, -inf for ignore) or boolean (True for ignore)
    # We will use additive: 0.0 for connected, -inf for not connected/padding
    attn_mask = torch.full((batch_size, max_nodes, max_nodes), fill_value=float('-inf'), dtype=torch.float)
    
    # Padding Mask for Loss (ignore padded positions)
    padding_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool) # True = is padding
    
    for i, item in enumerate(batch):
        n = item['num_nodes']
        
        # Copy IDs
        padded_ids[i, :n] = item['node_ids']
        
        # Copy Adjacency to Attention Mask
        # Where adj is 1, mask is 0. Where adj is 0, mask is -inf.
        # item['adj'] is 1s and 0s.
        # We want: 1 -> 0, 0 -> -inf
        current_adj = item['adj']
        # Create mask for the valid subgraph
        sub_mask = torch.where(current_adj > 0, torch.tensor(0.0), torch.tensor(float('-inf')))
        attn_mask[i, :n, :n] = sub_mask
        
        # Mark valid positions
        padding_mask[i, :n] = False
        
    return {
        'input_ids': padded_ids,
        'attn_mask': attn_mask,
        'padding_mask': padding_mask,
        'batch_size': batch_size
    }

class GraphBERT(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_idx = vocab_size
        self.pad_token_idx = vocab_size + 1
        
        # Embeddings: Cows + MASK + PAD
        self.embedding = nn.Embedding(vocab_size + 2, hidden_dim, padding_idx=self.pad_token_idx)
        
        # Transformer Encoder
        # batch_first=True is important
        self.num_heads = num_heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction Head
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, attn_mask):
        """
        input_ids: (Batch, Max_Nodes)
        attn_mask: (Batch, Max_Nodes, Max_Nodes)
        """
        x = self.embedding(input_ids) # (Batch, Max_Nodes, Hidden)
        
        # Expand mask for multi-head attention: (Batch * NumHeads, S, S)
        # attn_mask is (B, S, S). We need (B*H, S, S).
        bsz, seq_len, _ = attn_mask.shape
        # repeat_interleave repeats elements: [M1, M2] -> [M1, M1, M2, M2]
        attn_mask_expanded = attn_mask.repeat_interleave(self.num_heads, dim=0)
        
        x = self.transformer(x, mask=attn_mask_expanded)
        logits = self.fc_out(x)
        return logits

def train_graph_bert(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = CowGraphDataset(args.input, split='train', window_size=args.window_size)
    val_dataset = CowGraphDataset(args.input, split='val', window_size=args.window_size)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = GraphBERT(
        vocab_size=train_dataset.vocab_size,
        hidden_dim=args.embedding_dim,
        num_layers=args.layers,
        num_heads=args.heads
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_masked = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device) # (B, N)
            attn_mask = batch['attn_mask'].to(device) # (B, N, N)
            padding_mask = batch['padding_mask'].to(device) # (B, N)
            
            targets = input_ids.clone()
            masked_input_ids = input_ids.clone()
            
            # Generate mask indices
            batch_size, max_nodes = input_ids.shape
            mask_indices = []
            
            for i in range(batch_size):
                # Find valid indices that have neighbors (degree > 1 because of self-loop)
                # attn_mask[i, j, :] is 0 if connected.
                # We check how many 0s are in the row.
                # Note: attn_mask contains -inf.
                
                # Get connectivity for this graph
                # shape (N, N)
                graph_mask = attn_mask[i] 
                
                # Count connections (where mask == 0)
                degrees = (graph_mask == 0).sum(dim=1) # (N,)
                
                # Valid candidates: not padding AND degree > 1
                # padding_mask[i] is True for padding
                is_node = ~padding_mask[i]
                has_neighbors = degrees > 1
                candidates = torch.where(is_node & has_neighbors)[0]
                
                if len(candidates) > 0:
                    # Pick one random candidate
                    idx_to_mask = candidates[random.randint(0, len(candidates) - 1)].item()
                    mask_indices.append((i, idx_to_mask))
                    masked_input_ids[i, idx_to_mask] = model.mask_token_idx
                else:
                    mask_indices.append(None)
            
            # We only compute loss on the masked tokens
            optimizer.zero_grad()
            logits = model(masked_input_ids, attn_mask) # (B, N, Vocab)
            
            pred_list = []
            target_list = []
            
            for k, mask_pos in enumerate(mask_indices):
                if mask_pos is not None:
                    r, c = mask_pos
                    pred = logits[r, c] # (Vocab)
                    true_id = targets[r, c]
                    
                    pred_list.append(pred)
                    target_list.append(true_id)
            
            if not pred_list:
                continue
                
            pred_tensor = torch.stack(pred_list)
            target_tensor = torch.stack(target_list)
            
            loss = criterion(pred_tensor, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = pred_tensor.argmax(dim=1)
            correct += (preds == target_tensor).sum().item()
            total_masked += len(target_tensor)
            
        avg_loss = total_loss / len(train_loader)
        acc = correct / total_masked if total_masked > 0 else 0
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}")
        
        # Validation
        evaluate(model, val_loader, device, criterion)

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total_masked = 0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            targets = input_ids.clone()
            masked_input_ids = input_ids.clone()
            
            batch_size, max_nodes = input_ids.shape
            mask_indices = []
            
            for i in range(batch_size):
                graph_mask = attn_mask[i]
                degrees = (graph_mask == 0).sum(dim=1)
                is_node = ~padding_mask[i]
                has_neighbors = degrees > 1
                candidates = torch.where(is_node & has_neighbors)[0]
                
                if len(candidates) > 0:
                    idx_to_mask = candidates[random.randint(0, len(candidates) - 1)].item()
                    mask_indices.append((i, idx_to_mask))
                    masked_input_ids[i, idx_to_mask] = model.mask_token_idx
                else:
                    mask_indices.append(None)
            
            logits = model(masked_input_ids, attn_mask)
            
            pred_list = []
            target_list = []
            
            for k, mask_pos in enumerate(mask_indices):
                if mask_pos is not None:
                    r, c = mask_pos
                    pred = logits[r, c]
                    true_id = targets[r, c]
                    pred_list.append(pred)
                    target_list.append(true_id)
            
            if not pred_list: continue
            
            pred_tensor = torch.stack(pred_list)
            target_tensor = torch.stack(target_list)
            
            loss = criterion(pred_tensor, target_tensor)
            total_loss += loss.item()
            
            preds = pred_tensor.argmax(dim=1)
            correct += (preds == target_tensor).sum().item()
            total_masked += len(target_tensor)
            
    avg_loss = total_loss / len(loader)
    acc = correct / total_masked if total_masked > 0 else 0
    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to pickle file')
    parser.add_argument('--output-dir', type=str, default='graph_bert_out')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--window-size', type=int, default=5, help='Number of snapshots to aggregate')
    
    args = parser.parse_args()
    train_graph_bert(args)
