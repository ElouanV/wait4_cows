#!/usr/bin/env python3
"""
Temporal Graph Network (TGN) for Cow Interaction Prediction

This module implements a simplified TGN to learn dynamic cow embeddings.
It processes a stream of timestamped interactions (u, v, t) and updates
node memories using a Recurrent Neural Network (GRU).

Architecture:
1. Memory Module: Stores state s_i(t) for each node i.
2. Message Function: Computes messages from interactions.
3. Memory Updater: Updates memory s_i(t) -> s_i(t') using GRU.
4. Embedding Module: Computes temporal embedding z_i(t) from memory and graph.
5. Link Predictor: Predicts probability of edge (u, v) at time t.

Usage:
    python src/tgn.py \
        --input network_sequence/network_sequence_rssi-68_*.pkl \
        --output-dir tgn_out \
        --epochs 10
"""

import argparse
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# --- Configuration ---
EXCLUDED_SENSORS = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}

class MemoryModule(nn.Module):
    """Stores and updates node memory."""
    def __init__(self, num_nodes, memory_dim, message_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        
        # Node Memory: (N, Memory_Dim)
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim), requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)
        
        # Message Function (MLP)
        # Input: [Memory_u, Memory_v, Time_Delta] -> Message
        self.msg_function = nn.Sequential(
            nn.Linear(memory_dim * 2 + 1, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # Memory Updater (GRU)
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def get_memory(self, node_idxs):
        return self.memory[node_idxs]

    def update_memory(self, node_idxs, messages):
        """Update memory for specific nodes using GRU."""
        # node_idxs: (B,)
        # messages: (B, Msg_Dim)
        
        current_memory = self.memory[node_idxs]
        updated_memory = self.gru(messages, current_memory)
        
        # Detach to prevent backprop through entire history (TGN trick)
        # But we need gradients for the current step.
        # In standard TGN, memory is updated *after* the batch prediction.
        
        with torch.no_grad():
            self.memory[node_idxs] = updated_memory
            
        return updated_memory

    def reset_memory(self):
        self.memory.fill_(0)
        self.last_update.fill_(0)

class TemporalEmbeddingModule(nn.Module):
    """Computes temporal embeddings from memory."""
    def __init__(self, memory_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(memory_dim, embedding_dim)
        
    def forward(self, memory):
        return self.linear(memory)

class LinkPredictor(nn.Module):
    """Predicts edge probability from two node embeddings."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 1)
        
    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=1)
        x = F.relu(self.fc1(x))
        score = self.fc2(x)
        return score

class TGN(nn.Module):
    def __init__(self, num_nodes, memory_dim=128, embedding_dim=128):
        super().__init__()
        self.memory_module = MemoryModule(num_nodes, memory_dim, memory_dim)
        self.embedding_module = TemporalEmbeddingModule(memory_dim, embedding_dim)
        self.link_predictor = LinkPredictor(embedding_dim)
        
    def forward(self, src, dst, t, neg_dst):
        """
        Process a batch of interactions.
        1. Compute embeddings using CURRENT memory (before update).
        2. Predict links (pos and neg).
        3. Compute messages and update memory for NEXT step.
        """
        # 1. Get current embeddings
        mem_src = self.memory_module.get_memory(src)
        mem_dst = self.memory_module.get_memory(dst)
        mem_neg = self.memory_module.get_memory(neg_dst)
        
        z_src = self.embedding_module(mem_src)
        z_dst = self.embedding_module(mem_dst)
        z_neg = self.embedding_module(mem_neg)
        
        # 2. Predict
        pos_score = self.link_predictor(z_src, z_dst)
        neg_score = self.link_predictor(z_src, z_neg)
        
        # 3. Update Memory (for future)
        # Message: [Mem_u, Mem_v, Delta_t]
        # We simplify and ignore Delta_t for now, or use 0
        delta_t = torch.zeros(len(src), 1, device=src.device)
        
        # Message for Source: Interaction with Dst
        msg_src = self.memory_module.msg_function(torch.cat([mem_src, mem_dst, delta_t], dim=1))
        # Message for Dest: Interaction with Src
        msg_dst = self.memory_module.msg_function(torch.cat([mem_dst, mem_src, delta_t], dim=1))
        
        # Update
        self.memory_module.update_memory(src, msg_src)
        self.memory_module.update_memory(dst, msg_dst)
        
        return pos_score, neg_score

def prepare_data(pickle_path):
    print(f"Loading data from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    data.sort(key=lambda x: x['timestamp'])
    
    # Build Vocab
    vocab = set()
    for entry in data:
        G = entry['graph']
        for n in G.nodes():
            if n not in EXCLUDED_SENSORS:
                vocab.add(n)
    
    vocab = sorted(list(vocab))
    cow_to_id = {c: i for i, c in enumerate(vocab)}
    
    # Extract Events: (u, v, t)
    events = []
    start_time = data[0]['timestamp'].timestamp()
    
    for entry in data:
        t = entry['timestamp'].timestamp() - start_time
        G = entry['graph']
        
        # We treat undirected edges as two directed events? 
        # Or just one? TGN usually handles directed.
        # Let's do undirected: u->v and v->u
        processed_edges = set()
        
        for u, v in G.edges():
            if u in EXCLUDED_SENSORS or v in EXCLUDED_SENSORS:
                continue
            
            # Canonical order to avoid duplicates if graph is undirected
            if u > v: u, v = v, u
            if (u, v) in processed_edges: continue
            processed_edges.add((u, v))
            
            u_idx = cow_to_id[u]
            v_idx = cow_to_id[v]
            
            events.append((u_idx, v_idx, t))
            
    print(f"Extracted {len(events)} interaction events.")
    return events, cow_to_id

def train_tgn(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    events, cow_to_id = prepare_data(args.input)
    num_nodes = len(cow_to_id)
    
    # Split
    split_idx = int(len(events) * 0.8)
    train_events = events[:split_idx]
    val_events = events[split_idx:]
    
    model = TGN(num_nodes, memory_dim=args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    batch_size = args.batch_size
    
    for epoch in range(args.epochs):
        model.memory_module.reset_memory()
        model.train()
        
        total_loss = 0
        
        # Train Loop
        for i in range(0, len(train_events), batch_size):
            batch = train_events[i:i+batch_size]
            
            src = torch.tensor([e[0] for e in batch], dtype=torch.long, device=device)
            dst = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
            t = torch.tensor([e[2] for e in batch], dtype=torch.float, device=device)
            
            # Negative Sampling
            neg_dst = torch.randint(0, num_nodes, (len(batch),), device=device)
            
            optimizer.zero_grad()
            pos_score, neg_score = model(src, dst, t, neg_dst)
            
            # Loss
            # Pos -> 1, Neg -> 0
            loss = criterion(pos_score, torch.ones_like(pos_score)) + \
                   criterion(neg_score, torch.zeros_like(neg_score))
                   
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / (len(train_events)/batch_size):.4f}")
        
        # Validation
        model.eval()
        val_aps = []
        with torch.no_grad():
            for i in range(0, len(val_events), batch_size):
                batch = val_events[i:i+batch_size]
                src = torch.tensor([e[0] for e in batch], dtype=torch.long, device=device)
                dst = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
                t = torch.tensor([e[2] for e in batch], dtype=torch.float, device=device)
                neg_dst = torch.randint(0, num_nodes, (len(batch),), device=device)
                
                pos_score, neg_score = model(src, dst, t, neg_dst)
                
                y_true = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).cpu().numpy()
                y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
                
                val_aps.append(average_precision_score(y_true, y_score))
                
        print(f"Val AP: {np.mean(val_aps):.4f}")
        
    # Save Embeddings (Final Memory State)
    final_memory = model.memory_module.memory.cpu().numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "tgn_embeddings.npy", final_memory)
    with open(output_dir / "tgn_embeddings.pkl", 'wb') as f:
        pickle.dump({
            'embeddings': final_memory,
            'cow_to_id': cow_to_id,
            'id_to_cow': {v: k for k, v in cow_to_id.items()}
        }, f)
    print(f"Saved TGN embeddings to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='tgn_out')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    args = parser.parse_args()
    train_tgn(args)
