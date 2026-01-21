#!/usr/bin/env python3
"""
CowContrastive: Contrastive Learning for Cow Embeddings (SimCLR-style)

This module implements a Contrastive Learning approach to learn cow embeddings.
Instead of predicting masked tokens (BERT), it learns to maximize the similarity
between two augmented views of the same sequence, while minimizing similarity
with other sequences.

Approach:
1. Extract sequences of cow interactions (trajectory).
2. For each sequence, generate two augmented views (random masking/cropping).
3. Encode both views using a Transformer Encoder.
4. Project embeddings to a latent space.
5. Minimize InfoNCE loss (maximize agreement between views of same cow).

Usage:
    python src/cow_contrastive.py \
        --input network_sequence/network_sequence_rssi-68_*.pkl \
        --output-dir cow_contrastive_out \
        --epochs 20
"""

import argparse
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Reuse extractor from CowBERT
from cowbert import TopKProximitySequenceExtractor, PositionalEncoding

class ContrastiveDataset(Dataset):
    def __init__(self, sequences: List[List[int]], vocab_size: int, max_len: int = 128):
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Special tokens
        self.PAD_TOKEN = vocab_size
        self.MASK_TOKEN = vocab_size + 1
        self.CLS_TOKEN = vocab_size + 2
        
    def __len__(self):
        return len(self.sequences)
    
    def augment(self, seq: List[int]) -> torch.Tensor:
        """Apply random augmentations to a sequence."""
        # 1. Random Crop (if long enough)
        if len(seq) > 10 and random.random() < 0.5:
            start = random.randint(0, len(seq) // 4)
            end = random.randint(len(seq) - len(seq) // 4, len(seq))
            seq = seq[start:end]
            
        # 2. Random Masking (15%)
        aug_seq = list(seq)
        for i in range(len(aug_seq)):
            if random.random() < 0.15:
                aug_seq[i] = self.MASK_TOKEN
                
        # Pad/Truncate
        if len(aug_seq) > self.max_len - 1: # -1 for CLS
            aug_seq = aug_seq[:self.max_len - 1]
            
        # Add CLS
        final_seq = [self.CLS_TOKEN] + aug_seq
        
        # Pad
        padding_len = self.max_len - len(final_seq)
        if padding_len > 0:
            final_seq = final_seq + [self.PAD_TOKEN] * padding_len
            
        return torch.tensor(final_seq, dtype=torch.long)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Generate two views
        view1 = self.augment(seq)
        view2 = self.augment(seq)
        
        return view1, view2

class CowEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings (Vocab + Special Tokens)
        self.embedding = nn.Embedding(vocab_size + 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Projection Head (for Contrastive Loss)
        # Map d_model -> d_model -> 64
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64) 
        )

    def forward(self, x):
        # x: (Batch, Seq_Len)
        mask = (x == (self.embedding.num_embeddings - 3)) # PAD token is vocab_size (index -3 from end?)
        # Actually let's pass padding mask explicitly if needed, or rely on index
        # PAD_TOKEN index is passed in dataset.
        
        x_emb = self.embedding(x) * math.sqrt(self.d_model)
        x_emb = x_emb.transpose(0, 1) # (Seq, Batch, D) for PosEncoder
        x_emb = self.pos_encoder(x_emb)
        x_emb = x_emb.transpose(0, 1) # (Batch, Seq, D)
        
        # Transformer
        # We need a padding mask. 
        # In dataset: PAD_TOKEN = vocab_size.
        # Embedding size = vocab_size + 3.
        # So PAD index is vocab_size.
        pad_idx = self.embedding.num_embeddings - 3
        src_key_padding_mask = (x == pad_idx)
        
        features = self.transformer(x_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Pooling: Take CLS token (index 0)
        cls_embedding = features[:, 0, :]
        
        # Projection
        projected = self.projection_head(cls_embedding)
        
        return cls_embedding, projected

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        # z_i, z_j: (Batch, D)
        batch_size = z_i.shape[0]
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate all representations: [z_i, z_j] -> (2*Batch, D)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix: (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # Positive pairs: (i, i+B) and (i+B, i)
        # We want to maximize sim(z[k], z[k+B])
        
        # Create targets
        # For index k in [0, B-1], target is k+B
        # For index k in [B, 2B-1], target is k-B
        targets = torch.cat([
            torch.arange(batch_size, 2*batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)
        
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

def train_contrastive(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'rb') as f:
        temporal_graphs = pickle.load(f)
        
    # Sort and Split
    temporal_graphs.sort(key=lambda x: x['timestamp'])
    split_idx = int(len(temporal_graphs) * 0.8)
    train_graphs = temporal_graphs[:split_idx]
    val_graphs = temporal_graphs[split_idx:]
    
    excluded_sensors = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
    
    # Extract Sequences
    extractor = TopKProximitySequenceExtractor(train_graphs, top_k=1, excluded_sensors=excluded_sensors)
    extractor.build_vocabulary()
    sequences_dict = extractor.extract_sequences()
    
    # Flatten sequences
    flat_sequences = []
    for cow, seqs in sequences_dict.items():
        # Flatten list of lists
        full_seq = []
        for neighbors in seqs:
            if neighbors:
                # neighbors is a list of cow names. Map to ID.
                # TopK=1, so usually 1 neighbor.
                for n in neighbors:
                    if n in extractor.cow_to_id:
                        full_seq.append(extractor.cow_to_id[n])
        
        # Chunk
        chunk_size = 64
        for i in range(0, len(full_seq), chunk_size):
            chunk = full_seq[i:i+chunk_size]
            if len(chunk) > 10:
                flat_sequences.append(chunk)
                
    print(f"Created {len(flat_sequences)} training sequences.")
    
    # Dataset & Loader
    dataset = ContrastiveDataset(flat_sequences, len(extractor.cow_vocabulary))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Model
    model = CowEncoder(len(extractor.cow_vocabulary)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = NTXentLoss(temperature=0.1)
    
    # Train
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for view1, view2 in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            
            _, proj1 = model(view1)
            _, proj2 = model(view2)
            
            loss = criterion(proj1, proj2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
        
    # Save Embeddings
    # We want the static embeddings (input layer) OR the contextual embeddings?
    # Usually for "Cow Identity", we might want the static embedding matrix.
    # But Contrastive Learning trains the Encoder.
    # The static embedding matrix `model.embedding` is learned.
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save static embeddings
    # Exclude special tokens (last 3)
    static_embeddings = model.embedding.weight.data[:-3].cpu().numpy()
    
    # Save
    np.save(output_dir / "cow_embeddings.npy", static_embeddings)
    
    # Save metadata
    with open(output_dir / "cow_embeddings.pkl", 'wb') as f:
        pickle.dump({
            'embeddings': static_embeddings,
            'id_to_cow': extractor.id_to_cow,
            'cow_to_id': extractor.cow_to_id
        }, f)
        
    print(f"Saved embeddings to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='cow_contrastive_out')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    train_contrastive(args)
