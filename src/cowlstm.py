#!/usr/bin/env python3
"""
CowLSTM: LSTM-based Cow Embeddings (Simpler Alternative to CowBERT)

This is a lightweight alternative to CowBERT using LSTM instead of Transformer.
Keeps the same data preparation, masking strategy, and training loop.
"""

import argparse
import json
import pickle
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import data extraction from cowbert
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cowbert import TopKProximitySequenceExtractor, CowBERTDataset, calculate_accuracy, calculate_per_cow_accuracy


class CowLSTMModel(nn.Module):
    """LSTM-based model for Cow Embeddings (simpler than CowBERT)."""
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 64, 
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings (same as CowBERT)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM Encoder
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Bidirectional to match BERT's bidirectionality
        )
        
        # MLM Head (same as CowBERT)
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.mlm_head[0].bias.data.zero_()
        self.mlm_head[0].weight.data.uniform_(-initrange, initrange)
        self.mlm_head[4].bias.data.zero_()
        self.mlm_head[4].weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: [batch_size, seq_len]
            src_mask: Not used, kept for compatibility with CowBERT
        Returns:
            output: [batch_size, seq_len, vocab_size]
        """
        # Embed
        embedded = self.embedding(src)  # [batch, seq_len, embed_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim*2]
        
        # MLM prediction
        prediction = self.mlm_head(lstm_out)  # [batch, seq_len, vocab_size]
        
        return prediction

    def get_cow_embeddings(self):
        """Return the learned static embeddings (input layer)."""
        # We exclude special tokens (last 4)
        return self.embedding.weight.data[:-4].cpu().numpy()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for input_ids, labels in pbar:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_ids)
        
        # Flatten for loss computation
        output_flat = output.view(-1, output.size(-1))
        labels_flat = labels.view(-1)
        
        # Loss and accuracy
        loss = criterion(output_flat, labels_flat)
        acc = calculate_accuracy(output_flat, labels_flat)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    return total_loss / len(dataloader), total_acc / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc='Validating'):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            output = model(input_ids)
            output_flat = output.view(-1, output.size(-1))
            labels_flat = labels.view(-1)
            
            loss = criterion(output_flat, labels_flat)
            acc = calculate_accuracy(output_flat, labels_flat)
            
            total_loss += loss.item()
            total_acc += acc
    
    return total_loss / len(dataloader), total_acc / len(dataloader)


def train_model(model, train_dataloader, val_dataloader, device, epochs=20, lr=0.001):
    """Train the model with validation."""
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0
    best_epoch = 0
    best_model_state = None
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_dataloader, criterion, device)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            print(f"   ⭐ New best model! Val Acc: {best_val_acc:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Restored best model from epoch {best_epoch+1} (Val Acc: {best_val_acc:.4f})")
    
    return {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc
    }


def save_results(model, history, per_cow_results, output_dir, metadata):
    """Save model, embeddings, and results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save model
    torch.save(model.state_dict(), output_path / 'model.pt')
    print(f"✅ Saved model to {output_path / 'model.pt'}")
    
    # Save embeddings
    embeddings = model.get_cow_embeddings()
    np.save(output_path / 'embeddings.npy', embeddings)
    print(f"✅ Saved embeddings to {output_path / 'embeddings.npy'}")
    
    # Save metadata
    embedding_metadata = {
        'embedding_dim': model.embed_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'epochs': len(history['train_loss']),
        'best_epoch': history['best_epoch'],
        'best_val_acc': history['best_val_acc'],
        'model': 'CowLSTM',
        'num_cows': len(embeddings),
        'cow_ids': metadata['cow_vocabulary']
    }
    
    with open(output_path / 'embedding_metadata.json', 'w') as f:
        json.dump(embedding_metadata, f, indent=2)
    print(f"✅ Saved metadata to {output_path / 'embedding_metadata.json'}")
    
    # Save training history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Saved training history to {output_path / 'training_history.json'}")
    
    # Save per-cow accuracy
    if per_cow_results:
        per_cow_df_data = []
        for cow_id, stats in per_cow_results.items():
            per_cow_df_data.append({
                'cow_id': cow_id,
                'accuracy': stats['accuracy'],
                'correct': stats['correct'],
                'total': stats['total']
            })
        
        import pandas as pd
        df = pd.DataFrame(per_cow_df_data)
        df = df.sort_values('accuracy', ascending=False)
        df.to_csv(output_path / 'per_cow_accuracy.csv', index=False)
        print(f"✅ Saved per-cow accuracy to {output_path / 'per_cow_accuracy.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Train CowLSTM (LSTM Alternative to CowBERT)")
    parser.add_argument('--input', type=str, required=True, help='Path to temporal graphs pickle')
    parser.add_argument('--output-dir', type=str, default='cowlstm_embeddings', help='Output directory')
    parser.add_argument('--embed-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--top-k', type=int, default=1, help="Number of closest neighbors to consider")
    
    args = parser.parse_args()
    
    print("="*70)
    print("COWLSTM: LSTM-BASED COW EMBEDDINGS")
    print("="*70)
    print(f"\nModel Configuration:")
    print(f"   Embedding dim: {args.embed_dim}")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Num layers: {args.num_layers}")
    print(f"   Bidirectional: True")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Epochs: {args.epochs}")
    
    # 1. Load Data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    with open(args.input, 'rb') as f:
        temporal_graphs = pickle.load(f)
    
    print(f"Loaded {len(temporal_graphs)} temporal snapshots")
    
    # Define excluded sensors
    excluded_sensors = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
    
    # Extract sequences
    extractor = TopKProximitySequenceExtractor(
        temporal_graphs, 
        top_k=args.top_k,
        excluded_sensors=excluded_sensors
    )
    extractor.build_vocabulary()
    sequences = extractor.extract_sequences()
    
    cow_to_id = extractor.cow_to_id
    id_to_cow = extractor.id_to_cow
    
    # 2. Create Datasets with Temporal Split
    print("\n" + "="*70)
    print("PREPARING DATASETS")
    print("="*70)
    print("Splitting data temporally (80/20)...")
    
    train_sequences = {}
    val_sequences = {}
    split_ratio = 0.8
    
    for cow, seqs in sequences.items():
        split_idx = int(len(seqs) * split_ratio)
        train_sequences[cow] = seqs[:split_idx]
        val_sequences[cow] = seqs[split_idx:]
    
    train_dataset = CowBERTDataset(train_sequences, cow_to_id, max_len=128)
    val_dataset = CowBERTDataset(val_sequences, cow_to_id, max_len=128)
    
    print(f"   Train sequences: {len(train_dataset):,}")
    print(f"   Val sequences: {len(val_dataset):,}")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 3. Create Model
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CowLSTMModel(
        vocab_size=train_dataset.full_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"   Embedding: {model.embedding.weight.numel():,}")
    print(f"   LSTM: {sum(p.numel() for n, p in model.named_parameters() if 'lstm' in n):,}")
    print(f"   MLM Head: {sum(p.numel() for n, p in model.named_parameters() if 'mlm_head' in n):,}")
    
    # 4. Train
    history = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        device,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # 5. Evaluate per-cow accuracy
    print("\n" + "="*70)
    print("PER-COW EVALUATION")
    print("="*70)
    
    per_cow_results = calculate_per_cow_accuracy(model, val_dataloader, device, id_to_cow)
    
    # Print top and bottom performers
    sorted_results = sorted(per_cow_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    print(f"\nTop 5 cows (highest accuracy):")
    for cow_id, stats in sorted_results[:5]:
        print(f"   {cow_id}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    print(f"\nBottom 5 cows (lowest accuracy):")
    for cow_id, stats in sorted_results[-5:]:
        print(f"   {cow_id}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    # 6. Save Results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    metadata = {
        'cow_vocabulary': extractor.cow_vocabulary,
        'cow_to_id': cow_to_id,
        'id_to_cow': id_to_cow,
        'num_cows': len(extractor.cow_vocabulary)
    }
    
    save_results(model, history, per_cow_results, args.output_dir, metadata)
    
    # 7. Final Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Best Validation Accuracy: {history['best_val_acc']:.4f}")
    print(f"   Best Epoch: {history['best_epoch']+1}/{args.epochs}")
    print(f"   Total Parameters: {total_params:,}")
    print(f"\n💾 Saved to: {args.output_dir}")
    
    # Compare to CowBERT
    print("\n" + "="*70)
    print("COMPARISON TO COWBERT")
    print("="*70)
    print(f"CowLSTM Parameters:  {total_params:,}")
    print(f"CowBERT Parameters:  ~424,000")
    print(f"Reduction:           {((424000 - total_params) / 424000 * 100):.1f}%")
    print(f"\nCowLSTM Val Acc:     {history['best_val_acc']:.4f}")
    print(f"CowBERT Val Acc:     ~0.73 (reported)")
    print(f"Difference:          {(history['best_val_acc'] - 0.73):.4f}")
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
