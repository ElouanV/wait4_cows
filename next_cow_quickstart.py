#!/usr/bin/env python3
"""
Quick start example for Next Cow Prediction task.

This script demonstrates:
1. Loading the dataset
2. Creating a PyTorch dataset
3. Building a simple model
4. Training loop

Usage:
    python next_cow_quickstart.py --dataset-dir next_cow_data
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class NextCowDataset(Dataset):
    """PyTorch Dataset for Next Cow Prediction."""
    
    def __init__(self, sequences):
        """
        Args:
            sequences: List of sequence dicts from pickle file
        """
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        return {
            'source_cow_id': torch.tensor(seq['source_cow_id'], dtype=torch.long),
            'input_sequence': torch.tensor(seq['input_sequence_ids'], dtype=torch.long),
            'target': torch.tensor(seq['target_id'], dtype=torch.long)
        }


class NextCowLSTM(nn.Module):
    """Simple LSTM-based model for next cow prediction."""
    
    def __init__(self, num_cows, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Cow embeddings
        self.embedding = nn.Embedding(num_cows, embedding_dim)
        
        # LSTM to process sequence
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_cows)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, source_cow_id, input_sequence):
        """
        Args:
            source_cow_id: (batch_size,) - source cow IDs
            input_sequence: (batch_size, seq_len) - sequence of cow IDs
        
        Returns:
            logits: (batch_size, num_cows) - prediction logits
        """
        # Embed input sequence
        embedded = self.embedding(input_sequence)  # (batch_size, seq_len, embedding_dim)
        
        # Process with LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        
        # Predict next cow
        logits = self.fc(last_hidden)  # (batch_size, num_cows)
        
        return logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        source_cow_ids = batch['source_cow_id'].to(device)
        input_sequences = batch['input_sequence'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(source_cow_ids, input_sequences)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device, k=5):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    top_k_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            source_cow_ids = batch['source_cow_id'].to(device)
            input_sequences = batch['input_sequence'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            logits = model(source_cow_ids, input_sequences)
            loss = criterion(logits, targets)
            
            # Metrics
            total_loss += loss.item()
            
            # Top-1 accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            
            # Top-K accuracy
            _, top_k_preds = logits.topk(k, dim=1)
            top_k_correct += top_k_preds.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
            
            total += targets.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        f'top_{k}_accuracy': top_k_correct / total
    }


def main():
    parser = argparse.ArgumentParser(
        description='Quick start example for Next Cow Prediction'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='next_cow_data',
        help='Dataset directory (default: next_cow_data)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs (default: 10)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='LSTM hidden dimension (default: 256)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEXT COW PREDICTION - QUICK START")
    print("=" * 70)
    
    # Check if dataset exists
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"\n❌ Dataset not found at {dataset_dir}")
        print(f"\n💡 Generate the dataset first:")
        print(f"   python generate_next_cow_dataset.py --output-dir {dataset_dir}")
        return
    
    # Load metadata
    print(f"\nLoading dataset from {dataset_dir}...")
    with open(dataset_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    vocab_size = metadata['num_cows']
    print(f"   Vocabulary size: {vocab_size} cows")
    
    # Load sequences
    with open(dataset_dir / 'train_sequences.pkl', 'rb') as f:
        train_sequences = pickle.load(f)
    
    with open(dataset_dir / 'val_sequences.pkl', 'rb') as f:
        val_sequences = pickle.load(f)
    
    print(f"   Train sequences: {len(train_sequences):,}")
    print(f"   Val sequences:   {len(val_sequences):,}")
    
    # Create datasets and dataloaders
    train_dataset = NextCowDataset(train_sequences)
    val_dataset = NextCowDataset(val_sequences)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = NextCowLSTM(
        num_cows=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\n   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"   Val Top-5:  {val_metrics['top_5_accuracy']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), dataset_dir / 'best_model.pt')
            print(f"   ✅ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {dataset_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
