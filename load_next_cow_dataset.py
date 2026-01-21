#!/usr/bin/env python3
"""
Helper script to load and explore the Next Cow Prediction dataset.

Usage:
    python load_next_cow_dataset.py --dataset-dir next_cow_data
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(dataset_dir: Path):
    """Load all components of the dataset."""
    print(f"Loading dataset from {dataset_dir}...")
    
    # Load metadata
    with open(dataset_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load config
    with open(dataset_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Load sequences
    with open(dataset_dir / 'train_sequences.pkl', 'rb') as f:
        train_seqs = pickle.load(f)
    
    with open(dataset_dir / 'val_sequences.pkl', 'rb') as f:
        val_seqs = pickle.load(f)
    
    with open(dataset_dir / 'test_sequences.pkl', 'rb') as f:
        test_seqs = pickle.load(f)
    
    print(f"✅ Loaded dataset")
    print(f"   Train: {len(train_seqs):,} sequences")
    print(f"   Val:   {len(val_seqs):,} sequences")
    print(f"   Test:  {len(test_seqs):,} sequences")
    print(f"   Vocabulary: {metadata['num_cows']} unique cows")
    
    return {
        'train': train_seqs,
        'val': val_seqs,
        'test': test_seqs,
        'metadata': metadata,
        'config': config
    }


def print_dataset_info(dataset):
    """Print detailed information about the dataset."""
    metadata = dataset['metadata']
    config = dataset['config']
    
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    
    print("\n📋 Configuration:")
    print(f"   Created: {config['created_at']}")
    print(f"   Sequence Length: {config['sequence_length']} (input: {config['sequence_length']-1}, target: 1)")
    print(f"   Time Window: {config['window_seconds']}s")
    print(f"   Min RSSI: {config['min_rssi']}")
    print(f"   Hours: {config['start_hours']}-{config['end_hours']}")
    print(f"   Excluded Sensors: {', '.join(config['excluded_sensors'])}")
    
    print("\n📊 Dataset Statistics:")
    print(f"   Total Sequences: {config['num_sequences']['total']:,}")
    print(f"   Train: {config['num_sequences']['train']:,} ({config['train_ratio']*100:.1f}%)")
    print(f"   Val:   {config['num_sequences']['val']:,} ({config['val_ratio']*100:.1f}%)")
    print(f"   Test:  {config['num_sequences']['test']:,} ({config['test_ratio']*100:.1f}%)")
    
    print("\n🐄 Vocabulary:")
    print(f"   Unique Cows: {metadata['num_cows']}")
    print(f"   Cow IDs: {', '.join(metadata['vocabulary'][:10])}...")
    
    # Analyze distribution
    train_seqs = dataset['train']
    
    # Source cow distribution
    source_counts = Counter([seq['source_cow'] for seq in train_seqs])
    print("\n📈 Training Set Source Cow Distribution:")
    print(f"   Most common: {source_counts.most_common(5)}")
    print(f"   Least common: {source_counts.most_common()[-5:]}")
    
    # Target distribution
    target_counts = Counter([seq['target'] for seq in train_seqs])
    print("\n🎯 Training Set Target Distribution:")
    print(f"   Most common: {target_counts.most_common(5)}")
    print(f"   Least common: {target_counts.most_common()[-5:]}")


def print_sample_sequences(dataset, n_samples=5):
    """Print sample sequences from the dataset."""
    train_seqs = dataset['train']
    metadata = dataset['metadata']
    
    print("\n" + "=" * 70)
    print(f"SAMPLE SEQUENCES (showing {n_samples} random examples)")
    print("=" * 70)
    
    samples = np.random.choice(train_seqs, min(n_samples, len(train_seqs)), replace=False)
    
    for i, seq in enumerate(samples, 1):
        print(f"\n📝 Example {i}:")
        print(f"   Source Cow: {seq['source_cow']} (ID: {seq['source_cow_id']})")
        print(f"   Input Sequence: {' → '.join(seq['input_sequence'])}")
        print(f"   Input IDs: {seq['input_sequence_ids']}")
        print(f"   Target: {seq['target']} (ID: {seq['target_id']})")
        print(f"   First Timestamp: {seq['timestamps'][0]}")
        print(f"   Last Timestamp: {seq['timestamps'][-1]}")


def create_pytorch_dataset_example():
    """Print example code for creating a PyTorch dataset."""
    print("\n" + "=" * 70)
    print("PYTORCH DATASET EXAMPLE")
    print("=" * 70)
    
    example_code = """
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class NextCowDataset(Dataset):
    \"\"\"PyTorch Dataset for Next Cow Prediction.\"\"\"
    
    def __init__(self, sequences):
        \"\"\"
        Args:
            sequences: List of sequence dicts from pickle file
        \"\"\"
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Convert to tensors
        source_cow_id = torch.tensor(seq['source_cow_id'], dtype=torch.long)
        input_ids = torch.tensor(seq['input_sequence_ids'], dtype=torch.long)
        target_id = torch.tensor(seq['target_id'], dtype=torch.long)
        
        return {
            'source_cow_id': source_cow_id,
            'input_sequence': input_ids,
            'target': target_id
        }

# Load dataset
with open('next_cow_data/train_sequences.pkl', 'rb') as f:
    train_sequences = pickle.load(f)

# Create PyTorch dataset
train_dataset = NextCowDataset(train_sequences)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Example: iterate through batches
for batch in train_loader:
    source_cow_ids = batch['source_cow_id']  # Shape: (batch_size,)
    input_sequences = batch['input_sequence']  # Shape: (batch_size, seq_len)
    targets = batch['target']  # Shape: (batch_size,)
    
    # Your model training code here
    # outputs = model(source_cow_ids, input_sequences)
    # loss = criterion(outputs, targets)
    break
"""
    print(example_code)


def create_model_architecture_example():
    """Print example model architectures."""
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE EXAMPLES")
    print("=" * 70)
    
    example_code = """
import torch
import torch.nn as nn

# Example 1: Simple LSTM-based model
class NextCowLSTM(nn.Module):
    def __init__(self, num_cows, embedding_dim=128, hidden_dim=256):
        super().__init__()
        
        # Embeddings for cows
        self.cow_embedding = nn.Embedding(num_cows, embedding_dim)
        
        # LSTM to process sequence
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_cows)
    
    def forward(self, source_cow_id, input_sequence):
        # Embed the input sequence
        # input_sequence shape: (batch_size, seq_len)
        embedded = self.cow_embedding(input_sequence)  # (batch_size, seq_len, embedding_dim)
        
        # Process with LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Predict next cow
        logits = self.fc(last_hidden)  # (batch_size, num_cows)
        
        return logits


# Example 2: Transformer-based model
class NextCowTransformer(nn.Module):
    def __init__(self, num_cows, embedding_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        
        # Embeddings
        self.cow_embedding = nn.Embedding(num_cows, embedding_dim)
        self.pos_encoding = nn.Embedding(100, embedding_dim)  # Max seq length 100
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(embedding_dim, num_cows)
    
    def forward(self, source_cow_id, input_sequence):
        batch_size, seq_len = input_sequence.shape
        
        # Embed cows and add positional encoding
        cow_embeds = self.cow_embedding(input_sequence)  # (batch_size, seq_len, embedding_dim)
        positions = torch.arange(seq_len, device=input_sequence.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_encoding(positions)
        
        embedded = cow_embeds + pos_embeds
        
        # Process with transformer
        transformed = self.transformer(embedded)  # (batch_size, seq_len, embedding_dim)
        
        # Use last token for prediction
        last_token = transformed[:, -1, :]  # (batch_size, embedding_dim)
        
        # Predict next cow
        logits = self.fc(last_token)  # (batch_size, num_cows)
        
        return logits


# Example 3: Model with source cow conditioning
class NextCowConditioned(nn.Module):
    def __init__(self, num_cows, embedding_dim=128, hidden_dim=256):
        super().__init__()
        
        # Embeddings
        self.cow_embedding = nn.Embedding(num_cows, embedding_dim)
        
        # LSTM for sequence
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Combine source cow info with sequence encoding
        self.fc = nn.Linear(hidden_dim + embedding_dim, num_cows)
    
    def forward(self, source_cow_id, input_sequence):
        # Embed source cow
        source_embed = self.cow_embedding(source_cow_id)  # (batch_size, embedding_dim)
        
        # Embed and process sequence
        seq_embeds = self.cow_embedding(input_sequence)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(seq_embeds)
        seq_encoding = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Combine source cow info with sequence
        combined = torch.cat([source_embed, seq_encoding], dim=1)  # (batch_size, hidden_dim + embedding_dim)
        
        # Predict next cow
        logits = self.fc(combined)  # (batch_size, num_cows)
        
        return logits
"""
    print(example_code)


def main():
    parser = argparse.ArgumentParser(
        description='Load and explore Next Cow Prediction dataset'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='next_cow_data',
        help='Dataset directory (default: next_cow_data)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5,
        help='Number of sample sequences to show (default: 5)'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print(f"\n💡 Generate the dataset first:")
        print(f"   python generate_next_cow_dataset.py --output-dir {dataset_dir}")
        return
    
    # Load dataset
    dataset = load_dataset(dataset_dir)
    
    # Print information
    print_dataset_info(dataset)
    
    # Show sample sequences
    print_sample_sequences(dataset, args.n_samples)
    
    # Print usage examples
    create_pytorch_dataset_example()
    create_model_architecture_example()
    
    print("\n" + "=" * 70)
    print("✅ DATASET EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
