#!/usr/bin/env python3
"""
Cow2Vec: Unsupervised Cow Embedding Model

This module implements a Word2Vec-inspired model to learn cow embeddings
from temporal proximity sequences. The model learns vector representations
where cows with similar social patterns are embedded close together.

Architecture:
- Skip-gram model: Predict context cows from target cow
- Negative sampling for efficient training
- Embedding dimension: configurable (default 128)

Training Data:
- Sequences of proximate cows extracted from temporal graphs
- Each cow's neighbors over time form a "sentence"
- Sliding window generates (target, context) pairs

Usage:
    python src/cow2vec.py \
        --input network_sequence/network_sequence_rssi-68_*.pkl \
        --output-dir cow2vec_embeddings \
        --embedding-dim 128 \
        --window-size 5 \
        --epochs 10
"""

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ProximitySequenceExtractor:
    """Extract proximity sequences from temporal graphs for Cow2Vec training."""
    
    def __init__(self, temporal_graphs: List[Dict], excluded_sensors: set = None):
        """
        Initialize extractor with temporal graph data.
        
        Args:
            temporal_graphs: List of dicts with 'timestamp', 'graph', etc.
            excluded_sensors: Set of sensor IDs to exclude.
        """
        self.temporal_graphs = temporal_graphs
        self.excluded_sensors = excluded_sensors or set()
        self.cow_vocabulary = set()
        self.cow_to_id = {}
        self.id_to_cow = {}
        self.proximity_sequences = defaultdict(list)
        
    def build_vocabulary(self):
        """Build cow vocabulary from all temporal graphs."""
        print("Building cow vocabulary...")
        
        for graph_info in tqdm(self.temporal_graphs, desc="Scanning graphs"):
            G = graph_info['graph']
            nodes = [n for n in G.nodes() if n not in self.excluded_sensors]
            self.cow_vocabulary.update(nodes)
        
        # Create bidirectional mappings
        self.cow_vocabulary = sorted(self.cow_vocabulary)
        self.cow_to_id = {cow: idx for idx, cow in enumerate(self.cow_vocabulary)}
        self.id_to_cow = {idx: cow for cow, idx in self.cow_to_id.items()}
        
        print(f"   Found {len(self.cow_vocabulary)} unique cows")
        return self.cow_vocabulary
    
    def extract_sequences(self) -> Dict[str, List[List[str]]]:
        """
        Extract proximity sequences for each cow.
        
        For each cow, create a sequence of its neighbors over time.
        Each timestep contributes the list of neighbors at that moment.
        
        Returns:
            Dict mapping cow_id -> list of neighbor sequences
        """
        print("\nExtracting proximity sequences...")
        
        for graph_info in tqdm(self.temporal_graphs, desc="Processing snapshots"):
            G = graph_info['graph']
            
            # For each cow in this graph, record its neighbors
            for cow in G.nodes():
                if cow in self.excluded_sensors:
                    continue
                    
                neighbors = [n for n in G.neighbors(cow) if n not in self.excluded_sensors]
                if neighbors:  # Only record if cow has neighbors
                    self.proximity_sequences[cow].append(neighbors)
        
        # Convert to regular dict and compute stats
        sequences_dict = dict(self.proximity_sequences)
        
        total_sequences = sum(len(seqs) for seqs in sequences_dict.values())
        avg_neighbors = np.mean([
            len(neighbors) 
            for seqs in sequences_dict.values() 
            for neighbors in seqs
        ])
        
        print(f"   Cows with proximity data: {len(sequences_dict)}")
        print(f"   Total proximity events: {total_sequences:,}")
        print(f"   Average neighbors per event: {avg_neighbors:.2f}")
        
        return sequences_dict
    
    def generate_training_pairs(
        self, 
        window_size: int = 5
    ) -> List[Tuple[int, int]]:
        """
        Generate (target, context) training pairs using sliding window.
        
        For each cow's sequence of neighbors over time, use a sliding window
        to generate pairs. Window size determines temporal context range.
        
        Args:
            window_size: Number of timesteps to consider as context
            
        Returns:
            List of (target_cow_id, context_cow_id) integer pairs
        """
        print(f"\nGenerating training pairs (window_size={window_size})...")
        
        training_pairs = []
        
        for cow, neighbor_sequences in tqdm(
            self.proximity_sequences.items(), 
            desc="Creating pairs"
        ):
            target_id = self.cow_to_id[cow]
            
            # Flatten sequences into timeline of neighbors
            for i, neighbors in enumerate(neighbor_sequences):
                # Look at neighbors in current + nearby timesteps (within window)
                start_idx = max(0, i - window_size)
                end_idx = min(len(neighbor_sequences), i + window_size + 1)
                
                context_neighbors = set()
                for j in range(start_idx, end_idx):
                    context_neighbors.update(neighbor_sequences[j])
                
                # Create pairs: target cow -> each context cow
                for neighbor in context_neighbors:
                    context_id = self.cow_to_id[neighbor]
                    training_pairs.append((target_id, context_id))
        
        print(f"   Generated {len(training_pairs):,} training pairs")
        
        return training_pairs


class Cow2VecDataset(Dataset):
    """PyTorch Dataset for Cow2Vec training with negative sampling."""
    
    def __init__(
        self, 
        training_pairs: List[Tuple[int, int]], 
        vocab_size: int,
        negative_samples: int = 5
    ):
        """
        Initialize dataset.
        
        Args:
            training_pairs: List of (target, context) cow ID pairs
            vocab_size: Total number of unique cows
            negative_samples: Number of negative samples per positive pair
        """
        self.pairs = training_pairs
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        
        # Compute sampling distribution (unigram^0.75 as in Word2Vec)
        cow_counts = Counter([pair[1] for pair in training_pairs])
        freqs = np.array([cow_counts.get(i, 0) for i in range(vocab_size)])
        self.sampling_probs = np.power(freqs, 0.75)
        self.sampling_probs /= self.sampling_probs.sum()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get training sample with negative samples.
        
        Returns:
            target: Target cow ID
            positive: Positive context cow ID
            negatives: List of negative sample cow IDs
        """
        target, positive = self.pairs[idx]
        
        # Sample negative examples (cows not in this context)
        negatives = np.random.choice(
            self.vocab_size,
            size=self.negative_samples,
            p=self.sampling_probs
        )
        
        return (
            torch.tensor(target, dtype=torch.long),
            torch.tensor(positive, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long)
        )


class Cow2VecModel(nn.Module):
    """Skip-gram model for learning cow embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Cow2Vec model.
        
        Args:
            vocab_size: Number of unique cows
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Target embeddings (main embeddings we'll use)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context embeddings (auxiliary embeddings for training)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        init_range = 0.5 / self.embedding_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, target, positive, negatives):
        """
        Forward pass with negative sampling.
        
        Args:
            target: Target cow IDs [batch_size]
            positive: Positive context cow IDs [batch_size]
            negatives: Negative sample cow IDs [batch_size, n_negative]
            
        Returns:
            loss: Negative sampling loss
        """
        batch_size = target.size(0)
        
        # Get embeddings
        target_embed = self.target_embeddings(target)  # [batch, embed_dim]
        positive_embed = self.context_embeddings(positive)  # [batch, embed_dim]
        negative_embed = self.context_embeddings(negatives)  # [batch, n_neg, embed_dim]
        
        # Positive score: target · positive
        positive_score = torch.sum(target_embed * positive_embed, dim=1)  # [batch]
        positive_loss = -torch.log(torch.sigmoid(positive_score))
        
        # Negative scores: target · negatives
        # target_embed: [batch, embed_dim] -> [batch, 1, embed_dim]
        # negative_embed: [batch, n_neg, embed_dim]
        negative_scores = torch.bmm(
            negative_embed, 
            target_embed.unsqueeze(2)
        ).squeeze(2)  # [batch, n_neg]
        
        negative_loss = -torch.sum(
            torch.log(torch.sigmoid(-negative_scores)), 
            dim=1
        )  # [batch]
        
        # Total loss
        loss = torch.mean(positive_loss + negative_loss)
        
        return loss
    
    def get_embeddings(self) -> np.ndarray:
        """
        Get learned cow embeddings.
        
        Returns:
            Embedding matrix [vocab_size, embedding_dim]
        """
        return self.target_embeddings.weight.data.cpu().numpy()


class Cow2VecTrainer:
    """Trainer for Cow2Vec model."""
    
    def __init__(
        self,
        model: Cow2VecModel,
        train_dataset: Cow2VecDataset,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Cow2Vec model
            train_dataset: Training dataset
            learning_rate: Learning rate for optimizer
            batch_size: Batch size
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.dataset = train_dataset
        self.device = device
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        self.loss_history = []
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.dataloader, desc="Training")
        for target, positive, negatives in pbar:
            # Move to device
            target = target.to(self.device)
            positive = positive.to(self.device)
            negatives = negatives.to(self.device)
            
            # Forward pass
            loss = self.model(target, positive, negatives)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.dataloader)
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def train(self, epochs: int, save_dir: Path = None):
        """
        Train for multiple epochs.
        
        Args:
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        print(f"\nTraining on {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {epochs}\n")
        
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            avg_loss = self.train_epoch()
            print(f"Average loss: {avg_loss:.4f}\n")
            
            # Save checkpoint
            if save_dir and epoch % 5 == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch{epoch}.pt"
                self.save_checkpoint(checkpoint_path, epoch)
    
    def save_checkpoint(self, path: Path, epoch: int):
        """Save training checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
        }, path)
        print(f"   Saved checkpoint: {path.name}")
    
    def plot_loss_curve(self, save_path: Path = None):
        """Plot training loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Cow2Vec Training Loss', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved loss curve: {save_path.name}")
        else:
            plt.show()
        
        plt.close()


class EmbeddingEvaluator:
    """Evaluate and visualize cow embeddings."""
    
    def __init__(
        self, 
        embeddings: np.ndarray, 
        id_to_cow: Dict[int, str],
        cow_to_id: Dict[str, int]
    ):
        """
        Initialize evaluator.
        
        Args:
            embeddings: Embedding matrix [vocab_size, embedding_dim]
            id_to_cow: Mapping from ID to cow name
            cow_to_id: Mapping from cow name to ID
        """
        self.embeddings = embeddings
        self.id_to_cow = id_to_cow
        self.cow_to_id = cow_to_id
    
    def find_similar_cows(self, cow_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar cows using cosine similarity.
        
        Args:
            cow_id: Target cow ID
            top_k: Number of similar cows to return
            
        Returns:
            List of (cow_id, similarity_score) tuples
        """
        if cow_id not in self.cow_to_id:
            raise ValueError(f"Cow {cow_id} not in vocabulary")
        
        target_idx = self.cow_to_id[cow_id]
        target_vec = self.embeddings[target_idx]
        
        # Compute cosine similarity with all cows
        similarities = np.dot(self.embeddings, target_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(target_vec)
        )
        
        # Get top-k most similar (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = [
            (self.id_to_cow[idx], similarities[idx]) 
            for idx in top_indices
        ]
        
        return results
    
    def visualize_embeddings_tsne(
        self, 
        save_path: Path = None,
        perplexity: int = 30,
        n_samples: int = None
    ):
        """
        Visualize embeddings using t-SNE.
        
        Args:
            save_path: Path to save visualization
            perplexity: t-SNE perplexity parameter
            n_samples: Number of cows to visualize (None = all)
        """
        print("\nGenerating t-SNE visualization...")
        
        # Sample if too many cows
        if n_samples and len(self.embeddings) > n_samples:
            indices = np.random.choice(len(self.embeddings), n_samples, replace=False)
            embeddings_subset = self.embeddings[indices]
            labels = [self.id_to_cow[i] for i in indices]
        else:
            embeddings_subset = self.embeddings
            labels = [self.id_to_cow[i] for i in range(len(self.embeddings))]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_subset)
        
        # Plot
        plt.figure(figsize=(14, 10))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=100)
        
        # Annotate points
        for i, label in enumerate(labels):
            plt.annotate(
                label, 
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
        plt.title('Cow2Vec Embeddings (t-SNE Projection)', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved t-SNE plot: {save_path.name}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_similarity_report(
        self, 
        sample_cows: List[str], 
        save_path: Path = None,
        top_k: int = 5
    ):
        """
        Generate similarity report for sample cows.
        
        Args:
            sample_cows: List of cow IDs to analyze
            save_path: Path to save report
            top_k: Number of similar cows to show per cow
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("COW2VEC SIMILARITY REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        for cow_id in sample_cows:
            if cow_id not in self.cow_to_id:
                continue
            
            similar = self.find_similar_cows(cow_id, top_k=top_k)
            
            report_lines.append(f"Cow: {cow_id}")
            report_lines.append(f"Most similar cows:")
            for i, (similar_cow, score) in enumerate(similar, 1):
                report_lines.append(f"  {i}. {similar_cow} (similarity: {score:.4f})")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"   Saved similarity report: {save_path.name}")
        else:
            print(report_text)


def save_embeddings(
    embeddings: np.ndarray,
    id_to_cow: Dict[int, str],
    cow_to_id: Dict[str, int],
    output_dir: Path,
    config: Dict
):
    """
    Save embeddings in multiple formats.
    
    Args:
        embeddings: Embedding matrix
        id_to_cow: ID to cow mapping
        cow_to_id: Cow to ID mapping
        output_dir: Output directory
        config: Training configuration
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving embeddings...")
    
    # 1. Numpy format
    np_path = output_dir / "cow_embeddings.npy"
    np.save(np_path, embeddings)
    print(f"✅ Saved numpy embeddings: {np_path.name}")
    
    # 2. CSV format
    csv_path = output_dir / "cow_embeddings.csv"
    df = pd.DataFrame(
        embeddings,
        index=[id_to_cow[i] for i in range(len(embeddings))]
    )
    df.index.name = 'cow_id'
    df.to_csv(csv_path)
    print(f"✅ Saved CSV embeddings: {csv_path.name}")
    
    # 3. Pickle format (includes mappings)
    pkl_path = output_dir / "cow_embeddings.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'id_to_cow': id_to_cow,
            'cow_to_id': cow_to_id,
            'config': config
        }, f)
    print(f"✅ Saved pickle embeddings: {pkl_path.name}")
    
    # 4. Metadata
    meta_path = output_dir / "embedding_metadata.json"
    metadata = {
        'vocab_size': len(id_to_cow),
        'embedding_dim': embeddings.shape[1],
        'cow_ids': [id_to_cow[i] for i in range(len(id_to_cow))],
        'config': config,
        'created_at': datetime.now().isoformat()
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {meta_path.name}")


def evaluate_model(model, val_pairs, device='cuda'):
    model.eval()
    context_embed = model.context_embeddings.weight.data # (V, D)
    
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    total = 0
    
    # Process in batches
    batch_size = 1000
    print(f"\nEvaluating Cow2Vec on {len(val_pairs)} validation pairs...")
    
    with torch.no_grad():
        for i in range(0, len(val_pairs), batch_size):
            batch = val_pairs[i:i+batch_size]
            targets = torch.tensor([p[0] for p in batch], device=device)
            true_contexts = torch.tensor([p[1] for p in batch], device=device)
            
            # Get embeddings
            t_emb = model.target_embeddings(targets) # (B, D)
            
            # Scores: (B, D) @ (V, D).T -> (B, V)
            scores = torch.matmul(t_emb, context_embed.t())
            
            # Get ranks
            true_scores = scores.gather(1, true_contexts.unsqueeze(1)) # (B, 1)
            ranks = (scores > true_scores).sum(dim=1) + 1
            
            hits_1 += (ranks <= 1).sum().item()
            hits_5 += (ranks <= 5).sum().item()
            hits_10 += (ranks <= 10).sum().item()
            total += len(batch)
            
    print(f"Cow2Vec Hit@1:  {hits_1/total:.4f}")
    print(f"Cow2Vec Hit@5:  {hits_5/total:.4f}")
    print(f"Cow2Vec Hit@10: {hits_10/total:.4f}")

def evaluate_baseline(train_pairs, val_pairs):
    print(f"\nEvaluating Frequency Baseline on {len(val_pairs)} validation pairs...")
    # Count frequencies in train
    counts = Counter([p[1] for p in train_pairs])
    top_k = [c for c, _ in counts.most_common(10)]
    
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    total = len(val_pairs)
    
    top_1_set = set(top_k[:1])
    top_5_set = set(top_k[:5])
    top_10_set = set(top_k[:10])
    
    for _, true_ctx in val_pairs:
        if true_ctx in top_1_set: hits_1 += 1
        if true_ctx in top_5_set: hits_5 += 1
        if true_ctx in top_10_set: hits_10 += 1
        
    print(f"Baseline Hit@1:  {hits_1/total:.4f}")
    print(f"Baseline Hit@5:  {hits_5/total:.4f}")
    print(f"Baseline Hit@10: {hits_10/total:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Train Cow2Vec embeddings from temporal proximity graphs"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to temporal graph pickle file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='cow2vec_embeddings',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Temporal window size for context (default: 5)'
    )
    parser.add_argument(
        '--negative-samples',
        type=int,
        default=5,
        help='Number of negative samples (default: 5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size (default: 512)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("COW2VEC: UNSUPERVISED COW EMBEDDING MODEL")
    print("="*70)
    print()
    
    # Load temporal graphs
    print(f"Loading temporal graphs from {args.input}...")
    with open(args.input, 'rb') as f:
        temporal_graphs = pickle.load(f)
    
    # Sort by timestamp
    temporal_graphs.sort(key=lambda x: x['timestamp'])
    
    # Temporal Split (80/20)
    split_idx = int(len(temporal_graphs) * 0.8)
    train_graphs = temporal_graphs[:split_idx]
    val_graphs = temporal_graphs[split_idx:]
    
    print(f"   Total snapshots: {len(temporal_graphs):,}")
    print(f"   Train snapshots: {len(train_graphs):,}")
    print(f"   Val snapshots:   {len(val_graphs):,}\n")
    
    excluded_sensors = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
    
    # Initialize extractors
    # Build vocab from FULL dataset to ensure indices match
    full_extractor = ProximitySequenceExtractor(temporal_graphs, excluded_sensors)
    full_extractor.build_vocabulary()
    
    # Extract sequences for Train
    print("\n--- Processing Training Data ---")
    train_extractor = ProximitySequenceExtractor(train_graphs, excluded_sensors)
    train_extractor.cow_vocabulary = full_extractor.cow_vocabulary
    train_extractor.cow_to_id = full_extractor.cow_to_id
    train_extractor.id_to_cow = full_extractor.id_to_cow
    train_extractor.extract_sequences()
    train_pairs = train_extractor.generate_training_pairs(window_size=args.window_size)
    
    # Extract sequences for Val
    print("\n--- Processing Validation Data ---")
    val_extractor = ProximitySequenceExtractor(val_graphs, excluded_sensors)
    val_extractor.cow_vocabulary = full_extractor.cow_vocabulary
    val_extractor.cow_to_id = full_extractor.cow_to_id
    val_extractor.id_to_cow = full_extractor.id_to_cow
    val_extractor.extract_sequences()
    val_pairs = val_extractor.generate_training_pairs(window_size=args.window_size)
    
    # Create dataset
    print("\nCreating training dataset...")
    dataset = Cow2VecDataset(
        train_pairs, 
        vocab_size=len(full_extractor.cow_vocabulary),
        negative_samples=args.negative_samples
    )
    print(f"   Dataset size: {len(dataset):,} samples")
    
    # Create model
    print(f"\nInitializing Cow2Vec model...")
    print(f"   Vocabulary size: {len(full_extractor.cow_vocabulary)}")
    print(f"   Embedding dimension: {args.embedding_dim}")
    
    model = Cow2VecModel(
        vocab_size=len(full_extractor.cow_vocabulary),
        embedding_dim=args.embedding_dim
    )
    
    # Train model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Cow2VecTrainer(
        model=model,
        train_dataset=dataset,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    trainer.train(epochs=args.epochs, save_dir=output_dir)
    
    # Save training loss curve
    trainer.plot_loss_curve(save_path=output_dir / "training_loss.png")
    
    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate_model(model, val_pairs, device=device)
    evaluate_baseline(train_pairs, val_pairs)
    
    # Get embeddings
    embeddings = model.get_embeddings()
    
    # Save embeddings
    config = {
        'embedding_dim': args.embedding_dim,
        'window_size': args.window_size,
        'negative_samples': args.negative_samples,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'input_file': args.input
    }
    
    save_embeddings(
        embeddings,
        full_extractor.id_to_cow,
        full_extractor.cow_to_id,
        output_dir,
        config
    )
    
    # Evaluate embeddings (Intrinsic)
    print("\n" + "="*70)
    print("INTRINSIC EVALUATION")
    print("="*70)
    
    evaluator = EmbeddingEvaluator(
        embeddings,
        full_extractor.id_to_cow,
        full_extractor.cow_to_id
    )
    
    # t-SNE visualization
    evaluator.visualize_embeddings_tsne(
        save_path=output_dir / "embeddings_tsne.png",
        perplexity=min(30, len(full_extractor.cow_vocabulary) - 1)
    )
    
    # Similarity report for sample cows
    sample_cows = list(full_extractor.cow_vocabulary)[:10]  # First 10 cows
    evaluator.generate_similarity_report(
        sample_cows,
        save_path=output_dir / "similarity_report.txt",
        top_k=5
    )
    
    print("\n" + "="*70)
    print("✅ COW2VEC TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 All outputs saved to: {output_dir}/")
    print("\nFiles created:")
    print("  - cow_embeddings.npy (numpy format)")
    print("  - cow_embeddings.csv (CSV format)")
    print("  - cow_embeddings.pkl (pickle with mappings)")
    print("  - embedding_metadata.json (configuration)")
    print("  - training_loss.png (loss curve)")
    print("  - embeddings_tsne.png (t-SNE visualization)")
    print("  - similarity_report.txt (similarity analysis)")
    print("\n💡 Use embeddings for downstream tasks:")
    print("   - Cow similarity analysis")
    print("   - Social group detection")
    print("   - Behavior prediction")
    print("   - Feature engineering for ML models")


if __name__ == "__main__":
    main()
