  #!/usr/bin/env python3


import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import extractor from cow2vec to reuse logic
from cow2vec import ProximitySequenceExtractor, EmbeddingEvaluator, save_embeddings


class TopKProximitySequenceExtractor(ProximitySequenceExtractor):
    """
    Extracts proximity sequences considering only the top-k closest neighbors.
    Also handles filtering of excluded sensors during extraction.
    """
    def __init__(self, temporal_graphs, top_k=1, excluded_sensors=None):
        super().__init__(temporal_graphs)
        self.top_k = top_k
        self.excluded_sensors = excluded_sensors or set()

    def extract_sequences(self) -> Dict[str, List[List[str]]]:
        print(f"\nExtracting proximity sequences (top_k={self.top_k})...")
        if self.excluded_sensors:
            print(f"Excluding sensors: {self.excluded_sensors}")
        
        for graph_info in tqdm(self.temporal_graphs, desc="Processing snapshots"):
            G = graph_info['graph']
            
            for cow in G.nodes():
                if cow in self.excluded_sensors:
                    continue

                # Get neighbors with RSSI
                neighbors_data = []
                for neighbor in G.neighbors(cow):
                    if neighbor in self.excluded_sensors:
                        continue
                    # RSSI is usually negative, closer to 0 is better
                    rssi = G.edges[cow, neighbor].get('rssi', -100)
                    neighbors_data.append((neighbor, rssi))
                
                # Sort by RSSI descending (closest first)
                neighbors_data.sort(key=lambda x: x[1], reverse=True)
                
                # Take top_k
                if self.top_k is not None and self.top_k > 0:
                    neighbors_data = neighbors_data[:self.top_k]
                
                neighbors = [n for n, r in neighbors_data]
                
                if neighbors:
                    self.proximity_sequences[cow].append(neighbors)
        
        return self.proximity_sequences

    def build_vocabulary(self):
        """Build vocabulary excluding specific sensors."""
        print("Building cow vocabulary...")
        self.cow_vocabulary = set()
        
        for graph_info in tqdm(self.temporal_graphs, desc="Scanning graphs"):
            G = graph_info['graph']
            nodes = [n for n in G.nodes() if n not in self.excluded_sensors]
            self.cow_vocabulary.update(nodes)
        
        self.cow_vocabulary = sorted(self.cow_vocabulary)
        self.cow_to_id = {cow: idx for idx, cow in enumerate(self.cow_vocabulary)}
        self.id_to_cow = {idx: cow for cow, idx in self.cow_to_id.items()}
        
        print(f"   Found {len(self.cow_vocabulary)} unique cows (after filtering)")
        return self.cow_vocabulary


class CowBERTDataset(Dataset):
    """
    Dataset for Masked Language Modeling.
    Takes sequences of cow interactions and applies masking.
    """
    
    def __init__(
        self, 
        sequences: Dict[str, List[List[str]]], 
        cow_to_id: Dict[str, int],
        max_len: int = 128,
        mask_prob: float = 0.15
    ):
        """
        Args:
            sequences: Dict mapping cow_id -> list of neighbor lists
            cow_to_id: Mapping from cow name to ID
            max_len: Maximum sequence length
            mask_prob: Probability of masking a token
        """
        self.cow_to_id = cow_to_id
        self.max_len = max_len
        self.mask_prob = mask_prob
        
        # Special tokens
        self.vocab_size = len(cow_to_id)
        self.PAD_TOKEN = self.vocab_size
        self.MASK_TOKEN = self.vocab_size + 1
        self.CLS_TOKEN = self.vocab_size + 2
        self.SEP_TOKEN = self.vocab_size + 3
        self.full_vocab_size = self.vocab_size + 4
        
        self.samples = self._prepare_samples(sequences)
        
    def _prepare_samples(self, sequences: Dict[str, List[List[str]]]) -> List[Tuple[int, List[int]]]:
        """Flatten neighbor lists into linear sequences and chunk them.
        Returns list of (ego_cow_id, neighbor_sequence) tuples.
        """
        samples = []
        
        print("Preparing BERT sequences...")
        for cow, neighbor_seqs in tqdm(sequences.items(), desc="Processing cows"):
            if cow not in self.cow_to_id:
                continue
                
            ego_cow_id = self.cow_to_id[cow]
            
            # Flatten: [ [A,B], [C], [A] ] -> [A, B, C, A]
            flat_seq = []
            for neighbors in neighbor_seqs:
                # Shuffle neighbors within same timestep to avoid artificial order
                current_neighbors = list(neighbors)
                random.shuffle(current_neighbors)
                for neighbor in current_neighbors:
                    if neighbor in self.cow_to_id:
                        flat_seq.append(self.cow_to_id[neighbor])
            
            # Chunk into max_len - 3 (for EGO, CLS and SEP)
            chunk_size = self.max_len - 3
            for i in range(0, len(flat_seq), chunk_size):
                chunk = flat_seq[i:i + chunk_size]
                if len(chunk) > 5:  # Ignore very short sequences
                    samples.append((ego_cow_id, chunk))
                    
        print(f"   Created {len(samples):,} sequences")
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return (masked_input, target_labels)
        Sequence structure: [EGO_COW, CLS, neighbor1, neighbor2, ..., SEP]
        The EGO_COW token is never masked - it identifies whose sequence this is.
        """
        ego_cow_id, seq = self.samples[idx]
        
        # Add special tokens: EGO first, then CLS, then neighbors, then SEP
        input_ids = [ego_cow_id, self.CLS_TOKEN] + seq + [self.SEP_TOKEN]
        labels = [-100] * len(input_ids)  # -100 ignored by CrossEntropyLoss
        
        # Apply masking (skip EGO at position 0, CLS at position 1, and SEP at the end)
        for i in range(2, len(input_ids) - 1):  # Skip EGO, CLS and SEP
            prob = random.random()
            if prob < self.mask_prob:
                # 80% replace with MASK
                if random.random() < 0.8:
                    labels[i] = input_ids[i]
                    input_ids[i] = self.MASK_TOKEN
                # 10% replace with random token
                elif random.random() < 0.5:
                    labels[i] = input_ids[i]
                    input_ids[i] = random.randint(0, self.vocab_size - 1)
                # 10% keep original (but predict it)
                else:
                    labels[i] = input_ids[i]
        
        # Padding
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.PAD_TOKEN] * padding_len
            labels = labels + [-100] * padding_len
            
        return torch.tensor(input_ids), torch.tensor(labels)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class CowBERTModel(nn.Module):
    """Transformer Encoder for Cow Embeddings."""
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 128, 
        nhead: int = 4, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # MLM Head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.mlm_head[0].bias.data.zero_()
        self.mlm_head[0].weight.data.uniform_(-initrange, initrange)
        self.mlm_head[3].bias.data.zero_()
        self.mlm_head[3].weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: [batch_size, seq_len]
        Returns:
            output: [batch_size, seq_len, vocab_size]
        """
        # Permute for Transformer (seq_len, batch, d_model)
        src = src.transpose(0, 1)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        
        # Back to (batch, seq_len, d_model)
        output = output.transpose(0, 1)
        
        # MLM prediction
        prediction = self.mlm_head(output)
        return prediction

    def get_cow_embeddings(self):
        """Return the learned static embeddings (input layer)."""
        # We exclude special tokens
        return self.embedding.weight.data[:-4].cpu().numpy()


def calculate_accuracy(output, labels):
    """
    Calculate accuracy for masked tokens only.
    output: [batch_size * seq_len, vocab_size]
    labels: [batch_size * seq_len]
    """
    predictions = output.argmax(dim=-1)
    mask = labels != -100
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct / total


def calculate_per_cow_accuracy(model, dataloader, device, id_to_cow):
    """
    Calculate accuracy per cow ID.
    Returns a dictionary mapping cow_id -> (correct, total, accuracy).
    """
    model.eval()
    
    # Track correct and total predictions per cow
    cow_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc="Computing per-cow accuracy"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            output = model(input_ids, src_mask=None)
            predictions = output.argmax(dim=-1)  # [batch_size, seq_len]
            
            # Process each sample in the batch
            batch_size, seq_len = labels.shape
            for b in range(batch_size):
                for i in range(seq_len):
                    if labels[b, i] != -100:  # This position is masked
                        true_cow_id = labels[b, i].item()
                        pred_cow_id = predictions[b, i].item()
                        
                        # Only count if it's a valid cow ID (not special token)
                        if true_cow_id < len(id_to_cow):
                            cow_name = id_to_cow[true_cow_id]
                            cow_stats[cow_name]['total'] += 1
                            if pred_cow_id == true_cow_id:
                                cow_stats[cow_name]['correct'] += 1
    
    # Calculate accuracy for each cow
    results = {}
    for cow_name, stats in cow_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            results[cow_name] = {
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy': accuracy
            }
    
    return results


def plot_per_cow_accuracy(per_cow_results, save_path, top_n=None):
    """
    Create a bar plot of per-cow accuracy.
    
    Args:
        per_cow_results: Dict mapping cow_id -> {'correct', 'total', 'accuracy'}
        save_path: Path to save the plot
        top_n: If specified, only plot top_n and bottom_n cows
    """
    # Sort by accuracy
    sorted_cows = sorted(per_cow_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    if top_n and len(sorted_cows) > top_n * 2:
        # Show top N and bottom N
        selected_cows = sorted_cows[:top_n] + sorted_cows[-top_n:]
    else:
        selected_cows = sorted_cows
    
    cow_names = [cow for cow, _ in selected_cows]
    accuracies = [stats['accuracy'] for _, stats in selected_cows]
    totals = [stats['total'] for _, stats in selected_cows]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color bars based on accuracy
    colors = plt.cm.RdYlGn([acc for acc in accuracies])
    bars = ax.bar(range(len(cow_names)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, acc, total) in enumerate(zip(bars, accuracies, totals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}\n(n={total})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Cow ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('CowBERT: Per-Cow Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(cow_names)))
    ax.set_xticklabels(cow_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=np.mean(accuracies), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.1%}')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved per-cow accuracy plot: {save_path}")
    
    return fig

def train_cowbert(
    model, 
    train_dataloader,
    val_dataloader=None,
    epochs=10, 
    lr=0.001, 
    device='cuda'
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    # Track best model
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    print(f"\nTraining CowBERT on {device}...", flush=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_acc = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}")
        for input_ids, labels in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Create padding mask (True where value is PAD)
            pad_token_id = model.embedding.num_embeddings - 4
            
            output = model(input_ids, src_mask=None)
            
            # Flatten for loss
            output_flat = output.view(-1, output.size(-1))
            labels_flat = labels.view(-1)
            
            loss = criterion(output_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            acc = calculate_accuracy(output_flat, labels_flat)
            
            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}', 'train_acc': f'{acc:.4f}'})
            
        avg_loss = total_loss / len(train_dataloader)
        avg_acc = total_acc / len(train_dataloader)
        train_loss_history.append(avg_loss)
        train_acc_history.append(avg_acc)
        
        # Validation
        val_msg = ""
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            total_val_acc = 0
            with torch.no_grad():
                for input_ids, labels in val_dataloader:
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    
                    output = model(input_ids, src_mask=None)
                    output_flat = output.view(-1, output.size(-1))
                    labels_flat = labels.view(-1)
                    
                    loss = criterion(output_flat, labels_flat)
                    acc = calculate_accuracy(output_flat, labels_flat)
                    
                    total_val_loss += loss.item()
                    total_val_acc += acc
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_acc = total_val_acc / len(val_dataloader)
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(avg_val_acc)
            val_msg = f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"
            
            # Save best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                print(f"   ⭐ New best model! Val Acc: {best_val_acc:.4f}", flush=True)
            
        print(f"   Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}{val_msg}", flush=True)
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Restored best model from epoch {best_epoch} (Val Acc: {best_val_acc:.4f})", flush=True)
    
    return {
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'val_loss': val_loss_history,
        'val_acc': val_acc_history,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc
    }


def evaluate_baseline(dataloader):
    """
    Evaluate a dummy baseline that predicts the token at index i-1.
    """
    print("\n" + "="*70, flush=True)
    print("BASELINE EVALUATION (Last Observation Carried Forward)", flush=True)
    print("="*70, flush=True)
    
    total_correct = 0
    total_masked = 0
    
    for input_ids, labels in tqdm(dataloader, desc="Evaluating Baseline"):
        # input_ids: [batch, seq_len]
        # labels: [batch, seq_len] (-100 where not masked)
        
        batch_size, seq_len = input_ids.shape
        
        for b in range(batch_size):
            for i in range(1, seq_len): # Skip CLS at 0
                if labels[b, i] != -100: # This position is masked/target
                    target = labels[b, i].item()
                    
                    # Predict value at i-1
                    # Note: input_ids[b, i-1] might be MASK or special token
                    pred = input_ids[b, i-1].item()
                    
                    if pred == target:
                        total_correct += 1
                    total_masked += 1
                    
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    print(f"Baseline Accuracy: {accuracy:.4f}", flush=True)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train CowBERT")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='cowbert_embeddings')
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=1, help="Number of closest neighbors to consider (default: 1)")
    
    args = parser.parse_args()
    
    print("="*70, flush=True)
    print("COWBERT: TRANSFORMER-BASED COW EMBEDDINGS", flush=True)
    print("="*70, flush=True)
    
    # 1. Load Data
    with open(args.input, 'rb') as f:
        temporal_graphs = pickle.load(f)
        
    # Define excluded sensors
    excluded_sensors = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
    
    # Use TopKProximitySequenceExtractor
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
    print("Splitting data temporally (80/20)...")
    train_sequences = {}
    val_sequences = {}
    split_ratio = 0.8
    
    for cow, seqs in sequences.items():
        # seqs is a list of neighbor lists ordered by time
        split_idx = int(len(seqs) * split_ratio)
        train_sequences[cow] = seqs[:split_idx]
        val_sequences[cow] = seqs[split_idx:]
    
    train_dataset = CowBERTDataset(
        train_sequences, 
        cow_to_id,
        max_len=128
    )
    
    val_dataset = CowBERTDataset(
        val_sequences, 
        cow_to_id,
        max_len=128
    )
    
    print(f"Dataset split: {len(train_dataset)} train sequences, {len(val_dataset)} validation sequences")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # --- Run Baseline ---
    evaluate_baseline(val_dataloader)
    # --------------------
    
    # 3. Initialize Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CowBERTModel(
        vocab_size=train_dataset.full_vocab_size,
        d_model=args.embedding_dim,
        nhead=4,
        num_layers=2
    )
    
    # 4. Train
    history = train_cowbert(
        model, 
        train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs, 
        device=device
    )
    
    # 5. Save Results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss')
        # Mark best epoch
        best_epoch = history.get('best_epoch', 0)
        if best_epoch > 0:
            ax1.axvline(x=best_epoch-1, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7,
                       label=f'Best Epoch ({best_epoch})')
    ax1.set_title('CowBERT Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    if history['val_acc']:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        # Mark best epoch
        if best_epoch > 0:
            ax2.axvline(x=best_epoch-1, color='red', linestyle='--',
                       linewidth=2, alpha=0.7,
                       label=f'Best Epoch ({best_epoch})')
            ax2.scatter([best_epoch-1], [history['best_val_acc']], 
                       color='red', s=100, zorder=5,
                       label=f'Best Val Acc: {history["best_val_acc"]:.2%}')
    ax2.set_title('CowBERT Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=300)
    
    # Extract embeddings (Input embeddings)
    # Note: In BERT, input embeddings are context-independent.
    # Contextual embeddings come from output of encoder.
    # For "Cow Embeddings", we usually want the static ones, 
    # or we can average contextual ones over many samples.
    # Here we save the static input embeddings for direct comparison with Cow2Vec.
    embeddings = model.get_cow_embeddings()
    
    config = {
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'model': 'CowBERT'
    }
    
    save_embeddings(
        embeddings,
        id_to_cow,
        cow_to_id,
        output_dir,
        config
    )
    
    # 6. Evaluate
    print("\n" + "="*70, flush=True)
    print("EVALUATION", flush=True)
    print("="*70, flush=True)
    
    evaluator = EmbeddingEvaluator(
        embeddings,
        id_to_cow,
        cow_to_id
    )
    
    evaluator.visualize_embeddings_tsne(
        save_path=output_dir / "embeddings_tsne.png",
        perplexity=30
    )
    
    sample_cows = list(cow_to_id.keys())[:10]
    evaluator.generate_similarity_report(
        sample_cows,
        save_path=output_dir / "similarity_report.txt"
    )
    
    # Final evaluation on validation set
    print("\n" + "="*70, flush=True)
    print("FINAL EVALUATION ON VALIDATION SET", flush=True)
    print("="*70, flush=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():
        for input_ids, labels in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            output = model(input_ids, src_mask=None)
            output_flat = output.view(-1, output.size(-1))
            labels_flat = labels.view(-1)
            
            loss = criterion(output_flat, labels_flat)
            acc = calculate_accuracy(output_flat, labels_flat)
            
            total_val_loss += loss.item()
            total_val_acc += acc
    
    final_val_loss = total_val_loss / len(val_dataloader)
    final_val_acc = total_val_acc / len(val_dataloader)
    
    # Per-cow accuracy analysis
    print("\n" + "="*70, flush=True)
    print("PER-COW ACCURACY ANALYSIS", flush=True)
    print("="*70, flush=True)
    
    per_cow_results = calculate_per_cow_accuracy(
        model, val_dataloader, device, id_to_cow
    )
    
    # Sort by accuracy for display
    sorted_cows = sorted(
        per_cow_results.items(), 
        key=lambda x: x[1]['accuracy'], 
        reverse=True
    )
    
    # Print top 10 and bottom 10
    print("\n🔝 TOP 10 EASIEST COWS TO PREDICT:")
    for i, (cow_name, stats) in enumerate(sorted_cows[:10], 1):
        print(f"  {i}. {cow_name}: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    print("\n🔻 BOTTOM 10 HARDEST COWS TO PREDICT:")
    for i, (cow_name, stats) in enumerate(sorted_cows[-10:], 1):
        print(f"  {i}. {cow_name}: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    # Calculate statistics
    accuracies = [stats['accuracy'] for stats in per_cow_results.values()]
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  - Mean accuracy: {np.mean(accuracies):.2%}")
    print(f"  - Median accuracy: {np.median(accuracies):.2%}")
    print(f"  - Std deviation: {np.std(accuracies):.2%}")
    print(f"  - Min accuracy: {np.min(accuracies):.2%}")
    print(f"  - Max accuracy: {np.max(accuracies):.2%}")
    
    # Plot per-cow accuracy
    plot_per_cow_accuracy(
        per_cow_results, 
        output_dir / 'per_cow_accuracy.png',
        top_n=None  # Plot all cows
    )
    
    # Save detailed per-cow results to CSV
    per_cow_df = pd.DataFrame([
        {
            'cow_id': cow_name,
            'accuracy': stats['accuracy'],
            'correct': stats['correct'],
            'total': stats['total']
        }
        for cow_name, stats in sorted_cows
    ])
    per_cow_csv = output_dir / 'per_cow_accuracy.csv'
    per_cow_df.to_csv(per_cow_csv, index=False)
    print(f"✅ Saved per-cow accuracy data: {per_cow_csv}")

    
    
    # Print results to console
    print(f"\n{'='*70}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Final Validation Loss: {final_val_loss:.4f}", flush=True)
    print(f"Final Validation Accuracy: {final_val_acc:.4f}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Save results to file
    results_file = output_dir / 'training_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write("COWBERT TRAINING RESULTS\n")
        f.write(f"{'='*70}\n\n")
        f.write("Configuration:\n")
        f.write(f"  - Embedding dimension: {args.embedding_dim}\n")
        f.write(f"  - Epochs: {args.epochs}\n")
        f.write(f"  - Batch size: {args.batch_size}\n")
        f.write(f"  - Top-k neighbors: {args.top_k}\n\n")
        f.write("Training Results:\n")
        f.write(f"  - Best epoch: {history.get('best_epoch', 'N/A')}\n")
        f.write(f"  - Best validation accuracy: "
                f"{history.get('best_val_acc', 0.0):.4f}\n\n")
        f.write("Final Evaluation (Best Model):\n")
        f.write(f"  - Validation Loss: {final_val_loss:.4f}\n")
        f.write(f"  - Validation Accuracy: {final_val_acc:.4f}\n\n")
        f.write("Per-Cow Accuracy Statistics:\n")
        accuracies_list = [s['accuracy'] for s in per_cow_results.values()]
        f.write(f"  - Mean: {np.mean(accuracies_list):.2%}\n")
        f.write(f"  - Median: {np.median(accuracies_list):.2%}\n")
        f.write(f"  - Std: {np.std(accuracies_list):.2%}\n")
        f.write(f"  - Min: {np.min(accuracies_list):.2%}\n")
        f.write(f"  - Max: {np.max(accuracies_list):.2%}\n\n")
        f.write("Output Files:\n")
        f.write(f"  - Embeddings: "
                f"{(output_dir / 'cow_embeddings.pkl').absolute()}\n")
        f.write(f"  - Training metrics: "
                f"{(output_dir / 'training_metrics.png').absolute()}\n")
        f.write(f"  - t-SNE visualization: "
                f"{(output_dir / 'embeddings_tsne.png').absolute()}\n")
        f.write(f"  - Similarity report: "
                f"{(output_dir / 'similarity_report.txt').absolute()}\n")
        f.write(f"  - Per-cow accuracy plot: "
                f"{(output_dir / 'per_cow_accuracy.png').absolute()}\n")
        f.write(f"  - Per-cow accuracy data: "
                f"{(output_dir / 'per_cow_accuracy.csv').absolute()}\n")
    
    print(f"\n✅ CowBERT training complete!", flush=True)
    print(f"\nResults saved to: {output_dir.absolute()}", flush=True)
    print(f"  - Training results: {results_file.absolute()}", flush=True)
    print(f"  - Embeddings: "
          f"{(output_dir / 'cow_embeddings.pkl').absolute()}", flush=True)
    print(f"  - Training metrics plot: "
          f"{(output_dir / 'training_metrics.png').absolute()}", flush=True)
    print(f"  - t-SNE visualization: "
          f"{(output_dir / 'embeddings_tsne.png').absolute()}", flush=True)
    print(f"  - Similarity report: "
          f"{(output_dir / 'similarity_report.txt').absolute()}", flush=True)
    print(f"  - Per-cow accuracy: "
          f"{(output_dir / 'per_cow_accuracy.png').absolute()}", flush=True)
    print(f"\n{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
