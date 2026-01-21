"""
Evaluate baselines for Masked Language Modeling task.
1. Most frequent baseline: Always predict the most frequent cow
2. Previous token baseline: Predict the token right before the masked position
"""

import pickle
import numpy as np
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def load_network_sequence(filepath):
    """Load network sequence data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def extract_sequences(sequences, top_k=1):
    """
    Extract flat sequences exactly like CowBERT does with TopKProximitySequenceExtractor.
    Returns list of (ego_cow, neighbor_sequence) tuples.
    """
    from collections import defaultdict
    
    proximity_sequences = defaultdict(list)
    cow_to_id = {}
    
    # Build vocabulary
    all_cows = set()
    for seq_data in sequences:
        graph = seq_data['graph']
        all_cows.update(graph.nodes())
    
    for i, cow in enumerate(sorted(all_cows)):
        cow_to_id[cow] = i
    
    # Extract sequences with top_k neighbors per timestep (like CowBERT)
    for seq_data in sequences:
        graph = seq_data['graph']
        
        for cow in graph.nodes():
            # Get neighbors with RSSI
            neighbors_data = []
            for neighbor in graph.neighbors(cow):
                rssi = graph.edges[cow, neighbor].get('rssi', -100)
                neighbors_data.append((neighbor, rssi))
            
            # Sort by RSSI descending (closest first)
            neighbors_data.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k
            if top_k is not None and top_k > 0:
                neighbors_data = neighbors_data[:top_k]
            
            neighbors = [n for n, r in neighbors_data]
            
            if neighbors:
                proximity_sequences[cow].append(neighbors)
    
    # Flatten sequences like CowBERT does
    samples = []
    max_len = 128
    chunk_size = max_len - 3  # For EGO, CLS, SEP
    
    for cow, neighbor_seqs in proximity_sequences.items():
        if cow not in cow_to_id:
            continue
        
        ego_cow_id = cow_to_id[cow]
        
        # Flatten temporal sequence: [[A], [B], [A]] -> [A, B, A]
        flat_seq = []
        for neighbors in neighbor_seqs:
            for neighbor in neighbors:
                if neighbor in cow_to_id:
                    flat_seq.append(cow_to_id[neighbor])
        
        # Chunk into sequences
        for i in range(0, len(flat_seq), chunk_size):
            chunk = flat_seq[i:i + chunk_size]
            if len(chunk) > 5:  # Ignore very short sequences
                samples.append((ego_cow_id, chunk))
    
    return samples, cow_to_id

def compute_cow_frequencies(samples):
    """Compute frequency of each cow across all sequences."""
    cow_counter = Counter()
    
    for ego_id, neighbor_ids in samples:
        cow_counter[ego_id] += 1
        cow_counter.update(neighbor_ids)
    
    return cow_counter

def evaluate_most_frequent_baseline(samples, most_frequent_id, mask_prob=0.15):
    """
    Baseline 1: Always predict the most frequent cow for masked positions.
    """
    correct = 0
    total = 0
    
    random.seed(42)
    
    for ego_id, neighbor_ids in tqdm(samples, desc="Most Frequent Baseline"):
        # Build sequence like CowBERT: [EGO, CLS, neighbors, SEP]
        # We mask positions 2 onwards (skip EGO and CLS)
        # CLS, SEP, etc are special tokens, but we focus on actual cow tokens
        
        sequence = neighbor_ids.copy()
        
        if len(sequence) == 0:
            continue
        
        # Simulate masking ~15% of tokens
        num_to_mask = max(1, int(len(sequence) * mask_prob))
        mask_indices = random.sample(range(len(sequence)), min(num_to_mask, len(sequence)))
        
        for idx in mask_indices:
            true_id = sequence[idx]
            predicted_id = most_frequent_id
            
            if predicted_id == true_id:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, total

def evaluate_previous_token_baseline(samples, mask_prob=0.15):
    """
    Baseline 2: Predict the token right before the masked position.
    If masked token is first in sequence, predict EGO token.
    
    NOTE: This baseline assumes tokens are in a meaningful sequential order.
    In CowBERT, neighbors within the same timestep are shuffled, so this
    baseline mainly captures temporal autocorrelation (same cow appearing
    in consecutive timesteps).
    """
    correct = 0
    total = 0
    
    random.seed(42)
    
    for ego_id, neighbor_ids in tqdm(samples, desc="Previous Token Baseline"):
        sequence = neighbor_ids.copy()
        
        if len(sequence) == 0:
            continue
        
        # Simulate masking
        num_to_mask = max(1, int(len(sequence) * mask_prob))
        mask_indices = random.sample(range(len(sequence)), min(num_to_mask, len(sequence)))
        
        for idx in mask_indices:
            true_id = sequence[idx]
            
            # Predict previous token
            if idx == 0:
                # First token in sequence, predict ego
                predicted_id = ego_id
            else:
                # Predict previous token
                predicted_id = sequence[idx - 1]
            
            if predicted_id == true_id:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, total

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate MLM baselines')
    parser.add_argument('--input', required=True, 
                       help='Path to network_sequence pickle file')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio for train/val split')
    args = parser.parse_args()
    
    print("="*70)
    print("MASKED LANGUAGE MODELING BASELINE EVALUATION")
    print("="*70)
    
    # Load data
    print(f"\nLoading sequences from {args.input}...")
    sequences = load_network_sequence(args.input)
    print(f"Total snapshots: {len(sequences)}")
    
    # Split train/val
    split_idx = int(len(sequences) * args.train_ratio)
    train_seq = sequences[:split_idx]
    val_seq = sequences[split_idx:]
    
    print(f"Train snapshots: {len(train_seq)}")
    print(f"Val snapshots: {len(val_seq)}")
    
    # Extract sequences
    print("\nExtracting sequences...")
    train_samples, cow_to_id = extract_sequences(train_seq)
    val_samples, _ = extract_sequences(val_seq)
    
    vocab_size = len(cow_to_id)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Compute cow frequencies from training set
    print("\nComputing cow frequencies from training set...")
    cow_freqs = compute_cow_frequencies(train_samples)
    
    # Get most frequent cow
    most_frequent_id = cow_freqs.most_common(1)[0][0]
    most_frequent_count = cow_freqs[most_frequent_id]
    
    print(f"Most frequent cow ID: {most_frequent_id} ({most_frequent_count} occurrences)")
    
    print(f"\nTop 10 most frequent cow IDs:")
    for cow_id, count in cow_freqs.most_common(10):
        print(f"  ID {cow_id}: {count} ({count/sum(cow_freqs.values())*100:.1f}%)")
    
    # Evaluate baselines
    print("\n" + "="*70)
    print("BASELINE 1: MOST FREQUENT TOKEN")
    print("="*70)
    print(f"Strategy: Always predict cow ID {most_frequent_id}")
    
    train_freq_acc, train_freq_total = evaluate_most_frequent_baseline(
        train_samples, most_frequent_id
    )
    val_freq_acc, val_freq_total = evaluate_most_frequent_baseline(
        val_samples, most_frequent_id
    )
    
    print(f"\nResults:")
    print(f"  Train: {train_freq_acc:.4f} ({train_freq_acc*100:.2f}%) on {train_freq_total} masked tokens")
    print(f"  Val:   {val_freq_acc:.4f} ({val_freq_acc*100:.2f}%) on {val_freq_total} masked tokens")
    
    print("\n" + "="*70)
    print("BASELINE 2: PREVIOUS TOKEN")
    print("="*70)
    print("Strategy: Predict the token right before the masked position")
    
    train_prev_acc, train_prev_total = evaluate_previous_token_baseline(train_samples)
    val_prev_acc, val_prev_total = evaluate_previous_token_baseline(val_samples)
    
    print(f"\nResults:")
    print(f"  Train: {train_prev_acc:.4f} ({train_prev_acc*100:.2f}%) on {train_prev_total} masked tokens")
    print(f"  Val:   {val_prev_acc:.4f} ({val_prev_acc*100:.2f}%) on {val_prev_total} masked tokens")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON WITH NEURAL MODELS")
    print("="*70)
    
    print(f"\nMasked Language Modeling Accuracy (Validation):")
    print(f"  Random (1/{vocab_size}):    {100/vocab_size:.2f}%")
    print(f"  Most Frequent:       {val_freq_acc*100:.2f}%")
    print(f"  Previous Token:      {val_prev_acc*100:.2f}%")
    print(f"  CowLSTM (h=32):     73.79%")
    print(f"  CowBERT:            ~73.00%")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
