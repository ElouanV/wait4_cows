#!/usr/bin/env python3
"""
Next Cow Prediction Dataset Generator

This script creates a dataset for predicting the next closest cow to a given cow.
For each cow, we extract temporal sequences of their closest neighbors over time,
creating training examples of the form: [cow_1, cow_2, ..., cow_n-1] -> cow_n

The sequences are extracted from RSSI proximity data, where closeness is 
determined by signal strength (higher RSSI = closer proximity).

Usage:
    python generate_next_cow_dataset.py \
        --output-dir next_cow_data \
        --sequence-length 10 \
        --window-seconds 60 \
        --min-rssi -100
"""

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_rssi_data(rssi_dir: Path, start_hours: int = None, end_hours: int = None) -> pd.DataFrame:
    """
    Load all RSSI parquet files from the directory.
    
    Args:
        rssi_dir: Path to directory containing RSSI parquet files
        start_hours: Optional hour to start filtering (e.g., 9 for 9am)
        end_hours: Optional hour to end filtering (e.g., 23 for 11pm)
    
    Returns:
        Combined DataFrame with columns: receiver_id, emitter_id, RSSI, relative_DateTime
    """
    print(f"Loading RSSI data from {rssi_dir}")
    
    parquet_files = sorted(rssi_dir.glob("*_RSSI_*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No RSSI parquet files found in {rssi_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    all_data = []
    for file_path in tqdm(parquet_files, desc="Loading files"):
        # Extract receiver ID from filename (e.g., "3660_RSSI_elevage_3_cut.parquet" -> "3660")
        receiver_id = file_path.stem.split("_")[0]
        
        df = pd.read_parquet(file_path)
        
        # Rename columns for consistency
        df = df.rename(columns={
            'accelero_id': 'emitter_id',
            'ble_id': 'emitter_ble_id'
        })
        
        # Add receiver ID column
        df['receiver_id'] = receiver_id
        
        # Select only needed columns
        df = df[['receiver_id', 'emitter_id', 'RSSI', 'relative_DateTime']]
        
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['relative_DateTime'] = pd.to_datetime(combined['relative_DateTime'])
    
    # Filter by time if specified
    if start_hours is not None or end_hours is not None:
        combined['hour'] = combined['relative_DateTime'].dt.hour
        
        if start_hours is not None and end_hours is not None:
            combined = combined[(combined['hour'] >= start_hours) & (combined['hour'] <= end_hours)]
        elif start_hours is not None:
            combined = combined[combined['hour'] >= start_hours]
        else:
            combined = combined[combined['hour'] <= end_hours]
        
        combined = combined.drop('hour', axis=1)
        print(f"   Filtered to hours {start_hours}-{end_hours}")
    
    print(f"✅ Loaded {len(combined):,} RSSI measurements")
    print(f"   Time range: {combined['relative_DateTime'].min()} to {combined['relative_DateTime'].max()}")
    print(f"   Unique receivers: {combined['receiver_id'].nunique()}")
    print(f"   Unique emitters: {combined['emitter_id'].nunique()}")
    
    return combined


def create_temporal_snapshots(rssi_df: pd.DataFrame, window_seconds: int, min_rssi: float) -> List[Dict]:
    """
    Create temporal snapshots of closest cows at regular intervals.
    
    For each time window, for each cow, determine their closest neighbor(s)
    based on RSSI signal strength.
    
    Args:
        rssi_df: DataFrame with RSSI measurements
        window_seconds: Duration of each time window
        min_rssi: Minimum RSSI threshold to consider
    
    Returns:
        List of dicts with timestamp and closest neighbors for each cow
    """
    print(f"\nCreating temporal snapshots (window={window_seconds}s, min_rssi={min_rssi})...")
    
    # Filter by minimum RSSI
    rssi_df = rssi_df[rssi_df['RSSI'] >= min_rssi].copy()
    
    # Create time bins
    rssi_df['time_bin'] = rssi_df['relative_DateTime'].dt.floor(f'{window_seconds}s')
    
    unique_times = sorted(rssi_df['time_bin'].unique())
    print(f"   Found {len(unique_times)} time windows")
    
    snapshots = []
    
    for timestamp in tqdm(unique_times, desc="Processing windows"):
        window_df = rssi_df[rssi_df['time_bin'] == timestamp]
        
        # For each receiver, find their closest neighbor(s)
        # Average RSSI within the window for each receiver-emitter pair
        agg_df = window_df.groupby(['receiver_id', 'emitter_id'])['RSSI'].mean().reset_index()
        
        # For each cow (receiver), get closest neighbor(s)
        closest_neighbors = {}
        
        for receiver_id in agg_df['receiver_id'].unique():
            cow_df = agg_df[agg_df['receiver_id'] == receiver_id]
            
            # Sort by RSSI (higher is better/closer)
            cow_df = cow_df.sort_values('RSSI', ascending=False)
            
            # Get the closest neighbor
            if len(cow_df) > 0:
                closest = cow_df.iloc[0]['emitter_id']
                closest_rssi = cow_df.iloc[0]['RSSI']
                
                closest_neighbors[receiver_id] = {
                    'closest_cow': closest,
                    'rssi': closest_rssi,
                    'all_neighbors': list(zip(cow_df['emitter_id'].tolist(), 
                                             cow_df['RSSI'].tolist()))
                }
        
        snapshots.append({
            'timestamp': timestamp,
            'closest_neighbors': closest_neighbors
        })
    
    print(f"✅ Created {len(snapshots)} temporal snapshots")
    return snapshots


def extract_sequences(snapshots: List[Dict], sequence_length: int, 
                     excluded_sensors: set = None) -> Tuple[List, Dict]:
    """
    Extract sequences of closest cows for next-cow prediction.
    
    For each cow, create sequences of their closest neighbors over time.
    Each sequence is of length `sequence_length`, where the first n-1 items
    are inputs and the last item is the target to predict.
    
    Args:
        snapshots: List of temporal snapshots
        sequence_length: Length of sequences to generate
        excluded_sensors: Set of sensor IDs to exclude
    
    Returns:
        Tuple of (sequences list, metadata dict)
    """
    print(f"\nExtracting sequences (length={sequence_length})...")
    
    if excluded_sensors is None:
        excluded_sensors = set()
    
    # Build temporal sequences for each cow
    cow_sequences = defaultdict(list)  # cow -> list of (timestamp, closest_neighbor) tuples
    
    for snapshot in snapshots:
        timestamp = snapshot['timestamp']
        neighbors = snapshot['closest_neighbors']
        
        for cow, neighbor_info in neighbors.items():
            if cow in excluded_sensors:
                continue
            
            closest_cow = neighbor_info['closest_cow']
            
            # Skip if closest cow is in excluded sensors
            if closest_cow in excluded_sensors:
                continue
            
            cow_sequences[cow].append((timestamp, closest_cow))
    
    # Convert to sequences for prediction
    sequences = []
    vocab = set()
    
    for cow, temporal_seq in tqdm(cow_sequences.items(), desc="Creating sequences"):
        # temporal_seq is a list of (timestamp, closest_cow) tuples
        # Extract just the closest cows
        closest_cows = [closest for _, closest in temporal_seq]
        
        # Create sliding window sequences
        for i in range(len(closest_cows) - sequence_length + 1):
            seq = closest_cows[i:i + sequence_length]
            
            # Input: first n-1 cows, Target: last cow
            input_seq = seq[:-1]
            target = seq[-1]
            
            sequences.append({
                'source_cow': cow,
                'input_sequence': input_seq,
                'target': target,
                'timestamps': [t for t, _ in temporal_seq[i:i + sequence_length]]
            })
            
            # Add to vocabulary
            vocab.add(cow)
            vocab.update(input_seq)
            vocab.add(target)
    
    # Create vocabulary mappings
    vocab = sorted(vocab)
    cow_to_id = {cow: idx for idx, cow in enumerate(vocab)}
    id_to_cow = {idx: cow for cow, idx in cow_to_id.items()}
    
    # Convert sequences to integer IDs
    for seq in sequences:
        seq['source_cow_id'] = cow_to_id[seq['source_cow']]
        seq['input_sequence_ids'] = [cow_to_id[cow] for cow in seq['input_sequence']]
        seq['target_id'] = cow_to_id[seq['target']]
    
    metadata = {
        'num_sequences': len(sequences),
        'num_cows': len(vocab),
        'vocabulary': vocab,
        'cow_to_id': cow_to_id,
        'id_to_cow': id_to_cow,
        'sequence_length': sequence_length,
        'excluded_sensors': list(excluded_sensors)
    }
    
    # Print statistics
    print(f"\n✅ Extracted {len(sequences):,} sequences")
    print(f"   Unique cows (vocabulary): {len(vocab)}")
    print(f"   Sequences per cow: {len(sequences) / len(cow_sequences):.1f} avg")
    
    # Distribution of source cows
    source_cow_counts = defaultdict(int)
    for seq in sequences:
        source_cow_counts[seq['source_cow']] += 1
    
    print(f"\n   Sequences distribution:")
    print(f"      Min: {min(source_cow_counts.values())} sequences/cow")
    print(f"      Max: {max(source_cow_counts.values())} sequences/cow")
    print(f"      Mean: {np.mean(list(source_cow_counts.values())):.1f} sequences/cow")
    print(f"      Median: {np.median(list(source_cow_counts.values())):.1f} sequences/cow")
    
    return sequences, metadata


def split_sequences(sequences: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Split sequences into train/val/test sets temporally.
    
    Args:
        sequences: List of sequence dicts
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
    
    Returns:
        Tuple of (train_sequences, val_sequences, test_sequences)
    """
    print(f"\nSplitting sequences (train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio})...")
    
    # Sort by first timestamp
    sequences_sorted = sorted(sequences, key=lambda x: x['timestamps'][0])
    
    n_total = len(sequences_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_seqs = sequences_sorted[:n_train]
    val_seqs = sequences_sorted[n_train:n_train + n_val]
    test_seqs = sequences_sorted[n_train + n_val:]
    
    print(f"   Train: {len(train_seqs):,} sequences ({len(train_seqs)/n_total*100:.1f}%)")
    print(f"   Val:   {len(val_seqs):,} sequences ({len(val_seqs)/n_total*100:.1f}%)")
    print(f"   Test:  {len(test_seqs):,} sequences ({len(test_seqs)/n_total*100:.1f}%)")
    
    return train_seqs, val_seqs, test_seqs


def save_dataset(sequences: List[Dict], metadata: Dict, output_path: Path, split_name: str):
    """
    Save sequences and metadata to disk.
    
    Args:
        sequences: List of sequence dicts
        metadata: Metadata dict
        output_path: Base output path
        split_name: Name of the split (train/val/test)
    """
    # Save sequences
    seq_file = output_path / f'{split_name}_sequences.pkl'
    with open(seq_file, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"   ✅ Saved {len(sequences)} sequences to {seq_file}")
    
    # Save metadata (only once for training set)
    if split_name == 'train':
        meta_file = output_path / 'metadata.json'
        # Convert sets to lists for JSON serialization
        metadata_serializable = {
            k: (list(v) if isinstance(v, set) else v)
            for k, v in metadata.items()
        }
        with open(meta_file, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        print(f"   ✅ Saved metadata to {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Next Cow Prediction Dataset'
    )
    parser.add_argument(
        '--rssi-dir',
        type=str,
        default='data/RSSI',
        help='Directory containing RSSI parquet files (default: data/RSSI)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='next_cow_data',
        help='Output directory for dataset (default: next_cow_data)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=10,
        help='Length of sequences (including target) (default: 10)'
    )
    parser.add_argument(
        '--window-seconds',
        type=int,
        default=60,
        help='Duration of time windows in seconds (default: 60)'
    )
    parser.add_argument(
        '--min-rssi',
        type=float,
        default=-100.0,
        help='Minimum RSSI threshold (default: -100.0)'
    )
    parser.add_argument(
        '--start-hours',
        type=int,
        default=9,
        help='Start hour for data filtering (default: 9)'
    )
    parser.add_argument(
        '--end-hours',
        type=int,
        default=23,
        help='End hour for data filtering (default: 23)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEXT COW PREDICTION DATASET GENERATOR")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"   RSSI Directory: {args.rssi_dir}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Sequence Length: {args.sequence_length}")
    print(f"   Time Window: {args.window_seconds}s")
    print(f"   Min RSSI: {args.min_rssi}")
    print(f"   Hours Filter: {args.start_hours}-{args.end_hours}")
    print(f"   Train/Val/Test Split: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Known problematic sensors to exclude
    excluded_sensors = {'3668', '3cf7', '3cfd', '366b', '3cf4', '3662'}
    
    # 1. Load RSSI data
    rssi_dir = Path(args.rssi_dir)
    rssi_df = load_rssi_data(rssi_dir, args.start_hours, args.end_hours)
    
    # 2. Create temporal snapshots
    snapshots = create_temporal_snapshots(rssi_df, args.window_seconds, args.min_rssi)
    
    # 3. Extract sequences
    sequences, metadata = extract_sequences(snapshots, args.sequence_length, excluded_sensors)
    
    # 4. Split into train/val/test
    train_seqs, val_seqs, test_seqs = split_sequences(
        sequences, args.train_ratio, args.val_ratio
    )
    
    # 5. Save datasets
    print(f"\nSaving datasets to {output_path}...")
    save_dataset(train_seqs, metadata, output_path, 'train')
    save_dataset(val_seqs, metadata, output_path, 'val')
    save_dataset(test_seqs, metadata, output_path, 'test')
    
    # 6. Save configuration
    config = {
        'created_at': datetime.now().isoformat(),
        'rssi_dir': str(args.rssi_dir),
        'sequence_length': args.sequence_length,
        'window_seconds': args.window_seconds,
        'min_rssi': args.min_rssi,
        'start_hours': args.start_hours,
        'end_hours': args.end_hours,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': 1 - args.train_ratio - args.val_ratio,
        'excluded_sensors': list(excluded_sensors),
        'num_sequences': {
            'train': len(train_seqs),
            'val': len(val_seqs),
            'test': len(test_seqs),
            'total': len(sequences)
        },
        'num_cows': metadata['num_cows']
    }
    
    config_file = output_path / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   ✅ Saved configuration to {config_file}")
    
    print("\n" + "=" * 70)
    print("✅ DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nDataset saved to: {output_path}")
    print(f"\nFiles created:")
    print(f"   - train_sequences.pkl ({len(train_seqs):,} sequences)")
    print(f"   - val_sequences.pkl ({len(val_seqs):,} sequences)")
    print(f"   - test_sequences.pkl ({len(test_seqs):,} sequences)")
    print(f"   - metadata.json (vocabulary and mappings)")
    print(f"   - config.json (generation parameters)")
    
    print(f"\n💡 To load the dataset in your model:")
    print(f"   import pickle")
    print(f"   with open('{output_path}/train_sequences.pkl', 'rb') as f:")
    print(f"       train_data = pickle.load(f)")
    print(f"\nEach sequence contains:")
    print(f"   - 'source_cow': ID of the cow being tracked")
    print(f"   - 'input_sequence': List of closest cows (length {args.sequence_length - 1})")
    print(f"   - 'target': The next closest cow to predict")
    print(f"   - 'input_sequence_ids': Integer IDs for input")
    print(f"   - 'target_id': Integer ID for target")


if __name__ == '__main__':
    main()
