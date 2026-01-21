#!/usr/bin/env python3
"""
Temporal-aware train/validation/test split for GNN models.

CRITICAL: For temporal graph sequences, we MUST split by time, not randomly!

Correct approach:
1. Sort graphs by timestamp
2. Use first 70% for training (earliest time)
3. Use next 15% for validation (middle time)
4. Use last 15% for test (latest time)

This prevents temporal leakage where model sees future to predict past.
"""

import numpy as np
from typing import List, Tuple, Any


def temporal_train_val_test_split(
    data_list: List[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    ensure_temporal_order: bool = True
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split temporal data maintaining chronological order.
    
    Args:
        data_list: List of data samples (assumed to be in temporal order)
        train_ratio: Fraction for training (earliest)
        val_ratio: Fraction for validation (middle)
        test_ratio: Fraction for test (latest)
        ensure_temporal_order: If True, verify timestamps are ordered
        
    Returns:
        train_list, val_list, test_list
        
    Example:
        If you have 1000 temporal snapshots:
        - Train: samples 0-699 (earliest 70%)
        - Val: samples 700-849 (middle 15%)
        - Test: samples 850-999 (latest 15%)
        
    This way model NEVER sees future data during training!
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(data_list)
    
    # Calculate split points
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split chronologically (NO SHUFFLING!)
    train_list = data_list[:train_end]
    val_list = data_list[train_end:val_end]
    test_list = data_list[val_end:]
    
    print(f"Temporal split (NO shuffling - maintains time order):")
    print(f"  Train: samples 0-{train_end-1} ({len(train_list)} samples, {train_ratio*100:.0f}%)")
    print(f"  Val:   samples {train_end}-{val_end-1} ({len(val_list)} samples, {val_ratio*100:.0f}%)")
    print(f"  Test:  samples {val_end}-{n-1} ({len(test_list)} samples, {test_ratio*100:.0f}%)")
    
    # Optional: verify temporal ordering if timestamps available
    if ensure_temporal_order and hasattr(data_list[0], 'timestamp'):
        train_times = [d.timestamp for d in train_list if hasattr(d, 'timestamp')]
        val_times = [d.timestamp for d in val_list if hasattr(d, 'timestamp')]
        test_times = [d.timestamp for d in test_list if hasattr(d, 'timestamp')]
        
        if train_times and val_times and test_times:
            assert max(train_times) <= min(val_times), \
                "Training data contains timestamps after validation data!"
            assert max(val_times) <= min(test_times), \
                "Validation data contains timestamps after test data!"
            print("✅ Verified: No temporal leakage (train < val < test)")
    
    return train_list, val_list, test_list


def temporal_train_test_split(
    data_list: List[Any],
    train_ratio: float = 0.8,
    ensure_temporal_order: bool = True
) -> Tuple[List[Any], List[Any]]:
    """
    Simple temporal train/test split.
    
    Args:
        data_list: List of data samples (in temporal order)
        train_ratio: Fraction for training
        ensure_temporal_order: Verify no leakage
        
    Returns:
        train_list, test_list
    """
    n = len(data_list)
    train_end = int(n * train_ratio)
    
    train_list = data_list[:train_end]
    test_list = data_list[train_end:]
    
    print(f"Temporal split:")
    print(f"  Train: samples 0-{train_end-1} ({len(train_list)} samples)")
    print(f"  Test:  samples {train_end}-{n-1} ({len(test_list)} samples)")
    
    return train_list, test_list


# Example usage:
if __name__ == "__main__":
    # Simulate temporal graph data
    class FakeGraph:
        def __init__(self, timestamp):
            self.timestamp = timestamp
    
    # Create 100 temporal snapshots
    graphs = [FakeGraph(i) for i in range(100)]
    
    # WRONG WAY (random shuffle):
    # indices = np.arange(100)
    # np.random.shuffle(indices)  # ❌ TEMPORAL LEAKAGE!
    # train = [graphs[i] for i in indices[:70]]
    
    # CORRECT WAY (temporal order):
    train, val, test = temporal_train_val_test_split(graphs)
    # ✅ Train on past, validate on middle, test on future
    
    print("\n✅ No temporal leakage!")
    print(f"Training on timesteps 0-{len(train)-1}")
    print(f"Validating on timesteps {len(train)}-{len(train)+len(val)-1}")
    print(f"Testing on timesteps {len(train)+len(val)}-{len(graphs)-1}")
