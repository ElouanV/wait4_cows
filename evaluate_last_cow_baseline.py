"""
Evaluate a simple baseline for next cow prediction: predict the last cow seen.
This baseline assumes the next cow will be the same as the most recent cow in the sequence.
"""

import pickle
import numpy as np
from collections import Counter
from pathlib import Path

def load_sequences(data_dir, split):
    """Load sequences from pickle file."""
    filepath = Path(data_dir) / f'{split}_sequences.pkl'
    with open(filepath, 'rb') as f:
        sequences = pickle.load(f)
    return sequences

def evaluate_last_cow_baseline(sequences, split_name):
    """
    Evaluate the last cow baseline.
    Predicts that the next cow will be the last cow in the current sequence.
    """
    correct = 0
    total = len(sequences)
    
    # Track predictions for confusion analysis
    predictions = []
    
    for seq_data in sequences:
        # Get the input sequence and target
        input_seq = seq_data['input_sequence_ids']
        target = seq_data['target_id']
        
        # Get the last cow in the sequence (most recent)
        last_cow = input_seq[-1]
        
        # Predict this cow as the next one
        prediction = last_cow
        predictions.append(prediction)
        
        if prediction == target:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{split_name} Set Results:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, predictions

def analyze_baseline(sequences, predictions, split_name):
    """Analyze the baseline predictions."""
    
    # Count how often last cow == next cow
    last_eq_next = sum(1 for seq_data in sequences 
                       if seq_data['input_sequence_ids'][-1] == seq_data['target_id'])
    total = len(sequences)
    
    print(f"\n{split_name} Analysis:")
    print(f"  Times last_cow == next_cow: {last_eq_next} / {total} ({last_eq_next/total*100:.2f}%)")
    
    # Most common predictions
    pred_counter = Counter(predictions)
    print(f"\n  Top 10 most predicted cows:")
    for cow_id, count in pred_counter.most_common(10):
        print(f"    {cow_id}: {count} times ({count/total*100:.1f}%)")
    
    # Most common targets
    targets = [seq_data['target_id'] for seq_data in sequences]
    target_counter = Counter(targets)
    print(f"\n  Top 10 most common target cows:")
    for cow_id, count in target_counter.most_common(10):
        print(f"    {cow_id}: {count} times ({count/total*100:.1f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate last cow baseline')
    parser.add_argument('--dataset-dir', default='next_cow_data',
                       help='Path to next cow dataset directory')
    args = parser.parse_args()
    
    print("="*70)
    print("LAST COW BASELINE EVALUATION")
    print("="*70)
    print("\nBaseline: Predict that the next cow will be the last cow in the sequence")
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_dir}...")
    train_sequences = load_sequences(args.dataset_dir, 'train')
    val_sequences = load_sequences(args.dataset_dir, 'val')
    test_sequences = load_sequences(args.dataset_dir, 'test')
    
    print(f"\nDataset info:")
    print(f"  Train samples: {len(train_sequences)}")
    print(f"  Val samples: {len(val_sequences)}")
    print(f"  Test samples: {len(test_sequences)}")
    print(f"  Sequence length: {len(train_sequences[0]['input_sequence_ids'])}")
    
    # Evaluate on each split
    train_acc, train_preds = evaluate_last_cow_baseline(train_sequences, "Train")
    val_acc, val_preds = evaluate_last_cow_baseline(val_sequences, "Val")
    test_acc, test_preds = evaluate_last_cow_baseline(test_sequences, "Test")
    
    # Analyze predictions
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    analyze_baseline(train_sequences, train_preds, "Train")
    analyze_baseline(val_sequences, val_preds, "Val")
    analyze_baseline(test_sequences, test_preds, "Test")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nLast Cow Baseline Accuracy:")
    print(f"  Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val:   {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nComparison with neural models:")
    print("  Pure MLP:     51.60%")
    print("  MLP + Logic:  51.53%")
    print("  One-Hot MLP:  51.80%")
    print("  Logic Only:   25.70%")
    print(f"  Last Cow:     {test_acc*100:.2f}%")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
