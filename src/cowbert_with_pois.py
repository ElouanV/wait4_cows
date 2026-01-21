#!/usr/bin/env python3
"""
CowBERT with POI (Point of Interest) Integration

This version of CowBERT includes fixed points of interest (like brushes, water spots)
in the training sequences. This allows the model to learn behavioral patterns related
to resource usage.

Key differences from standard CowBERT:
- POIs (366b=brush, 3cf7=water, etc.) are included in vocabulary
- Cow-to-POI proximities are part of the training sequences
- POI tokens can appear in sequences but are never masked (like [EGO] token)

Usage:
    python src/cowbert_with_pois.py \\
        --input network_sequence/network_sequence_rssi-68_*.pkl \\
        --output-dir cowbert_out_with_pois \\
        --poi-ids 366b,3cf7 \\
        --embedding-dim 16 \\
        --epochs 200
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from original cowbert
from cowbert import *  # noqa: F403, F401


class TopKProximitySequenceExtractorWithPOIs(TopKProximitySequenceExtractor):  # noqa: F405
    """
    Extended extractor that includes POI nodes in sequences.
    POIs are treated as special behavioral markers.
    """
    def __init__(self, temporal_graphs, top_k=1, excluded_sensors=None, poi_ids=None):
        # Don't exclude POIs from extraction
        self.poi_ids = set(poi_ids or [])
        
        # Remove POIs from excluded sensors if they were there
        if excluded_sensors:
            excluded_sensors = set(excluded_sensors) - self.poi_ids
        
        super().__init__(temporal_graphs, top_k, excluded_sensors)
    
    def build_vocabulary(self):
        """Build vocabulary including POIs as special tokens."""
        print("Building cow vocabulary (including POIs)...")
        self.cow_vocabulary = set()
        
        for graph_info in tqdm(self.temporal_graphs, desc="Scanning graphs"):  # noqa: F405
            G = graph_info['graph']
            nodes = [n for n in G.nodes() if n not in self.excluded_sensors]
            self.cow_vocabulary.update(nodes)
        
        # Separate cows and POIs
        self.poi_nodes = sorted(self.cow_vocabulary & self.poi_ids)
        self.cow_nodes = sorted(self.cow_vocabulary - self.poi_ids)
        
        # Build vocabulary: cows first, then POIs
        self.cow_vocabulary = self.cow_nodes + self.poi_nodes
        self.cow_to_id = {cow: idx for idx, cow in enumerate(self.cow_vocabulary)}
        self.id_to_cow = {idx: cow for cow, idx in self.cow_to_id.items()}
        
        print(f"   Found {len(self.cow_nodes)} cows and {len(self.poi_nodes)} POIs")
        print(f"   POIs: {self.poi_nodes}")
        return self.cow_vocabulary


class CowBERTDatasetWithPOIs(CowBERTDataset):  # noqa: F405
    """
    Extended dataset that handles POI tokens.
    POIs are never masked (similar to [EGO] token).
    """
    def __init__(
        self,
        sequences: Dict[str, List[List[str]]],  # noqa: F405
        cow_to_id: Dict[str, int],  # noqa: F405
        poi_ids: set,
        max_len: int = 128,
        mask_prob: float = 0.15
    ):
        self.poi_token_ids = {cow_to_id[poi] for poi in poi_ids if poi in cow_to_id}
        super().__init__(sequences, cow_to_id, max_len, mask_prob)
    
    def __getitem__(self, idx):
        """
        Return (masked_input, target_labels)
        Sequence structure: [EGO_COW, CLS, neighbor1, neighbor2, ..., SEP]
        POI tokens are never masked (like [EGO]).
        """
        ego_cow_id, seq = self.samples[idx]
        
        # Add special tokens
        input_ids = [ego_cow_id, self.CLS_TOKEN] + seq + [self.SEP_TOKEN]
        labels = [-100] * len(input_ids)  # -100 ignored by CrossEntropyLoss
        
        # Apply masking (skip EGO at position 0, CLS at position 1, SEP at end, and POIs)
        for i in range(2, len(input_ids) - 1):
            # Skip POI tokens (don't mask them)
            if input_ids[i] in self.poi_token_ids:
                continue
                
            prob = random.random()  # noqa: F405
            if prob < self.mask_prob:
                # 80% replace with MASK
                if random.random() < 0.8:  # noqa: F405
                    labels[i] = input_ids[i]
                    input_ids[i] = self.MASK_TOKEN
                # 10% replace with random token (but not a POI)
                elif random.random() < 0.5:  # noqa: F405
                    labels[i] = input_ids[i]
                    # Sample from cow tokens only (not POIs or special tokens)
                    num_cow_tokens = min(self.poi_token_ids) if self.poi_token_ids else self.vocab_size
                    input_ids[i] = random.randint(0, num_cow_tokens - 1)  # noqa: F405
                # 10% keep original
                else:
                    labels[i] = input_ids[i]
        
        # Padding
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.PAD_TOKEN] * padding_len
            labels = labels + [-100] * padding_len
            
        return torch.tensor(input_ids), torch.tensor(labels)  # noqa: F405


def main():
    parser = argparse.ArgumentParser(description="Train CowBERT with POIs")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='cowbert_with_pois')
    parser.add_argument('--poi-ids', type=str, default='366b,3cf7',
                        help="Comma-separated list of POI sensor IDs")
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print("="*70, flush=True)
    print("COWBERT WITH POIs: BEHAVIORAL EMBEDDINGS", flush=True)
    print("="*70, flush=True)
    
    # Parse POI IDs
    poi_ids = set(args.poi_ids.split(','))
    print(f"\n📍 Including POIs: {poi_ids}")
    
    # 1. Load Data
    with open(args.input, 'rb') as f:
        temporal_graphs = pickle.load(f)  # noqa: F405
        
    # Define excluded sensors (remove POIs from exclusion list)
    excluded_sensors = {'3668', '3cfd', '3cf4', '3662'}  # Removed 366b, 3cf7
    
    # Use extractor with POIs
    extractor = TopKProximitySequenceExtractorWithPOIs(
        temporal_graphs,
        top_k=args.top_k,
        excluded_sensors=excluded_sensors,
        poi_ids=poi_ids
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
        split_idx = int(len(seqs) * split_ratio)
        train_sequences[cow] = seqs[:split_idx]
        val_sequences[cow] = seqs[split_idx:]
    
    train_dataset = CowBERTDatasetWithPOIs(
        train_sequences,
        cow_to_id,
        poi_ids,
        max_len=128
    )
    
    val_dataset = CowBERTDatasetWithPOIs(
        val_sequences,
        cow_to_id,
        poi_ids,
        max_len=128
    )
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val sequences")

    train_dataloader = DataLoader(  # noqa: F405
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(  # noqa: F405
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # --- Run Baseline ---
    evaluate_baseline(val_dataloader)  # noqa: F405
    # --------------------
    
    # 3. Initialize Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # noqa: F405
    model = CowBERTModel(  # noqa: F405
        vocab_size=train_dataset.full_vocab_size,
        d_model=args.embedding_dim,
        nhead=4,
        num_layers=2
    )
    
    # 4. Train
    history = train_cowbert(  # noqa: F405
        model,
        train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # 5. Save Results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot with best epoch marker (from updated cowbert.py)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # noqa: F405
    
    ax1.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Validation Loss')
        best_epoch = history.get('best_epoch', 0)
        if best_epoch > 0:
            ax1.axvline(x=best_epoch-1, color='red', linestyle='--',
                       linewidth=2, alpha=0.7,
                       label=f'Best Epoch ({best_epoch})')
    ax1.set_title('CowBERT+POIs Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    if history['val_acc']:
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        if best_epoch > 0:
            ax2.axvline(x=best_epoch-1, color='red', linestyle='--',
                       linewidth=2, alpha=0.7,
                       label=f'Best Epoch ({best_epoch})')
            ax2.scatter([best_epoch-1], [history['best_val_acc']],
                       color='red', s=100, zorder=5,
                       label=f'Best: {history["best_val_acc"]:.2%}')
    ax2.set_title('CowBERT+POIs Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()  # noqa: F405
    plt.savefig(output_dir / 'training_metrics.png', dpi=300)  # noqa: F405
    
    # Extract embeddings
    embeddings = model.get_cow_embeddings()
    
    config = {
        'embedding_dim': args.embedding_dim,
        'epochs': args.epochs,
        'model': 'CowBERT+POIs',
        'poi_ids': list(poi_ids)
    }
    
    save_embeddings(  # noqa: F405
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
    
    evaluator = EmbeddingEvaluator(  # noqa: F405
        embeddings,
        id_to_cow,
        cow_to_id
    )
    
    evaluator.visualize_embeddings_tsne(
        save_path=output_dir / "embeddings_tsne.png",
        perplexity=30
    )
    
    sample_cows = [c for c in list(cow_to_id.keys())[:10] if c not in poi_ids]
    evaluator.generate_similarity_report(
        sample_cows,
        save_path=output_dir / "similarity_report.txt"
    )
    
    # Final evaluation
    print("\n" + "="*70, flush=True)
    print("FINAL EVALUATION ON VALIDATION SET", flush=True)
    print("="*70, flush=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()  # noqa: F405
    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():  # noqa: F405
        for input_ids, labels in tqdm(val_dataloader, desc="Evaluating"):  # noqa: F405
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            output = model(input_ids, src_mask=None)
            output_flat = output.view(-1, output.size(-1))
            labels_flat = labels.view(-1)
            
            loss = criterion(output_flat, labels_flat)
            acc = calculate_accuracy(output_flat, labels_flat)  # noqa: F405
            
            total_val_loss += loss.item()
            total_val_acc += acc
    
    final_val_loss = total_val_loss / len(val_dataloader)
    final_val_acc = total_val_acc / len(val_dataloader)
    
    # Per-cow accuracy
    print("\n" + "="*70, flush=True)
    print("PER-COW ACCURACY ANALYSIS", flush=True)
    print("="*70, flush=True)
    
    per_cow_results = calculate_per_cow_accuracy(  # noqa: F405
        model, val_dataloader, device, id_to_cow
    )
    
    sorted_cows = sorted(
        per_cow_results.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    print("\n🔝 TOP 10 EASIEST TO PREDICT:")
    for i, (cow_name, stats) in enumerate(sorted_cows[:10], 1):
        poi_marker = " [POI]" if cow_name in poi_ids else ""
        print(f"  {i}. {cow_name}{poi_marker}: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    print("\n🔻 BOTTOM 10 HARDEST TO PREDICT:")
    for i, (cow_name, stats) in enumerate(sorted_cows[-10:], 1):
        poi_marker = " [POI]" if cow_name in poi_ids else ""
        print(f"  {i}. {cow_name}{poi_marker}: {stats['accuracy']:.2%} "
              f"({stats['correct']}/{stats['total']})")
    
    accuracies = [stats['accuracy'] for stats in per_cow_results.values()]
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  - Mean accuracy: {np.mean(accuracies):.2%}")  # noqa: F405
    print(f"  - Median accuracy: {np.median(accuracies):.2%}")  # noqa: F405
    print(f"  - Std deviation: {np.std(accuracies):.2%}")  # noqa: F405
    
    plot_per_cow_accuracy(  # noqa: F405
        per_cow_results,
        output_dir / 'per_cow_accuracy.png',
        top_n=None
    )
    
    per_cow_df = pd.DataFrame([  # noqa: F405
        {
            'cow_id': cow_name,
            'is_poi': cow_name in poi_ids,
            'accuracy': stats['accuracy'],
            'correct': stats['correct'],
            'total': stats['total']
        }
        for cow_name, stats in sorted_cows
    ])
    per_cow_csv = output_dir / 'per_cow_accuracy.csv'
    per_cow_df.to_csv(per_cow_csv, index=False)
    print(f"✅ Saved per-cow results: {per_cow_csv}")
    
    # Print results
    print(f"\n{'='*70}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Final Validation Loss: {final_val_loss:.4f}", flush=True)
    print(f"Final Validation Accuracy: {final_val_acc:.4f}", flush=True)
    print(f"Best Epoch: {history.get('best_epoch', 'N/A')}", flush=True)
    print(f"Best Val Accuracy: {history.get('best_val_acc', 0.0):.4f}", flush=True)
    print(f"{'='*70}", flush=True)
    
    print(f"\n✅ Training complete! Results in: {output_dir.absolute()}")
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
