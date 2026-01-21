#!/usr/bin/env python3
"""
Train Next Cow Prediction with pure MLP architecture (no LSTM).

Since sequences are fixed length, we can use a simpler MLP instead of LSTM.

Usage:
    python train_mlp_next_cow.py --dataset-dir next_cow_data
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

from src.models.logic_layer import LogicalLayer


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


class NextCowMLPModel(nn.Module):
    """Pure MLP model for next cow prediction (no LSTM)."""
    
    def __init__(self, num_cows, embedding_dim=64, hidden_dims=[256, 256], 
                 dropout=0.3, use_logic_layer=False, logic_dim=128, skip_mlp=False):
        super().__init__()
        
        self.use_logic_layer = use_logic_layer
        self.skip_mlp = skip_mlp
        self.seq_length = 9  # Fixed sequence length
        
        # Cow embeddings
        self.embedding = nn.Embedding(num_cows, embedding_dim)
        
        # Input dimension: seq_length * embedding_dim
        input_dim = self.seq_length * embedding_dim
        
        if skip_mlp:
            # Skip MLP layers - use one-hot bag-of-cows representation
            if not use_logic_layer:
                raise ValueError("skip_mlp=True requires use_logic_layer=True")
            
            # For logic layer: use one-hot representation (sum of one-hot vectors)
            # Input will be num_cows dimensional (frequency of each cow in sequence)
            self.logic_layer = LogicalLayer(
                in_features=num_cows,  # One-hot bag representation
                out_features=num_cows,
                dummy_phi_in=False,
                use_weight_sigma=True,
                use_weight_exp=True
            )
            self.mlp = None  # No MLP
            self.fc_pre_logic = None  # No projection layer
            self.bn_pre_logic = None  # No batch norm
            self.embedding = None  # Don't use embeddings for logic-only
        else:
            # Build MLP layers
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            self.mlp = nn.Sequential(*layers)
            
            # Output layer
            if use_logic_layer:
                # Logic layer for interpretable prediction
                self.fc_pre_logic = nn.Linear(prev_dim, logic_dim)
                self.bn_pre_logic = nn.BatchNorm1d(logic_dim)
                
                self.logic_layer = LogicalLayer(
                    in_features=logic_dim,
                    out_features=num_cows,
                    dummy_phi_in=False,
                    use_weight_sigma=True,
                    use_weight_exp=True
                )
            else:
                # Standard linear output
                self.fc_out = nn.Linear(prev_dim, num_cows)
    
    def forward(self, source_cow_id, input_sequence):
        """
        Args:
            source_cow_id: (batch_size,) - source cow IDs
            input_sequence: (batch_size, seq_len) - sequence of cow IDs
        
        Returns:
            logits: (batch_size, num_cows) - prediction logits
        """
        batch_size = input_sequence.size(0)
        
        if self.skip_mlp:
            # Create one-hot bag-of-cows representation
            # Sum of one-hot vectors = frequency count of each cow in sequence
            num_cows = self.logic_layer.in_features
            
            # One-hot encode: (batch_size, seq_len, num_cows)
            one_hot = torch.zeros(
                batch_size, 
                self.seq_length, 
                num_cows, 
                device=input_sequence.device
            )
            one_hot.scatter_(2, input_sequence.unsqueeze(2), 1)
            
            # Sum across sequence: (batch_size, num_cows)
            x = one_hot.sum(dim=1)
            
            # Pass through logic layer
            probs = self.logic_layer(x)
            logits = torch.log(probs + 1e-8)
        else:
            # Embed input sequence
            embedded = self.embedding(input_sequence)  # (batch_size, seq_len, embedding_dim)
            
            # Flatten sequence embeddings
            x = embedded.view(batch_size, -1)  # (batch_size, seq_len * embedding_dim)
        
            # Pass through MLP
            x = self.mlp(x)  # (batch_size, hidden_dim)
            
            if self.use_logic_layer:
                # Project to logic dimension
                x = self.fc_pre_logic(x)
                x = self.bn_pre_logic(x)
                x = torch.relu(x)
                
                # Pass through logic layer (outputs probabilities)
                probs = self.logic_layer(x)
                logits = torch.log(probs + 1e-8)
            else:
                # Standard prediction
                logits = self.fc_out(x)
        
        return logits
    
    def get_regularization_loss(self):
        """Get regularization losses from logic layer."""
        if not self.use_logic_layer:
            return 0.0
        return self.logic_layer.reg_loss
    
    def get_entropy_loss(self):
        """Get entropy loss from logic layer."""
        if not self.use_logic_layer:
            return 0.0
        return self.logic_layer.entropy_loss
    
    def extract_rules(self):
        """Extract interpretable logic rules."""
        if not self.use_logic_layer:
            raise ValueError("Rules can only be extracted when using logic layer")
        return self.logic_layer.extract_rules()


def train_epoch(model, dataloader, criterion, optimizer, device, 
                reg_weight=0.001, entropy_weight=0.0001):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_entropy_loss = 0
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
        
        # Cross entropy loss
        ce_loss = criterion(logits, targets)
        
        # Regularization losses
        reg_loss = model.get_regularization_loss()
        entropy_loss = model.get_entropy_loss()
        
        # Total loss
        loss = ce_loss + reg_weight * reg_loss + entropy_weight * entropy_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        if isinstance(reg_loss, torch.Tensor):
            total_reg_loss += reg_loss.item()
        if isinstance(entropy_loss, torch.Tensor):
            total_entropy_loss += entropy_loss.item()
            
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'ce_loss': total_ce_loss / len(dataloader),
        'reg_loss': total_reg_loss / len(dataloader),
        'entropy_loss': total_entropy_loss / len(dataloader),
        'accuracy': correct / total
    }


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
        description='Train Next Cow Prediction with Pure MLP'
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
        default=128,
        help='Batch size (default: 128)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs (default: 20)'
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
        default=64,
        help='Embedding dimension (default: 64)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        default=[256, 256],
        help='Hidden layer dimensions (default: 256 256)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    parser.add_argument(
        '--use-logic',
        action='store_true',
        help='Use logic layer for interpretable predictions'
    )
    parser.add_argument(
        '--logic-dim',
        type=int,
        default=128,
        help='Logic layer dimension (default: 128)'
    )
    parser.add_argument(
        '--skip-mlp',
        action='store_true',
        help='Skip MLP layers and train logic layer alone (requires --use-logic)'
    )
    parser.add_argument(
        '--reg-weight',
        type=float,
        default=0.001,
        help='Regularization weight (default: 0.001)'
    )
    parser.add_argument(
        '--entropy-weight',
        type=float,
        default=0.0001,
        help='Entropy weight (default: 0.0001)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='pure_mlp_next_cow_output',
        help='Output directory (default: pure_mlp_next_cow_output)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_mlp and not args.use_logic:
        print("\n❌ Error: --skip-mlp requires --use-logic")
        return
    
    print("=" * 70)
    if args.skip_mlp:
        print("NEXT COW PREDICTION - LOGIC LAYER ONLY")
    elif args.use_logic:
        print("NEXT COW PREDICTION - MLP + LOGIC LAYER")
    else:
        print("NEXT COW PREDICTION - PURE MLP ARCHITECTURE")
    print("=" * 70)
    
    # Check dataset
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"\n❌ Dataset not found at {dataset_dir}")
        print(f"\n💡 Generate the dataset first:")
        print(f"   python generate_next_cow_dataset.py --output-dir {dataset_dir}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
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
    
    if args.skip_mlp:
        model_type = "Logic Layer Only"
    elif args.use_logic:
        model_type = "MLP + Logic Layer"
    else:
        model_type = "Pure MLP"
    
    print(f"Model type: {model_type}")
    if not args.skip_mlp:
        print(f"Hidden layers: {args.hidden_dims}")
    
    model = NextCowMLPModel(
        num_cows=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_logic_layer=args.use_logic,
        logic_dim=args.logic_dim,
        skip_mlp=args.skip_mlp
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save config
    config = {
        'model_type': model_type,
        'architecture': 'pure_mlp',
        'use_logic_layer': args.use_logic,
        'skip_mlp': args.skip_mlp,
        'num_cows': vocab_size,
        'embedding_dim': args.embedding_dim,
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'logic_dim': args.logic_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'reg_weight': args.reg_weight,
        'entropy_weight': args.entropy_weight,
        'epochs': args.epochs,
        'dataset_dir': str(dataset_dir),
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_ce_loss': [],
        'train_reg_loss': [],
        'train_entropy_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top_5_acc': [],
    }
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            reg_weight=args.reg_weight,
            entropy_weight=args.entropy_weight
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        print(f"\n   Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        if args.use_logic:
            print(f"   Train CE:   {train_metrics['ce_loss']:.4f} | Reg: {train_metrics['reg_loss']:.4f} | Entropy: {train_metrics['entropy_loss']:.4f}")
        print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"   Val Top-5:  {val_metrics['top_5_accuracy']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_ce_loss'].append(train_metrics['ce_loss'])
        history['train_reg_loss'].append(train_metrics['reg_loss'])
        history['train_entropy_loss'].append(train_metrics['entropy_loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_top_5_acc'].append(val_metrics['top_5_accuracy'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"   ✅ Saved best model (val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f})")
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Extract logic rules if applicable
    if args.use_logic:
        print("\n" + "=" * 70)
        print("EXTRACTING LOGIC RULES")
        print("=" * 70)
        
        try:
            model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
            model.eval()
            
            rules = model.extract_rules()
            
            rules_serializable = []
            for i, rule_set in enumerate(rules):
                rules_serializable.append({
                    'output_neuron': i,
                    'num_rules': len(rule_set),
                    'rules': [list(rule) for rule in rule_set]
                })
            
            with open(output_dir / 'logic_rules.json', 'w') as f:
                json.dump(rules_serializable, f, indent=2)
            
            print(f"\n✅ Extracted logic rules for {len(rules)} output neurons")
            total_rules = sum(len(rule_set) for rule_set in rules)
            print(f"   Total rules: {total_rules}")
            print(f"   Average rules per neuron: {total_rules / len(rules):.2f}")
        
        except Exception as e:
            print(f"\n⚠️  Could not extract rules: {e}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
