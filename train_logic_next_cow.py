#!/usr/bin/env python3
"""
Train Next Cow Prediction model with Logic Layer.

This script demonstrates training a next cow prediction model that uses
the interpretable LogiX logic layer for the final prediction layer.

Usage:
    python train_logic_next_cow.py --dataset-dir next_cow_data
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


class NextCowLogicModel(nn.Module):
    """LSTM-based model with interpretable Logic Layer for next cow prediction."""
    
    def __init__(self, num_cows, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.2, logic_dim=128,
                 use_logic_layer=True):
        super().__init__()
        
        self.use_logic_layer = use_logic_layer
        
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
        
        self.dropout = nn.Dropout(dropout)
        
        # Logic Layer architecture
        if use_logic_layer:
            # First project to logic_dim
            self.fc_pre_logic = nn.Linear(hidden_dim, logic_dim)
            self.bn = nn.BatchNorm1d(logic_dim)
            
            # Logic layer for interpretable prediction
            self.logic_layer = LogicalLayer(
                in_features=logic_dim,
                out_features=num_cows,
                dummy_phi_in=False,
                use_weight_sigma=True,
                use_weight_exp=True
            )
        else:
            # Standard MLP baseline
            self.fc = nn.Linear(hidden_dim, num_cows)
    
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
        
        if self.use_logic_layer:
            # Project to logic dimension
            x = self.fc_pre_logic(last_hidden)
            x = self.bn(x)
            x = torch.relu(x)
            
            # Pass through logic layer (outputs probabilities directly)
            probs = self.logic_layer(x)
            
            # Convert to logits for cross entropy
            logits = torch.log(probs + 1e-8)
        else:
            # Standard prediction
            logits = self.fc(last_hidden)
        
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
        """Extract interpretable logic rules (only available with logic layer)."""
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
        
        # Regularization losses from logic layer
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
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
        description='Train Next Cow Prediction with Logic Layer'
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
        default=128,
        help='Embedding dimension (default: 128)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='LSTM hidden dimension (default: 256)'
    )
    parser.add_argument(
        '--logic-dim',
        type=int,
        default=128,
        help='Logic layer dimension (default: 128)'
    )
    parser.add_argument(
        '--reg-weight',
        type=float,
        default=0.001,
        help='Regularization weight for logic layer (default: 0.001)'
    )
    parser.add_argument(
        '--entropy-weight',
        type=float,
        default=0.0001,
        help='Entropy weight for logic layer (default: 0.0001)'
    )
    parser.add_argument(
        '--no-logic',
        action='store_true',
        help='Disable logic layer (use standard MLP baseline)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='logic_next_cow_output',
        help='Output directory for results (default: logic_next_cow_output)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEXT COW PREDICTION WITH LOGIC LAYER")
    print("=" * 70)
    
    # Check if dataset exists
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
    
    use_logic = not args.no_logic
    model_type = "Logic Layer" if use_logic else "MLP Baseline"
    print(f"Model type: {model_type}")
    
    model = NextCowLogicModel(
        num_cows=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        logic_dim=args.logic_dim,
        use_logic_layer=use_logic
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save config
    config = {
        'model_type': model_type,
        'use_logic_layer': use_logic,
        'num_cows': vocab_size,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
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
        if use_logic:
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
    
    # Extract and save logic rules (if using logic layer)
    if use_logic:
        print("\n" + "=" * 70)
        print("EXTRACTING LOGIC RULES")
        print("=" * 70)
        
        try:
            # Load best model
            model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
            model.eval()
            
            # Extract rules
            rules = model.extract_rules()
            
            # Save rules
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
            
            # Print some statistics
            total_rules = sum(len(rule_set) for rule_set in rules)
            avg_rules = total_rules / len(rules) if len(rules) > 0 else 0
            print(f"   Total rules: {total_rules}")
            print(f"   Average rules per neuron: {avg_rules:.2f}")
            
            # Show sample rules
            print(f"\n   Sample rules (first 3 neurons):")
            for i in range(min(3, len(rules))):
                num_rules = len(rules[i])
                print(f"   Neuron {i}: {num_rules} rules")
                if num_rules > 0:
                    sample_rules = list(rules[i])[:3]
                    for j, rule in enumerate(sample_rules):
                        print(f"      Rule {j+1}: features {rule}")
        
        except Exception as e:
            print(f"\n⚠️  Could not extract rules: {e}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"   - best_model.pt: Best model weights")
    print(f"   - config.json: Training configuration")
    print(f"   - history.json: Training history")
    if use_logic:
        print(f"   - logic_rules.json: Extracted logic rules")


if __name__ == '__main__':
    main()
