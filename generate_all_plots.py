#!/usr/bin/env python3
"""
Generate all presentation plots from experimental results
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import shutil
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Create output directory
output_dir = Path('presentation_plots_final')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("GENERATING PRESENTATION PLOTS")
print("="*80)
print(f"\n📁 Output directory: {output_dir.absolute()}\n")

# ============================================================================
# 1. Load Next Cow Prediction Results
# ============================================================================
print("1️⃣  Loading Next Cow Prediction experiments...")

experiments = {
    'Pure MLP': 'pure_mlp_next_cow_output',
    'MLP + Logic': 'mlp_next_cow_output',
    'Logic Only': 'logic_only_onehot_lr01_3000ep',
    'One-Hot MLP': 'onehot_mlp_output',
}

results = {}
for name, path in experiments.items():
    exp_path = Path(path)
    if exp_path.exists():
        try:
            with open(exp_path / 'config.json', 'r') as f:
                config = json.load(f)
            
            history_file = exp_path / 'history.json'
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                results[name] = {'config': config, 'history': history}
                print(f"   ✓ {name}: {len(history['train_loss'])} epochs")
        except Exception as e:
            print(f"   ✗ {name}: {e}")

print(f"\n   ✅ Loaded {len(results)} experiments\n")

# ============================================================================
# 2. Plot Training Curves
# ============================================================================
print("2️⃣  Generating training curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Next Cow Prediction: Training Comparison', fontsize=16, fontweight='bold', y=0.995)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Training Loss
for idx, (name, data) in enumerate(results.items()):
    if 'train_loss' in data['history']:
        values = data['history']['train_loss']
        if len(values) > 500:
            step = max(1, len(values) // 500)
            epochs = list(range(0, len(values), step))
            values = values[::step]
        else:
            epochs = list(range(len(values)))
        axes[0, 0].plot(epochs, values, label=name, linewidth=2, color=colors[idx], alpha=0.9)

axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss', fontweight='bold')
axes[0, 0].legend(framealpha=0.9)
axes[0, 0].grid(True, alpha=0.3)

# Validation Loss
for idx, (name, data) in enumerate(results.items()):
    if 'val_loss' in data['history']:
        values = data['history']['val_loss']
        if len(values) > 500:
            step = max(1, len(values) // 500)
            epochs = list(range(0, len(values), step))
            values = values[::step]
        else:
            epochs = list(range(len(values)))
        axes[0, 1].plot(epochs, values, label=name, linewidth=2, color=colors[idx], alpha=0.9)

axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Validation Loss', fontweight='bold')
axes[0, 1].legend(framealpha=0.9)
axes[0, 1].grid(True, alpha=0.3)

# Training Accuracy
for idx, (name, data) in enumerate(results.items()):
    if 'train_acc' in data['history']:
        values = data['history']['train_acc']
        if len(values) > 500:
            step = max(1, len(values) // 500)
            epochs = list(range(0, len(values), step))
            values = values[::step]
        else:
            epochs = list(range(len(values)))
        axes[1, 0].plot(epochs, values, label=name, linewidth=2, color=colors[idx], alpha=0.9)

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Training Accuracy', fontweight='bold')
axes[1, 0].legend(framealpha=0.9)
axes[1, 0].grid(True, alpha=0.3)

# Validation Accuracy
for idx, (name, data) in enumerate(results.items()):
    if 'val_acc' in data['history']:
        values = data['history']['val_acc']
        if len(values) > 500:
            step = max(1, len(values) // 500)
            epochs = list(range(0, len(values), step))
            values = values[::step]
        else:
            epochs = list(range(len(values)))
        axes[1, 1].plot(epochs, values, label=name, linewidth=2, color=colors[idx], alpha=0.9)

axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Validation Accuracy', fontweight='bold')
axes[1, 1].legend(framealpha=0.9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '01_next_cow_training_curves.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / '01_next_cow_training_curves.pdf', bbox_inches='tight')
plt.close()

print("   ✅ Saved: 01_next_cow_training_curves.png/.pdf\n")

# ============================================================================
# 3. Performance Comparison
# ============================================================================
print("3️⃣  Generating performance comparison...")

performance_data = []
for name, data in results.items():
    if data['history']:
        best_val_acc = max(data['history']['val_acc']) if 'val_acc' in data['history'] else 0
        final_train_acc = data['history']['train_acc'][-1] if 'train_acc' in data['history'] else 0
        
        performance_data.append({
            'Model': name,
            'Validation': best_val_acc * 100,
            'Training': final_train_acc * 100
        })

df_perf = pd.DataFrame(performance_data)

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df_perf))
width = 0.35

bars1 = ax.bar(x - width/2, df_perf['Training'], width, label='Training Accuracy', 
               color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, df_perf['Validation'], width, label='Validation Accuracy', 
               color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Next Cow Prediction: Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_perf['Model'], fontsize=11)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(df_perf['Training'].max(), df_perf['Validation'].max()) * 1.15)

plt.tight_layout()
plt.savefig(output_dir / '02_performance_comparison.png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / '02_performance_comparison.pdf', bbox_inches='tight')
plt.close()

df_perf.to_csv(output_dir / '02_performance_comparison.csv', index=False)
print("   ✅ Saved: 02_performance_comparison.png/.pdf/.csv")
print(f"\n   📊 Performance Summary:")
print(df_perf.to_string(index=False))
print()

# ============================================================================
# 4. Temporal Graph Analysis
# ============================================================================
print("\n4️⃣  Loading temporal experiments...")

experiments_temporal = {
    'Brush': 'brush_experiment',
    'Water Spot': 'water_spot_experiment',
    'Lactation': 'lactation_experiment'
}

temporal_data = {}
for name, exp_dir in experiments_temporal.items():
    exp_path = Path(exp_dir)
    if exp_path.exists():
        summary_files = list(exp_path.glob('*_summary.csv'))
        if summary_files:
            try:
                df = pd.read_csv(summary_files[0])
                temporal_data[name] = df
                print(f"   ✓ {name}: {len(df)} time windows")
            except Exception as e:
                print(f"   ✗ {name}: {e}")

print(f"\n   ✅ Loaded {len(temporal_data)} temporal experiments\n")

# Plot temporal evolution
if temporal_data:
    print("5️⃣  Generating temporal graph evolution...")
    
    fig, axes = plt.subplots(len(temporal_data), 2, figsize=(14, 5*len(temporal_data)))
    if len(temporal_data) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Temporal Graph Evolution: POI Experiments', fontsize=16, fontweight='bold')
    
    for idx, (name, df) in enumerate(temporal_data.items()):
        # Plot edges
        if 'num_edges' in df.columns:
            axes[idx, 0].plot(df['num_edges'], linewidth=2, color='#1f77b4')
            axes[idx, 0].fill_between(range(len(df)), df['num_edges'], alpha=0.3, color='#1f77b4')
            axes[idx, 0].set_title(f'{name}: Number of Edges Over Time', fontsize=12, fontweight='bold')
            axes[idx, 0].set_xlabel('Time Window', fontsize=11)
            axes[idx, 0].set_ylabel('Number of Edges', fontsize=11)
            axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot nodes
        if 'num_nodes' in df.columns:
            axes[idx, 1].plot(df['num_nodes'], linewidth=2, color='#ff7f0e')
            axes[idx, 1].fill_between(range(len(df)), df['num_nodes'], alpha=0.3, color='#ff7f0e')
            axes[idx, 1].set_title(f'{name}: Active Nodes Over Time', fontsize=12, fontweight='bold')
            axes[idx, 1].set_xlabel('Time Window', fontsize=11)
            axes[idx, 1].set_ylabel('Number of Active Nodes', fontsize=11)
            axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_temporal_graph_evolution.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / '03_temporal_graph_evolution.pdf', bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: 03_temporal_graph_evolution.png/.pdf\n")

# ============================================================================
# 6. Copy Existing Figures
# ============================================================================
print("6️⃣  Copying existing figures...")

source_dirs = ['presentation_figures', 'presentation_results']
copied = 0

for source_dir in source_dirs:
    source_path = Path(source_dir)
    if source_path.exists():
        for file in source_path.glob('*.*'):
            if file.suffix in ['.png', '.pdf', '.csv']:
                dest = output_dir / f'existing_{file.name}'
                shutil.copy2(file, dest)
                copied += 1

print(f"   ✅ Copied {copied} existing figures\n")

# ============================================================================
# 7. Generate Summary
# ============================================================================
print("7️⃣  Generating summary report...")

summary_text = f"""
PRESENTATION PLOTS SUMMARY
{'='*80}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {output_dir.absolute()}

EXPERIMENTS ANALYZED:
{'-'*80}

1. NEXT COW PREDICTION
"""

for name, data in results.items():
    if data['history']:
        best_val = max(data['history']['val_acc'])
        summary_text += f"   - {name:20s}: {best_val*100:5.2f}% validation accuracy\n"

summary_text += "\n2. TEMPORAL GRAPH EXPERIMENTS\n"
for name, df in temporal_data.items():
    avg_nodes = df['num_nodes'].mean() if 'num_nodes' in df else 0
    avg_edges = df['num_edges'].mean() if 'num_edges' in df else 0
    summary_text += f"   - {name:20s}: {avg_nodes:.1f} avg nodes, {avg_edges:.1f} avg edges\n"

summary_text += f"""
{'-'*80}

GENERATED FILES:
"""

for file in sorted(output_dir.glob('*')):
    size = file.stat().st_size / 1024
    summary_text += f"  - {file.name:50s} ({size:>8.1f} KB)\n"

summary_text += f"""
{'='*80}
"""

with open(output_dir / 'README.txt', 'w') as f:
    f.write(summary_text)

print(summary_text)
print("\n" + "="*80)
print("✅ ALL PLOTS GENERATED SUCCESSFULLY")
print("="*80)
print(f"\n📁 All files saved to: {output_dir.absolute()}")
print("\nReady for presentation! 🎉")
