#!/usr/bin/env python3
"""
Visualize Cow2Vec embedding quality and patterns.
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Load embeddings
with open('cow2vec_embeddings/cow_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']
id_to_cow = data['id_to_cow']
cow_to_id = data['cow_to_id']

# Normalize for cosine similarity
normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Compute full similarity matrix
n_cows = len(cow_to_id)
similarity_matrix = np.dot(normalized, normalized.T)

# Create dataframe
cow_ids = [id_to_cow[i] for i in range(n_cows)]
sim_df = pd.DataFrame(similarity_matrix, index=cow_ids, columns=cow_ids)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Full similarity heatmap
ax1 = axes[0, 0]
sns.heatmap(sim_df, cmap='RdYlBu_r', center=0, vmin=-0.5, vmax=1.0,
            square=True, linewidths=0.1, cbar_kws={'label': 'Cosine Similarity'},
            ax=ax1)
ax1.set_title('Cow-to-Cow Similarity Matrix', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Cow ID', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cow ID', fontsize=11, fontweight='bold')

# 2. Similarity distribution
ax2 = axes[0, 1]
# Get upper triangle (excluding diagonal)
triu_indices = np.triu_indices(n_cows, k=1)
similarities = similarity_matrix[triu_indices]

ax2.hist(similarities, bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
ax2.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(similarities):.3f}')
ax2.axvline(np.median(similarities), color='orange', linestyle='--', linewidth=2,
            label=f'Median: {np.median(similarities):.3f}')
ax2.set_xlabel('Cosine Similarity', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Pairwise Similarities', fontsize=14, fontweight='bold', pad=10)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 3. Top similar pairs
ax3 = axes[1, 0]
top_pairs = []
for i in range(n_cows):
    for j in range(i+1, n_cows):
        top_pairs.append((cow_ids[i], cow_ids[j], similarity_matrix[i, j]))

top_pairs.sort(key=lambda x: x[2], reverse=True)
top_10 = top_pairs[:10]

pair_labels = [f"{p[0]}\n↔\n{p[1]}" for p in top_10]
pair_scores = [p[2] for p in top_10]

bars = ax3.barh(range(len(pair_labels)), pair_scores, color='coral', edgecolor='darkred', linewidth=1.5)
ax3.set_yticks(range(len(pair_labels)))
ax3.set_yticklabels(pair_labels, fontsize=9)
ax3.set_xlabel('Similarity Score', fontsize=11, fontweight='bold')
ax3.set_title('Top 10 Most Similar Cow Pairs', fontsize=14, fontweight='bold', pad=10)
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, pair_scores)):
    ax3.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{score:.3f}', va='center', fontweight='bold', fontsize=9)

# 4. Average similarity per cow
ax4 = axes[1, 1]
avg_similarities = []
for i in range(n_cows):
    # Average similarity to all other cows (excluding self)
    sims = similarity_matrix[i, :]
    avg_sim = (np.sum(sims) - sims[i]) / (n_cows - 1)
    avg_similarities.append((cow_ids[i], avg_sim))

avg_similarities.sort(key=lambda x: x[1], reverse=True)

# Show top and bottom 10
top_bottom = avg_similarities[:10] + avg_similarities[-10:]
labels = [x[0] for x in top_bottom]
scores = [x[1] for x in top_bottom]
colors = ['green']*10 + ['red']*10

bars = ax4.barh(range(len(labels)), scores, color=colors, alpha=0.6, edgecolor='black', linewidth=1)
ax4.set_yticks(range(len(labels)))
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel('Average Similarity to Others', fontsize=11, fontweight='bold')
ax4.set_title('Most/Least Socially Similar Cows', fontsize=14, fontweight='bold', pad=10)
ax4.axvline(np.mean([s for _, s in avg_similarities]), color='blue', 
           linestyle='--', linewidth=2, alpha=0.7, label='Overall mean')
ax4.legend(fontsize=9)
ax4.grid(axis='x', alpha=0.3)

# Add separator line
ax4.axhline(9.5, color='black', linestyle='-', linewidth=2)
ax4.text(0.02, 9.5, 'Most Social', ha='left', va='bottom', fontweight='bold', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax4.text(0.02, 9.5, 'Least Social', ha='left', va='top', fontweight='bold', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.suptitle('Cow2Vec Embedding Quality Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = Path('cow2vec_embeddings/embedding_quality_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved quality analysis: {output_path}")
plt.close()

# Print statistics
print("\n" + "="*70)
print("EMBEDDING QUALITY STATISTICS")
print("="*70)
print(f"Number of cows: {n_cows}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"\nSimilarity Statistics:")
print(f"  Mean similarity: {np.mean(similarities):.4f}")
print(f"  Median similarity: {np.median(similarities):.4f}")
print(f"  Std deviation: {np.std(similarities):.4f}")
print(f"  Min similarity: {np.min(similarities):.4f}")
print(f"  Max similarity: {np.max(similarities):.4f}")
print(f"\nTop 3 most similar pairs:")
for i, (cow1, cow2, score) in enumerate(top_pairs[:3], 1):
    print(f"  {i}. {cow1} ↔ {cow2}: {score:.4f}")
print(f"\nTop 3 most socially similar cows (high avg similarity):")
for i, (cow, score) in enumerate(avg_similarities[:3], 1):
    print(f"  {i}. {cow}: {score:.4f}")
print(f"\nTop 3 least socially similar cows (low avg similarity):")
for i, (cow, score) in enumerate(avg_similarities[-3:], 1):
    print(f"  {i}. {cow}: {score:.4f}")
print("="*70)
