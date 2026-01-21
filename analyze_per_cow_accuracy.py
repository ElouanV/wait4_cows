#!/usr/bin/env python3
"""
Analyze per-cow accuracy results from CowBERT training.

This script loads the per-cow accuracy CSV generated during CowBERT training
and creates additional visualizations and analysis.

Usage:
    python analyze_per_cow_accuracy.py --input cowbert_out/per_cow_accuracy.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_per_cow_results(csv_path):
    """Load per-cow accuracy results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def plot_accuracy_distribution(df, save_path=None):
    """Plot distribution of per-cow accuracies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(df['accuracy'], bins=20, color='steelblue', 
             edgecolor='black', alpha=0.7)
    ax1.axvline(df['accuracy'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["accuracy"].mean():.1%}')
    ax1.axvline(df['accuracy'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {df["accuracy"].median():.1%}')
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Cows', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Per-Cow Accuracies', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    ax2.boxplot([df['accuracy']], vert=True, widths=0.5)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Cow Accuracy Box Plot', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['All Cows'])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved distribution plot: {save_path}")
    else:
        plt.show()
    
    return fig


def plot_accuracy_vs_samples(df, save_path=None):
    """Plot accuracy vs number of samples per cow."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plot with size based on total samples
    scatter = ax.scatter(df['total'], df['accuracy'], 
                        s=100, alpha=0.6, c=df['accuracy'],
                        cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Accuracy', fontsize=12, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(df['total'], df['accuracy'], 1)
    p = np.poly1d(z)
    ax.plot(df['total'], p(df['total']), "r--", alpha=0.8, linewidth=2,
            label=f'Trend: y={z[0]:.6f}x+{z[1]:.3f}')
    
    ax.set_xlabel('Number of Predictions (Sample Size)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Cow Accuracy vs Sample Size', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved accuracy vs samples plot: {save_path}")
    else:
        plt.show()
    
    return fig


def print_detailed_stats(df):
    """Print detailed statistics about per-cow accuracy."""
    print("\n" + "="*70)
    print("DETAILED PER-COW ACCURACY STATISTICS")
    print("="*70)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  - Number of cows: {len(df)}")
    print(f"  - Mean accuracy: {df['accuracy'].mean():.2%}")
    print(f"  - Median accuracy: {df['accuracy'].median():.2%}")
    print(f"  - Std deviation: {df['accuracy'].std():.2%}")
    print(f"  - Min accuracy: {df['accuracy'].min():.2%} "
          f"(cow: {df.loc[df['accuracy'].idxmin(), 'cow_id']})")
    print(f"  - Max accuracy: {df['accuracy'].max():.2%} "
          f"(cow: {df.loc[df['accuracy'].idxmax(), 'cow_id']})")
    
    print(f"\n📈 Sample Size Statistics:")
    print(f"  - Total predictions: {df['total'].sum():,}")
    print(f"  - Mean samples per cow: {df['total'].mean():.1f}")
    print(f"  - Median samples per cow: {df['total'].median():.1f}")
    print(f"  - Min samples: {df['total'].min()} "
          f"(cow: {df.loc[df['total'].idxmin(), 'cow_id']})")
    print(f"  - Max samples: {df['total'].max()} "
          f"(cow: {df.loc[df['total'].idxmax(), 'cow_id']})")
    
    # Correlation between accuracy and sample size
    correlation = df['accuracy'].corr(df['total'])
    print(f"\n🔗 Correlation between accuracy and sample size: {correlation:.3f}")
    
    # Quartile analysis
    print(f"\n📊 Accuracy Quartiles:")
    quartiles = df['accuracy'].quantile([0.25, 0.5, 0.75])
    print(f"  - Q1 (25th percentile): {quartiles[0.25]:.2%}")
    print(f"  - Q2 (50th percentile): {quartiles[0.5]:.2%}")
    print(f"  - Q3 (75th percentile): {quartiles[0.75]:.2%}")
    
    # Group cows by accuracy ranges
    print(f"\n🎯 Cows by Accuracy Range:")
    ranges = [
        (0.0, 0.5, "Very Low (<50%)"),
        (0.5, 0.6, "Low (50-60%)"),
        (0.6, 0.7, "Medium (60-70%)"),
        (0.7, 0.8, "Good (70-80%)"),
        (0.8, 0.9, "Very Good (80-90%)"),
        (0.9, 1.0, "Excellent (>90%)")
    ]
    
    for min_acc, max_acc, label in ranges:
        count = len(df[(df['accuracy'] >= min_acc) & (df['accuracy'] < max_acc)])
        percentage = 100 * count / len(df)
        print(f"  - {label}: {count} cows ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-cow accuracy from CowBERT training"
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help="Path to per_cow_accuracy.csv file"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Directory to save plots (default: same as input CSV)"
    )
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"❌ Error: File not found: {csv_path}")
        return
    
    print(f"Loading per-cow accuracy data from: {csv_path}")
    df = load_per_cow_results(csv_path)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print detailed statistics
    print_detailed_stats(df)
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    plot_accuracy_distribution(
        df, 
        save_path=output_dir / 'per_cow_accuracy_distribution.png'
    )
    
    plot_accuracy_vs_samples(
        df,
        save_path=output_dir / 'per_cow_accuracy_vs_samples.png'
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Analysis complete! Plots saved to: {output_dir.absolute()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
