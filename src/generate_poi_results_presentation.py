#!/usr/bin/env python3
"""
Generate presentation-ready plots and tables from POI analysis results.

Creates:
- Time spent near each POI (bar charts)
- Top visitors comparison across POIs
- Pattern mining results visualization
- Summary tables for presentations
- Network visualizations

Usage:
    python src/generate_poi_results_presentation.py
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_poi_data(base_dir: Path):
    """Load all POI analysis data."""
    # Find the most recent files
    time_stats_files = list(base_dir.glob("*_time_stats.csv"))
    metadata_files = list(base_dir.glob("*_metadata.json"))
    frequent_itemsets_files = list(base_dir.glob("*_frequent_itemsets.csv"))
    
    if not time_stats_files or not metadata_files:
        return None
    
    # Get most recent
    time_stats_file = sorted(time_stats_files)[-1]
    metadata_file = sorted(metadata_files)[-1]
    
    # Load data
    time_stats = pd.read_csv(time_stats_file)
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Load frequent itemsets if available
    frequent_itemsets = None
    if frequent_itemsets_files:
        freq_file = sorted(frequent_itemsets_files)[-1]
        try:
            frequent_itemsets = pd.read_csv(freq_file)
        except:
            pass
    
    return {
        'time_stats': time_stats,
        'metadata': metadata,
        'frequent_itemsets': frequent_itemsets,
        'poi_name': metadata.get('poi_name', metadata.get('zone_name', 'unknown')),
        'poi_id': metadata.get('poi_id', metadata.get('poi_ids', 'unknown'))
    }


def create_top_visitors_chart(poi_data_dict, output_dir, top_n=15):
    """Create bar chart of top visitors for each POI."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    poi_names = ['Brush', 'Water Spot Zone', 'Lactation Machine']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (poi_key, poi_name, color) in enumerate(zip(['brush', 'water_spot_zone', 'lactation'], poi_names, colors)):
        if poi_key not in poi_data_dict:
            continue
        
        data = poi_data_dict[poi_key]['time_stats'].head(top_n).copy()
        data = data.sort_values('total_time_minutes')  # Sort ascending for horizontal bar
        
        ax = axes[idx]
        bars = ax.barh(data['cow_id'], data['total_time_minutes'], color=color, alpha=0.7)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, data['total_time_minutes'])):
            ax.text(val + 5, i, f'{val:.0f}m', va='center', fontsize=8)
        
        ax.set_xlabel('Time (minutes)', fontweight='bold')
        ax.set_ylabel('Cow ID', fontweight='bold')
        ax.set_title(f'{poi_name}\nTop {top_n} Visitors', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_visitors_comparison.png', bbox_inches='tight')
    plt.savefig(output_dir / 'top_visitors_comparison.pdf', bbox_inches='tight')
    print(f"✅ Created: top_visitors_comparison.png/pdf")
    plt.close()


def create_summary_table(poi_data_dict, output_dir):
    """Create compact summary table for presentation."""
    summary_data = []
    
    for poi_key in ['brush', 'water_spot_zone', 'lactation']:
        if poi_key not in poi_data_dict:
            continue
        
        data = poi_data_dict[poi_key]
        metadata = data['metadata']
        time_stats = data['time_stats']
        
        # Get statistics
        stats = metadata.get('statistics', {})
        
        # Top 3 visitors
        top3 = time_stats.head(3)
        top_visitors = ', '.join([f"{row['cow_id']} ({row['total_time_minutes']:.0f}m)" 
                                  for _, row in top3.iterrows()])
        
        summary_data.append({
            'POI': poi_key.replace('_', ' ').title(),
            'ID': str(metadata.get('poi_id', metadata.get('poi_ids', 'N/A'))),
            'Total Snapshots': f"{stats.get('total_snapshots', 0):,}",
            'Snapshots w/ Activity': f"{stats.get('snapshots_with_neighbors', 0):,}",
            'Unique Visitors': len(data['time_stats']),
            'Activity Rate': f"{stats.get('snapshots_with_neighbors', 0) / max(stats.get('total_snapshots', 1), 1) * 100:.1f}%",
            'Top 3 Visitors': top_visitors
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    df.to_csv(output_dir / 'poi_summary_table.csv', index=False)
    
    # Create styled table image
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='left', loc='center',
                     colWidths=[0.12, 0.08, 0.12, 0.14, 0.1, 0.1, 0.34])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'poi_summary_table.png', bbox_inches='tight', dpi=300)
    print(f"✅ Created: poi_summary_table.png and .csv")
    plt.close()
    
    return df


def create_time_distribution_plot(poi_data_dict, output_dir):
    """Create distribution of visit durations across POIs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    poi_names = ['Brush', 'Water Spot Zone', 'Lactation Machine']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (poi_key, poi_name, color) in enumerate(zip(['brush', 'water_spot_zone', 'lactation'], poi_names, colors)):
        if poi_key not in poi_data_dict:
            continue
        
        data = poi_data_dict[poi_key]['time_stats']['total_time_minutes']
        
        ax = axes[idx]
        ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add statistics
        mean_time = data.mean()
        median_time = data.median()
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}m')
        ax.axvline(median_time, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_time:.1f}m')
        
        ax.set_xlabel('Total Time (minutes)', fontweight='bold')
        ax.set_ylabel('Number of Cows', fontweight='bold')
        ax.set_title(f'{poi_name}\nVisit Duration Distribution', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_distribution.png', bbox_inches='tight')
    plt.savefig(output_dir / 'time_distribution.pdf', bbox_inches='tight')
    print(f"✅ Created: time_distribution.png/pdf")
    plt.close()


def create_activity_timeline(poi_data_dict, output_dir):
    """Create timeline showing activity levels across POIs."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    poi_configs = [
        ('brush', 'Brush', '#2ecc71'),
        ('water_spot_zone', 'Water Spot Zone', '#3498db'),
        ('lactation', 'Lactation Machine', '#e74c3c')
    ]
    
    for idx, (poi_key, poi_name, color) in enumerate(poi_configs):
        if poi_key not in poi_data_dict:
            continue
        
        # Load summary file
        base_dir = Path('brush_experiment' if poi_key == 'brush' else 
                       'water_spot_experiment' if 'water' in poi_key else 
                       'lactation_experiment')
        
        summary_files = list(base_dir.glob(f"*_summary.csv"))
        if not summary_files:
            continue
        
        summary = pd.read_csv(sorted(summary_files)[-1])
        summary['timestamp'] = pd.to_datetime(summary['timestamp'])
        
        # Plot activity (number of neighbors over time)
        ax = axes[idx]
        ax.fill_between(range(len(summary)), summary['num_neighbors'], 
                        alpha=0.6, color=color, label=poi_name)
        ax.plot(summary['num_neighbors'], color=color, linewidth=1, alpha=0.8)
        
        ax.set_ylabel('# Cows Nearby', fontweight='bold')
        ax.set_title(f'{poi_name} Activity Over Time', fontweight='bold', loc='left')
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add average line
        avg = summary['num_neighbors'].mean()
        ax.axhline(avg, color='red', linestyle='--', linewidth=1.5, 
                  label=f'Avg: {avg:.1f} cows')
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time Snapshot Index', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activity_timeline.png', bbox_inches='tight')
    plt.savefig(output_dir / 'activity_timeline.pdf', bbox_inches='tight')
    print(f"✅ Created: activity_timeline.png/pdf")
    plt.close()


def create_visitor_overlap_analysis(poi_data_dict, output_dir):
    """Analyze and visualize which cows visit multiple POIs."""
    # Get visitor sets for each POI
    visitors = {}
    for poi_key in ['brush', 'water_spot_zone', 'lactation']:
        if poi_key in poi_data_dict:
            visitors[poi_key] = set(poi_data_dict[poi_key]['time_stats']['cow_id'].values)
    
    if len(visitors) < 2:
        return
    
    # Create Venn diagram data
    from matplotlib_venn import venn3
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    brush_set = visitors.get('brush', set())
    water_set = visitors.get('water_spot_zone', set())
    lact_set = visitors.get('lactation', set())
    
    venn = venn3([brush_set, water_set, lact_set], 
                 ('Brush', 'Water Spot', 'Lactation'),
                 ax=ax)
    
    # Color the circles
    if venn.get_patch_by_id('100'):
        venn.get_patch_by_id('100').set_color('#2ecc71')
        venn.get_patch_by_id('100').set_alpha(0.5)
    if venn.get_patch_by_id('010'):
        venn.get_patch_by_id('010').set_color('#3498db')
        venn.get_patch_by_id('010').set_alpha(0.5)
    if venn.get_patch_by_id('001'):
        venn.get_patch_by_id('001').set_color('#e74c3c')
        venn.get_patch_by_id('001').set_alpha(0.5)
    
    ax.set_title('Visitor Overlap Across POIs', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visitor_overlap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'visitor_overlap.pdf', bbox_inches='tight')
    print(f"✅ Created: visitor_overlap.png/pdf")
    plt.close()
    
    # Create detailed overlap table
    overlap_stats = []
    
    all_visitors = brush_set | water_set | lact_set
    
    for cow in sorted(all_visitors):
        visits = []
        if cow in brush_set:
            visits.append('Brush')
        if cow in water_set:
            visits.append('Water')
        if cow in lact_set:
            visits.append('Lactation')
        
        overlap_stats.append({
            'Cow ID': cow,
            'POIs Visited': ', '.join(visits),
            'Count': len(visits)
        })
    
    df = pd.DataFrame(overlap_stats)
    df = df.sort_values('Count', ascending=False)
    df.to_csv(output_dir / 'visitor_overlap_details.csv', index=False)
    print(f"✅ Created: visitor_overlap_details.csv")


def create_compact_stats_table(poi_data_dict, output_dir):
    """Create ultra-compact stats table for slides."""
    data = []
    
    for poi_key, poi_display in [('brush', 'Brush'), 
                                  ('water_spot_zone', 'Water Spot'), 
                                  ('lactation', 'Lactation')]:
        if poi_key not in poi_data_dict:
            continue
        
        stats = poi_data_dict[poi_key]['metadata']['statistics']
        time_stats = poi_data_dict[poi_key]['time_stats']
        
        data.append({
            'POI': poi_display,
            'Visitors': len(time_stats),
            'Activity %': f"{stats['snapshots_with_neighbors'] / stats['total_snapshots'] * 100:.0f}%",
            'Avg Time/Cow': f"{time_stats['total_time_minutes'].mean():.0f}m",
            'Top Visitor': f"{time_stats.iloc[0]['cow_id']} ({time_stats.iloc[0]['total_time_minutes']:.0f}m)"
        })
    
    df = pd.DataFrame(data)
    
    # Save as both CSV and formatted table
    df.to_csv(output_dir / 'compact_summary.csv', index=False)
    
    # Create minimal table image
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    
    plt.savefig(output_dir / 'compact_summary_table.png', bbox_inches='tight', dpi=300)
    print(f"✅ Created: compact_summary_table.png and .csv")
    plt.close()


def main():
    print("\n" + "="*70)
    print("GENERATING PRESENTATION MATERIALS")
    print("="*70 + "\n")
    
    # Load all POI data
    poi_data = {}
    
    dirs = {
        'brush': Path('brush_experiment'),
        'water_spot_zone': Path('water_spot_experiment'),
        'lactation': Path('lactation_experiment')
    }
    
    for poi_key, poi_dir in dirs.items():
        if poi_dir.exists():
            data = load_poi_data(poi_dir)
            if data:
                poi_data[poi_key] = data
                print(f"✅ Loaded {poi_key} data")
    
    if not poi_data:
        print("❌ No POI data found!")
        return
    
    # Create output directory
    output_dir = Path('presentation_results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Generating visualizations...\n")
    
    # Generate all visualizations
    create_compact_stats_table(poi_data, output_dir)
    create_summary_table(poi_data, output_dir)
    create_top_visitors_chart(poi_data, output_dir, top_n=15)
    create_time_distribution_plot(poi_data, output_dir)
    create_activity_timeline(poi_data, output_dir)
    
    try:
        create_visitor_overlap_analysis(poi_data, output_dir)
    except ImportError:
        print("⚠️  Skipped visitor overlap (install matplotlib-venn)")
    
    print(f"\n{'='*70}")
    print(f"✅ ALL VISUALIZATIONS CREATED!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  📊 compact_summary_table.png - Ultra-compact stats table")
    print(f"  📊 poi_summary_table.png - Detailed summary table")
    print(f"  📊 top_visitors_comparison.png - Top visitors for each POI")
    print(f"  📊 time_distribution.png - Visit duration distributions")
    print(f"  📊 activity_timeline.png - Activity over time")
    print(f"  📊 visitor_overlap.png - Venn diagram of visitors")
    print(f"\n  📄 All plots also saved as PDF for high-quality printing")
    print(f"  📄 CSV files for custom table creation\n")


if __name__ == "__main__":
    main()
