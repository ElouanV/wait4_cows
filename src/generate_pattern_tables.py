#!/usr/bin/env python3
"""
Generate presentation-ready tables from pattern mining results.
Creates clean LaTeX and CSV tables for presentations.
"""

import pandas as pd
from pathlib import Path


def create_pattern_tables(poi_name, results_dir):
    """Generate clean tables for a POI."""
    results_path = Path(results_dir)
    
    # Find the result files
    edge_files = list(results_path.glob(f"{poi_name}_*_edge_patterns.csv"))
    itemset_files = list(results_path.glob(f"{poi_name}_*_frequent_itemsets.csv"))
    
    if not edge_files or not itemset_files:
        print(f"No results found for {poi_name}")
        return
    
    edge_file = edge_files[0]
    itemset_file = itemset_files[0]
    
    # Load data
    edges = pd.read_csv(edge_file)
    itemsets = pd.read_csv(itemset_file)
    
    print(f"\n{'='*70}")
    print(f"{poi_name.upper().replace('_', ' ')} - PATTERN MINING RESULTS")
    print(f"{'='*70}\n")
    
    # Frequent Itemsets Table
    print("TABLE 1: Most Frequent Cows")
    print("-" * 50)
    
    # Filter single cows only
    single_cows = itemsets[itemsets['length'] == 1].copy()
    single_cows['Support (%)'] = (single_cows['support'] * 100).round(1)
    single_cows = single_cows.rename(columns={'itemsets': 'Cow ID'})
    single_cows = single_cows[['Cow ID', 'Support (%)']].head(10)
    
    print(single_cows.to_string(index=False))
    print()
    
    # Save as CSV
    output_csv = results_path / f"{poi_name}_table1_frequent_cows.csv"
    single_cows.to_csv(output_csv, index=False)
    print(f"✓ Saved: {output_csv.name}")
    
    # Save as LaTeX
    output_tex = results_path / f"{poi_name}_table1_frequent_cows.tex"
    with open(output_tex, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Most Frequent Cows at " + poi_name.replace('_', ' ').title() + "}\n")
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\hline\n")
        f.write("Cow ID & Support (\\%) \\\\\n")
        f.write("\\hline\n")
        for _, row in single_cows.iterrows():
            f.write(f"{row['Cow ID']} & {row['Support (%)']:.1f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"✓ Saved: {output_tex.name}")
    print()
    
    # Cow Pairs Table
    print("TABLE 2: Most Frequent Cow Pairs (Together at POI)")
    print("-" * 60)
    
    pairs = edges.head(10).copy()
    pairs['Pair'] = pairs['cow1'] + ' + ' + pairs['cow2']
    pairs['Occurrences'] = pairs['count']
    pairs['Frequency (%)'] = (pairs['frequency'] * 100).round(2)
    pairs = pairs[['Pair', 'Occurrences', 'Frequency (%)']]
    
    print(pairs.to_string(index=False))
    print()
    
    # Save as CSV
    output_csv = results_path / f"{poi_name}_table2_cow_pairs.csv"
    pairs.to_csv(output_csv, index=False)
    print(f"✓ Saved: {output_csv.name}")
    
    # Save as LaTeX
    output_tex = results_path / f"{poi_name}_table2_cow_pairs.tex"
    with open(output_tex, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Most Frequent Cow Pairs at " + poi_name.replace('_', ' ').title() + "}\n")
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\hline\n")
        f.write("Cow Pair & Occurrences & Frequency (\\%) \\\\\n")
        f.write("\\hline\n")
        for _, row in pairs.iterrows():
            f.write(f"{row['Pair']} & {row['Occurrences']} & {row['Frequency (%)']:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"✓ Saved: {output_tex.name}")
    print()
    
    # Multi-cow groups (if any)
    multi_cows = itemsets[itemsets['length'] > 1].copy()
    if len(multi_cows) > 0:
        print("TABLE 3: Frequent Cow Groups (2+ cows)")
        print("-" * 60)
        
        multi_cows['Group'] = multi_cows['itemsets']
        multi_cows['Size'] = multi_cows['length']
        multi_cows['Support (%)'] = (multi_cows['support'] * 100).round(2)
        multi_cows = multi_cows[['Group', 'Size', 'Support (%)']].head(10)
        
        print(multi_cows.to_string(index=False))
        print()
        
        # Save as CSV
        output_csv = results_path / f"{poi_name}_table3_cow_groups.csv"
        multi_cows.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv.name}")
        
        # Save as LaTeX
        output_tex = results_path / f"{poi_name}_table3_cow_groups.tex"
        with open(output_tex, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Frequent Cow Groups at " + poi_name.replace('_', ' ').title() + "}\n")
            f.write("\\begin{tabular}{lrr}\n")
            f.write("\\hline\n")
            f.write("Cow Group & Size & Support (\\%) \\\\\n")
            f.write("\\hline\n")
            for _, row in multi_cows.iterrows():
                f.write(f"{row['Group']} & {row['Size']} & {row['Support (%)']:.2f} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        print(f"✓ Saved: {output_tex.name}")
        print()


def main():
    """Generate tables for all POIs."""
    
    # Brush
    create_pattern_tables("brush_proximity", "brush_experiment")
    
    # Lactation
    create_pattern_tables("lactation_proximity", "lactation_experiment")
    
    # Water spot (first sensor)
    create_pattern_tables("water_spot_proximity", "water_spot_experiment")
    
    print("\n" + "="*70)
    print("ALL TABLES GENERATED!")
    print("="*70)
    print("\nFiles created:")
    print("  - *_table1_frequent_cows.csv/tex")
    print("  - *_table2_cow_pairs.csv/tex")
    print("  - *_table3_cow_groups.csv/tex")
    print("\nUse .csv files for Excel/Google Sheets")
    print("Use .tex files for LaTeX presentations")


if __name__ == "__main__":
    main()
