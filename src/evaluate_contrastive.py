#!/usr/bin/env python3
"""
Evaluate Contrastive Embeddings
"""
import pickle
import argparse
from pathlib import Path
import numpy as np
from cow2vec import EmbeddingEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to cow_embeddings.pkl')
    parser.add_argument('--output-dir', type=str, default='cow_contrastive_out')
    args = parser.parse_args()
    
    print(f"Loading embeddings from {args.input}...")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
        
    embeddings = data['embeddings']
    id_to_cow = data['id_to_cow']
    cow_to_id = data['cow_to_id']
    
    print(f"Loaded {len(embeddings)} embeddings.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = EmbeddingEvaluator(
        embeddings,
        id_to_cow,
        cow_to_id
    )
    
    # t-SNE visualization
    print("Generating t-SNE...")
    evaluator.visualize_embeddings_tsne(
        save_path=output_dir / "embeddings_tsne.png",
        perplexity=min(30, len(embeddings) - 1)
    )
    
    # Similarity report
    print("Generating similarity report...")
    sample_cows = list(cow_to_id.keys())[:10]
    evaluator.generate_similarity_report(
        sample_cows,
        save_path=output_dir / "similarity_report.txt",
        top_k=5
    )
    
    print(f"Evaluation complete. Results in {output_dir}")

if __name__ == "__main__":
    main()