#!/usr/bin/env python3
"""
Example script showing how to extract cow embeddings from CowLSTM model.

The cow embedding in CowLSTM is the static embedding from the embedding layer
(the first layer of the model). This is a learned lookup table that maps each
cow ID to a vector of size embed_dim.
"""

import json
import numpy as np
import torch
from pathlib import Path

def load_cowlstm_embeddings(model_dir: str):
    """
    Load cow embeddings from a trained CowLSTM model.
    
    Args:
        model_dir: Path to the directory containing model outputs
        
    Returns:
        embeddings: numpy array of shape [num_cows, embed_dim]
        cow_ids: list of cow IDs corresponding to each row in embeddings
        metadata: dictionary with model configuration
    """
    model_path = Path(model_dir)
    
    # Load embeddings (already extracted during training)
    embeddings = np.load(model_path / 'embeddings.npy')
    
    # Load metadata to get cow ID mapping
    with open(model_path / 'embedding_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    cow_ids = metadata['cow_ids']
    
    print(f"Loaded embeddings for {len(cow_ids)} cows")
    print(f"Embedding dimension: {metadata['embedding_dim']}")
    print(f"Model type: {metadata['model']}")
    print(f"Best validation accuracy: {metadata['best_val_acc']:.4f}")
    
    return embeddings, cow_ids, metadata


def get_embedding_for_cow(cow_id: str, embeddings: np.ndarray, cow_ids: list) -> np.ndarray:
    """
    Get the embedding vector for a specific cow.
    
    Args:
        cow_id: The cow identifier (e.g., '3cf5')
        embeddings: The embeddings array [num_cows, embed_dim]
        cow_ids: List of cow IDs
        
    Returns:
        embedding: Vector of shape [embed_dim]
    """
    if cow_id not in cow_ids:
        raise ValueError(f"Cow ID {cow_id} not found in vocabulary")
    
    idx = cow_ids.index(cow_id)
    return embeddings[idx]


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def find_most_similar_cows(target_cow: str, embeddings: np.ndarray, cow_ids: list, top_k: int = 5):
    """
    Find the most similar cows to a target cow based on embedding similarity.
    
    Args:
        target_cow: The cow ID to find similar cows for
        embeddings: The embeddings array [num_cows, embed_dim]
        cow_ids: List of cow IDs
        top_k: Number of most similar cows to return
        
    Returns:
        List of (cow_id, similarity_score) tuples
    """
    target_emb = get_embedding_for_cow(target_cow, embeddings, cow_ids)
    
    similarities = []
    for i, cow_id in enumerate(cow_ids):
        if cow_id == target_cow:
            continue  # Skip self
        sim = compute_similarity(target_emb, embeddings[i])
        similarities.append((cow_id, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


# Example usage
if __name__ == '__main__':
    # Load CowLSTM embeddings
    print("="*70)
    print("LOADING COWLSTM EMBEDDINGS")
    print("="*70)
    embeddings, cow_ids, metadata = load_cowlstm_embeddings('/home/elouan/wait4_data/cowlstm_embeddings_dim16')
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Number of cows: {len(cow_ids)}")
    print(f"\nFirst 10 cow IDs: {cow_ids[:10]}")
    
    # Example 1: Get embedding for a specific cow
    print("\n" + "="*70)
    print("EXAMPLE 1: GET EMBEDDING FOR SPECIFIC COW")
    print("="*70)
    target_cow = cow_ids[0]  # Take first cow as example
    cow_emb = get_embedding_for_cow(target_cow, embeddings, cow_ids)
    print(f"\nEmbedding for cow {target_cow}:")
    print(f"Shape: {cow_emb.shape}")
    print(f"Vector: {cow_emb}")
    
    # Example 2: Find most similar cows
    print("\n" + "="*70)
    print("EXAMPLE 2: FIND MOST SIMILAR COWS")
    print("="*70)
    similar_cows = find_most_similar_cows(target_cow, embeddings, cow_ids, top_k=5)
    print(f"\nTop 5 cows most similar to {target_cow}:")
    for cow_id, similarity in similar_cows:
        print(f"  {cow_id}: {similarity:.4f}")
    
    # Example 3: Embedding statistics
    print("\n" + "="*70)
    print("EXAMPLE 3: EMBEDDING STATISTICS")
    print("="*70)
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")
    print(f"  Min: {embeddings.min():.4f}")
    print(f"  Max: {embeddings.max():.4f}")
    
    # Compute pairwise similarity matrix
    print(f"\nComputing pairwise similarities...")
    n = len(cow_ids)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = compute_similarity(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    print(f"Average pairwise similarity: {(similarity_matrix.sum() / (n*(n-1))):.4f}")
    print(f"Max pairwise similarity: {similarity_matrix.max():.4f}")
    print(f"Min pairwise similarity: {similarity_matrix.min():.4f}")
