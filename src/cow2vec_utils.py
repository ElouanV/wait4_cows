#!/usr/bin/env python3
"""
Cow2Vec Embedding Loader and Utilities

Helper functions to load and use trained Cow2Vec embeddings.

Usage:
    from cow2vec_utils import Cow2VecEmbeddings
    
    # Load embeddings
    embeddings = Cow2VecEmbeddings('cow2vec_embeddings/cow_embeddings.pkl')
    
    # Find similar cows
    similar = embeddings.most_similar('3cf1', top_k=5)
    
    # Get embedding vector
    vector = embeddings.get_vector('3cf1')
    
    # Compute similarity between two cows
    sim = embeddings.similarity('3cf1', '3cf3')
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class Cow2VecEmbeddings:
    """Wrapper class for Cow2Vec embeddings with utility methods."""
    
    def __init__(self, embedding_path: str):
        """
        Load Cow2Vec embeddings.
        
        Args:
            embedding_path: Path to cow_embeddings.pkl file
        """
        with open(embedding_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.id_to_cow = data['id_to_cow']
        self.cow_to_id = data['cow_to_id']
        self.config = data.get('config', {})
        
        # Normalize embeddings for faster cosine similarity
        self.normalized_embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        print(f"Loaded embeddings: {len(self.cow_to_id)} cows, "
              f"{self.embeddings.shape[1]}-dimensional")
    
    def get_vector(self, cow_id: str) -> np.ndarray:
        """
        Get embedding vector for a cow.
        
        Args:
            cow_id: Cow ID
            
        Returns:
            Embedding vector
        """
        if cow_id not in self.cow_to_id:
            raise ValueError(f"Cow {cow_id} not in vocabulary")
        
        idx = self.cow_to_id[cow_id]
        return self.embeddings[idx]
    
    def similarity(self, cow1: str, cow2: str) -> float:
        """
        Compute cosine similarity between two cows.
        
        Args:
            cow1: First cow ID
            cow2: Second cow ID
            
        Returns:
            Similarity score [-1, 1]
        """
        if cow1 not in self.cow_to_id or cow2 not in self.cow_to_id:
            raise ValueError("One or both cows not in vocabulary")
        
        idx1 = self.cow_to_id[cow1]
        idx2 = self.cow_to_id[cow2]
        
        return float(np.dot(
            self.normalized_embeddings[idx1],
            self.normalized_embeddings[idx2]
        ))
    
    def most_similar(
        self, 
        cow_id: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar cows.
        
        Args:
            cow_id: Target cow ID
            top_k: Number of similar cows to return
            
        Returns:
            List of (cow_id, similarity_score) tuples
        """
        if cow_id not in self.cow_to_id:
            raise ValueError(f"Cow {cow_id} not in vocabulary")
        
        idx = self.cow_to_id[cow_id]
        target_vec = self.normalized_embeddings[idx]
        
        # Compute similarities
        similarities = np.dot(self.normalized_embeddings, target_vec)
        
        # Get top-k (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [
            (self.id_to_cow[i], float(similarities[i]))
            for i in top_indices
        ]
    
    def get_all_cows(self) -> List[str]:
        """Get list of all cow IDs in vocabulary."""
        return sorted(self.cow_to_id.keys())
    
    def similarity_matrix(self, cow_ids: List[str] = None) -> pd.DataFrame:
        """
        Compute pairwise similarity matrix.
        
        Args:
            cow_ids: List of cow IDs (None = all cows)
            
        Returns:
            DataFrame with pairwise similarities
        """
        if cow_ids is None:
            cow_ids = self.get_all_cows()
        
        n = len(cow_ids)
        sim_matrix = np.zeros((n, n))
        
        for i, cow1 in enumerate(cow_ids):
            for j, cow2 in enumerate(cow_ids):
                sim_matrix[i, j] = self.similarity(cow1, cow2)
        
        return pd.DataFrame(sim_matrix, index=cow_ids, columns=cow_ids)
    
    def cluster_cows(self, n_clusters: int = 5) -> Dict[str, int]:
        """
        Cluster cows using K-means on embeddings.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dict mapping cow_id -> cluster_id
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        return {
            cow_id: int(cluster_labels[idx])
            for cow_id, idx in self.cow_to_id.items()
        }
    
    def save_as_csv(self, output_path: str):
        """
        Save embeddings as CSV with cow IDs.
        
        Args:
            output_path: Path to save CSV
        """
        df = pd.DataFrame(
            self.embeddings,
            index=[self.id_to_cow[i] for i in range(len(self.embeddings))]
        )
        df.index.name = 'cow_id'
        df.to_csv(output_path)
        print(f"Saved embeddings to {output_path}")


def load_embeddings(embedding_path: str) -> Cow2VecEmbeddings:
    """
    Convenience function to load embeddings.
    
    Args:
        embedding_path: Path to embedding pickle file
        
    Returns:
        Cow2VecEmbeddings object
    """
    return Cow2VecEmbeddings(embedding_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cow2vec_utils.py <embedding_path> [cow_id]")
        sys.exit(1)
    
    embedding_path = sys.argv[1]
    embeddings = load_embeddings(embedding_path)
    
    if len(sys.argv) >= 3:
        # Show similar cows for specified cow
        cow_id = sys.argv[2]
        print(f"\nMost similar cows to {cow_id}:")
        similar = embeddings.most_similar(cow_id, top_k=10)
        for i, (similar_cow, score) in enumerate(similar, 1):
            print(f"{i:2d}. {similar_cow} (similarity: {score:.4f})")
    else:
        # Show general info
        print(f"\nAvailable cows: {', '.join(embeddings.get_all_cows()[:20])}...")
        print(f"Total cows: {len(embeddings.get_all_cows())}")
        print(f"Embedding dimension: {embeddings.embeddings.shape[1]}")
