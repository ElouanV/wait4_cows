"""
Alternative unsupervised methods for cow embeddings.

This module implements multiple unsupervised learning approaches:
1. Node2Vec: Biased random walks with Skip-gram
2. DeepWalk: Uniform random walks with Skip-gram
3. Graph Autoencoder: GCN-based encoder-decoder
4. Temporal Contrastive Learning: SimCLR-style contrastive method

Author: AI Assistant
Date: December 2025
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx, to_dense_adj

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================================
# Node2Vec Implementation
# ============================================================================

class Node2VecWalker:
    """Generates biased random walks for Node2Vec."""
    
    def __init__(self, p: float = 1.0, q: float = 1.0):
        """
        Initialize Node2Vec walker.
        
        Args:
            p: Return parameter (likelihood of returning to previous node)
            q: In-out parameter (BFS vs DFS behavior)
               q > 1: BFS-like (local exploration)
               q < 1: DFS-like (global exploration)
        """
        self.p = p
        self.q = q
        
    def _get_alias_edge(self, src: str, dst: str, G: nx.Graph) -> Tuple[List[float], List[int]]:
        """
        Precompute transition probabilities for an edge.
        
        Returns alias sampling tables for efficient sampling.
        """
        neighbors = list(G.neighbors(dst))
        unnormalized_probs = []
        
        for neighbor in neighbors:
            if neighbor == src:
                # Return to source: probability 1/p
                unnormalized_probs.append(1.0 / self.p)
            elif G.has_edge(neighbor, src):
                # Common neighbor: probability 1
                unnormalized_probs.append(1.0)
            else:
                # Further away: probability 1/q
                unnormalized_probs.append(1.0 / self.q)
        
        # Normalize
        norm_const = sum(unnormalized_probs)
        normalized_probs = [p / norm_const for p in unnormalized_probs]
        
        return normalized_probs, neighbors
    
    def node2vec_walk(self, G: nx.Graph, start_node: str, walk_length: int) -> List[str]:
        """
        Generate a single biased random walk.
        
        Args:
            G: NetworkX graph
            start_node: Starting node
            walk_length: Length of walk
            
        Returns:
            List of node IDs representing the walk
        """
        walk = [start_node]
        
        while len(walk) < walk_length:
            cur = walk[-1]
            neighbors = list(G.neighbors(cur))
            
            if len(neighbors) == 0:
                break
            
            if len(walk) == 1:
                # First step: uniform random
                walk.append(random.choice(neighbors))
            else:
                # Biased step based on previous node
                prev = walk[-2]
                probs, neighbor_list = self._get_alias_edge(prev, cur, G)
                walk.append(np.random.choice(neighbor_list, p=probs))
        
        return walk
    
    def generate_walks(self, G: nx.Graph, num_walks: int, walk_length: int) -> List[List[str]]:
        """
        Generate multiple random walks from each node.
        
        Args:
            G: NetworkX graph
            num_walks: Number of walks per node
            walk_length: Length of each walk
            
        Returns:
            List of walks (each walk is a list of node IDs)
        """
        walks = []
        nodes = list(G.nodes())
        
        print(f"Generating {num_walks} walks per node (p={self.p}, q={self.q})...")
        for _ in tqdm(range(num_walks)):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(G, node, walk_length))
        
        return walks


class Node2VecModel(nn.Module):
    """Skip-gram model for Node2Vec embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        nn.init.uniform_(self.target_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            target: Target node IDs [batch_size]
            context: Context node IDs [batch_size]
            
        Returns:
            Similarity scores [batch_size]
        """
        target_emb = self.target_embeddings(target)  # [batch, emb_dim]
        context_emb = self.context_embeddings(context)  # [batch, emb_dim]
        
        # Dot product
        scores = (target_emb * context_emb).sum(dim=1)  # [batch]
        return scores


class Node2VecTrainer:
    """Trainer for Node2Vec embeddings."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        negative_samples: int = 5,
        batch_size: int = 512,
        epochs: int = 5,
        lr: float = 0.01,
        device: str = 'cpu'
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        
        self.walker = Node2VecWalker(p=p, q=q)
        self.model = None
        self.node_to_idx = None
        self.idx_to_node = None
    
    def _build_training_pairs(self, walks: List[List[str]]) -> List[Tuple[int, int]]:
        """Build (target, context) pairs from walks."""
        pairs = []
        
        for walk in walks:
            walk_indices = [self.node_to_idx[node] for node in walk]
            
            for i, target in enumerate(walk_indices):
                # Context window
                start = max(0, i - self.window_size)
                end = min(len(walk_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context = walk_indices[j]
                        pairs.append((target, context))
        
        return pairs
    
    def train(self, temporal_graphs: List[nx.Graph]) -> Dict[str, np.ndarray]:
        """
        Train Node2Vec on temporal graphs.
        
        Args:
            temporal_graphs: List of NetworkX graphs (temporal snapshots)
            
        Returns:
            Dictionary mapping cow IDs to embeddings
        """
        print("\n" + "="*70)
        print("NODE2VEC TRAINING")
        print("="*70)
        
        # Aggregate temporal graphs into single graph
        print("\n[1/5] Aggregating temporal graphs...")
        G_agg = nx.Graph()
        for G in tqdm(temporal_graphs):
            G_agg.add_nodes_from(G.nodes())
            for u, v, data in G.edges(data=True):
                if G_agg.has_edge(u, v):
                    G_agg[u][v]['weight'] += data.get('rssi', 1)
                else:
                    G_agg.add_edge(u, v, weight=data.get('rssi', 1))
        
        print(f"Aggregated graph: {G_agg.number_of_nodes()} nodes, {G_agg.number_of_edges()} edges")
        
        # Build vocabulary
        nodes = sorted(G_agg.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        vocab_size = len(nodes)
        
        # Generate walks
        print(f"\n[2/5] Generating random walks...")
        walks = self.walker.generate_walks(G_agg, self.num_walks, self.walk_length)
        print(f"Generated {len(walks)} walks")
        
        # Build training pairs
        print("\n[3/5] Building training pairs...")
        pairs = self._build_training_pairs(walks)
        print(f"Created {len(pairs)} training pairs")
        
        # Initialize model
        print("\n[4/5] Initializing model...")
        self.model = Node2VecModel(vocab_size, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        print(f"\n[5/5] Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(pairs)
            
            for i in tqdm(range(0, len(pairs), self.batch_size), desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch_pairs = pairs[i:i+self.batch_size]
                
                # Positive samples
                targets = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long, device=self.device)
                contexts = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long, device=self.device)
                
                # Positive scores
                pos_scores = self.model(targets, contexts)
                pos_loss = -F.logsigmoid(pos_scores).mean()
                
                # Negative samples
                neg_contexts = torch.randint(0, vocab_size, (len(batch_pairs) * self.negative_samples,), device=self.device)
                neg_targets = targets.repeat_interleave(self.negative_samples)
                neg_scores = self.model(neg_targets, neg_contexts)
                neg_loss = -F.logsigmoid(-neg_scores).mean()
                
                # Total loss
                loss = pos_loss + neg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(pairs) // self.batch_size)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
        
        # Extract embeddings
        print("\n[DONE] Extracting embeddings...")
        embeddings = {}
        self.model.eval()
        with torch.no_grad():
            for node, idx in self.node_to_idx.items():
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=self.device)
                emb = self.model.target_embeddings(idx_tensor).cpu().numpy()[0]
                embeddings[node] = emb
        
        return embeddings


# ============================================================================
# DeepWalk Implementation (Simplified Node2Vec with p=1, q=1)
# ============================================================================

class DeepWalkTrainer:
    """DeepWalk: uniform random walks with Skip-gram."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 10,
        negative_samples: int = 5,
        batch_size: int = 512,
        epochs: int = 5,
        lr: float = 0.01,
        device: str = 'cpu'
    ):
        # Use Node2Vec with p=1, q=1 (uniform walks)
        self.trainer = Node2VecTrainer(
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            num_walks=num_walks,
            window_size=window_size,
            p=1.0,  # Uniform
            q=1.0,  # Uniform
            negative_samples=negative_samples,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device
        )
    
    def train(self, temporal_graphs: List[nx.Graph]) -> Dict[str, np.ndarray]:
        """Train DeepWalk (wrapper around Node2Vec with p=1, q=1)."""
        print("\n" + "="*70)
        print("DEEPWALK TRAINING (Node2Vec with p=1, q=1)")
        print("="*70)
        return self.trainer.train(temporal_graphs)


# ============================================================================
# Graph Autoencoder Implementation
# ============================================================================

class GCNEncoder(nn.Module):
    """GCN encoder for graph autoencoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GraphAutoencoder(nn.Module):
    """Graph Autoencoder with GCN encoder and inner product decoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, embedding_dim)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            z: Node embeddings [num_nodes, embedding_dim]
        """
        z = self.encoder(x, edge_index)
        return z
    
    def decode(self, z, edge_index):
        """
        Decode edges from embeddings.
        
        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Edge probabilities [num_edges]
        """
        # Inner product decoder
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)
    
    def decode_all(self, z):
        """
        Decode all possible edges (for full adjacency matrix).
        
        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            
        Returns:
            Adjacency matrix [num_nodes, num_nodes]
        """
        return torch.sigmoid(torch.matmul(z, z.t()))


class GraphAutoencoderTrainer:
    """Trainer for Graph Autoencoder."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        epochs: int = 50,
        lr: float = 0.01,
        device: str = 'cpu'
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device
        
        self.model = None
        self.node_to_idx = None
        self.idx_to_node = None
    
    def _prepare_graph_data(self, G_agg: nx.Graph):
        """Convert NetworkX graph to PyG data."""
        from torch_geometric.data import Data
        
        # Build vocabulary
        nodes = sorted(G_agg.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Node features: degree (normalized)
        degrees = np.array([G_agg.degree(node) for node in nodes], dtype=np.float32)
        degrees = degrees.reshape(-1, 1) / (degrees.max() + 1e-8)
        x = torch.tensor(degrees, dtype=torch.float32)
        
        # Edge index
        edge_list = []
        for u, v in G_agg.edges():
            edge_list.append([self.node_to_idx[u], self.node_to_idx[v]])
            edge_list.append([self.node_to_idx[v], self.node_to_idx[u]])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def train(self, temporal_graphs: List[nx.Graph]) -> Dict[str, np.ndarray]:
        """
        Train Graph Autoencoder.
        
        Args:
            temporal_graphs: List of NetworkX graphs
            
        Returns:
            Dictionary mapping cow IDs to embeddings
        """
        print("\n" + "="*70)
        print("GRAPH AUTOENCODER TRAINING")
        print("="*70)
        
        # Aggregate temporal graphs
        print("\n[1/4] Aggregating temporal graphs...")
        G_agg = nx.Graph()
        for G in tqdm(temporal_graphs):
            G_agg.add_nodes_from(G.nodes())
            for u, v, data in G.edges(data=True):
                if G_agg.has_edge(u, v):
                    G_agg[u][v]['weight'] += data.get('rssi', 1)
                else:
                    G_agg.add_edge(u, v, weight=data.get('rssi', 1))
        
        print(f"Aggregated graph: {G_agg.number_of_nodes()} nodes, {G_agg.number_of_edges()} edges")
        
        # Prepare data
        print("\n[2/4] Preparing graph data...")
        data = self._prepare_graph_data(G_agg).to(self.device)
        
        # Initialize model
        print("\n[3/4] Initializing model...")
        input_dim = data.x.shape[1]
        self.model = GraphAutoencoder(input_dim, self.hidden_dim, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        print(f"\n[4/4] Training for {self.epochs} epochs...")
        self.model.train()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            z = self.model(data.x, data.edge_index)
            
            # Decode edges
            edge_probs = torch.sigmoid(self.model.decode(z, data.edge_index))
            
            # Positive loss (existing edges)
            pos_loss = -torch.log(edge_probs + 1e-15).mean()
            
            # Negative sampling
            num_neg = data.edge_index.shape[1]
            neg_edge_index = torch.randint(0, data.x.shape[0], (2, num_neg), device=self.device)
            neg_probs = torch.sigmoid(self.model.decode(z, neg_edge_index))
            neg_loss = -torch.log(1 - neg_probs + 1e-15).mean()
            
            # Total loss
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.4f} (Pos: {pos_loss.item():.4f}, Neg: {neg_loss.item():.4f})")
        
        # Extract embeddings
        print("\n[DONE] Extracting embeddings...")
        self.model.eval()
        with torch.no_grad():
            z = self.model(data.x, data.edge_index).cpu().numpy()
        
        embeddings = {node: z[idx] for node, idx in self.node_to_idx.items()}
        return embeddings


# ============================================================================
# Temporal Contrastive Learning Implementation
# ============================================================================

class TemporalContrastiveModel(nn.Module):
    """GCN encoder for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 64)  # Projection for contrastive loss
        )
    
    def forward(self, x, edge_index):
        # Encoder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        z = self.conv2(x, edge_index)
        
        # Projection head
        h = self.projection_head(z)
        
        return z, h


class TemporalContrastiveTrainer:
    """
    Temporal Contrastive Learning for cow embeddings.
    
    Idea: Same cow at different times should have similar embeddings.
    Uses NT-Xent loss (InfoNCE) like SimCLR.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.5,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 0.001,
        device: str = 'cpu'
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        
        self.model = None
        self.node_to_idx = None
        self.idx_to_node = None
    
    def _nt_xent_loss(self, z_i, z_j):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
        
        Args:
            z_i: Embeddings from view 1 [batch, dim]
            z_j: Embeddings from view 2 [batch, dim]
            
        Returns:
            Contrastive loss
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch, dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch, 2*batch]
        
        # Remove self-similarities (diagonal)
        sim_matrix = sim_matrix.fill_diagonal_(-float('inf'))
        
        # Positive pairs: (i, batch+i) and (batch+i, i)
        pos_sim = torch.cat([
            sim_matrix[torch.arange(batch_size), torch.arange(batch_size) + batch_size],
            sim_matrix[torch.arange(batch_size) + batch_size, torch.arange(batch_size)]
        ])  # [2*batch]
        
        # All similarities (negatives included)
        all_sim = torch.cat([
            sim_matrix[:batch_size],
            sim_matrix[batch_size:]
        ])  # [2*batch, 2*batch]
        
        # NT-Xent loss
        loss = -pos_sim + torch.logsumexp(all_sim, dim=1)
        return loss.mean()
    
    def train(self, temporal_graphs: List[nx.Graph]) -> Dict[str, np.ndarray]:
        """
        Train temporal contrastive model.
        
        Args:
            temporal_graphs: List of NetworkX graphs (temporal snapshots)
            
        Returns:
            Dictionary mapping cow IDs to embeddings
        """
        print("\n" + "="*70)
        print("TEMPORAL CONTRASTIVE LEARNING")
        print("="*70)
        
        from torch_geometric.data import Data
        
        # Build vocabulary from all graphs
        print("\n[1/5] Building vocabulary...")
        all_nodes = set()
        for G in temporal_graphs:
            all_nodes.update(G.nodes())
        nodes = sorted(all_nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        num_nodes = len(nodes)
        print(f"Vocabulary: {num_nodes} cows")
        
        # Convert graphs to PyG format
        print("\n[2/5] Converting graphs to PyG format...")
        data_list = []
        for G in tqdm(temporal_graphs):
            # Node features: degree (normalized)
            x = torch.zeros(num_nodes, 1)
            for node in G.nodes():
                idx = self.node_to_idx[node]
                x[idx, 0] = G.degree(node)
            x = x / (x.max() + 1e-8)
            
            # Edge index
            edge_list = []
            for u, v in G.edges():
                edge_list.append([self.node_to_idx[u], self.node_to_idx[v]])
                edge_list.append([self.node_to_idx[v], self.node_to_idx[u]])
            
            if len(edge_list) > 0:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            data_list.append(Data(x=x, edge_index=edge_index))
        
        print(f"Converted {len(data_list)} temporal snapshots")
        
        # Initialize model
        print("\n[3/5] Initializing model...")
        self.model = TemporalContrastiveModel(1, self.hidden_dim, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        print(f"\n[4/5] Training for {self.epochs} epochs...")
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            # Sample pairs of temporal snapshots
            indices = list(range(len(data_list)))
            random.shuffle(indices)
            
            for i in tqdm(range(0, len(indices) - 1, 2), desc=f"Epoch {epoch+1}/{self.epochs}"):
                # Two different time snapshots (augmentations)
                idx1 = indices[i]
                idx2 = indices[i + 1]
                
                data1 = data_list[idx1].to(self.device)
                data2 = data_list[idx2].to(self.device)
                
                # Forward pass
                z1, h1 = self.model(data1.x, data1.edge_index)
                z2, h2 = self.model(data2.x, data2.edge_index)
                
                # Find common cows in both snapshots
                nodes1 = set(data1.edge_index.flatten().cpu().numpy())
                nodes2 = set(data2.edge_index.flatten().cpu().numpy())
                common_nodes = list(nodes1 & nodes2)
                
                if len(common_nodes) < 2:
                    continue
                
                # Sample a batch of common nodes
                if len(common_nodes) > self.batch_size:
                    sampled_nodes = random.sample(common_nodes, self.batch_size)
                else:
                    sampled_nodes = common_nodes
                
                sampled_nodes = torch.tensor(sampled_nodes, dtype=torch.long, device=self.device)
                
                # Get embeddings for common nodes
                h1_batch = h1[sampled_nodes]
                h2_batch = h2[sampled_nodes]
                
                # Contrastive loss
                loss = self._nt_xent_loss(h1_batch, h2_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
        
        # Extract embeddings (average over all snapshots)
        print("\n[5/5] Extracting embeddings...")
        self.model.eval()
        embedding_sum = torch.zeros(num_nodes, self.embedding_dim, device=self.device)
        embedding_count = torch.zeros(num_nodes, device=self.device)
        
        with torch.no_grad():
            for data in tqdm(data_list, desc="Averaging embeddings"):
                data = data.to(self.device)
                z, _ = self.model(data.x, data.edge_index)
                
                # Accumulate embeddings for nodes present in this snapshot
                active_nodes = set(data.edge_index.flatten().cpu().numpy())
                for node_idx in active_nodes:
                    embedding_sum[node_idx] += z[node_idx]
                    embedding_count[node_idx] += 1
        
        # Average
        embedding_count[embedding_count == 0] = 1  # Avoid division by zero
        final_embeddings = embedding_sum / embedding_count.unsqueeze(1)
        final_embeddings = final_embeddings.cpu().numpy()
        
        embeddings = {node: final_embeddings[idx] for node, idx in self.node_to_idx.items()}
        return embeddings


# ============================================================================
# Comparison and Evaluation
# ============================================================================

class EmbeddingComparator:
    """Compare multiple embedding methods."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_similarity_matrix(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute cosine similarity matrix."""
        nodes = sorted(embeddings.keys())
        emb_matrix = np.array([embeddings[node] for node in nodes])
        
        # Cosine similarity
        emb_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        sim_matrix = emb_norm @ emb_norm.T
        
        return sim_matrix, nodes
    
    def plot_tsne(self, method_embeddings: Dict[str, Dict[str, np.ndarray]]):
        """Plot t-SNE for all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, (method_name, embeddings) in enumerate(method_embeddings.items()):
            nodes = sorted(embeddings.keys())
            emb_matrix = np.array([embeddings[node] for node in nodes])
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(nodes)-1))
            emb_2d = tsne.fit_transform(emb_matrix)
            
            # Plot
            ax = axes[idx]
            scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=range(len(nodes)), cmap='tab20', s=100, alpha=0.7)
            
            # Annotate some points
            for i, node in enumerate(nodes):
                if i % 5 == 0:  # Annotate every 5th point
                    ax.annotate(node, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8, alpha=0.7)
            
            ax.set_title(f"{method_name} - t-SNE Visualization", fontsize=14, fontweight='bold')
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "tsne_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'tsne_comparison.png'}")
        plt.close()
    
    def plot_similarity_heatmaps(self, method_embeddings: Dict[str, Dict[str, np.ndarray]]):
        """Plot similarity heatmaps for all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        axes = axes.flatten()
        
        for idx, (method_name, embeddings) in enumerate(method_embeddings.items()):
            sim_matrix, nodes = self.compute_similarity_matrix(embeddings)
            
            ax = axes[idx]
            sns.heatmap(sim_matrix, xticklabels=nodes, yticklabels=nodes, 
                       cmap='RdYlBu_r', center=0, vmin=-1, vmax=1, 
                       square=True, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
            ax.set_title(f"{method_name} - Similarity Matrix", fontsize=14, fontweight='bold')
            ax.set_xlabel("Cow ID")
            ax.set_ylabel("Cow ID")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "similarity_heatmaps.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'similarity_heatmaps.png'}")
        plt.close()
    
    def compare_top_similarities(self, method_embeddings: Dict[str, Dict[str, np.ndarray]], top_k: int = 5):
        """Find top similar pairs for each method."""
        results = {}
        
        for method_name, embeddings in method_embeddings.items():
            sim_matrix, nodes = self.compute_similarity_matrix(embeddings)
            
            # Find top-k similar pairs (excluding self-similarity)
            np.fill_diagonal(sim_matrix, -float('inf'))
            
            pairs = []
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    pairs.append((nodes[i], nodes[j], sim_matrix[i, j]))
            
            pairs.sort(key=lambda x: x[2], reverse=True)
            results[method_name] = pairs[:top_k]
        
        # Print results
        print("\n" + "="*70)
        print("TOP SIMILAR PAIRS COMPARISON")
        print("="*70)
        
        for method_name, top_pairs in results.items():
            print(f"\n{method_name}:")
            for cow1, cow2, sim in top_pairs:
                print(f"  {cow1} ↔ {cow2}: {sim:.4f}")
        
        return results


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_all_methods(
    network_pkl: str,
    output_dir: str = "alternative_embeddings_results",
    embedding_dim: int = 128,
    device: str = 'cpu'
):
    """
    Train all embedding methods and compare results.
    
    Args:
        network_pkl: Path to network sequence pickle
        output_dir: Output directory for results
        embedding_dim: Embedding dimension
        device: Device for training ('cpu' or 'cuda')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load temporal graphs
    print("\n" + "="*70)
    print("LOADING TEMPORAL GRAPHS")
    print("="*70)
    with open(network_pkl, 'rb') as f:
        network_sequence = pickle.load(f)
    
    temporal_graphs = [data['graph'] for data in network_sequence]
    print(f"Loaded {len(temporal_graphs)} temporal graphs")
    
    # Train all methods
    all_embeddings = {}
    
    # 1. Node2Vec
    print("\n" + "="*70)
    print("METHOD 1/4: NODE2VEC")
    print("="*70)
    node2vec_trainer = Node2VecTrainer(
        embedding_dim=embedding_dim,
        walk_length=80,
        num_walks=10,
        window_size=10,
        p=1.0,
        q=0.5,  # BFS-like (local exploration)
        epochs=5,
        device=device
    )
    all_embeddings['Node2Vec'] = node2vec_trainer.train(temporal_graphs)
    
    # Save
    with open(output_dir / 'node2vec_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings['Node2Vec'], f)
    print(f"Saved: {output_dir / 'node2vec_embeddings.pkl'}")
    
    # 2. DeepWalk
    print("\n" + "="*70)
    print("METHOD 2/4: DEEPWALK")
    print("="*70)
    deepwalk_trainer = DeepWalkTrainer(
        embedding_dim=embedding_dim,
        walk_length=80,
        num_walks=10,
        window_size=10,
        epochs=5,
        device=device
    )
    all_embeddings['DeepWalk'] = deepwalk_trainer.train(temporal_graphs)
    
    # Save
    with open(output_dir / 'deepwalk_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings['DeepWalk'], f)
    print(f"Saved: {output_dir / 'deepwalk_embeddings.pkl'}")
    
    # 3. Graph Autoencoder
    print("\n" + "="*70)
    print("METHOD 3/4: GRAPH AUTOENCODER")
    print("="*70)
    gae_trainer = GraphAutoencoderTrainer(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        epochs=50,
        device=device
    )
    all_embeddings['Graph_Autoencoder'] = gae_trainer.train(temporal_graphs)
    
    # Save
    with open(output_dir / 'gae_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings['Graph_Autoencoder'], f)
    print(f"Saved: {output_dir / 'gae_embeddings.pkl'}")
    
    # 4. Temporal Contrastive
    print("\n" + "="*70)
    print("METHOD 4/4: TEMPORAL CONTRASTIVE LEARNING")
    print("="*70)
    contrastive_trainer = TemporalContrastiveTrainer(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        temperature=0.5,
        epochs=30,
        batch_size=32,
        device=device
    )
    all_embeddings['Temporal_Contrastive'] = contrastive_trainer.train(temporal_graphs)
    
    # Save
    with open(output_dir / 'contrastive_embeddings.pkl', 'wb') as f:
        pickle.dump(all_embeddings['Temporal_Contrastive'], f)
    print(f"Saved: {output_dir / 'contrastive_embeddings.pkl'}")
    
    # Compare methods
    print("\n" + "="*70)
    print("COMPARING ALL METHODS")
    print("="*70)
    comparator = EmbeddingComparator(output_dir)
    
    # Plot t-SNE
    print("\nGenerating t-SNE plots...")
    comparator.plot_tsne(all_embeddings)
    
    # Plot similarity heatmaps
    print("\nGenerating similarity heatmaps...")
    comparator.plot_similarity_heatmaps(all_embeddings)
    
    # Compare top similarities
    print("\nComparing top similar pairs...")
    comparator.compare_top_similarities(all_embeddings, top_k=10)
    
    print("\n" + "="*70)
    print("ALL METHODS TRAINED AND COMPARED!")
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train alternative cow embedding methods")
    parser.add_argument('--network-pkl', type=str, required=True, help='Path to network sequence pickle')
    parser.add_argument('--output-dir', type=str, default='alternative_embeddings_results', help='Output directory')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for training')
    
    args = parser.parse_args()
    
    train_all_methods(
        network_pkl=args.network_pkl,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        device=args.device
    )
