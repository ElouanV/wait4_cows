import argparse
import torch
import numpy as np
import pandas as pd
import random
import time
import os
import json
from datetime import datetime

from torch import nn
from torch.optim import AdamW
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score

from torch_geometric.nn import LGConv, GCNConv, GATConv, GINConv

# Global tensor type
dtype = torch.float32

def make_output_dir(hparams):
    """
    Create a directory based on dataset name and sorted hyperparameters.
    This helps in organizing experiment outputs clearly.
    """
    dataset_name = os.path.splitext(os.path.basename(hparams['dataset_path']))[0]
    seed = hparams.pop('seed')
    param_strs = [f"{key}={int(val) if isinstance(val, bool) else val}" 
                  for key in sorted(hparams) if key != "dataset_path"]
    param_path = "|".join(param_strs)
    output_dir = os.path.join("results", dataset_name, param_path, str(seed))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

class GNN(nn.Module):
    """
    Graph Neural Network supporting LightGCN, GCN, GAT, and GIN layers for user-item interaction modeling.
    """
    def __init__(self, num_users, num_items, input_dimx, input_dimy, num_compx, num_compy,
                 output_dim, data, num_layers=2, dim_h=64, model_type='gcn', 
                 use_edge_weight=True, user_init=None, item_init=None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.dim = dim_h
        self.use_edge_weight = use_edge_weight

        # User embedding
        if user_init is not None:
            self.emb_users = nn.Embedding.from_pretrained(torch.tensor(user_init, dtype=torch.float), freeze=False)
        else:
            self.emb_users = nn.Embedding(num_users, dim_h)
            nn.init.normal_(self.emb_users.weight, std=0.01)

        # Item embedding
        if item_init is not None:
            self.emb_items = nn.Embedding.from_pretrained(torch.tensor(item_init, dtype=torch.float), freeze=False)
        else:
            self.emb_items = nn.Embedding(num_items, dim_h)
            nn.init.normal_(self.emb_items.weight, std=0.01)

        # Define GNN layers
        convs = []
        for _ in range(num_layers):
            if model_type == 'lgc':
                convs.append(LGConv())
            elif model_type == 'gcn':
                convs.append(GCNConv(dim_h, dim_h))
            elif model_type == 'gat':
                convs.append(GATConv(dim_h, dim_h, heads=2, concat=False))
            elif model_type == 'gin':
                convs.append(GINConv(nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_h))))
        self.convs = nn.ModuleList(convs)

        # Store input sparse matrix
        self.data = data.to(dtype).transpose(0, 1).coalesce()

    def forward(self, edge_index, edge_weight):
        """
        Compute user and item embeddings using multiple graph convolution layers.
        """
        emb = torch.cat([self.emb_users.weight, self.emb_items.weight])
        embs = [emb]
        edge_weight = edge_weight if edge_weight is not None and self.use_edge_weight else None
        for conv in self.convs:
            emb = conv(emb, edge_index)
            embs.append(emb)
            emb = emb.relu()
        return embs

def soft_arg_max(logits):
    """
    Differentiable version of argmax using softmax with a straight-through estimator.
    """
    y_soft = logits.softmax(-1)
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    return y_hard - y_soft.detach() + y_soft

def loss_fct(T, tauyx=False, hard=False, only_numerator=False):
    """
    Computes tau_x or tau_y objective based on a joint distribution matrix T.
    """
    total = torch.sum(T)
    p = torch.sum(T, dim=1) / total
    q = torch.sum(T, dim=0) / total
    r = T / total
    r_sq = torch.square(r)
    if tauyx:
        mask = p != 0
        num1 = torch.sum(r_sq.T[:, mask] / p[mask])
        q_sqr = torch.sum(q**2)
        num = num1 - q_sqr
        denom = 1 - q_sqr
        tau_y = num / (denom + 1e-10)
        return -num if only_numerator else (-num / denom, 0.0, tau_y)
    else:
        mask = q != 0
        num1 = torch.sum(r_sq[:, mask] / q[mask])
        p_sqr = torch.sum(p**2)
        num = num1 - p_sqr
        denom = 1 - p_sqr
        tau_x = num / (denom + 1e-10 * hard)
        return -num if only_numerator else (-num / denom, tau_x, 0.0)

def data_loader(path):
    """
    Load bipartite dataset and generate required tensors for GNN input.
    """
    df = pd.read_csv(path, sep=',', encoding='latin-1',index_col="cow_id")

    # Create mappings
    user_mapping = {userid: i for i, userid in enumerate(np.unique(df.index))}
    item_mapping = {isbn: i for i, isbn in enumerate(np.unique(df.columns))}

    num_users = len(user_mapping)
    num_items = len(item_mapping)
    num_total = num_users + num_items

    print("num_users : ", num_users)
    print("num_items : ", num_items)
    print("num_total : ", num_total)

    # Create edge_index
    user_ids = torch.LongTensor([user_mapping[i] for i in df.index])
    item_ids = torch.LongTensor([item_mapping[i] + num_users for i in df.columns])  # shift item IDs
    edge_index = torch.stack((user_ids, item_ids))  # shape: [2, num_edges]

    # Create edge weights from the 'cluster' column
    edge_weight = torch.tensor(df.values, dtype=torch.float32)

    matrix = df.values
    input_data = torch.tensor(matrix.values, dtype=torch.float32)


    train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=0)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0)

    return (
        num_users, num_items,
        edge_index, edge_weight,
        edge_index[:, train_idx], edge_weight[train_idx],
        edge_index[:, val_idx], edge_weight[val_idx],
        edge_index[:, test_idx], edge_weight[test_idx],
        train_idx, input_data
    )

def normalize_embeddings(embeddings, normalize=True):
    """
    Normalize embeddings to be on simplex (non-negative, sum to 1)
    """
    if normalize:
        min_vals, _ = torch.min(embeddings, dim=1)
        emb = embeddings - min_vals.unsqueeze(1)
        emb = emb / (emb.sum(dim=1, keepdim=True) + 1e-9)
    else:
        from entmax import sparsemax
        emb = sparsemax(embeddings, dim=-1)
    return emb

def train_model_with_hparams(hparams):
    """
    Main training loop: loads data, initializes model, trains with MI loss.
    """
    start = time.process_time()
    seed = hparams['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (
        num_users, num_items,
        edge_index, edge_weight,
        train_edge_index, train_edge_weight,
        val_edge_index, val_edge_weight,
        test_edge_index, test_edge_weight,
        train_index, input_table
    ) = data_loader(hparams['dataset_path'])

    # Optional: PCA/SVD init
    user_init, item_init = None, None
    if hparams['init'] in ['pca', 'svd']:
        pca_class = IncrementalPCA if hparams['init'] == 'pca' else TruncatedSVD
        user_pca = pca_class(n_components=hparams['embedding_dim'])
        item_pca = pca_class(n_components=hparams['embedding_dim'])

        shuffled_rows = np.random.permutation(input_table.shape[0])
        user_init = user_pca.fit_transform(input_table[shuffled_rows])
        user_init = user_init[np.argsort(shuffled_rows)]

        shuffled_cols = np.random.permutation(input_table.shape[1])
        item_init = item_pca.fit_transform(input_table[:, shuffled_cols].T)
        item_init = item_init[np.argsort(shuffled_cols)]

    # Sparse data for matrix multiplication
    dense_data = input_table.T
    indices = dense_data.nonzero(as_tuple=False).T
    values = dense_data[dense_data != 0]
    data = torch.sparse_coo_tensor(indices, values, size=dense_data.shape, dtype=dtype).coalesce().to(device)

    # Model init
    model = GNN(
        num_users, num_items, num_users, num_items, num_users, num_items,
        hparams['embedding_dim'], data,
        num_layers=hparams['num_layers'], dim_h=hparams['embedding_dim'],
        model_type=hparams['model_type'], use_edge_weight=hparams['use_edge_weight'],
        user_init=user_init, item_init=item_init
    ).to(dtype).to(device)

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    optimizer = AdamW(model.parameters(), lr=hparams['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=150)

    # Training state
    best = {
        'tau_x': float('-inf'), 'tau_y': float('-inf'),
        'tau_x_hard': float('-inf'), 'tau_y_hard': float('-inf'),
        'tau_x_hard_num': float('-inf'), 'tau_y_hard_num': float('-inf'),
        'tau_avg_hard': float('-inf'), 'tau_avg_hard_num': float('-inf'),
        'emb_users_final': None, 'emb_items_final': None,
        'ari': 0
    }

    patience = max_patience = 400

    for epoch in range(hparams['epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(edge_index, edge_weight)
        epoch_loss = float('inf')

        for layer, output in enumerate(outputs):
            emb_users, emb_items = torch.split(output, [num_users, num_items])
            row = normalize_embeddings(emb_users, normalize=hparams.get('normalize', True))
            col = normalize_embeddings(emb_items, normalize=hparams.get('normalize', True))
            row_hard = soft_arg_max(emb_users)
            col_hard = soft_arg_max(emb_items)

            Z = torch.sparse.mm(data, row)
            Z_hard = torch.sparse.mm(data, row_hard)
            T = torch.matmul(col.T, Z)
            T_hard = torch.matmul(col_hard.T, Z_hard)

            loss1, tau_x, _ = loss_fct(T, tauyx=False, only_numerator=hparams['only_numerator'])
            loss2, _, tau_y = loss_fct(T, tauyx=True, only_numerator=hparams['only_numerator'])
            loss1_hard, tau_x_hard, _ = loss_fct(T_hard, tauyx=False, hard=True)
            loss2_hard, _, tau_y_hard = loss_fct(T_hard, tauyx=True, hard=True)
            _, tau_x_hard_num, _ = loss_fct(T_hard, tauyx=False, hard=True, only_numerator=True)
            _, _, tau_y_hard_num = loss_fct(T_hard, tauyx=True, hard=True, only_numerator=True)

            alpha = (loss2 / (loss1 + loss2)).detach()
            loss = loss1 * alpha + loss2 * (1 - alpha)
            loss.backward()

            if loss.item() < epoch_loss:
                epoch_loss = loss.item()
                best.update({
                    'tau_x': tau_x.item(),
                    'tau_y': tau_y.item(),
                    'tau_x_hard': tau_x_hard.item(),
                    'tau_y_hard': tau_y_hard.item(),
                    'tau_x_hard_num': tau_x_hard_num.item(),
                    'tau_y_hard_num': tau_y_hard_num.item(),
                    'tau_avg_hard': (2 * tau_x_hard * tau_y_hard / (tau_x_hard + tau_y_hard)).item(),
                    'tau_avg_hard_num': (2 * tau_x_hard_num * tau_y_hard_num / (tau_x_hard_num + tau_y_hard_num)).item(),
                    'emb_users_final': emb_users.detach().cpu(),
                    'emb_items_final': emb_items.detach().cpu(),
                })

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(epoch_loss)

        # Early stopping
        if patience <= 0:
            break
        patience = max_patience if best['tau_avg_hard'] > epoch_loss else patience - 1

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] tau_x: {best['tau_x']:.4f}, tau_y: {best['tau_y']:.4f}, patience: {patience}")

    end = time.process_time()
    output_dir = make_output_dir(hparams)
    torch.save(best['emb_users_final'], os.path.join(output_dir, 'best_emb_users_final.pt'))
    torch.save(best['emb_items_final'], os.path.join(output_dir, 'best_emb_items_final.pt'))
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({k: v for k, v in best.items() if k != 'emb_users_final' and k != 'emb_items_final'}, f, indent=4)
    with open(os.path.join(output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=4)
    return best['tau_avg_hard']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./datasets/data.csv')
    parser.add_argument('--model_type', type=str, default='gcn')
    parser.add_argument('--init', type=str, default='pca')
    parser.add_argument('--use_edge_weight', action='store_true')
    parser.add_argument('--only_numerator', action='store_true')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--normalize', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    hparams = vars(args)
    print(f"Running with params: {hparams}")
    train_model_with_hparams(hparams)

if __name__ == "__main__":
    main()