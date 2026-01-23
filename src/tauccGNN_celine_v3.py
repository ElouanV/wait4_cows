import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import random
import torch.nn.functional as F
from torch import nn, optim, Tensor

from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import LGConv, GCNConv, GATConv, GINConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import normalized_mutual_info_score

import time

import os
import torch
import json
from datetime import datetime

def make_output_dir(hparams):
    # Estrai nome del dataset dal path
    dataset_name = os.path.splitext(os.path.basename(hparams['dataset_path']))[0]
    seed = hparams.pop('seed')
    # Costruisci stringa dei parametri ordinati
    param_strs = []
    for key in sorted(hparams):
        if key == "dataset_path":
            continue
        val = hparams[key]
        if isinstance(val, bool):
            val = int(val)  # True -> 1, False -> 0
        param_strs.append(f"{key}={val}")
    
    # Unisci con | tra parametri
    param_path = "|".join(param_strs)

    # Percorso finale
    output_dir = os.path.join("lightgnn", dataset_name, param_path, str(seed))
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

dtype = torch.float32

class GNN(nn.Module):
    def __init__(self, num_users, num_items,  input_dimx, input_dimy, num_compx, num_compy, output_dim,
                 data, num_layers=2, dim_h=64, model_type='lgc', use_edge_weight=True):
        super().__init__()       
        #  We store the number of users and items and create user and item embedding layers.
        # The shape of the emb_users ($e_u^0$) is (num_users, dim_h) and the shape of the emb_items ($e_i^0$)  is (num_items, dim_h)
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.emb_users = nn.Embedding(num_embeddings=self.num_users, embedding_dim=dim_h)
        self.emb_items = nn.Embedding(num_embeddings=self.num_items, embedding_dim=dim_h)
        self.dim = dim_h
        self.use_edge_weight = use_edge_weight

        # We create a list of num_layers (previously called K) LightGCN layers using PyTorch Geometric’s LGConv().
        # This will be used to perform the light graph convolution operations:
        convs = []
        for _ in range(num_layers):
            if model_type == 'lgc':
                convs.append(LGConv())
            elif model_type == 'gcn':
                convs.append(GCNConv(dim_h, dim_h))
            elif model_type == 'gat':
                convs.append(GATConv(dim_h, dim_h, heads=4, concat=False))
            elif model_type == 'gin':
                convs.append(GINConv(nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU(), nn.Linear(dim_h, dim_h))))
        
        
        self.convs = nn.ModuleList(convs)

        # We initialize the user and item embedding layers with normal distributions with a standard deviation of 0.01.
        # This helps to prevent the model from getting stuck in poor local optima when it is trained:
        nn.init.normal_(self.emb_users.weight, std=0.01)
        nn.init.normal_(self.emb_items.weight, std=0.01)
        self.linear = nn.Linear(dim_h, dim_h)
        self.data = data.to(dtype).T
        # self.col_labels_ = torch.full((input_dimy, output_dim), fill_value=0.0)#, requires_grad=False) 
        # self.row_labels_ = torch.full((input_dimx, output_dim), fill_value=0.0)#, requires_grad=False)
        # self.best_partition = torch.full((input_dimx, output_dim), fill_value=0.0)#, requires_grad=False)
        # for i in range(input_dimx):
        #     j = torch.randint(0, output_dim, (1,))
        #     self.row_labels_[i,j] = 1
        # for i in range(input_dimy):
        #     j = torch.randint(0, output_dim, (1,))
        #     self.col_labels_[i,j] = 1
        
    # The forward() method takes in an edge index tensor and returns the final user and item embedding vectors, e_u^(K) and e_i^(K).
    # It starts by concatenating the user and item embedding layers and storing the result in the emb tensor. It then creates a list, **embs**, with emb as its first element:

    def forward(self, edge_index, edge_weight):
        emb = torch.cat([self.emb_users.weight, self.emb_items.weight])
        embs = [emb]
        edge_weight = edge_weight if edge_weight is not None and self.use_edge_weight else None

        # We then apply the LightGCN layers in a loop and store the output of each layer in the embs list:
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index, edge_weight=edge_weight).relu()
            embs.append(emb)

        # We perform layer combination by calculating the final embedding vectors by taking 
        # the mean of the tensors in the embs list along the second dimension:
        emb_final = 1/(self.num_layers+1) * torch.sum(torch.stack(embs, dim=1), dim=1)
        emb_final = self.linear(emb_final)
        # We split emb_final into user e_u and item e_i embedding vectors
        emb_users_final, emb_items_final = torch.split(emb_final, [self.num_users, self.num_items])
        return emb_users_final, self.emb_users.weight, emb_items_final, self.emb_items.weight

def soft_arg_max(logits): 
    y_soft = logits.softmax(-1)
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(-1, index, 1.0)
    return y_hard - y_soft.detach() + y_soft

def loss_fct(T, tauyx = False, hard=False, only_numerator=False):
    # self.data (m,n), Px (n,k), Py(m,k)
    #print(Px.shape)
    total = torch.sum(T)
    tot_row = torch.sum(T, dim=1)
    tot_col = torch.sum(T, dim=0)
    p = torch.div(tot_row, total)
    q = torch.div(tot_col, total)
    r = T / total

    r_sq = torch.square(r)                          # r^2
    tau_x = 0.0
    tau_y = 0.0
    if tauyx:
        #print(p)
        mask = (p != 0) 
        num1 = torch.sum(r_sq.T[:,mask] / p[mask])  # first part of the numerator
        q_sqr = torch.sum(torch.square(q))          # q^2
        num = num1 - q_sqr
        denom = 1 - q_sqr
        tau_y = num / (denom+ 1e-10)
    else:
        mask = (q != 0)
        #print(q, mask, r_sq.shape)
        num1 = torch.sum(r_sq[:,mask] / q[mask])
        p_sqr = torch.sum(torch.square(p))
        num = num1 - p_sqr
        denom = 1 - p_sqr
        tau_x = num / (denom+ 1e-10*hard)
    if only_numerator: return -num, tau_x*(denom+ 1e-10*hard), tau_y*(denom+ 1e-10*hard)
    return -num / (denom), tau_x, tau_y #compute tau

# def data_loader(path):
#     # We load the datasets:
#     df = pd.read_csv(path, sep=',', encoding='latin-1') 
            
#     # Create mappings
#     user_mapping = {userid: i for i, userid in enumerate(np.unique(df.loc[:,['doc']].values.reshape(1,-1)))}
#     item_mapping = {isbn: i for i, isbn in enumerate(np.unique(df.loc[:,['word']].values.reshape(1,-1)))}
    
#     # Count users and items
#     num_users = len(user_mapping)
#     num_items = len(item_mapping)
#     num_total = num_users + num_items
    
#     print("num_users : ", num_users)
#     print("num_items : ", num_items)
#     print("num_total : ", num_total)

#     #print(user_mapping)

#     ## We create a tensor of user and item indices based on the user ratings in the dataset. The edge_index tensor is created by stacking these two tensors:
    
#     # Build the adjacency matrix based on user ratings
#     user_ids = torch.LongTensor([user_mapping[i] for i in df['doc']])
#     item_ids = torch.LongTensor([item_mapping[i] for i in df['word']])
#     edge_index = torch.stack((user_ids, item_ids))

#     matrix = df.pivot(index="doc", columns="word", values="cluster")
#     matrix = matrix.fillna(0)
#     input_data = torch.tensor(matrix.values, dtype=dtype)#torch.float32)
    

#     # Create training, validation, and test adjacency matrices
#     train_index, test_index = train_test_split(range(len(df)), test_size=0.2, random_state=0)
#     val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=0)

#     print(edge_index)
#     train_edge_index = edge_index[:, train_index]
#     val_edge_index = edge_index[:, val_index]
#     test_edge_index = edge_index[:, test_index]
#     print(train_edge_index.shape, val_edge_index.shape, test_edge_index.shape)
#     return num_users, num_items, edge_index, train_edge_index, val_edge_index, test_edge_index, train_index, input_data

def data_loader(path):
    # Load the dataset
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

    # Optional: Create input matrix for downstream tasks
    matrix = df.values
    input_data = torch.tensor(matrix.values, dtype=torch.float32)

    # Split indices for train/val/test
    train_index, test_index = train_test_split(range(len(df)), test_size=0.2, random_state=0)
    val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=0)

    train_edge_index = edge_index[:, train_index]
    val_edge_index = edge_index[:, val_index]
    test_edge_index = edge_index[:, test_index]

    train_edge_weight = edge_weight[train_index]
    val_edge_weight = edge_weight[val_index]
    test_edge_weight = edge_weight[test_index]

    print(train_edge_index.shape, val_edge_index.shape, test_edge_index.shape)

    return (
        num_users,
        num_items,
        edge_index,
        edge_weight,  # full edge weights
        train_edge_index,
        train_edge_weight,
        val_edge_index,
        val_edge_weight,
        test_edge_index,
        test_edge_weight,
        train_index,
        input_data
    )


def normalize_embeddings(embeddings, normalize=True):
    if normalize:
        min_vals, _ = torch.min(embeddings, dim=1)
        emb = embeddings - min_vals.view(-1, 1)
        emb = emb / (emb.sum(dim=1, keepdim=True) + 1e-9)
    else:
        from entmax import sparsemax
        emb = sparsemax(embeddings, dim=-1)
    return emb



def train_model_with_hparams(hparams):
    start = time.process_time()
    seed = hparams['seed']
    # Python built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    
    # For deterministic behavior (optional but useful)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eps = 1e-9

    # Data
    num_users, num_items, edge_index, edge_weight, train_edge_index, _, val_edge_index, _, test_edge_index, _, train_index, input_table = data_loader(hparams['dataset_path'])
    input_dimx, input_dimy = num_users, num_items 
    data = input_table.to(device).T
    output_dim = hparams['embedding_dim']

    # Model
    model = GNN(
        num_users, num_items, input_dimx, input_dimy,
        data.shape[1], data.T.shape[1], output_dim, data,
        num_layers=hparams['num_layers'],
        dim_h=hparams['embedding_dim'], model_type=hparams['model_type'], use_edge_weight=hparams['use_edge_weight']
    ).to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['lr'])

    # Init best tracking
    best = {
        'tau_x': float('-inf'),
        'tau_y': float('-inf'),
        'tau_x_hard': float('-inf'),
        'tau_y_hard': float('-inf'),
        'tau_x_hard_num': float('-inf'),
        'tau_y_hard_num': float('-inf'),
        'tau_avg_hard': float('-inf'),
        'emb_users_final': None,
        'emb_items_final': None
    }

    for epoch in range(hparams['epochs']):
        model.train()
        optimizer.zero_grad()

        emb_users_final, _, emb_items_final, _ = model(edge_index, edge_weight)
        row = normalize_embeddings(emb_users_final, normalize=hparams.get('normalize', True))
        col = normalize_embeddings(emb_items_final, normalize=hparams.get('normalize', True))

        row_a = soft_arg_max(emb_users_final)
        col_a = soft_arg_max(emb_items_final)

        Z = torch.matmul(data, row)
        Z_hard = torch.matmul(data, row_a)
        T = torch.matmul(col.T, Z)
        T_hard = torch.matmul(col_a.T, Z_hard)

        loss1, tau_x, _ = loss_fct(T, tauyx=False, only_numerator=hparams['only_numerator'])
        loss2, _, tau_y = loss_fct(T, tauyx=True, only_numerator=hparams['only_numerator'])
        loss1_hard, tau_x_hard, _ = loss_fct(T_hard, tauyx=False, hard=True)
        loss2_hard, _, tau_y_hard = loss_fct(T_hard, tauyx=True, hard=True)
        _, tau_x_hard_num, _ = loss_fct(T_hard, tauyx=False, hard=True, only_numerator=True)
        _, _, tau_y_hard_num = loss_fct(T_hard, tauyx=True, hard=True, only_numerator=True)

        if epoch % 100 == 0 or epoch == hparams['epochs'] - 1:
            print(f"[Epoch {epoch}] tau_x: {tau_x:.4f}, tau_y: {tau_y:.4f}, tau_x_hard: {tau_x_hard:.4f}, tau_y_hard: {tau_y_hard:.4f}")

        # Joint loss
        alpha = (loss2 / (loss1 + loss2)).detach()
        loss = loss1 * alpha + loss2 * (1 - alpha)
        loss.backward()

        for _, param in model.named_parameters():
            if param.grad is not None:
                param.grad[torch.isnan(param.grad)] = 0.0

        optimizer.step()

        # Update bests
        if tau_x_hard < 1 and tau_y_hard < 1 and (2*tau_x_hard*tau_y_hard)/(tau_x_hard+tau_y_hard) > best['tau_avg_hard']:
            best['tau_x'] = tau_x.item()
            best['tau_y'] = tau_y.item()
            best['tau_x_hard'] = tau_x_hard.item()
            best['tau_y_hard'] = tau_y_hard.item()
            best['tau_x_hard_num'] = tau_x_hard_num.item()
            best['tau_y_hard_num'] = tau_y_hard_num.item()
            best['tau_avg_hard'] = (2*tau_x_hard.item()*tau_y_hard.item())/(tau_x_hard.item() + tau_y_hard.item())/2
            best['emb_users_final'] = emb_users_final.detach().cpu()
            best['emb_items_final'] = emb_items_final.detach().cpu()
    end = time.process_time()
    ### SAVE EVERYTHING
    output_dir = make_output_dir(hparams)
    print(f"\nSaving results to: {output_dir}")
    torch.save(best['emb_users_final'].to(torch.float8_e4m3fn), os.path.join(output_dir, 'best_emb_users_final.pt'))
    torch.save(best['emb_items_final'].to(torch.float8_e4m3fn), os.path.join(output_dir, 'best_emb_items_final.pt'))
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'tau_x': best['tau_x'],
            'tau_y': best['tau_y'],
            'tau_x_hard': best['tau_x_hard'],
            'tau_y_hard': best['tau_y_hard'],
            'tau_x_hard_num': best['tau_x_hard_num'],
            'tau_y_hard_num': best['tau_y_hard_num'],
            'time': end - start
        }, f, indent=4)
    
    with open(os.path.join(output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=4)

    return best['tau_avg_hard']


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGCN with Mutual Info Loss")

    # Data and training
    parser.add_argument('--dataset_path', type=str, default='./datasets/hitech.txt')
    parser.add_argument('--model_type', type=str, default='lgc')
    parser.add_argument('--use_edge_weight', action='store_false')
    parser.add_argument('--only_numerator', action='store_true')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)

    # Model architecture
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)

    # Others
    parser.add_argument('--no_normalize', action='store_true', help="Disable custom normalization")

    return parser.parse_args()

def main():
    args = parse_args()

    hparams = {
        'dataset_path': args.dataset_path,
        'model_type': args.model_type,
        'use_edge_weight': args.use_edge_weight,
        'epochs': args.epochs,
        'only_numerator': args.only_numerator,
        'lr': args.lr,
        'embedding_dim': args.embedding_dim,
        'num_layers': args.num_layers,
        'normalize': not args.no_normalize,
        'seed': args.seed
    }

    print(f"Running with params: {hparams}")
    train_model_with_hparams(hparams)

if __name__ == "__main__":
    main()
