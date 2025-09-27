import numpy as np
import torch
from torch_geometric.data import Data
from torch_cluster import knn_graph


def image_to_pointcloud(event):

    H, W, C = event.shape
    
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)   # (H*W, 2)
    features = event.reshape(-1, C)                       # (H*W, C)
    
    mask = features.sum(axis=1) != 0
    coords = coords[mask]
    features = features[mask]
    
    return coords, features


def pointcloud_to_knn_graph(coords, features, k=5):
    x = torch.tensor(features, dtype=torch.float)   
    pos = torch.tensor(coords, dtype=torch.float)  
    
    edge_index = knn_graph(pos, k=k, loop=False) 
    
    data = Data(x=x, pos=pos, edge_index=edge_index)
    return data
