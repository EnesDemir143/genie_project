import h5py
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
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


class JetEventDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, k=5):
        self.k = k
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["quark-gluon_data-set_n139306.hdf5"]   # sen dosyayı root/raw içine koyacaksın

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = []
        
        with h5py.File(self.raw_paths[0], "r") as f:
            X_jets = f["X_jets"][:]   # (N, H, W, C)
            labels = f["y"][:]

        for event, y in zip(X_jets, labels):
            coords, features = image_to_pointcloud(event)   # (H,W,C) zaten
            graph = pointcloud_to_knn_graph(coords, features, k=self.k)
            graph.y = torch.tensor([int(y)], dtype=torch.long)
            data_list.append(graph)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])