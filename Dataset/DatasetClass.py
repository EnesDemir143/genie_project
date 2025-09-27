import h5py
import os
import torch
from torch.utils.data import Dataset

class quarkGluonEvent(Dataset):
    def __init__(self, DatasetPath, transform=None, include_others=False):
        self.dataset_path = DatasetPath
        self.transform = transform
        self.include_others = include_others
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"{self.dataset_path} not found.")
        
        with h5py.File(self.dataset_path, 'r') as f:
            assert len(f['X_jets']) == len(f['y']) == len(f['m0']) == len(f['pt']), "Dataset arrays must have the same length."
            self.length = len(f['X_jets'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as f:
            X = torch.tensor(f['X_jets'][idx], dtype=torch.float32).permute(2,0,1)      
            if self.transform:
                X = self.transform(X)
                
            if self.include_others:
                y = torch.tensor(int(f['y'][idx]), dtype=torch.long)
                m0 = torch.tensor(f['m0'][idx], dtype=torch.float32)
                pt = torch.tensor(f['pt'][idx], dtype=torch.float32)
                
                return X, y, m0, pt
            else:
                return X, None, None, None
