import h5py
import os
import torch
from torch.utils.data.dataset import Dataset


class quarkGluonEvent(Dataset):
    def __init__(self, DatasetPath):
        self.dataset_path = DatasetPath
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"The dataset path {self.dataset_path} does not exist.")
        else:
            with h5py.File(self.dataset_path, 'r') as f:
                self.X_jets = f['X_jets'][:]
                self.y_labels = f['y'][:]
                self.m0 = f['m0'][:]
                self.pt = f['pt'][:]
                print(f"Loaded dataset with {len(self.X_jets)} samples.")
        
    def __len__(self):
        return len(self.X_jets)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X_jets[idx], dtype=torch.float32)
        X = X.permute(2, 0, 1)
        y = torch.tensor(int(self.y_labels[idx]), dtype=torch.long)
        m0 = torch.tensor(self.m0[idx], dtype=torch.float32)
        pt = torch.tensor(self.pt[idx], dtype=torch.float32)
        
 
        return X, y, m0, pt