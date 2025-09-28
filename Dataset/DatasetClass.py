import h5py
import torch
from torch.utils.data import Dataset, get_worker_info
import torch.nn.functional as F

class ResizeTensor:
    def __init__(self, size):
        self.size = size
    def __call__(self, tensor):
        if tensor.dim() == 3:  # (C,H,W)
            tensor = tensor.unsqueeze(0)  # (1,C,H,W)
            out = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)
            return out.squeeze(0)  # (C,H,W)
        elif tensor.dim() == 4:  # (N,C,H,W)
            return F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


class QuarkGluonEvent(Dataset):
    def __init__(self, dataset_path, transform=None, include_others=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.include_others = include_others
        with h5py.File(dataset_path, 'r') as f:
            self.X_jets = torch.tensor(f['X_jets'][:], dtype=torch.float32).permute(0, 3, 1, 2) 
            self.length = len(self.X_jets)
            if include_others:
                self.y = torch.tensor(int(f['y'][:]), dtype=torch.long)
                self.m0 = torch.tensor(f['m0'][:], dtype=torch.float32)
                self.pt = torch.tensor(f['pt'][:], dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = self.X_jets[idx]
        if self.transform:
            X = self.transform(X)
        if self.include_others:
            return X, self.y[idx], self.m0[idx], self.pt[idx]
        return X