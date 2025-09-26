import torch
import pickle
from tqdm import tqdm

def calculate_std_mean(dataloader):
    n_samples = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    
    for data, _, _, _ in tqdm(dataloader):   # data: (N, C, H, W)
        N, C, H, W = data.shape
        num = N * H * W   
        print(f"Processing batch with {N} samples, shape=({C},{H},{W}), total elements per channel={num}")
        
        # Flatten: (N, C, -1)
        data = data.view(N, C, -1)
        batch_sum = data.sum(dim=(0,2))         # (C,)
        batch_sq_sum = (data**2).sum(dim=(0,2)) # (C,)

        total_sum += batch_sum
        total_sq_sum += batch_sq_sum
        n_samples += num
    
    mean = total_sum / n_samples   # (C,)
    std = torch.sqrt(total_sq_sum / n_samples - mean ** 2)  # (C,)

    print(f"Final mean: {mean}, Final std: {std}")
    
    with open("mean_std.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
    print("Saved to mean_std.pkl")
    
    return mean, std
