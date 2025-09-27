import torch
import pickle
from tqdm.notebook import tqdm
from multiprocessing import RLock
tqdm.set_lock(RLock())

def calculate_std_mean(dataloader):
    n_samples = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader), desc="Batches"):
            N, C, H, W = data.shape
            num = N * H * W   

            batch_sum = data.sum(dim=(0,2,3))         # (C,)
            batch_sq_sum = (data**2).sum(dim=(0,2,3)) # (C,)

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
