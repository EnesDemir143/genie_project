import torch
import torch.nn as nn
from pytorch_msssim import ssim
from train import denormalize    
import matplotlib.pyplot as plt
import numpy as np
import h5py


def test_autoencoder(model, test_loader, device, loss_fn, mean, std, alpha):
    model.eval()
    test_loss = 0.0
    mse_loss = nn.MSELoss()
    epoch_ssim_test = 0.0
    epoch_l2_test = 0.0
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            l2_batch = mse_loss(outputs, inputs)
            epoch_l2_test += l2_batch.item() * inputs.size(0)
            
            outputs_denorm = denormalize(outputs, mean, std).clamp(0, 1)
            inputs_denorm  = denormalize(inputs, mean, std).clamp(0, 1)
            ssim_batch = ssim(outputs_denorm, inputs_denorm, data_range=1.0, size_average=True)
            epoch_ssim_test += ssim_batch.item() * inputs.size(0)
            
            loss = loss_fn(outputs, inputs) + alpha * (1 - ssim_batch)
            test_loss += loss.item() * inputs.size(0)
    
    test_loss /= len(test_loader.dataset)
    test_l2_loss = epoch_l2_test / len(test_loader.dataset)
    test_ssim_metric = (epoch_ssim_test / len(test_loader.dataset))
    
    return test_loss, test_l2_loss, test_ssim_metric


def compare_results(model, h5_path, mean, std, n=5, transform=None, channel_names=None):
    model.eval()
    
    with h5py.File(h5_path, 'r') as f:
        X_jets = torch.tensor(f['X_jets'][:], dtype=torch.float32).permute(0, 3, 1, 2)  
        # (N, C, H, W)
        
    with torch.no_grad():
        inputs = X_jets[:n].to(next(model.parameters()).device)
        if transform:
            inputs = transform(inputs)
        outputs = model(inputs)
        
    inputs_denorm = denormalize(inputs, mean, std).clamp(0, 1)
    outputs_denorm = denormalize(outputs, mean, std).clamp(0, 1)
    
    inputs_np = inputs_denorm.cpu().numpy()
    outputs_np = outputs_denorm.cpu().numpy()
    
    N, C, H, W = inputs_np.shape
    if channel_names is None or len(channel_names) != C:
        channel_names = [f"Channel {i}" for i in range(C)]
    
    fig, axes = plt.subplots(N*C, 2, figsize=(6, 3*N*C))
    
    # Eğer n=1 ve C=3 ise axes shape (3,2) olacak, 
    # eğer n>1 ise flatten etmek lazım
    axes = axes.reshape(N*C, 2)
    
    for i in range(N):
        for c in range(C):
            idx = i*C + c
            axes[idx, 0].imshow(inputs_np[i, c], cmap="inferno")
            axes[idx, 0].set_title(f"Original - {channel_names[c]} (sample {i})")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(outputs_np[i, c], cmap="inferno")
            axes[idx, 1].set_title(f"Reconstructed - {channel_names[c]} (sample {i})")
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
