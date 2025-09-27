import torch
import torch.nn as nn
from pytorch_msssim import ssim
from train import denormalize    
import matplotlib.pyplot as plt
import numpy as np


def test_autoencoder(model, test_loader, device, loss_fn, mean, std, alpha):
    model.eval()
    test_loss = 0.0
    mse_loss = nn.MSELoss()
    epoch_ssim_test = 0.0
    epoch_l2_test = 0.0
    
    with torch.no_grad():
        for inputs, _, _, _ in test_loader:
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


def compare_results(model, original, mean, std, num_images=5, channel_names=None):
    model.eval()
    with torch.no_grad():
        original = original.to(next(model.parameters()).device)
        reconstructed = model(original)

    original = denormalize(original.cpu(), mean, std).clamp(0, 1)
    reconstructed = denormalize(reconstructed.cpu(), mean, std).clamp(0, 1)

    num_images = min(num_images, original.shape[0])

    if num_images > 1:
        original_np = original.permute(0, 2, 3, 1).numpy()
        reconstructed_np = reconstructed.permute(0, 2, 3, 1).numpy()

        plt.figure(figsize=(12, 6))
        for i in range(num_images):
            plt.subplot(2, num_images, i + 1)
            plt.imshow(original_np[i], cmap='inferno')
            plt.title("Original")
            plt.axis('off')

            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructed_np[i], cmap='inferno')
            plt.title("Reconstructed")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        orig = original[0].numpy()   # (C,H,W)
        recon = reconstructed[0].numpy()

        C = orig.shape[0]
        if channel_names is None or len(channel_names) != C:
            channel_names = [f"Channel {i}" for i in range(C)]

        plt.figure(figsize=(12, 4*C))
        for c in range(C):
            plt.subplot(C, 2, 2*c + 1)
            plt.imshow(orig[c], cmap="inferno")
            plt.title(f"Original - {channel_names[c]}")
            plt.axis('off')

            plt.subplot(C, 2, 2*c + 2)
            plt.imshow(recon[c], cmap="inferno")
            plt.title(f"Reconstructed - {channel_names[c]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()