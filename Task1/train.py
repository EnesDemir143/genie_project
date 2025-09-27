import torch
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
from pytorch_msssim import ssim
import sys
import pickle
import logging
import os


def create_log(log_path):    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s')

    app_handler = logging.FileHandler(log_path, mode='w')
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    app_logger = logging.getLogger('app_logger')
    app_logger.setLevel(logging.INFO)
    if not app_logger.hasHandlers():
        app_logger.addHandler(app_handler)

    return app_logger  


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean


def train_autoencoder(model, train_loader, val_loader, device, loss_fn, optimizer, mean, std, early_stopping_patience=15, EPOCHS=100, scheduler=None, alpha=0.5):
    
    log_path='autoencoder.log'
    model_save_path='best_model.pth'
    
    app_logger = create_log(log_path)
    
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_l2_losses = []
    val_l2_losses = []
    train_ssim = []
    val_ssim = []
    metrics = {}
    best_epoch = None
    mse_loss = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_ssim_val = 0.0
        epoch_ssim_train = 0.0
        epoch_l2_val = 0.0
        epoch_l2_train = 0.0
        
        app_logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        app_logger.info("-" * 15)
        
        for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            outputs_denorm = denormalize(outputs, mean, std).clamp(0, 1)
            inputs_denorm  = denormalize(inputs, mean, std).clamp(0, 1)
            ssim_batch = ssim(outputs_denorm, inputs_denorm, data_range=1.0, size_average=True)
            epoch_ssim_train += ssim_batch.item() * inputs.size(0)
            
            l2_batch = mse_loss(outputs, inputs)
            epoch_l2_train += l2_batch.item() * inputs.size(0)
            
            loss = loss_fn(outputs, inputs) + alpha * (1 - ssim_batch)
            
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        
        train_l2_loss = epoch_l2_train / len(train_loader.dataset)
        train_ssim_metric = (epoch_ssim_train / len(train_loader.dataset))
        
        train_ssim.append(train_ssim_metric)
        train_l2_losses.append(train_l2_loss)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation", leave=False):
                inputs = inputs.to(device)
                outputs = model(inputs)
                                
                l2_batch = mse_loss(outputs, inputs)
                epoch_l2_val += l2_batch.item() * inputs.size(0)
                
                outputs_denorm = denormalize(outputs, mean, std).clamp(0, 1)
                inputs_denorm  = denormalize(inputs, mean, std).clamp(0, 1)
                ssim_batch = ssim(outputs_denorm, inputs_denorm, data_range=1.0, size_average=True)
                epoch_ssim_val += ssim_batch.item() * inputs.size(0)
                
                loss = loss_fn(outputs, inputs) + alpha * (1 - ssim_batch)
                val_running_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        
        val_l2_loss = epoch_l2_val / len(val_loader.dataset)
        val_ssim_metric = (epoch_ssim_val / len(val_loader.dataset))
        
        val_ssim.append(val_ssim_metric)
        val_l2_losses.append(val_l2_loss)
        
        if scheduler:
            scheduler.step(epoch_val_loss)
            
        current_lr = optimizer.param_groups[0]['lr']
        app_logger.info(f"Current Learning Rate: {current_lr:.6f}")
        
        app_logger.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train L2 Loss: {train_l2_loss:.4f}, Val L2 Loss: {val_l2_loss:.4f}, Train SSIM: {train_ssim_metric:.4f}, Val SSIM: {val_ssim_metric:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            
            best_epoch = f'epoch_{epoch+1} complated - val_loss: {best_val_loss:.4f}, train_loss: {epoch_train_loss:.4f}, train_l2_loss: {train_l2_loss:.4f}, val_l2_loss: {val_l2_loss:.4f}, train_ssim: {train_ssim_metric:.4f}, val_ssim: {val_ssim_metric:.4f}, lr: {current_lr:.6f}'
            
            app_logger.info(f"New best model saved at epoch {epoch+1} with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                app_logger.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break
            
        metrics[f'epoch_{epoch+1}'] = {
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_l2_loss': train_l2_loss,
            'val_l2_loss': val_l2_loss,
            'train_ssim_metric': train_ssim_metric,
            'val_ssim_metric': val_ssim_metric
        }
    
        epoch_l2_val = 0.0
        epoch_l2_train = 0.0
        epoch_ssim_val = 0.0
        epoch_ssim_train = 0.0    

    all_metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_l2_losses': train_l2_losses,
        'val_l2_losses': val_l2_losses,
        'train_ssim': train_ssim,
        'val_ssim': val_ssim
    }
    
    with open('all_metrics.pkl', 'wb') as f:
        pickle.dump(all_metrics, f)

    with open('saved_metrics_per_epoch.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("Training complete.")
    print(f"Best model saved at {model_save_path} with {best_epoch}")
    
    app_logger.info('Best epoch so far: {}'.format(best_epoch))
