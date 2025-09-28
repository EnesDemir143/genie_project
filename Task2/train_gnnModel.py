import logging
import torch

from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score
)
import sys
sys.path.append('..')
import shutil
import pickle
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import average_precision_score, precision_recall_curve
import logging

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

def train_model(model, data_loader, validation_loader, device, loss_fn, optimizer, threshold = None, early_stopping_patience=10, EPOCHS=100, scheduler=None):
        
    log_path = 'training_gnn_model.log'
    model_save_path = 'best_model.pth'
    
    app_logger = create_log(log_path)
    model.to(device)
    app_logger.info(f'Starting training for {EPOCHS} epochs')

    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    accuracies = [] 
    accuracies_train = []
    specificitys_val = []
    recall_val = []
    precision_val = []
    fn_val = []
    fp_val = []
    tn_val = []
    tp_val = []
    f1_val = []
    f1_train_list = []
    auc_train_list = []
    auc_val = []
    roc_train = []
    roc_val = []
    mcc_val = []
    mcc_train = []
    balanced_acc_train = []
    balanced_acc_val = []
    pr_auc_train = []
    pr_auc_val = []
    metrics = {}
    
    best_value = None  
    
    best_epoch = None
    
    if os.path.exists("roc_curves"):
        shutil.rmtree("roc_curves")

    os.makedirs("roc_curves", exist_ok=True)  

    for epoch in range(EPOCHS):
        model.train()

        all_pred_train = []
        all_labels_train = []
        all_probs_train = []

        all_preds = []
        all_labels = []
        all_probs = [] 

        total_loss = 0 
        batch_count = 0

        app_logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        app_logger.info("-" * 15)

        for data in tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training", leave=False):
            data = data.to(device)
            labels = data.y.to(device).float()

            predictions = model(data)
            predictions =predictions.squeeze()
            probs_train = torch.sigmoid(predictions)

            
            if threshold is not None:
                preds_train = (probs_train > threshold).long()  
            else:
                preds_train = (probs_train > 0.5).long() 

            all_pred_train.extend(preds_train.cpu().numpy())
            all_labels_train.extend(labels.cpu().numpy())
            all_probs_train.extend(probs_train.detach().cpu().numpy())

            loss = loss_fn(predictions, labels.float())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count += 1
            if batch_count % 10 == 0:
                app_logger.info(f"Batch {batch_count}: Loss = {loss.item():.4f}")

        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        with torch.no_grad():

            for val_data in tqdm(validation_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation", leave=False):
                val_data = val_data.to(device) 
                val_labels = val_data.y.to(device).float()
                
                val_predictions = model(val_data)
                val_predictions = val_predictions.squeeze()
                val_loss = loss_fn(val_predictions, val_labels.float())
                total_val_loss += val_loss.item()
                val_batch_count += 1

                probs = torch.sigmoid(val_predictions)

                if threshold is not None and 0 < threshold < 1:
                    preds = (probs > threshold).long()  
                else:
                    preds = (probs > 0.5).long() 

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
        val_losses.append(avg_val_loss)

        if scheduler is not None:
            if hasattr(scheduler, 'step') and scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        app_logger.info(f"Current Learning Rate: {current_lr:.6f}")

        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_val.append(f1)
        acc = accuracy_score(all_labels, all_preds)
        accuracies.append(acc)

        f1_train = f1_score(all_labels_train, all_pred_train, average='macro')
        f1_train_list.append(f1_train)
        acc_train = accuracy_score(all_labels_train, all_pred_train)
        accuracies_train.append(acc_train)
        
        mcc_train_value = matthews_corrcoef(all_labels_train, all_pred_train)
        mcc_train.append(mcc_train_value)

        mcc_value = matthews_corrcoef(all_labels, all_preds)
        mcc_val.append(mcc_value)
        
        balanced_acc_train_value = balanced_accuracy_score(all_labels_train, all_pred_train)
        balanced_acc_train.append(balanced_acc_train_value)
        
        balanced_acc_value = balanced_accuracy_score(all_labels, all_preds)
        balanced_acc_val.append(balanced_acc_value)

        auc = 0.0
        auc_train = 0.0
        fpr_train = []
        tpr_train = []
        fpr_val = []
        tpr_val = []

        try:
            auc = roc_auc_score(all_labels, all_probs)
            auc_train = roc_auc_score(all_labels_train, all_probs_train)
        except ValueError:
            app_logger.warning("ROC AUC couldn't be calculated. Probably only one class present.")

        auc_train_list.append(auc_train)
        auc_val.append(auc)
        
        try:

            fpr_train, tpr_train, _ = roc_curve(all_labels_train, all_probs_train)

            fpr_val, tpr_val, _ = roc_curve(all_labels, all_probs)
            
            roc_train.append({
                'fpr_train': fpr_train,
                'tpr_train': tpr_train
            })
            roc_val.append({
                'fpr_val': fpr_val,
                'tpr_val': tpr_val
            })
            
            pr_auc_train_value = average_precision_score(all_labels_train, all_probs_train)
            pr_auc_train.append(pr_auc_train_value)
            
            pr_auc_val_value = average_precision_score(all_labels, all_probs)   
            pr_auc_val.append(pr_auc_val_value)
            
            precisions, recalls, _ = precision_recall_curve(all_labels, all_probs)
            
            plt.figure()
            plt.plot(recalls, precisions, label=f"PR AUC = {pr_auc_val_value:.4f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve for val - Epoch {epoch+1}")
            plt.legend()
            plt.savefig(f"roc_curves/epoch_{epoch+1}_pr.png")
            plt.close()
            
            plt.figure()
            plt.plot(fpr_val, tpr_val, label=f"AUC = {auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve for val - Epoch {epoch+1}")
            plt.legend()
            plt.savefig(f"roc_curves/epoch_{epoch+1}_roc.png")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='o')
            plt.title('Train vs Validation Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig("loss_over_epochs.png")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(accuracies) + 1), accuracies, label='Validation Accuracy', marker='o', color='green')
            plt.plot(range(1, len(accuracies_train) + 1), accuracies_train, label='Train Accuracy', marker='o', color='blue')
            plt.title('Validation Accuracy vs Train Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.savefig("accuracy_over_epochs.png")
            plt.close()

        except ValueError:
            app_logger.warning("ROC curve couldn't be plotted due to invalid data.")
        
        app_logger.info(f"Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.4f}, Train_Acc: {acc_train:.4f}, Train_AUC: {auc_train:.4f}, F1_train: {f1_train:.4f}, Val Loss: {avg_val_loss:.4f}, Val_F1: {f1:.4f}, Val_Acc: {acc:.4f}, Val_AUC: {auc:.4f}")
        
        cm = confusion_matrix(all_labels, all_preds)
        

        if len(set(all_labels)) == 2 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif len(set(all_labels)) == 1:

            if all_labels[0] == 1:  
                tp = len(all_labels)
                tn = fp = fn = 0
            else:  
                tn = len(all_labels)
                tp = fp = fn = 0
        else:
            tp = tn = fp = fn = 0
            app_logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")
            
        tp_val.append(tp)
        tn_val.append(tn)
        fp_val.append(fp)
        fn_val.append(fn)
        
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        precision_val.append(precision)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        recall_val.append(recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificitys_val.append(specificity)
        
        app_logger.info(
        f"Confusion Matrix:\n{cm.tolist()}\n"
        f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} | "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}"
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            app_logger.info(f" Validation loss improved. Model saved to {model_save_path}")
            early_stop_counter = 0

            best_epoch = (f"Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.4f}, Train_Acc: {acc_train:.4f}, "
                         f"Train_AUC: {auc_train:.4f}, F1_train: {f1_train:.4f}, Val Loss: {avg_val_loss:.4f}, "
                         f"Val_F1: {f1:.4f}, Val_Acc: {acc:.4f}, Val_AUC: {auc:.4f}\n"
                         f"Confusion Matrix:\n{cm.tolist()}\n"
                         f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} | "
                         f"Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}")
        else:
            early_stop_counter += 1
            app_logger.info(f" No improvement. Early stopping counter: {early_stop_counter}/{early_stopping_patience}")

        if early_stop_counter >= early_stopping_patience:
            app_logger.info(" Early stopping triggered. Stopping training.")
            break
            
        metrics[f'Epoch_{epoch+1}'] = {
            "train_losses": avg_train_loss,
            "val_losses": avg_val_loss,
            "accuracies_val": acc,
            "accuracies_train": acc_train,
            "auc_train": auc_train,
            "auc_val": auc,
            "f1_score_val": f1,
            "f1_score_train": f1_train,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "fpr_train": fpr_train.tolist() if len(fpr_train) > 0 else [],
            "tpr_train": tpr_train.tolist() if len(tpr_train) > 0 else [],
            "fpr_val": fpr_val.tolist() if len(fpr_val) > 0 else [],
            "tpr_val": tpr_val.tolist() if len(tpr_val) > 0 else [],
            "specificity": specificity,
            "mcc_train": mcc_train_value,
            "mcc_val": mcc_value,
            "balanced_acc_train": balanced_acc_train_value,
            "balanced_acc_val": balanced_acc_value,
            "pr_auc_train": pr_auc_train_value,
            "pr_auc_val": pr_auc_val_value,
            "best_epoch_log": best_epoch,
        }

    all_datas = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "accuracies_val": accuracies,
            "accuracies_train": accuracies_train,
            "auc_train": auc_train_list,
            "auc_val": auc_val,
            "f1_score_val": f1_val,
            "f1_score_train": f1_train_list,
            "TP": tp_val,
            "TN": tn_val,
            "FP": fp_val,
            "FN": fn_val,
            "precision": precision_val,
            "recall": recall_val,
            "specificity": specificitys_val,
            'roc_train': roc_train,
            'roc_val': roc_val,
            "mcc_train": mcc_train,
            "mcc_val": mcc_val,
            "balanced_acc_train": balanced_acc_train,
            "balanced_acc_val": balanced_acc_val,
            "pr_auc_train": pr_auc_train,
            "pr_auc_val": pr_auc_val,
            "best_epoch_log": best_epoch,
        }
    
    
    with open('training_metrics_per_epoch.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        
    with open('all_data.pkl', 'wb') as f:
        pickle.dump(all_datas, f)

    print("\nTraining completed!")
    app_logger.info('Best epoch so far: {}'.format(best_epoch))
    
    return best_value