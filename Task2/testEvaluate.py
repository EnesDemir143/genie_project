import torch
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    precision_score, recall_score, roc_curve
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model_on_testset(model, test_loader, device, loss_fn, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_test_loss = 0
    test_batch_count = 0

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for test_input, test_label in tqdm(test_loader, desc="Testing", leave=False):
            test_input, test_label = test_input.to(device), test_label.to(device)
            test_output = model(test_input)

            loss = loss_fn(test_output, test_label)
            total_test_loss += loss.item()
            test_batch_count += 1

            probs = torch.softmax(test_output, dim=1)[:, 1]
            preds = torch.argmax(test_output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_test_loss = total_test_loss / test_batch_count
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve.png")
    plt.close()



    roc_test = {
        'fpr': fpr,
        'tpr': tpr
    }
    
    test_results = {
            "loss": avg_test_loss,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "roc": roc_test,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "confusion_matrix": cm,
        }
    
    return test_results
