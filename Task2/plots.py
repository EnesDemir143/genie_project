import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall(all_models):
    """
    Birden fazla modelin test sonuçlarından Precision-Recall eğrisini çizer.

    Parametre:
    ----------
    all_models : dict
        Anahtarları model isimleri (örn. "Model A"), değerleri ise o modele ait dict.
        Model dict'leri şu yapıya sahip olmalı:
            - 'all_model_test_outputs' adlı dict içinde:
                - 'test_real_labels' : Gerçek test etiketleri (list veya numpy array)
                - 'test_probs'       : Modelin test için tahmin olasılıkları (list veya numpy array)

    İşleyiş:
    --------
    Her model için precision, recall değerleri hesaplanır ve aynı grafikte çizilir.
    """
    plt.figure(figsize=(20, 12))

    for model_name, model_dict in all_models.items():
        y_true = model_dict["all_model_test_outputs"]["test_real_labels"]
        y_scores = model_dict["all_model_test_outputs"]["test_probs"]

        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        plt.plot(recall, precision, label=f"{model_name}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(all_models):
    """
    Birden fazla modelin test sonuçlarından ROC eğrisini çizer.

    Parametre:
    ----------
    all_models : dict
        Anahtarları model isimleri (örn. "Model A"), değerleri ise o modele ait dict.
        Model dict'leri şu yapıya sahip olmalı:
            - 'all_model_test_outputs' adlı dict içinde:
                - 'test_real_labels' : Gerçek test etiketleri (list veya numpy array)
                - 'test_probs' : Modelin test için tahmin olasılıkları (list veya numpy array)

    İşleyiş:
    --------
    Her model için FPR, TPR değerleri hesaplanır ve aynı grafikte çizilir.
    """
    plt.figure(figsize=(20, 12))

    for model_name, model_dict in all_models.items():
        y_true = model_dict["all_model_test_outputs"]["test_real_labels"]
        y_scores = model_dict["all_model_test_outputs"]["test_probs"]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{model_name}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Eğrisi (TPR vs FPR)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def plot_net_benefit(all_models):
    """
    Birden fazla model için Net Benefit (Decision Curve Analysis) Threshold grafiği çizer.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    plt.figure(figsize=(20, 12))

    any_model = next(iter(all_models.values()))  # İlk modelin verisini al
    y_true_any = np.array(any_model["all_model_test_outputs"]["test_real_labels"])
    prevalence = np.mean(y_true_any)

    for model_name, model_dict in all_models.items():
        y_true = np.array(model_dict["all_model_test_outputs"]["test_real_labels"])
        y_scores = np.array(model_dict["all_model_test_outputs"]["test_probs"])
        N = len(y_true)

        net_benefits = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            nb = (TP / N) - (FP / N) * (thresh / (1 - thresh))
            net_benefits.append(nb)

        plt.plot(thresholds, net_benefits, label=model_name)

    # "All positive" ve "None" stratejilerini bir kez çiz
    all_pos_net_benefit = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, all_pos_net_benefit, 'k--', label="All positive")
    plt.plot(thresholds, np.zeros_like(thresholds), 'k:', label="None")

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (Net Benefit vs Threshold)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
