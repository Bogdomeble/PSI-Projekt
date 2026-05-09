import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from src.config import BASE_DIR

# Set a nice style for all plots
sns.set_theme(style="whitegrid")

def get_plots_dir():
    """Ensures that the plots folder exists in the main project directory."""
    plots_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_confusion_matrix(y_true, y_pred, model_name="model"):
    """Generates and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Stays (0)', 'Leaves (1)'],
                yticklabels=['Stays (0)', 'Leaves (1)'])
    plt.ylabel('Reality')
    plt.xlabel('Model Prediction')
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.tight_layout()

    save_path = os.path.join(get_plots_dir(), f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, model_name="model"):
    """Generates and saves the ROC-AUC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc_score:.3f}', color='darkorange', lw=2)
    plt.plot([0, 1],[0, 1], color='navy', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve - {model_name.upper()}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(get_plots_dir(), f"{model_name.lower()}_roc_curve.png")
    plt.savefig(save_path)
    plt.close()

def plot_xgboost_importance(xgb_model, model_name="xgboost"):
    """Generates and saves the feature importance plot for the XGBoost model."""
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight',
                        title='XGBoost - Top 10 Most Important Features')
    plt.tight_layout()

    save_path = os.path.join(get_plots_dir(), "{model_name}_feature_importance.png")
    plt.savefig(save_path)
    plt.close()

def print_metrics_table(title, nn_res, xgb_res):
    """Function to print metrics for both models"""
    print("\n" + "="*55)
    print(f" {title}")
    print("="*55)
    print(f"{'Metric':<15} | {'PyTorch (NN)':<15} | {'XGBoost':<15}")
    print("-" * 55)
    for metric in["Accuracy", "F1-Score", "Recall", "ROC-AUC"]:
        nn_val = f"{nn_res[metric]:.4f}"
        xgb_val = f"{xgb_res[metric]:.4f}"
        print(f"{metric:<15} | {nn_val:<15} | {xgb_val:<15}")
    print("="*55)