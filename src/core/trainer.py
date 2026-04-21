import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


# helper function for plot directory
def get_plots_dir():
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "plots",
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def train_pytorch(model, train_loader, val_loader, device, epochs, lr):
    print("\n--- Starting Neural Network Training (PyTorch) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    print(
        f"PyTorch training finished! Final loss: {train_loss / len(train_loader):.4f}"
    )
    return model


def evaluate_pytorch(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # --- CONFUSION MATRIX (PyTorch) ---
    save_dir = get_plots_dir()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Neural Network (PyTorch)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_pytorch.png"))
    plt.close()
    print("PyTorch Confusion Matrix saved.")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def train_and_eval_xgboost(xgb_model, xgb_data):
    print("\n--- Starting XGBoost Training ---")
    features = xgb_data["feature_names"]
    X_train_df = pd.DataFrame(xgb_data["X_train"], columns=features)
    X_val_df = pd.DataFrame(xgb_data["X_val"], columns=features)
    X_test_df = pd.DataFrame(xgb_data["X_test"], columns=features)

    xgb_model.fit(
        X_train_df,
        xgb_data["y_train"],
        eval_set=[(X_val_df, xgb_data["y_val"])],
        verbose=False,
    )
    print("XGBoost training finished!")

    preds = xgb_model.predict(X_test_df)

    # --- CONFUSION MATRIX (XGBoost) ---
    save_dir = get_plots_dir()
    cm = confusion_matrix(xgb_data["y_test"], preds)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(cmap="Greens", values_format="d")  # different colour
    plt.title("Confusion Matrix - XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_xgboost.png"))
    plt.close()

    # --- FEATURE IMPORTANCE  ---
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(
        xgb_model,
        max_num_features=10,
        importance_type="weight",
        title="XGBoost - Top 10 Most Important Features",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "xgboost_feature_importance.png"))
    plt.close()
    print("XGBoost plots (CM & Feature Importance) saved.")

    return {
        "Accuracy": accuracy_score(xgb_data["y_test"], preds),
        "F1-Score": f1_score(xgb_data["y_test"], preds),
        "Recall": recall_score(xgb_data["y_test"], preds),
    }
