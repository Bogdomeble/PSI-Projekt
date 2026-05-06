import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# We import our plotting tools
from src.utils.plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
)


def train_pytorch(
    model, train_loader, val_loader, device, epochs, lr, pos_weight_val=1.0
):
    print("\n--- Starting Neural Network Training (PyTorch) ---")

    # Added weight for the minority class (Churn)
    # pos_weight = torch.tensor([pos_weight_val]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = FocalLoss()

    # Added weight_decay (L2 regularization) to reduce overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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


def evaluate_pytorch(model, test_loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()  # Dynamic classification threshold

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # Generate plots using our new module
    plot_confusion_matrix(y_true, y_pred, model_name="pytorch")
    plot_roc_curve(y_true, y_probs, model_name="pytorch")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_probs),
    }


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


# Użycie w train_pytorch:
# criterion = FocalLoss(alpha=0.73, gamma=2.0) # alpha = waga klasy (np. odsetek klasy dominującej)
