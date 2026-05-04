# src/core/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

# We import our plotting tools
from src.utils.plotting import plot_confusion_matrix, plot_roc_curve, plot_xgboost_importance

def train_pytorch(model, train_loader, val_loader, device, epochs, lr, pos_weight_val=1.0):
    print("\n--- Starting Neural Network Training (PyTorch) ---")

    # Added weight for the minority class (Churn)
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    print(f"PyTorch training finished! Final loss: {train_loss/len(train_loader):.4f}")
    return model

def evaluate_pytorch(model, test_loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred, y_probs = [],[],[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float() # Dynamic classification threshold

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
        "ROC-AUC": roc_auc_score(y_true, y_probs)
    }

def train_and_eval_xgboost(xgb_model, xgb_data):
    print("\n--- Starting XGBoost Training ---")

    features = xgb_data['feature_names']
    X_train_df = pd.DataFrame(xgb_data['X_train'], columns=features)
    X_val_df = pd.DataFrame(xgb_data['X_val'], columns=features)
    X_test_df = pd.DataFrame(xgb_data['X_test'], columns=features)

    xgb_model.fit(
        X_train_df, xgb_data['y_train'],
        eval_set=[(X_val_df, xgb_data['y_val'])],
        verbose=False
    )
    print("XGBoost training finished!")

    # Predictions and Probabilities
    preds = xgb_model.predict(xgb_data['X_test'])
    probs = xgb_model.predict_proba(xgb_data['X_test'])[:, 1] # Get probability for Churn class (1)

    # Generate all plots for XGBoost
    plot_confusion_matrix(xgb_data['y_test'], preds, model_name="xgboost")
    plot_roc_curve(xgb_data['y_test'], probs, model_name="xgboost")
    plot_xgboost_importance(xgb_model)

    print("Plots for models (ROC, Confusion Matrix, Feature Importance) have been saved in the 'plots/' folder.")

    return {
        "Accuracy": accuracy_score(xgb_data['y_test'], preds),
        "F1-Score": f1_score(xgb_data['y_test'], preds),
        "Recall": recall_score(xgb_data['y_test'], preds),
        "ROC-AUC": roc_auc_score(xgb_data['y_test'], probs)
    }
