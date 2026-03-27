import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score
import xgboost as xgb
import matplotlib.pyplot as plt
import os

def train_pytorch(model, train_loader, val_loader, device, epochs, lr):
    print("\n--- Starting Neural Network Training (PyTorch) ---")
    criterion = nn.BCEWithLogitsLoss() # Ideal loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Loop over batches of training data
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

def evaluate_pytorch(model, test_loader, device):
    model.eval()
    y_true, y_pred = [],[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float() # Convert probability to 0 or 1

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

def train_and_eval_xgboost(xgb_model, xgb_data):
    print("\n--- Starting XGBoost Training ---")

    # feature names for the xgboost model plot
    features = xgb_data['feature_names']
    X_train_df = pd.DataFrame(xgb_data['X_train'], columns=features)
    X_val_df = pd.DataFrame(xgb_data['X_val'], columns=features)
    X_test_df = pd.DataFrame(xgb_data['X_test'], columns=features)

    # Training (In XGBoost just one line)
    xgb_model.fit(
        X_train_df, xgb_data['y_train'],
        eval_set=[(X_val_df, xgb_data['y_val'])],
        verbose=False # Set to True if we want to see logs from each tree iteration
    )
    print("XGBoost training finished!")

    # Evaluation
    preds = xgb_model.predict(xgb_data['X_test'])
    metrics = {
        "Accuracy": accuracy_score(xgb_data['y_test'], preds),
        "F1-Score": f1_score(xgb_data['y_test'], preds),
        "Recall": recall_score(xgb_data['y_test'], preds)
    }

    # FEATURE IMPORTANCE VISUALIZATION
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "plots")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight', title='XGBoost - Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "xgboost_feature_importance.png"))
    plt.close()
    print("Feature importance plot saved in the 'plots' folder.")

    return metrics
