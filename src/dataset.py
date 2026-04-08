import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import *


def load_and_clean_data(filepath):
    print(f"Reading data from: {filepath}")
    df = pd.read_csv(filepath)

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # empty cells -> NaN -> 0

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # map main variable to {0,1}

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # one-hot encoding

    df = pd.get_dummies(df, drop_first=True)

    # cast everything needed to float

    df = df.astype(float)

    return df

def prepare_data_splits(df):
    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values

    # Divide the data into train,test,validate subsets

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    # scale data for NNs

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class ChurnDataset(Dataset):
    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)

        # For binary classification pytorch need someithing like (n,1) instead of just (n)

        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders():

    df = load_and_clean_data(RAW_DATA_PATH)

    feature_names = df.drop(columns=[TARGET_COLUMN]).columns.tolist()

    #save modified data to temporaty file

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits(df)

    # Dataloaders for python

    train_loader = DataLoader(ChurnDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
    """drop_last=True fixes the Expected more than 1 value per channel when training, got input size torch.Size([1, 64])"""
    val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ChurnDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # XGboost dictionary for the selected training sets

    xgb_data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names  # <--- this is for the xgboost plot file
    }

    # we return input_dims here

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, xgb_data, input_dim
