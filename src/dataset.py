import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import *
import joblib 
import json 

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

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # also return the scaler at the end
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

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

    # check if directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    # get scaler from function
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_data_splits(df)

    # save the columns and the scaler
    export_dir = os.path.join(BASE_DIR, "exported-models")
    os.makedirs(export_dir, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(export_dir, "scaler.pkl"))
    with open(os.path.join(export_dir, "columns.json"), "w") as f:
        json.dump(feature_names, f)

    # scaler weights - these stay the same
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(ChurnDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ChurnDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    xgb_data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names
    }

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, xgb_data, input_dim