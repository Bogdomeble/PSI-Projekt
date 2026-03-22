import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import *

"""
todo 22.03.2026r.  translate comments to english

"""
def load_and_clean_data(filepath):
    print(f"Reading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Zamieniamy puste znaki na NaN, a potem NaN na zera (dla nowych klientów)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Zamiana zmiennej docelowej na 1 i 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Zamiana kolumn tekstowych na zera i jedynki (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)
    
    # Rzutowanie wszystkiego na liczby zmiennoprzecinkowe
    df = df.astype(float)
    
    return df

def prepare_data_splits(df):
    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values

    # Dzielimy na Zbiór Testowy i reszte
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Dzielimy reszte na Zbiór Treningowy i Walidacyjny
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Skalowanie danych (ważne dla SSN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Przy klasyfikacji binarnej PyTorch woli format kształtu (N, 1) a nie tylko (N)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders():
    df = load_and_clean_data(RAW_DATA_PATH)
    
    # Opcjonalny zapis na dysk, żebyśmy widzieli jak teraz wyglądają dane
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data_splits(df)

    # Tworzenie DataLoaderów dla PyTorcha
    train_loader = DataLoader(ChurnDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ChurnDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Przygotowanie słownika dla XGBoosta, który akceptuje  tablice NumPy
    xgb_data = {
        "X_train": X_train, "y_train": y_train, 
        "X_val": X_val, "y_val": y_val, 
        "X_test": X_test, "y_test": y_test
    }

    # Zwracamy ilość cech wejściowych (input_dim), żeby sieć wiedziała ile ma "wejść"
    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, xgb_data, input_dim