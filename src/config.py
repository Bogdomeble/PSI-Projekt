import os
import torch

# --- Path dirs ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# --- Parameters ---
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# --- Pytorch (SSN) ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# ---  (GPU/CPU) ---
# checks if the device supports cuda
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"