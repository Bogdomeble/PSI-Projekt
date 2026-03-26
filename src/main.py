from src.dataset import get_dataloaders
from src.config import DEVICE

def main():
    print("=========================================")
    print(f"Using compute device: {DEVICE.upper()}")
    print("=========================================\n")
    
    try:
        # main function
        train_loader, val_loader, test_loader, xgb_data, input_dim = get_dataloaders()
        
        print("\n--- Data processed succesfully ---")
        print(f" Number of input values: {input_dim}")
        print(f"Training dataset: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
        print(f"Validation dataset:      {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
        print(f"Testing dataset:    {len(test_loader.dataset)} samples ({len(test_loader)} batches)")
        
        # one batch for testing the data flow
        X_batch, y_batch = next(iter(train_loader))
        print(f"\n Shape of single X batch: {X_batch.shape} ->[batch_size, input_size]")
        print(f"Kształt pojedynczego batcha y: {y_batch.shape} -> [batch_size, 1]")
        
    except FileNotFoundError:
        print("\n[Error] File not found! Make sure it is in data/raw/")
    except Exception as e:
        print(f"\n[Critical Error] main() threw exception:\n{e}")

if __name__ == "__main__":
    main()