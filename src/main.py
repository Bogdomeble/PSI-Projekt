from src.dataset import get_dataloaders
from src.models.neural_net import ChurnNeuralNet
from src.models.xgboost_model import get_xgboost_model
from src.core.trainer import train_pytorch, evaluate_pytorch, train_and_eval_xgboost
from src.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE
from torchinfo import summary

def main():
    print("=========================================")
    print(f"Using compute device: {DEVICE.upper()}")
    print("=========================================\n")

    try:
        # main function

        train_loader, val_loader, test_loader, xgb_data, input_dim = get_dataloaders()

        # pytorch nn_model init and summary

        nn_model = ChurnNeuralNet(input_dim).to(DEVICE)

        # xgboost model init

        xgb_model = get_xgboost_model()

        # testing and Training

        nn_model = train_pytorch(nn_model, train_loader, val_loader, DEVICE, EPOCHS, LEARNING_RATE)
        nn_results = evaluate_pytorch(nn_model, test_loader, DEVICE)
        xgb_results = train_and_eval_xgboost(xgb_model, xgb_data)

        print(f"After training (PyTorch):\n")
        summary(nn_model, input_size=(BATCH_SIZE, input_dim))

        """ comment: this can be uncommented for dataset loading testing"""

        # print("\n--- Data processed succesfully ---")
        # print(f" Number of input values: {input_dim}")
        # print(f"Training dataset: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
        # print(f"Validation dataset: {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
        # print(f"Testing dataset: {len(test_loader.dataset)} samples ({len(test_loader)} batches)")

        # one batch for testing the data flow

        # X_batch, y_batch = next(iter(train_loader))
        # print(f"\n Shape of single X batch: {X_batch.shape} ->[batch_size, input_size]")
        # print(f" Shape of single y batch: {y_batch.shape} -> [batch_size, 1]")

        """end comment"""

        # ==========================================
        # Results summary
        # ==========================================
        print("\n" + "="*40)
        print("\nFinal results (test set)\n")
        print("="*40)
        print(f"{'Metric':<15} | {'PyTorch (NN)':<15} | {'XGBoost':<15}")
        print("-" * 45)

        for metric in ["Accuracy", "F1-Score", "Recall"]:
            nn_val = f"{nn_results[metric]:.4f}"
            xgb_val = f"{xgb_results[metric]:.4f}"
            print(f"{metric:<15} | {nn_val:<15} | {xgb_val:<15}")

        print("="*40)

    except FileNotFoundError:
        print("\n[Error] File not found! Make sure it is in data/raw/")
    except Exception as e:
        print(f"\n[Critical Error] main() threw exception:\n{e}")

if __name__ == "__main__":
    main()
