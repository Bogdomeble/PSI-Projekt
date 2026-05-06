from src.dataset import get_dataloaders
from src.models.neural_net import ChurnNeuralNet
from src.models.xgboost_model import get_xgboost_model
from src.core.trainer_pytorch import train_pytorch, evaluate_pytorch
from src.core.trainer_xgboost import train_and_eval_xgboost, train_and_eval_xgboost_with_feature_selection
from src.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, XGBOOST_DROP_PERCENT
from torchinfo import summary

def main():
    print("=========================================")
    print(f"Using compute device: {DEVICE.upper()}")
    print("=========================================\n")

    try:
        train_loader, val_loader, test_loader, xgb_data, input_dim = get_dataloaders()

        nn_model = ChurnNeuralNet(input_dim).to(DEVICE)
        xgb_model = get_xgboost_model()

        # PYTORCH TRAINING - with added weight 2.76 for the Churn class
        nn_model = train_pytorch(
            nn_model, train_loader, val_loader, DEVICE,
            EPOCHS, LEARNING_RATE, pos_weight_val=2.76
        )

        # PYTORCH EVALUATION - with lowered threshold to 0.4 (better recall for imbalanced data)
        nn_results = evaluate_pytorch(nn_model, test_loader, DEVICE, threshold=0.4)

        # XGBOOST TRAINING AND EVALUATION
        #xgb_results = train_and_eval_xgboost(xgb_model, xgb_data)

        # XGBOOST RE-TRAINING WITH FEATURE DROPPING
        xgb_results = train_and_eval_xgboost_with_feature_selection(
            xgb_model, xgb_data,
            drop_percent=XGBOOST_DROP_PERCENT,
        )

        print(f"\nAfter training (PyTorch):\n")
        summary(nn_model, input_size=(BATCH_SIZE, input_dim))

        # ==========================================
        # Results summary
        # ==========================================
        print("\n" + "="*55)
        print(" Final results (test set)")
        print("="*55)
        print(f"{'Metric':<15} | {'PyTorch (NN)':<15} | {'XGBoost':<15}")
        print("-" * 55)

        # Added ROC-AUC to the list of metrics
        for metric in["Accuracy", "F1-Score", "Recall", "ROC-AUC"]:
            nn_val = f"{nn_results[metric]:.4f}"
            xgb_val = f"{xgb_results[metric]:.4f}"
            print(f"{metric:<15} | {nn_val:<15} | {xgb_val:<15}")

        print("="*55)

    except FileNotFoundError:
        print("\n[Error] File not found! Make sure it is in data/raw/")
    except Exception as e:
        print(f"\n[Critical Error] main() threw exception:\n{e}")

if __name__ == "__main__":
    main()
