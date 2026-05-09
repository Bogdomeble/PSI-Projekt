import os
import json
import torch
from torchinfo import summary

from src.dataset import get_dataloaders
from src.models.neural_net import ChurnNeuralNet
from src.models.xgboost_model import get_xgboost_model
from src.core.trainer_pytorch import train_pytorch, evaluate_pytorch
from src.core.trainer_xgboost import train_and_eval_xgboost_with_feature_selection
from src.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, XGBOOST_DROP_PERCENT

from src.utils.plotting import print_metrics_table


def main():
    print("=========================================")
    print(f"Using compute device: {DEVICE.upper()}")
    print("=========================================\n")

    try:
        train_loader, val_loader, test_loader, xgb_data, input_dim = get_dataloaders()

        nn_model = ChurnNeuralNet(input_dim).to(DEVICE)
        xgb_model_raw = get_xgboost_model()

        # --- PYTORCH ---
        nn_model = train_pytorch(
            nn_model, train_loader, val_loader, DEVICE,
            EPOCHS, LEARNING_RATE, pos_weight_val=2.76
        )
        
        # evaluate on validation set without saving plots
        nn_val_results = evaluate_pytorch(nn_model, val_loader, DEVICE, threshold=0.4, save_plots=False)
        # evaluate on test set with saving plots
        nn_test_results = evaluate_pytorch(nn_model, test_loader, DEVICE, threshold=0.4, save_plots=True)


        # --- XGBOOST ---
        #  get the filtered model and results for the test and validation sets
        xgb_model_filtered, xgb_val_results, xgb_test_results = train_and_eval_xgboost_with_feature_selection(
            xgb_model_raw, xgb_data,
            drop_percent=XGBOOST_DROP_PERCENT,
        )

        print(f"\nAfter training (PyTorch):\n")
        summary(nn_model, input_size=(BATCH_SIZE, input_dim))


        # --- print results ---
        print_metrics_table("RESULTS: VALIDATION SET", nn_val_results, xgb_val_results)
        print_metrics_table("RESULTS: TEST SET", nn_test_results, xgb_test_results)


        # --- export the models ---
        export_dir = "exported-models"
        os.makedirs(export_dir, exist_ok=True) 

        # torch.save(...) for the neural network model
        torch.save(nn_model.state_dict(), os.path.join(export_dir, "nn_model.pt"))
        
        # model.save_model(...) for the xgboost .json model
        xgb_model_filtered.save_model(os.path.join(export_dir, "xgb_model.json"))
        
        # get the columns for xgboost to interpret correctly
        with open(os.path.join(export_dir, "xgb_columns.json"), "w") as f:
            json.dump(xgb_test_results["top_features"], f)
            
        print(f"\n[Success] Models exported to directory: '{export_dir}/'")

    except FileNotFoundError:
        print("\n[Error] File not found! Make sure it is in data/raw/")
    except Exception as e:
        print(f"\n[Critical Error] main() threw exception:\n{e}")

if __name__ == "__main__":
    main()