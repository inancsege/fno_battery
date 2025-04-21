# battery_fno/main.py
import torch
import os

# --- Project Imports ---
import config
from utils.helpers import set_seed, create_dir_if_not_exists
from utils.plotting import plot_loss_curves, plot_predictions

from preprocessing.nasa_preprocessor import NasaVitPreprocessor, NasaRulPreprocessor
from preprocessing.ieee_fc_preprocessor import IeeeFcPreprocessor
from preprocessing.xjtu_preprocessor import XjtuPreprocessor
# Import other preprocessors as needed

from models.fno import FNO1D, FNO_RUL_Hybrid # Import relevant model(s)
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

def main():
    # --- Setup ---
    set_seed(config.SEED)
    create_dir_if_not_exists(config.OUTPUT_DIR)
    create_dir_if_not_exists(config.MODEL_SAVE_DIR)
    create_dir_if_not_exists(config.FIGURE_SAVE_DIR)
    create_dir_if_not_exists(config.RESULTS_SAVE_DIR)

    print(f"Using device: {config.DEVICE}")
    print(f"Selected Dataset Type: {config.DATASET_TYPE}")
    print(f"Model Name: {config.MODEL_NAME}")

    # --- Data Preprocessing ---
    # Select the appropriate preprocessor based on config
    if config.DATASET_TYPE == 'NASA_VIT':
        preprocessor = NasaVitPreprocessor(config)
    elif config.DATASET_TYPE == 'NASA_RUL':
        preprocessor = NasaRulPreprocessor(config)
    elif config.DATASET_TYPE == 'IEEE_FC':
        preprocessor = IeeeFcPreprocessor(config)
    elif config.DATASET_TYPE == 'XJTU':
        preprocessor = XjtuPreprocessor(config)
    else:
        raise ValueError(f"Unsupported DATASET_TYPE: {config.DATASET_TYPE}")

    # Run preprocessing to get data loaders and scaler
    (train_loader, val_loader, test_loader), scaler_obj = preprocessor.run()
    # Note: scaler_obj is the preprocessor instance, which holds scalers internally
    # Pass the preprocessor instance to Trainer/Evaluator for inverse transform capability

    # --- Model Initialization ---
    # Select the appropriate model based on config/dataset
    if config.DATASET_TYPE == 'NASA_RUL':
        print("Initializing FNO_RUL_Hybrid model...")
        # Ensure input_dims are correctly set in config for NASA_RUL
        input_dims_rul = {
            'v': config.INPUT_DIM_V, 'i': config.INPUT_DIM_I,
            't': config.INPUT_DIM_T, 'c': config.INPUT_DIM_C
        }
        model = FNO_RUL_Hybrid(
            modes=config.MODES,
            width=config.WIDTH,
            seq_len_cnn=config.SEQ_LEN_CNN,
            seq_len_lstm=config.SEQ_LEN_LSTM,
            input_dims=input_dims_rul
        )
    else: # For NASA_VIT, IEEE_FC, XJTU
        print("Initializing standard FNO1D model...")
        model = FNO1D(
            input_dim=config.INPUT_DIM,
            output_dim=config.OUTPUT_DIM,
            modes=config.MODES,
            width=config.WIDTH,
            depth=config.DEPTH,
            seq_len=config.SEQ_LEN # Pass seq_len if model needs it
        )

    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # --- Training ---
    trainer = Trainer(model, train_loader, val_loader, config, scaler=preprocessor)
    train_losses, val_losses = trainer.train()

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, config.LOSS_PLOT_PATH)

    # --- Evaluation ---
    print("\n--- Evaluating on Validation Set ---")
    # Load the best model saved during training
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    val_evaluator = Evaluator(model, val_loader, config, scaler=preprocessor)
    y_true_val, y_pred_val, val_metrics = val_evaluator.evaluate()

    # Plot validation predictions
    plot_predictions(y_true_val, y_pred_val, "Validation Set", config.PRED_PLOT_PATH_VAL)

    print("\n--- Evaluating on Test Set ---")
    test_evaluator = Evaluator(model, test_loader, config, scaler=preprocessor)
    y_true_test, y_pred_test, test_metrics = test_evaluator.evaluate()

    # Plot test predictions
    plot_predictions(y_true_test, y_pred_test, "Test Set", config.PRED_PLOT_PATH_TEST)

    # Save test results
    test_evaluator.save_results(test_metrics)

    print("\n--- Run Finished ---")

if __name__ == "__main__":
    main()