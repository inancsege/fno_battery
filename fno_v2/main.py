# battery_fno/main.py
import torch
import os
import sys
import argparse

# Add the parent directory to sys.path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Project Imports ---
import fno_v2.config as config
from fno_v2.utils.helpers import set_seed, create_dir_if_not_exists
from fno_v2.utils.plotting import plot_loss_curves, plot_predictions

from fno_v2.preprocessing.nasa_preprocessor import NasaVitPreprocessor, NasaRulPreprocessor
from fno_v2.preprocessing.ieee_fc_preprocessor import IeeeFcPreprocessor
from fno_v2.preprocessing.xjtu_preprocessor import XjtuPreprocessor
from fno_v2.preprocessing.golf_car_preprocessor import GolfCarPreprocessor
# Import other preprocessors as needed

from fno_v2.models.fno import FNO1D, FNO_RUL_Hybrid  # FNO models
from fno_v2.models.lstm import LSTM, LSTMAttention   # LSTM models
from fno_v2.models.tcn import TCN                    # TCN model
from fno_v2.models.ml_models import (
    XGBoostModel, RandomForestModel, LinearRegressionModel, SVRModel  # Traditional ML models
)

from fno_v2.training.trainer import Trainer
from fno_v2.evaluation.evaluator import Evaluator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train and evaluate battery models")
    parser.add_argument("--model", type=str, default=None, 
                        choices=["FNO", "LSTM", "LSTM_ATTN", "TCN", "XGBoost", 
                                 "RandomForest", "LinearRegression", "SVR"],
                        help="Optional: Override model architecture in config.py")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["NASA_VIT", "NASA_RUL", "IEEE_FC", "XJTU", "GOLF_CAR"],
                        help="Optional: Override dataset in config.py")
    return parser.parse_args()

def main():
    # --- Parse Arguments ---
    args = parse_arguments()
    
    # Use MODEL_TYPE from config, but allow command-line override
    selected_model = args.model if args.model else config.MODEL_TYPE
    
    # Override the dataset type if specified in the arguments
    if args.dataset:
        config.DATASET_TYPE = args.dataset
        # Update the active data config and related parameters
        config.ACTIVE_DATA_CONFIG = config.DATA_PATHS[config.DATASET_TYPE]
        if config.DATASET_TYPE == 'NASA_RUL':
            config.SEQ_LEN = max(config.SEQ_LEN_CNN, config.SEQ_LEN_LSTM)
        elif config.DATASET_TYPE in ['NASA_VIT', 'IEEE_FC', 'XJTU', 'GOLF_CAR']:
            config.INPUT_DIM = config.ACTIVE_DATA_CONFIG['input_dim']
            config.OUTPUT_DIM = config.ACTIVE_DATA_CONFIG['output_dim']
            
    # --- Setup ---
    set_seed(config.SEED)
    create_dir_if_not_exists(config.OUTPUT_DIR)
    create_dir_if_not_exists(config.MODEL_SAVE_DIR)
    create_dir_if_not_exists(config.FIGURE_SAVE_DIR)
    create_dir_if_not_exists(config.RESULTS_SAVE_DIR)

    # Update model name if command-line model was used
    if args.model:
        config.MODEL_NAME = f"{selected_model}_{config.DATASET_TYPE}"
        config.BEST_MODEL_PATH = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME}_best.pth")
        config.RESULTS_PATH = os.path.join(config.RESULTS_SAVE_DIR, f"{config.MODEL_NAME}_results.txt")
        config.LOSS_PLOT_PATH = os.path.join(config.FIGURE_SAVE_DIR, f"{config.MODEL_NAME}_loss.png")
        config.PRED_PLOT_PATH_VAL = os.path.join(config.FIGURE_SAVE_DIR, f"{config.MODEL_NAME}_preds_val.png")
        config.PRED_PLOT_PATH_TEST = os.path.join(config.FIGURE_SAVE_DIR, f"{config.MODEL_NAME}_preds_test.png")

    print(f"Using device: {config.DEVICE}")
    print(f"Selected Model: {selected_model}")
    print(f"Selected Dataset Type: {config.DATASET_TYPE}")
    print(f"Model Name: {config.MODEL_NAME}")
    
    # Print model parameters
    print(f"Regularization: Dropout={config.DROPOUT_RATE}, L2={config.WEIGHT_DECAY}, Noise={config.NOISE_LEVEL}")

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
    elif config.DATASET_TYPE == 'GOLF_CAR':
        preprocessor = GolfCarPreprocessor(config)
    else:
        raise ValueError(f"Unsupported DATASET_TYPE: {config.DATASET_TYPE}")

    # Run preprocessing to get data loaders and scaler
    (train_loader, val_loader, test_loader), scaler_obj = preprocessor.run()
    
    # Print dataset information where possible
    print("Data splits loaded for training, validation, and testing")

    # --- Model Initialization ---
    if selected_model == "FNO":
        if config.DATASET_TYPE == 'NASA_RUL':
            print("Initializing FNO_RUL_Hybrid model...")
            # Ensure input_dims are correctly set in config for NASA_RUL
            input_dims_rul = {
                'v': config.INPUT_DIM_V, 'i': config.INPUT_DIM_I,
                't': config.INPUT_DIM_T, 'c': config.INPUT_DIM_C
            }
            model = FNO_RUL_Hybrid(
                modes=config.ACTIVE_MODEL_CONFIG['modes'],
                width=config.ACTIVE_MODEL_CONFIG['width'],
                seq_len_cnn=config.SEQ_LEN_CNN,
                seq_len_lstm=config.SEQ_LEN_LSTM,
                input_dims=input_dims_rul
            )
        else:  # For NASA_VIT, IEEE_FC, XJTU, GOLF_CAR
            print("Initializing standard FNO1D model...")
            model = FNO1D(
                input_dim=config.INPUT_DIM,
                output_dim=config.OUTPUT_DIM,
                modes=config.ACTIVE_MODEL_CONFIG['modes'],
                width=config.ACTIVE_MODEL_CONFIG['width'],
                depth=config.ACTIVE_MODEL_CONFIG['depth'],
                seq_len=config.SEQ_LEN
            )
            
    elif selected_model == "LSTM":
        print("Initializing LSTM model...")
        model = LSTM(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.ACTIVE_MODEL_CONFIG['hidden_dim'],
            num_layers=config.ACTIVE_MODEL_CONFIG['num_layers'],
            output_dim=config.OUTPUT_DIM,
            dropout=config.ACTIVE_MODEL_CONFIG['dropout']
        )
        
    elif selected_model == "LSTM_ATTN":
        print("Initializing LSTM with Attention model...")
        model = LSTMAttention(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.ACTIVE_MODEL_CONFIG['hidden_dim'],
            num_layers=config.ACTIVE_MODEL_CONFIG['num_layers'],
            output_dim=config.OUTPUT_DIM,
            dropout=config.ACTIVE_MODEL_CONFIG['dropout']
        )
        
    elif selected_model == "TCN":
        print("Initializing TCN model...")
        model = TCN(
            input_dim=config.INPUT_DIM,
            output_dim=config.OUTPUT_DIM,
            num_channels=config.ACTIVE_MODEL_CONFIG['num_channels'],
            kernel_size=config.ACTIVE_MODEL_CONFIG['kernel_size'],
            dropout=config.ACTIVE_MODEL_CONFIG['dropout'],
            seq_len=config.SEQ_LEN
        )
        
    elif selected_model == "XGBoost":
        print("Initializing XGBoost model...")
        model = XGBoostModel(
            seq_len=config.SEQ_LEN,
            input_dim=config.INPUT_DIM,
            n_estimators=config.ACTIVE_MODEL_CONFIG['n_estimators'],
            max_depth=config.ACTIVE_MODEL_CONFIG['max_depth'],
            learning_rate=config.ACTIVE_MODEL_CONFIG['learning_rate'],
            use_cuda=config.ACTIVE_MODEL_CONFIG['use_cuda']
        )
        
    elif selected_model == "RandomForest":
        print("Initializing Random Forest model...")
        model = RandomForestModel(
            seq_len=config.SEQ_LEN,
            input_dim=config.INPUT_DIM,
            n_estimators=config.ACTIVE_MODEL_CONFIG['n_estimators'],
            max_depth=config.ACTIVE_MODEL_CONFIG['max_depth']
        )
        
    elif selected_model == "LinearRegression":
        print("Initializing Linear Regression model...")
        model = LinearRegressionModel(
            seq_len=config.SEQ_LEN,
            input_dim=config.INPUT_DIM
        )
        
    elif selected_model == "SVR":
        print("Initializing SVR model...")
        model = SVRModel(
            seq_len=config.SEQ_LEN,
            input_dim=config.INPUT_DIM,
            kernel=config.ACTIVE_MODEL_CONFIG['kernel'],
            C=config.ACTIVE_MODEL_CONFIG['C'],
            epsilon=config.ACTIVE_MODEL_CONFIG['epsilon']
        )
        
    else:
        raise ValueError(f"Unsupported model type: {selected_model}")

    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if hasattr(p, 'requires_grad') and p.requires_grad)} trainable parameters.")

    # --- Training ---
    trainer = Trainer(model, train_loader, val_loader, config, scaler=preprocessor)
    train_losses, val_losses = trainer.train()

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, config.LOSS_PLOT_PATH)

    # --- Evaluation ---
    print("\n--- Evaluating on Validation Set ---")
    
    # For PyTorch neural network models, load the best saved model
    # Skip for ML models which don't use state dictionaries
    if selected_model in ["FNO", "LSTM", "LSTM_ATTN", "TCN"]:
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
        model.eval()  # Set to evaluation mode
    
    # Initialize evaluator
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
    
    # Summary information
    print("\n--- Note on Results ---")
    print(f"Model Type: {selected_model}")
    print(f"Dataset: {config.DATASET_TYPE}")
    print("The model has been trained with:")
    if selected_model in ["FNO", "LSTM", "LSTM_ATTN", "TCN"]:
        print("- Dropout layers to prevent overfitting")
        print("- Noise during training")
        print("- L1 and L2 regularization")
        print("- Data augmentation")
    elif selected_model in ["XGBoost", "RandomForest"]:
        print("- Tree-based ensemble methods")
        print("- Optimal hyperparameters")
    elif selected_model == "LinearRegression":
        print("- Linear model serving as baseline")
    elif selected_model == "SVR":
        print("- Support Vector Regression with RBF kernel")

    print("\n--- Run Finished ---")

if __name__ == "__main__":
    main()