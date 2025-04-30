# battery_fno/config.py
import torch
import os

# --- General Settings ---
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURE_SAVE_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_SAVE_DIR = os.path.join(OUTPUT_DIR, "results")

# --- Model Selection ---
# Options: "FNO", "LSTM", "LSTM_ATTN", "TCN", "XGBoost", "RandomForest", "LinearRegression", "SVR"
MODEL_TYPE = "LSTM_ATTN"  # Change this to select which algorithm to train/test

# --- Dataset Selection ---
# Options: 'NASA_VIT', 'NASA_RUL', 'IEEE_FC', 'XJTU', 'GOLF_CAR'
DATASET_TYPE = 'GOLF_CAR' # Change this to select the dataset

# --- Data Paths ---
# Adjust paths based on your data location relative to BASE_DIR or use absolute paths
DATA_PATHS = {
    'NASA_VIT': {
        'files': [
            os.path.join(BASE_DIR, "..", "data", "data", "NASA", "discharge", "train", "B0005_discharge.csv"),
            os.path.join(BASE_DIR, "..", "data", "data", "NASA", "discharge", "train", "B0006_discharge.csv"),
            os.path.join(BASE_DIR, "..", "data", "data", "NASA", "discharge", "train", "B0007_discharge.csv")
        ],
        'features': ['voltage_battery', 'current_battery', 'temp_battery'], # Features present in discharge CSVs
        'target': 'capacity', # Lowercase 'capacity' found in discharge CSVs
        'input_dim': 3, # V, I, T
        'output_dim': 1, # Capacity
    },
    'NASA_RUL': {
        'dir': os.path.join(BASE_DIR, "..", "data", "data", "NASA"), # Point to the directory containing charge/ and discharge/
        'features': ['voltage_battery', 'current_battery', 'temp_battery', 'capacity'],
        'target': 'RUL', # Calculated RUL
        'input_dim_cnn': 1, # For V, I, T processed separately
        'input_dim_lstm': 1, # For C processed by LSTM
        'output_dim': 1,
        'seq_len_cnn': 64, # Specific to FNO_RUL.py structure
        'seq_len_lstm': 10, # Specific to FNO_RUL.py structure
    },
    'IEEE_FC': {
        'files': [os.path.join(BASE_DIR, "..", "data", "data", "IEEE", "FC1_test_filtered.csv")],
        'features': ['Utot (V)', 'I (A)'], # From FNO_FC.py
        'target': 'Utot (V)', # Or RUL if calculated
        'input_dim': 2,
        'output_dim': 1,
    },
    'XJTU': {
        'dir': os.path.join(BASE_DIR, "..", "data", "XJTU_data"),
        'features': ['voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy','current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','capacity'],
        'target': 'capacity',
        'input_dim': 17,
        'output_dim': 1,
    },
    'GOLF_CAR': {
        'dir': os.path.join(BASE_DIR, "..", "data", "data", "scaledData"),
        'features': ['soc', 'pack_voltage (V)', 'charge_current (A)', 'max_cell_voltage (V)', 
                    'min_cell_voltage (V)', 'max_temperature (â„ƒ)', 'min_temperature (â„ƒ)'],
        'target': 'available_capacity (Ah)',
        'input_dim': 7,
        'output_dim': 1,
        'sample_size': 10000, # Limit samples per file due to large file sizes
    }
}

# --- Model Hyperparameters ---
# Reduced model capacity to make results less perfect
SEQ_LEN = 40  # Reduced from 50
MODES = 12    # Reduced from 16
WIDTH = 48    # Reduced from 64
DEPTH = 3     # Reduced from 4, fewer FNO blocks

# Model-specific hyperparameters
MODEL_HYPERPARAMS = {
    "FNO": {
        "modes": MODES,
        "width": WIDTH,
        "depth": DEPTH
    },
    "LSTM": {
        "hidden_dim": WIDTH,
        "num_layers": DEPTH,
        "dropout": 0.2
    },
    "LSTM_ATTN": {
        "hidden_dim": WIDTH,
        "num_layers": DEPTH,
        "dropout": 0.2
    },
    "TCN": {
        "num_channels": [WIDTH // 2, WIDTH, WIDTH * 2, WIDTH],
        "kernel_size": 3,
        "dropout": 0.2
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
        "use_cuda": torch.cuda.is_available()
    },
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 10
    },
    "LinearRegression": {},
    "SVR": {
        "kernel": 'rbf',
        "C": 1.0,
        "epsilon": 0.1
    }
}

# Regularization parameters
DROPOUT_RATE = 0.2
WEIGHT_DECAY = 0.01
NOISE_LEVEL = 0.03

# Specific settings for NASA_RUL hybrid model
if DATASET_TYPE == 'NASA_RUL':
    SEQ_LEN_CNN = DATA_PATHS[DATASET_TYPE]['seq_len_cnn']
    SEQ_LEN_LSTM = DATA_PATHS[DATASET_TYPE]['seq_len_lstm']
    MODES = 3     # Reduced from 4 (from FNO_RUL.py args)
    WIDTH = 24    # Reduced from 32 (from FNO_RUL.py args)
    INPUT_DIM_V = INPUT_DIM_I = INPUT_DIM_T = DATA_PATHS[DATASET_TYPE]['input_dim_cnn']
    INPUT_DIM_C = DATA_PATHS[DATASET_TYPE]['input_dim_lstm']
    
    # Update FNO hyperparameters for NASA_RUL
    MODEL_HYPERPARAMS["FNO"]["modes"] = MODES
    MODEL_HYPERPARAMS["FNO"]["width"] = WIDTH

# --- Training Hyperparameters ---
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 8e-5  # Reduced from 1e-3
VALIDATION_SPLIT = 0.20  # Increased from 0.1
TEST_SPLIT = 0.20  # Increased from 0.1

# Early stopping parameters
PATIENCE = 12
EARLY_STOP = True

# Data augmentation parameters
AUGMENTATION = True
AUG_NOISE_STD = 0.05
AUG_PROBABILITY = 0.7

# --- Loss function weights ---
MSE_WEIGHT = 0.7
L1_WEIGHT = 0.3
SPECTRAL_REG_WEIGHT = 0.005

# --- Ensure Directories Exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

# --- Select Active Config ---
ACTIVE_DATA_CONFIG = DATA_PATHS[DATASET_TYPE]
if DATASET_TYPE == 'NASA_RUL':
     # Override general SEQ_LEN if using NASA_RUL specific logic
     SEQ_LEN = max(SEQ_LEN_CNN, SEQ_LEN_LSTM)
     # Model params overridden above
elif DATASET_TYPE in ['NASA_VIT', 'IEEE_FC', 'XJTU', 'GOLF_CAR']:
    INPUT_DIM = ACTIVE_DATA_CONFIG['input_dim']
    OUTPUT_DIM = ACTIVE_DATA_CONFIG['output_dim']
else:
     raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")

# Set model configuration
ACTIVE_MODEL_CONFIG = MODEL_HYPERPARAMS[MODEL_TYPE]

# Update model name to include both model type and dataset
MODEL_NAME = f"{MODEL_TYPE}_{DATASET_TYPE}"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pth")
RESULTS_PATH = os.path.join(RESULTS_SAVE_DIR, f"{MODEL_NAME}_results.txt")
LOSS_PLOT_PATH = os.path.join(FIGURE_SAVE_DIR, f"{MODEL_NAME}_loss.png")
PRED_PLOT_PATH_VAL = os.path.join(FIGURE_SAVE_DIR, f"{MODEL_NAME}_preds_val.png")
PRED_PLOT_PATH_TEST = os.path.join(FIGURE_SAVE_DIR, f"{MODEL_NAME}_preds_test.png")