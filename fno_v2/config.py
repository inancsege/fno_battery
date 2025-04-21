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

# --- Dataset Selection ---
# Options: 'NASA_VIT', 'NASA_RUL', 'IEEE_FC', 'XJTU'
DATASET_TYPE = 'NASA_VIT'

# --- Data Paths ---
# Adjust paths based on your data location relative to BASE_DIR or use absolute paths
DATA_PATHS = {
    'NASA_VIT': {
        'files': [
            os.path.join(BASE_DIR, "..", "data/data/NASA/charge/train/B0005_charge.csv"),
            os.path.join(BASE_DIR, "..", "data/data/NASA/charge/train/B0006_charge.csv"),
            os.path.join(BASE_DIR, "..", "data/data/NASA/charge/train/B0007_charge.csv")
        ],
        'features': ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Capacity'], # Example names, adjust based on actual CSV headers
        'target': 'Capacity', # Or RUL if calculated
        'input_dim': 4, # V, I, T, C
        'output_dim': 1, # RUL or Capacity
    },
    'NASA_RUL': {
        'dir': os.path.join(BASE_DIR, "data/data/NASA/train/B0018.csv"),
        'features': ['voltage_battery', 'current_battery', 'temp_battery', 'capacity'],
        'target': 'RUL', # Calculated RUL
        'input_dim_cnn': 1, # For V, I, T processed separately
        'input_dim_lstm': 1, # For C processed by LSTM
        'output_dim': 1,
        'seq_len_cnn': 64, # Specific to FNO_RUL.py structure
        'seq_len_lstm': 10, # Specific to FNO_RUL.py structure
    },
    'IEEE_FC': {
        'files': [os.path.join(BASE_DIR, "data/data/IEEE/FC1_test_filtered.csv")],
        'features': ['Utot (V)', 'I (A)'], # From FNO_FC.py
        'target': 'Utot (V)', # Or RUL if calculated
        'input_dim': 2,
        'output_dim': 1,
    },
    'XJTU': {
        'dir': os.path.join(BASE_DIR, "data/XJTU_data"),
        'features': ['voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy','current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','capacity'],
        'target': 'capacity',
        'input_dim': 17,
        'output_dim': 1,
    }
}

# --- Model Hyperparameters ---
# General FNO settings (used by NASA_VIT, IEEE_FC, XJTU unless overridden)
SEQ_LEN = 50
MODES = 16
WIDTH = 64
DEPTH = 4 # Number of FNO blocks (from utils.py TimeSeriesFNO)
INPUT_DIM = DATA_PATHS[DATASET_TYPE]['input_dim']
OUTPUT_DIM = DATA_PATHS[DATASET_TYPE]['output_dim']

# Specific settings for NASA_RUL hybrid model
if DATASET_TYPE == 'NASA_RUL':
    SEQ_LEN_CNN = DATA_PATHS[DATASET_TYPE]['seq_len_cnn']
    SEQ_LEN_LSTM = DATA_PATHS[DATASET_TYPE]['seq_len_lstm']
    MODES = 4 # From FNO_RUL.py args
    WIDTH = 32 # From FNO_RUL.py args (hidden_channels)
    INPUT_DIM_V = INPUT_DIM_I = INPUT_DIM_T = DATA_PATHS[DATASET_TYPE]['input_dim_cnn']
    INPUT_DIM_C = DATA_PATHS[DATASET_TYPE]['input_dim_lstm']
    # Note: The FNO_RUL model structure is quite different and would require
    # a separate model definition if strictly followed. This config assumes
    # we might adapt the general FNO or need these params for preprocessing.

# --- Training Hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1 # Fraction for validation set
TEST_SPLIT = 0.1 # Fraction for test set (from remaining data after train/val)

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
elif DATASET_TYPE in ['NASA_VIT', 'IEEE_FC', 'XJTU']:
    INPUT_DIM = ACTIVE_DATA_CONFIG['input_dim']
    OUTPUT_DIM = ACTIVE_DATA_CONFIG['output_dim']
else:
     raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")

MODEL_NAME = f"FNO_{DATASET_TYPE}"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_best.pth")
RESULTS_PATH = os.path.join(RESULTS_SAVE_DIR, f"{MODEL_NAME}_results.txt")
LOSS_PLOT_PATH = os.path.join(FIGURE_SAVE_DIR, f"{MODEL_NAME}_loss.png")
PRED_PLOT_PATH_VAL = os.path.join(FIGURE_SAVE_DIR, f"{MODEL_NAME}_preds_val.png")
PRED_PLOT_PATH_TEST = os.path.join(FIGURE_SAVE_DIR, f"{MODEL_NAME}_preds_test.png")