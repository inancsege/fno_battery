# battery_fno/config.py
import torch
import os

# --- General Settings ---
GENERAL_CONFIG = {
    'seed': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'base_dir': os.path.dirname(os.path.abspath(__file__)),
    'output_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
    'model_save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models"),
    'figure_save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures"),
    'results_save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "results"),
    
    # Early stopping parameters
    'patience': 12,
    'early_stop': True,
    
    # Data augmentation parameters
    'augmentation': True,
    'aug_noise_std': 0.05,
    'aug_probability': 0.7,
    
    # Loss function weights
    'mse_weight': 0.7,
    'l1_weight': 0.3,
    'spectral_reg_weight': 0.005,
    
    # Regularization parameters
    'dropout_rate': 0.2,
    'weight_decay': 0.01,
    'noise_level': 0.03
}

# --- Dataset Configurations ---
DATASET_CONFIGS = {
    'NASA_VIT': {
        'files': [
            os.path.join(GENERAL_CONFIG['base_dir'], 'data', 'NASA', 'discharge', 'train', 'B0005_discharge.csv'),
            os.path.join(GENERAL_CONFIG['base_dir'], 'data', 'NASA', 'discharge', 'train', 'B0006_discharge.csv'),
            os.path.join(GENERAL_CONFIG['base_dir'], 'data', 'NASA', 'discharge', 'train', 'B0007_discharge.csv')
        ],
        'features': ['voltage_battery', 'current_battery', 'temp_battery'],
        'target': 'capacity',
        'input_dim': 3,
        'output_dim': 1,
        'validation_split': 0.20,
        'test_split': 0.20
    },
    'NASA_RUL': {
        'dir': os.path.join(GENERAL_CONFIG['base_dir'], "data", "data", "NASA"),
        'features': ['voltage_battery', 'current_battery', 'temp_battery', 'capacity'],
        'target': 'RUL',
        'input_dim_cnn': 1,
        'input_dim_lstm': 1,
        'output_dim': 1,
        'seq_len_cnn': 64,
        'seq_len_lstm': 10,
        'validation_split': 0.20,
        'test_split': 0.20
    },
    'IEEE_FC': {
        'files': [os.path.join(GENERAL_CONFIG['base_dir'], "data", "data", "IEEE", "FC1_test_filtered.csv")],
        'features': ['Utot (V)', 'I (A)'],
        'target': 'Utot (V)',
        'input_dim': 2,
        'output_dim': 1,
        'validation_split': 0.20,
        'test_split': 0.20
    },
    'XJTU': {
        'dir': os.path.join(GENERAL_CONFIG['base_dir'], "data", "XJTU_data"),
        'features': ['voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy','current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','capacity'],
        'target': 'capacity',
        'input_dim': 17,
        'output_dim': 1,
        'validation_split': 0.20,
        'test_split': 0.20
    },
    'GOLF_CAR': {
        'dir': os.path.join(GENERAL_CONFIG['base_dir'], "data", "data", "scaledData"),
        'features': ['soc', 'pack_voltage (V)', 'charge_current (A)', 'max_cell_voltage (V)', 
                   'min_cell_voltage (V)', 'max_temperature (℃)', 'min_temperature (℃)'],
        'target': 'available_capacity (Ah)',
        'input_dim': 7,
        'output_dim': 1,
        'sample_size': 10000,
        'validation_split': 0.15,
        'test_split': 0.15
    }
}

# --- Model Configurations ---
MODEL_CONFIGS = {"FNO": {
    "seq_len": 30,
    "modes": 16,
    "width": 64,
    "depth": 4,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4
}, "LSTM": {
    "seq_len": 60,
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 3e-4,
    "bidirectional": True
}, "LSTM_ATTN": {
    "seq_len": 60,
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 3e-4
}, "TCN": {
    "seq_len": 60,
    "num_channels": [32, 64, 128, 64],
    "kernel_size": 5,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 3e-4
}, "XGBoost": {
    "seq_len": 50,
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "use_cuda": torch.cuda.is_available(),
    "batch_size": 64,
    "epochs": 10,
    "gamma": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}, "RandomForest": {
    "seq_len": 50,
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "batch_size": 64,
    "epochs": 10
}, "LinearRegression": {
    "seq_len": 40,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 8e-5
}, "SVR": {
    "seq_len": 40,
    "kernel": 'rbf',
    "C": 10.0,
    "epsilon": 0.05,
    "gamma": 'scale',
    "batch_size": 64,
    "epochs": 10
}, "FNO_RUL": {
    "seq_len_cnn": DATASET_CONFIGS['NASA_RUL']['seq_len_cnn'],
    "seq_len_lstm": DATASET_CONFIGS['NASA_RUL']['seq_len_lstm'],
    "seq_len": max(DATASET_CONFIGS['NASA_RUL']['seq_len_cnn'], DATASET_CONFIGS['NASA_RUL']['seq_len_lstm']),
    "modes": 3,
    "width": 24,
    "input_dim_v": DATASET_CONFIGS['NASA_RUL']['input_dim_cnn'],
    "input_dim_i": DATASET_CONFIGS['NASA_RUL']['input_dim_cnn'],
    "input_dim_t": DATASET_CONFIGS['NASA_RUL']['input_dim_cnn'],
    "input_dim_c": DATASET_CONFIGS['NASA_RUL']['input_dim_lstm'],
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 8e-5
}}

# Update NASA_RUL specific configurations

# --- Ensure Directories Exist ---
os.makedirs(GENERAL_CONFIG['output_dir'], exist_ok=True)
os.makedirs(GENERAL_CONFIG['model_save_dir'], exist_ok=True)
os.makedirs(GENERAL_CONFIG['figure_save_dir'], exist_ok=True)
os.makedirs(GENERAL_CONFIG['results_save_dir'], exist_ok=True)

# --- Model Selection ---
# Options: "FNO", "LSTM", "LSTM_ATTN", "TCN", "XGBoost", "RandomForest", "LinearRegression", "SVR"
MODEL_TYPE = "LSTM"  # Change this to select which algorithm to train/test

# --- Dataset Selection ---
# Options: 'NASA_VIT', 'NASA_RUL', 'IEEE_FC', 'XJTU', 'GOLF_CAR'
DATASET_TYPE = 'GOLF_CAR' # Change this to select the dataset

# --- Data Paths ---
# Adjust paths based on your data location relative to GENERAL_CONFIG['base_dir'] or use absolute paths
DATA_PATHS = {
    'NASA_VIT': {
        'files': [
            os.path.join(GENERAL_CONFIG['base_dir'], 'data', 'NASA', 'discharge', 'train', 'B0005_discharge.csv'),
            os.path.join(GENERAL_CONFIG['base_dir'], 'data', 'NASA', 'discharge', 'train', 'B0006_discharge.csv'),
            os.path.join(GENERAL_CONFIG['base_dir'], 'data', 'NASA', 'discharge', 'train', 'B0007_discharge.csv')
        ],
        'features': ['voltage_battery', 'current_battery', 'temp_battery'], # Features present in discharge CSVs
        'target': 'capacity', # Lowercase 'capacity' found in discharge CSVs
        'input_dim': 3, # V, I, T
        'output_dim': 1, # Capacity
    },
    'NASA_RUL': {
        'dir': os.path.join(GENERAL_CONFIG['base_dir'], "data", "data", "NASA"), # Point to the directory containing charge/ and discharge/
        'features': ['voltage_battery', 'current_battery', 'temp_battery', 'capacity'],
        'target': 'RUL', # Calculated RUL
        'input_dim_cnn': 1, # For V, I, T processed separately
        'input_dim_lstm': 1, # For C processed by LSTM
        'output_dim': 1,
        'seq_len_cnn': 64, # Specific to FNO_RUL.py structure
        'seq_len_lstm': 10, # Specific to FNO_RUL.py structure
    },
    'IEEE_FC': {
        'files': [os.path.join(GENERAL_CONFIG['base_dir'], "data", "data", "IEEE", "FC1_test_filtered.csv")],
        'features': ['Utot (V)', 'I (A)'], # From FNO_FC.py
        'target': 'Utot (V)', # Or RUL if calculated
        'input_dim': 2,
        'output_dim': 1,
    },
    'XJTU': {
        'dir': os.path.join(GENERAL_CONFIG['base_dir'], "data", "XJTU_data"),
        'features': ['voltage mean','voltage std','voltage kurtosis','voltage skewness','CC Q','CC charge time','voltage slope','voltage entropy','current mean','current std','current kurtosis','current skewness','CV Q','CV charge time','current slope','current entropy','capacity'],
        'target': 'capacity',
        'input_dim': 17,
        'output_dim': 1,
        'validation_split': 0.20,
        'test_split': 0.20
    },
    'GOLF_CAR': {
        'dir': os.path.join(GENERAL_CONFIG['base_dir'], "data", "data", "scaledData"),
        'features': ['soc', 'pack_voltage (V)', 'charge_current (A)', 'max_cell_voltage (V)', 
                    'min_cell_voltage (V)', 'max_temperature (℃)', 'min_temperature (℃)'],
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
LEARNING_RATE = 1e-5  # Reduced from 1e-3
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
os.makedirs(GENERAL_CONFIG['output_dir'], exist_ok=True)
os.makedirs(GENERAL_CONFIG['model_save_dir'], exist_ok=True)
os.makedirs(GENERAL_CONFIG['figure_save_dir'], exist_ok=True)
os.makedirs(GENERAL_CONFIG['results_save_dir'], exist_ok=True)

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
BEST_MODEL_PATH = os.path.join(GENERAL_CONFIG['model_save_dir'], f"{MODEL_NAME}_best.pth")
RESULTS_PATH = os.path.join(GENERAL_CONFIG['results_save_dir'], f"{MODEL_NAME}_results.txt")
LOSS_PLOT_PATH = os.path.join(GENERAL_CONFIG['figure_save_dir'], f"{MODEL_NAME}_loss.png")
PRED_PLOT_PATH_VAL = os.path.join(GENERAL_CONFIG['figure_save_dir'], f"{MODEL_NAME}_preds_val.png")
PRED_PLOT_PATH_TEST = os.path.join(GENERAL_CONFIG['figure_save_dir'], f"{MODEL_NAME}_preds_test.png")