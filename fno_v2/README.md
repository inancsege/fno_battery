# FNO Battery/Fuel Cell State Estimation (fno_v2)

This project implements a Fourier Neural Operator (FNO) based model for battery state estimation tasks, such as State of Health (SoH) or Remaining Useful Life (RUL) prediction. It is designed to work with various battery datasets like NASA, IEEE FC, and XJTU.

## Project Structure

```
fno_v2/
├── main.py                 # Main script to run training and evaluation
├── main_direct.py          # Alternative main script for direct execution
├── run.py                  # Runner script for simplified execution
├── run_models.sh           # Shell script to run multiple model configurations (Linux/Mac)
├── run_models.ps1          # PowerShell script to run multiple model configurations (Windows)
├── config.py               # Configuration file for datasets, models, and hyperparameters
├── datasets/
│   └── battery_dataset.py  # PyTorch Dataset classes
├── evaluation/
│   ├── evaluator.py        # Handles model evaluation
│   └── metrics.py          # Defines evaluation metrics (RMSE, MAE, R2, etc.)
├── models/
│   ├── fno.py              # FNO model definitions (FNO1D, FNO_RUL_Hybrid)
│   └── layers.py           # Custom layers (SpectralConv1d, FNO1DBlock)
├── preprocessing/
│   ├── base_preprocessor.py # Abstract base class for preprocessors
│   ├── nasa_preprocessor.py # Preprocessor for NASA datasets (VIT & RUL)
│   ├── ieee_fc_preprocessor.py # Preprocessor for IEEE FC dataset
│   ├── xjtu_preprocessor.py  # Preprocessor for XJTU dataset
│   └── utils.py            # Preprocessing utility functions (scaling, sequencing)
├── training/
│   └── trainer.py          # Handles the model training loop
└── utils/
    ├── helpers.py          # Helper functions (e.g., set_seed)
    └── plotting.py         # Plotting utilities (loss curves, predictions)
├── outputs/                # (Created automatically) Stores results, models, figures
│   ├── figures/
│   ├── models/
│   └── results/
└── README.md               # This file
```

Additional project files in the parent directory:
```
fno_battery/
├── FNO.py                  # Original FNO implementation
├── FNO_FC.py               # FNO for Fuel Cell applications
├── FNO_RUL.py              # FNO for Remaining Useful Life prediction
├── main.py                 # Original main script
├── preproc.py              # Original preprocessing utilities
├── utils.py                # Original utility functions
├── fno_forecasting.png     # Visualization of FNO forecasting
├── requirements.txt        # Project dependencies
├── __init__.py             # Package initialization
└── data/                   # Directory for datasets
```

## Features

*   **Fourier Neural Operators (FNO):** Utilizes FNO for efficient time series analysis and state estimation.
*   **Modular Design:** Clearly separated components for data preprocessing, model definition, training, and evaluation.
*   **Dataset Support:** Includes preprocessors for common battery datasets (NASA, IEEE FC, XJTU). Easily extendable to new datasets by implementing a new `BasePreprocessor`.
*   **Configurable:** Easily configure dataset paths, model hyperparameters, and training settings via `config.py`.
*   **Evaluation Metrics:** Computes standard regression metrics (RMSE, MAE, R2, MAPE, PCC, MDA).
*   **Visualization:** Generates plots for loss curves and prediction comparisons.
*   **Batch Execution:** Run multiple model configurations using the provided shell scripts (`run_models.sh` or `run_models.ps1`).

## Setup

1.  **Clone the repository (if applicable).**
2.  **Install dependencies:** Create a Python environment (e.g., using `conda` or `venv`) and install the required packages:
    ```bash
    # Example using pip:
    pip install -r requirements.txt
    ```
    *Note: Ensure you install the correct PyTorch version for your system (CPU/GPU).*
3.  **Download Data:** Place the required datasets (NASA, IEEE FC, XJTU) in the appropriate locations. The expected paths are relative to the `fno_v2` directory and are configured in `config.py`. Adjust the paths in `config.py` if your data is located elsewhere.
    *   Example: For XJTU, data is expected in `../data/XJTU_data/*.csv` relative to `config.py`.

## Configuration

Edit the `fno_v2/config.py` file to set up your experiment:

1.  **`DATASET_TYPE`:** Select the dataset to use ('NASA_VIT', 'NASA_RUL', 'IEEE_FC', 'XJTU').
2.  **`DATA_PATHS`:** Verify or update the file/directory paths for the selected `DATASET_TYPE`. Ensure feature names and target variable names match your CSV files.
3.  **Model Hyperparameters:** Adjust `SEQ_LEN`, `MODES`, `WIDTH`, `DEPTH` as needed. Specific parameters for `NASA_RUL` are handled separately.
4.  **Training Hyperparameters:** Set `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `VALIDATION_SPLIT`, `TEST_SPLIT`.
5.  **Output Directories:** `OUTPUT_DIR` and its subdirectories (`MODEL_SAVE_DIR`, `FIGURE_SAVE_DIR`, `RESULTS_SAVE_DIR`) are defined relative to the `config.py` location.

## Usage

There are several ways to run the model:

### Using the main script:
```bash
python -m fno_v2.main
```

### Using the run script (simplified interface):
```bash
python -m fno_v2.run
```

### Running multiple models with different configurations:
```bash
# On Linux/Mac:
./fno_v2/run_models.sh

# On Windows:
./fno_v2/run_models.ps1
```

The execution process:
1.  Loads the configuration from `config.py`.
2.  Initializes the appropriate preprocessor based on `DATASET_TYPE`.
3.  Loads and preprocesses the data, creating sequences and splitting into train/validation/test sets.
4.  Initializes the corresponding FNO model.
5.  Trains the model, saving the best version based on validation loss to `outputs/models/`.
6.  Plots training/validation loss curves and saves them to `outputs/figures/`.
7.  Evaluates the best model on the validation and test sets.
8.  Plots predictions vs. true values for validation and test sets and saves them to `outputs/figures/`.
9.  Saves evaluation metrics to `outputs/results/`.

## Extending

*   **Adding a New Dataset:**
    1.  Create a new preprocessor class inheriting from `BasePreprocessor` in `fno_v2/preprocessing/`.
    2.  Implement the `load_data`, `preprocess`, and `create_sequences` methods specific to your dataset format.
    3.  Add a new entry to the `DATA_PATHS` dictionary in `config.py`.
    4.  Update the `if/elif` block in `main.py` to instantiate your new preprocessor when its `DATASET_TYPE` is selected.
*   **Adding a New Model:**
    1.  Define your model class in `fno_v2/models/`.
    2.  Update the model initialization logic in `main.py` to select your new model based on configuration or dataset type.
*   **Customizing Training:**
    1.  Modify the `Trainer` class in `fno_v2/training/trainer.py` to implement custom training loops or loss functions. 