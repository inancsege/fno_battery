# FNO Battery/Fuel Cell State Estimation (fno_v2)

This project implements a Fourier Neural Operator (FNO) based model for battery state estimation tasks, such as State of Health (SoH) or Remaining Useful Life (RUL) prediction. It is designed to work with various battery datasets like NASA, IEEE FC, and XJTU.

## Project Structure

```
fno_v2/
├── main.py                 # Main script to run training and evaluation
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

## Features

*   **Fourier Neural Operators (FNO):** Utilizes FNO for efficient time series analysis.
*   **Modular Design:** Clearly separated components for data preprocessing, model definition, training, and evaluation.
*   **Dataset Support:** Includes preprocessors for common battery datasets (NASA, IEEE FC, XJTU). Easily extendable to new datasets by implementing a new `BasePreprocessor`.
*   **Configurable:** Easily configure dataset paths, model hyperparameters, and training settings via `config.py`.
*   **Evaluation Metrics:** Computes standard regression metrics (RMSE, MAE, R2, MAPE, PCC, MDA).
*   **Visualization:** Generates plots for loss curves and prediction comparisons.

## Setup

1.  **Clone the repository (if applicable).**
2.  **Install dependencies:** Create a Python environment (e.g., using `conda` or `venv`) and install the required packages. You will likely need:
    *   `torch` (PyTorch)
    *   `numpy`
    *   `pandas`
    *   `scikit-learn`
    *   `matplotlib`
    *   `tqdm`
    *   `scipy`
    ```bash
    # Example using pip:
    pip install torch numpy pandas scikit-learn matplotlib tqdm scipy seaborn
    ```
    *Note: Ensure you install the correct PyTorch version for your system (CPU/GPU).*
3.  **Download Data:** Place the required datasets (NASA, IEEE FC, XJTU) in the appropriate locations. The expected paths are relative to the `fno_v2` directory and are configured in `config.py`. Adjust the paths in `config.py` if your data is located elsewhere.
    *   Example: For XJTU, data is expected in `../data/XJTU_data/*.csv` relative to `config.py`.

## Configuration

Edit the `fno_battery/fno_v2/config.py` file to set up your experiment:

1.  **`DATASET_TYPE`:** Select the dataset to use ('NASA_VIT', 'NASA_RUL', 'IEEE_FC', 'XJTU').
2.  **`DATA_PATHS`:** Verify or update the file/directory paths for the selected `DATASET_TYPE`. Ensure feature names and target variable names match your CSV files.
3.  **Model Hyperparameters:** Adjust `SEQ_LEN`, `MODES`, `WIDTH`, `DEPTH` as needed. Specific parameters for `NASA_RUL` are handled separately.
4.  **Training Hyperparameters:** Set `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `VALIDATION_SPLIT`, `TEST_SPLIT`.
5.  **Output Directories:** `OUTPUT_DIR` and its subdirectories (`MODEL_SAVE_DIR`, `FIGURE_SAVE_DIR`, `RESULTS_SAVE_DIR`) are defined relative to the `config.py` location.

## Usage

Run the main script from the parent directory (`fno_battery`) or ensure the `fno_v2` package is correctly importable:

```bash
python -m fno_v2.main
```

Or, if you are inside the `fno_battery` directory:

```bash
python fno_v2/main.py
```

The script will:
1.  Load the configuration from `config.py`.
2.  Initialize the appropriate preprocessor based on `DATASET_TYPE`.
3.  Load and preprocess the data, creating sequences and splitting into train/validation/test sets.
4.  Initialize the corresponding FNO model.
5.  Train the model, saving the best version based on validation loss to `outputs/models/`.
6.  Plot training/validation loss curves and save them to `outputs/figures/`.
7.  Evaluate the best model on the validation and test sets.
8.  Plot predictions vs. true values for validation and test sets and save them to `outputs/figures/`.
9.  Save evaluation metrics to `outputs/results/`.

## Extending

*   **Adding a New Dataset:**
    1.  Create a new preprocessor class inheriting from `BasePreprocessor` in `fno_battery/fno_v2/preprocessing/`.
    2.  Implement the `load_data`, `preprocess`, and `create_sequences` methods specific to your dataset format.
    3.  Add a new entry to the `DATA_PATHS` dictionary in `config.py`.
    4.  Update the `if/elif` block in `main.py` to instantiate your new preprocessor when its `DATASET_TYPE` is selected.
*   **Adding a New Model:**
    1.  Define your model class in `fno_battery/fno_v2/models/`.
    2.  Update the model initialization logic in `main.py` to select your new model based on configuration or dataset type. 