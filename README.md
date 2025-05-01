# FNO Battery Capacity Prediction

A deep learning framework for battery state estimation using Fourier Neural Operators (FNO) and other machine learning models.

## Overview

This project implements various models for battery state estimation tasks such as State of Health (SoH) and Remaining Useful Life (RUL) prediction, with a focus on FNO-based architectures. The framework supports multiple battery datasets and offers a modular design for easy extension.

## Features

- **Multiple Models**: FNO, LSTM, LSTM with Attention, TCN, XGBoost, Random Forest, SVR, and Linear Regression
- **Dataset Support**: NASA (VIT & RUL), IEEE FC, XJTU, and GOLF_CAR datasets
- **Easy Configuration**: Simple configuration of models, datasets, and training parameters
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, R², etc.) and visualization tools
- **Extensible Design**: Modular architecture for adding new models and datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fno_battery.git
cd fno_battery

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with default settings (FNO model on NASA_VIT dataset)
python main.py

# Specify different model and dataset
python main.py --model LSTM --dataset NASA_RUL

# Run in evaluation-only mode (requires pre-trained model)
python main.py --model FNO --dataset NASA_VIT --eval_only
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Model type (FNO, LSTM, LSTM_ATTN, TCN, XGBoost, RandomForest, LinearRegression, SVR) |
| `--dataset` | Dataset to use (NASA_VIT, NASA_RUL, IEEE_FC, XJTU, GOLF_CAR) |
| `--seq_len` | Sequence length for time series data |
| `--batch_size` | Training batch size |
| `--epochs` | Number of training epochs |
| `--learning_rate` | Learning rate for optimization |
| `--eval_only` | Only run evaluation on test data |
| `--seed` | Random seed for reproducibility |
| `--gpu` | GPU device ID to use |
| `--debug` | Enable debug mode with verbose logging |

### Batch Execution

Run multiple model configurations:

```bash
# On Linux/Mac
./run_models.sh

# On Windows
.\run_models.ps1
```

## Project Structure

```
├── main.py                 # Main script for training and evaluation
├── config.py               # Configuration settings
├── models/                 # Model implementations
├── preprocessing/          # Data preprocessing modules
│   └── data_processor.py   # Main data processing class
├── training/               # Training utilities
│   └── trainer.py          # Model training implementation
├── evaluation/             # Evaluation utilities
│   └── evaluator.py        # Model evaluation implementation
├── utils/                  # Utility functions
│   ├── logger.py           # Logging configuration
│   ├── helpers.py          # Helper functions
│   └── plotting.py         # Visualization tools
├── outputs/                # Generated results (created automatically)
│   ├── figures/            # Saved plots and visualizations
│   ├── results/            # Metrics and numerical results
│   └── models/             # Saved model weights
└── requirements.txt        # Project dependencies
```

## Configuration

Edit `config.py` to customize model hyperparameters, dataset paths, and training settings. The file includes:

- `MODEL_CONFIGS`: Hyperparameters for each model type
- `DATASET_CONFIGS`: Paths and settings for each dataset
- `GENERAL_CONFIG`: General settings applicable to all experiments

## Extending the Framework

### Adding a New Model

1. Create a new model file in the `models` directory
2. Update `config.py` with the model's hyperparameters
3. Import the model in `models/__init__.py`

### Adding a New Dataset

1. Create a new preprocessor in the `preprocessing` directory
2. Update `config.py` with the dataset configuration
3. Update `preprocessing/data_processor.py` to handle the new dataset

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{yourarticle,
  title={Your paper title},
  author={Your name},
  journal={Journal name},
  year={Year}
}
``` 