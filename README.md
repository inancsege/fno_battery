# Battery Capacity Prediction with Multiple AI Models

This repository contains a collection of AI models for predicting battery capacity and state of health using time-series data. The project supports several state-of-the-art models and includes support for multiple battery datasets.

## Models Supported

1. **FNO (Fourier Neural Operator)**: A powerful model for learning operators in PDE-like systems, adapted for time-series data
2. **LSTM**: Standard and Attention-based LSTM models for sequential data
3. **TCN (Temporal Convolutional Network)**: Convolution-based architecture for time-series
4. **XGBoost**: Gradient boosting trees implementation, efficient for tabular data
5. **Random Forest**: Ensemble tree-based model
6. **Linear Regression**: Simple baseline model
7. **SVR (Support Vector Regression)**: Kernel-based regression model

## Datasets Supported

- **NASA_VIT**: NASA battery dataset with voltage, current, and temperature features
- **NASA_RUL**: NASA dataset for remaining useful life prediction
- **IEEE_FC**: IEEE fuel cell dataset
- **XJTU**: XJTU-SY battery dataset
- **GOLF_CAR**: Golf cart battery telemetry data

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fno_battery.git
   cd fno_battery
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

To train a model, use the `main.py` script with appropriate arguments:

```bash
python fno_v2/main.py --model [MODEL_TYPE] --dataset [DATASET_TYPE]
```

Where:
- `[MODEL_TYPE]` is one of: FNO, LSTM, LSTM_ATTN, TCN, XGBoost, RandomForest, LinearRegression, SVR
- `[DATASET_TYPE]` is one of: NASA_VIT, NASA_RUL, IEEE_FC, XJTU, GOLF_CAR

Example:
```bash
python fno_v2/main.py --model LSTM --dataset GOLF_CAR
```

### Model Configuration

Model parameters can be adjusted in the `fno_v2/config.py` file. Key parameters include:

- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate for optimization
- `SEQ_LEN`: Sequence length for time-series data
- Model-specific parameters (MODES, WIDTH, DEPTH, etc.)

### Directory Structure

```
fno_battery/
├── data/                 # Raw datasets
│   ├── NASA/
│   ├── IEEE/
│   ├── XJTU_data/
│   └── scaledData/
├── fno_v2/
│   ├── models/           # Model definitions
│   ├── preprocessing/    # Data preprocessing classes
│   ├── training/         # Training utilities
│   ├── evaluation/       # Evaluation utilities
│   ├── utils/            # Helper functions
│   ├── config.py         # Configuration settings
│   └── main.py           # Main execution script
└── requirements.txt      # Package dependencies
```

## Performance

Model performance varies based on the dataset and configuration. Typical metrics include:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
- PCC (Pearson Correlation Coefficient)
- MDA (Mean Directional Accuracy)

Evaluation results are saved in the `outputs/results/` directory.

## Example Commands

```bash
# Train FNO model on NASA_VIT data
python fno_v2/main.py --model FNO --dataset NASA_VIT

# Train LSTM model with golf car battery data
python fno_v2/main.py --model LSTM --dataset GOLF_CAR

# Train XGBoost model on XJTU data
python fno_v2/main.py --model XGBoost --dataset XJTU
```

## Results Visualization

Prediction visualizations and loss curves are saved in the `outputs/figures/` directory after training. 