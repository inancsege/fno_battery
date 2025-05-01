"""
Models for battery capacity prediction
"""

# Import models
from models.fno import FNO1D, FNO_RUL_Hybrid
from models.lstm import LSTM, LSTMAttention
from models.tcn import TCN
from models.ml_models import XGBoostModel, RandomForestModel, LinearRegressionModel, SVRModel

# Map models to their class names
FNO = FNO1D
FNO_RUL = FNO_RUL_Hybrid
LSTM = LSTM
LSTM_ATTN = LSTMAttention
TCN = TCN
XGBoost = XGBoostModel
RandomForest = RandomForestModel
LinearRegression = LinearRegressionModel
SVR = SVRModel

# All available models
__all__ = [
    'FNO',
    'FNO_RUL',
    'LSTM',
    'LSTM_ATTN',
    'TCN',
    'XGBoost',
    'RandomForest',
    'LinearRegression',
    'SVR'
] 