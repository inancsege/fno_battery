# battery_fno/evaluation/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.inf # Avoid division by zero if all true values are zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def calculate_r2(y_true, y_pred):
    """Calculates R-squared (Coefficient of Determination)."""
    return r2_score(y_true, y_pred)

def calculate_pcc(y_true, y_pred):
    """Calculates Pearson Correlation Coefficient."""
    pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return pcc

def calculate_mda(y_true, y_pred):
    """Calculates Mean Directional Accuracy."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2:
        return np.nan # Not enough points to calculate direction
    direction_true = np.sign(np.diff(y_true.flatten()))
    direction_pred = np.sign(np.diff(y_pred.flatten()))
    return np.mean(direction_true == direction_pred) * 100


def compute_all_metrics(y_true, y_pred):
    """Computes and returns a dictionary of all relevant metrics."""
    # Ensure inputs are flat numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    metrics = {
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'PCC': calculate_pcc(y_true, y_pred),
        'MDA': calculate_mda(y_true, y_pred),
    }
    return metrics