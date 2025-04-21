# battery_fno/preprocessing/utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def create_sequences(X, y, seq_len, stride=1):
    """Creates overlapping sequences and corresponding targets."""
    X_seq, y_seq = [], []
    num_samples = X.shape[0]
    for i in range(0, num_samples - seq_len + 1, stride):
        X_seq.append(X[i : i + seq_len])
        # Target is typically the value at the end of the sequence or just after
        # Adjust this logic based on the prediction task (e.g., RUL at end of seq)
        # Using the value corresponding to the *last* input time step
        y_seq.append(y[i + seq_len - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def create_multi_input_sequences(data_dict, seq_lens, target_rul, stride=1):
    """
    Creates sequences for multiple inputs with potentially different lengths.
    Specific to the structure needed by FNO_RUL.py's model.
    Assumes 'v', 'i', 't' use seq_len_cnn and 'c' uses seq_len_lstm.
    """
    v_seq, i_seq, t_seq, c_seq, rul_seq = [], [], [], [], []
    seq_len_cnn = seq_lens['cnn']
    seq_len_lstm = seq_lens['lstm']
    max_len = max(seq_len_cnn, seq_len_lstm)
    num_samples = data_dict['voltage'].shape[0]

    for i in range(0, num_samples - max_len + 1, stride):
        v = data_dict['voltage'][i : i + seq_len_cnn]
        current = data_dict['current'][i : i + seq_len_cnn] # Renamed 'i' to 'current'
        temp = data_dict['temperature'][i : i + seq_len_cnn] # Renamed 't' to 'temp'
        cap = data_dict['capacity'][i : i + seq_len_lstm] # Renamed 'c' to 'cap'

        v_seq.append(v)
        i_seq.append(current)
        t_seq.append(temp)
        c_seq.append(cap)
        # RUL corresponds to the state *at the end* of the longest sequence window
        rul_seq.append(target_rul[i + max_len - 1])

    return {
        'voltage': np.array(v_seq, dtype=np.float32),
        'current': np.array(i_seq, dtype=np.float32),
        'temperature': np.array(t_seq, dtype=np.float32),
        'capacity': np.array(c_seq, dtype=np.float32)
    }, np.array(rul_seq, dtype=np.float32)


def scale_data(data, scaler_type='standard'):
    """Scales the input data using the specified scaler."""
    original_shape = data.shape
    # Reshape data to 2D if it's 3D (batch, seq_len, features)
    if data.ndim == 3:
        data_reshaped = data.reshape(-1, original_shape[-1])
    elif data.ndim == 2:
         data_reshaped = data
    elif data.ndim == 1:
         data_reshaped = data.reshape(-1,1)
    else:
        raise ValueError("Data must be 1D, 2D or 3D")

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    scaled_data = scaler.fit_transform(data_reshaped)

    # Reshape back to original shape
    if data.ndim > 1:
      scaled_data = scaled_data.reshape(original_shape)

    return scaled_data, scaler

def scale_data_with_existing(data, scaler):
    """Scales data using a pre-fitted scaler."""
    original_shape = data.shape
    # Reshape data to 2D if it's 3D (batch, seq_len, features)
    if data.ndim == 3:
        data_reshaped = data.reshape(-1, original_shape[-1])
    elif data.ndim == 2:
         data_reshaped = data
    elif data.ndim == 1:
         data_reshaped = data.reshape(-1,1)
    else:
        raise ValueError("Data must be 1D, 2D or 3D")


    scaled_data = scaler.transform(data_reshaped)
    # Reshape back to original shape
    if data.ndim > 1:
        scaled_data = scaled_data.reshape(original_shape)

    return scaled_data

def split_data(X, y, val_split, test_split, shuffle=True, seed=None):
    """Splits data into training, validation, and test sets."""
    if seed is not None:
        np.random.seed(seed)

    num_samples = X.shape[0] if isinstance(X, np.ndarray) else len(X['voltage']) # Handle dict input
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)

    val_idx = int(num_samples * (1 - val_split - test_split))
    test_idx = int(num_samples * (1 - test_split))

    train_indices = indices[:val_idx]
    val_indices = indices[val_idx:test_idx]
    test_indices = indices[test_idx:]

    if isinstance(X, np.ndarray):
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
    elif isinstance(X, dict): # Handle dict input for multi-input models
        X_train = {key: val[train_indices] for key, val in X.items()}
        y_train = y[train_indices]
        X_val = {key: val[val_indices] for key, val in X.items()}
        y_val = y[val_indices]
        X_test = {key: val[test_indices] for key, val in X.items()}
        y_test = y[test_indices]
    else:
         raise TypeError("X must be a numpy array or a dictionary")


    print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    return X_train, y_train, X_val, y_val, X_test, y_test