import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# ------------------------------
# FNO
# ------------------------------

# SpectralConv1d class for spectral convolution
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to use
        self.scale = 1 / (in_channels * out_channels)
        self.weights_complex = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, modes) * (in_channel, out_channel, modes)
        return torch.einsum("bim, iom -> bom", input, weights)

    def forward(self, x):
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, L//2 + 1)
        
        # Limit modes to available size
        used_modes = min(self.modes, x_ft.shape[-1])

        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :used_modes] = self.compl_mul1d(
            x_ft[:, :, :used_modes], self.weights_complex[:, :, :used_modes]
        )

        x = torch.fft.irfft(out_ft, n=L, dim=-1)
        return x

# FNO1DBlock class for Fourier Neural Operator block
class FNO1DBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.fourier = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, 1)  # pointwise linear
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.fourier(x) + self.linear(x))

# TimeSeriesFNO class for time series forecasting using FNO
class TimeSeriesFNO(nn.Module):
    def __init__(self, input_features, output_features, seq_len, pred_len, width=64, modes=16, depth=4):
        super().__init__()
        self.input_proj = nn.Conv1d(input_features, width, 1)
        self.blocks = nn.Sequential(*[FNO1DBlock(width, modes) for _ in range(depth)])
        self.output_proj = nn.Conv1d(width, output_features, 1)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x):
        # x shape: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)  # (B, T, output_features)
        return x[:, -self.pred_len:]  # return prediction window

# ------------------------------
# DATA
# ------------------------------

# Function to create sequences from data
def create_sequences(X, y, seq_len):

    sequences = []
    targets = []
    
    for i in range(len(X) - seq_len * 2):
        sequences.append(X[i:i+seq_len])
        targets.append(y[i+seq_len:i+seq_len*2])
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

# Function to load and process data for FC model
def load_and_proc_data_FC(file_list,
                       features=['Utot (V)', 'I (A)'],
                       targets = ['Utot (V)'],
                       SEQ_LEN=100, 
                       BATCH_SIZE=32):
    
    X_seq = []
    y_seq = []

    for file in file_list:
        df = pd.read_csv(file)
        
        X = df[features].values
        y = df[targets[0]].values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler_data = StandardScaler()
        X = scaler_data.fit_transform(X)
        y = y / 2

        X_seq_temp, y_seq_temp = create_sequences(X, y, SEQ_LEN)
        X_seq.extend(X_seq_temp)
        y_seq.extend(y_seq_temp)
        ind = np.arange(len(X_seq))

    np.random.shuffle(ind)
    X_seq = [X_seq[i] for i in ind]
    y_seq = [y_seq[i] for i in ind]

    train_size = int(0.8 * len(X_seq))
    val_size = int(0.1 * len(X_seq))

    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

    X_train = torch.tensor(np.array(X_seq[:train_size]), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_seq[:train_size]), dtype=torch.float32)
    X_val = torch.tensor(np.array(X_seq[train_size:train_size+val_size]), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_seq[train_size:train_size+val_size]), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_seq[train_size+val_size:]), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_seq[train_size+val_size:]), dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return X, y, train_loader, val_loader, test_loader, scaler_data

# Function to load and process data for general use
def load_and_proc_data(file_list,
                       features=['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)', 'min_temperature (℃)', 'soc', 'available_capacity (Ah)'],
                       targets = ['available_capacity (Ah)'],
                       SEQ_LEN=100, 
                       BATCH_SIZE=32):
    
    X_seq = []
    y_seq = []

    for file in file_list:
        df = pd.read_csv(file)
        
        X = df[features].values
        y = df[targets[0]].values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler_data = StandardScaler()
        X = scaler_data.fit_transform(X)
        y = y / 2

        X_seq_temp, y_seq_temp = create_sequences(X, y, SEQ_LEN)
        X_seq.extend(X_seq_temp)
        y_seq.extend(y_seq_temp)

    ind = np.arange(len(X_seq))
    np.random.shuffle(ind)
    X_seq = [X_seq[i] for i in ind]
    y_seq = [y_seq[i] for i in ind]

    train_size = int(0.8 * len(X_seq))
    val_size = int(0.1 * len(X_seq))

    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

    X_train = torch.tensor(np.array(X_seq[:train_size]), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_seq[:train_size]), dtype=torch.float32)
    X_val = torch.tensor(np.array(X_seq[train_size:train_size+val_size]), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_seq[train_size:train_size+val_size]), dtype=torch.float32)
    X_test = torch.tensor(np.array(X_seq[train_size+val_size:]), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_seq[train_size+val_size:]), dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return X, y, train_loader, val_loader, test_loader, scaler_data

# ------------------------------
# TRAINING & TESTING
# ------------------------------

# Function to train the model
def train_model(model, train_loader, val_loader, loss_fn, optimizer, model_save_file="models/best_model.pth", device=torch.device("cpu"), num_epochs=20):
    
    model.to(device)
    
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            if batch_y.ndim == 2:
                batch_y = batch_y.unsqueeze(-1)
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.5f}")

# Function to evaluate the model
def evaluate_model(model, test_loader, model_save_file, output_save_file, plot_model_name='model', plot_fig = True, device=torch.device("cpu"), return_error_results = False, use_gpu = True):
    model.load_state_dict(torch.load(model_save_file))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            if use_gpu:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    pred_out = all_preds
    target_out = all_targets

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Ensure predictions and targets are compatible
    all_preds = all_preds.squeeze()
    all_targets = all_targets[:, -1]  # Use the last value of the target sequence

    # Compute error metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    # Compute MAPE (avoid division by zero)
    non_zero_mask = all_targets != 0
    mape = np.mean(np.abs((all_targets[non_zero_mask] - all_preds[non_zero_mask]) / all_targets[non_zero_mask])) * 100

    # Compute Pearson Correlation Coefficient (PCC)
    pcc = np.array([pearsonr(all_targets, all_preds)[0]])

    # Compute Mean Directional Accuracy (MDA)
    direction_actual = np.sign(np.diff(all_targets))
    direction_pred = np.sign(np.diff(all_preds))

    mda = np.mean(direction_actual == direction_pred)

    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Test PCC: {pcc}")
    print(f"Test MDA: {mda}")

    with open(output_save_file, "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test MAE: {mae:.4f}\n")
        f.write(f"Test R²: {r2:.4f}\n")
        f.write(f"Test MAPE: {mape:.2f}%\n")
        f.write(f"Test PCC: {pcc}\n")
        f.write(f"Test MDA: {mda}\n")

    if plot_fig:
        plt.figure(figsize=(16, 10))  # Increased figure size
        start = 0
        for i in range(1):
            length = pred_out[i].shape[0]
            x = np.arange(start, start + length)
            y_pred = pred_out[i].squeeze()
            y_target = target_out[i].squeeze()

            # Optional: Select only one feature if still 2D
            if y_pred.ndim == 2:
                y_pred = y_pred[:, 0]
                y_target = y_target[:, 0]

            plt.plot(x, y_pred, label="Predicted" if i == 0 else "", linestyle="dashed", alpha=0.7)
            plt.plot(x, y_target, label="Target" if i == 0 else "", linestyle="solid", alpha=0.7)
            start += length

        plt.ylabel("SOH (%)")
        plt.xticks([])
        plt.title(f'{plot_model_name} Example')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'outputs/figures/{plot_model_name}.png')

    if return_error_results:
        return rmse, mae, r2, mape, pcc, mda