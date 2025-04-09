import os
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.fft
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# ---------------- SpectralConv1D ------------------
class SpectralConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(modes, in_channels, out_channels, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize, seq_len, _ = x.shape
        x = x.permute(0, 2, 1)  # [batch, in_channels, seq_len]
        x_ft = torch.fft.rfft(x)  # [batch, in_channels, freq]

        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), device=x.device, dtype=torch.cfloat)

        x_ft_low = x_ft[:, :, :self.modes].permute(0, 2, 1)  # [batch, modes, in_channels]
        out_ft[:, :, :self.modes] = torch.einsum("bmi,mio->bmo", x_ft_low, self.weights).permute(0, 2, 1)

        x = torch.fft.irfft(out_ft, n=seq_len)
        x = x.permute(0, 2, 1)  # [batch, seq_len, out_channels]
        return x



# ---------------- FNOBlock ------------------
class FNOBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, modes, seq_length=64):
        super(FNOBlock, self).__init__()
        self.lifting = nn.Linear(in_channels, hidden_channels)
        self.conv = SpectralConv1D(hidden_channels, hidden_channels, modes)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x):
        x = self.lifting(x)
        identity = x
        x1 = self.conv(x)
        x2 = self.mlp(x)
        x = x1 + x2
        x = self.norm(x)
        return x + identity

# ---------------- FNOModel ------------------
class FNOModel(nn.Module):
    def __init__(self, seq_len_lstm, seq_len_cnn, input_dims, hidden_channels=32, modes=4):
        super(FNOModel, self).__init__()
        self.v_fno = FNOBlock(input_dims['v'], hidden_channels, modes, seq_len_cnn)
        self.i_fno = FNOBlock(input_dims['i'], hidden_channels, modes, seq_len_cnn)
        self.t_fno = FNOBlock(input_dims['t'], hidden_channels, modes, seq_len_cnn)
        self.fno_flat_dim = seq_len_cnn * hidden_channels
        self.lstm = nn.LSTM(input_size=input_dims['c'], hidden_size=hidden_channels, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.fno_flat_dim * 3 + hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, v, i, t, c):
        v_feat = self.v_fno(v).reshape(v.size(0), -1)
        i_feat = self.i_fno(i).reshape(i.size(0), -1)
        t_feat = self.t_fno(t).reshape(t.size(0), -1)
        _, (c_lstm_out, _) = self.lstm(c)
        c_feat = c_lstm_out.squeeze(0)
        x = torch.cat([v_feat, i_feat, t_feat, c_feat], dim=1)
        return self.fc(x)

# ---------------- Dataset ------------------
class BatteryRULDataset(Dataset):
    def __init__(self, X, y):
        self.v = torch.tensor(X['voltage_input'], dtype=torch.float32)
        self.i = torch.tensor(X['current_input'], dtype=torch.float32)
        self.t = torch.tensor(X['temperature_input'], dtype=torch.float32)
        self.c = torch.tensor(X['capacity_input'], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.v[idx], self.i[idx], self.t[idx], self.c[idx], self.y[idx]

# ---------------- Data Preprocessing ------------------
def load_and_preprocess_data(data_dir, seq_len_cnn, seq_len_lstm, test_size=0.2, random_state=42):
    discharge_files = glob.glob(os.path.join(data_dir, 'discharge/train/*.csv'))
    voltage_seqs, current_seqs, temp_seqs, capacity_seqs, rul_targets = [], [], [], [], []

    for discharge_file in discharge_files:
        df = pd.read_csv(discharge_file)
        max_cycle = df['cycle'].max()
        for i in range(len(df) - max(seq_len_cnn, seq_len_lstm)):
            v_seq = df['voltage_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            i_seq = df['current_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            t_seq = df['temp_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            c_seq = df['capacity'].iloc[i:i+seq_len_lstm].values.reshape(-1, 1)
            cycle = df['cycle'].iloc[i+seq_len_cnn-1]
            rul = (max_cycle - cycle) / max_cycle
            voltage_seqs.append(v_seq)
            current_seqs.append(i_seq)
            temp_seqs.append(t_seq)
            capacity_seqs.append(c_seq)
            rul_targets.append(rul)

    voltage_seqs = np.array(voltage_seqs)
    current_seqs = np.array(current_seqs)
    temp_seqs = np.array(temp_seqs)
    capacity_seqs = np.array(capacity_seqs)
    rul_targets = np.array(rul_targets)

    indices = np.arange(len(rul_targets))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    train_idx, val_idx = indices[:split], indices[split:]

    def subset(idx):
        return {
            'voltage_input': voltage_seqs[idx],
            'current_input': current_seqs[idx],
            'temperature_input': temp_seqs[idx],
            'capacity_input': capacity_seqs[idx]
        }, rul_targets[idx]

    return *subset(train_idx), *subset(val_idx)

# ---------------- Training ------------------
def train(model, train_loader, val_loader, device, epochs=100, lr=0.001):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for v, i, t, c, y in train_loader:
            v, i, t, c, y = v.to(device), i.to(device), t.to(device), c.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = model(v, i, t, c)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for v, i, t, c, y in val_loader:
                v, i, t, c, y = v.to(device), i.to(device), t.to(device), c.to(device), y.to(device).unsqueeze(1)
                output = model(v, i, t, c)
                loss = criterion(output, y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")

def load_test_data(data_dir, seq_len_cnn=64, seq_len_lstm=10):
    discharge_files = glob.glob(os.path.join(data_dir, 'discharge/test/*.csv'))
    print(f"Found {len(discharge_files)} test discharge files")

    voltage_seqs = []
    current_seqs = []
    temp_seqs = []
    capacity_seqs = []
    rul_targets = []

    for file in discharge_files:
        battery_id = os.path.basename(file).split('_')[0]
        print(f"Processing {battery_id} (test)...")

        df = pd.read_csv(file)
        max_cycle = df['cycle'].max()

        for i in range(len(df) - max(seq_len_cnn, seq_len_lstm)):
            v_seq = df['voltage_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            i_seq = df['current_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            t_seq = df['temp_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            c_seq = df['capacity'].iloc[i:i+seq_len_lstm].values.reshape(-1, 1)
            current_cycle = df['cycle'].iloc[i+seq_len_cnn-1]
            rul = (max_cycle - current_cycle) / max_cycle

            voltage_seqs.append(v_seq)
            current_seqs.append(i_seq)
            temp_seqs.append(t_seq)
            capacity_seqs.append(c_seq)
            rul_targets.append(rul)

    voltage_seqs = np.array(voltage_seqs)
    current_seqs = np.array(current_seqs)
    temp_seqs = np.array(temp_seqs)
    capacity_seqs = np.array(capacity_seqs)
    rul_targets = np.array(rul_targets)

    print(f"Loaded {len(rul_targets)} test samples.")

    X_test = {
        'voltage_input': voltage_seqs,
        'current_input': current_seqs,
        'temperature_input': temp_seqs,
        'capacity_input': capacity_seqs
    }
    y_test = rul_targets

    return X_test, y_test


def evaluate(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for v, i, t, c, y in data_loader:
            v, i, t, c = v.to(device), i.to(device), t.to(device), c.to(device)
            outputs = model(v, i, t, c)
            y_true.append(y.numpy())
            y_pred.append(outputs.cpu().numpy().squeeze())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"\n📊 Evaluation Metrics:")
    print(f"  ▪️ MSE  = {mse:.4f}")
    print(f"  ▪️ MAE  = {mae:.4f}")
    print(f"  ▪️ MAPE = {mape:.4f}")

    return y_true, y_pred

def plot_predictions(y_true, y_pred, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='True RUL')
    plt.plot(y_pred, label='Predicted RUL')
    plt.legend()
    plt.title('RUL Prediction')
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


# ---------------- Main ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/data/NASA')
    parser.add_argument('--seq_len_cnn', type=int, default=64)
    parser.add_argument('--seq_len_lstm', type=int, default=10)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--modes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val = load_and_preprocess_data(
        args.data_dir, args.seq_len_cnn, args.seq_len_lstm)

    input_dims = {
        'v': X_train['voltage_input'].shape[2],
        'i': X_train['current_input'].shape[2],
        't': X_train['temperature_input'].shape[2],
        'c': X_train['capacity_input'].shape[2]
    }

    train_loader = DataLoader(BatteryRULDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(BatteryRULDataset(X_val, y_val), batch_size=args.batch_size)

    model = FNOModel(args.seq_len_lstm, args.seq_len_cnn, input_dims, args.hidden_channels, args.modes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    print("\n🔍 Evaluating on val set...")
    y_true, y_pred = evaluate(model, val_loader, device)

    plot_predictions(y_true, y_pred, save_path='outputs/figures/val_rul_prediction.png')

    print("\n📦 Loading test data...")
    X_test, y_test = load_test_data(data_dir=args.data_dir, seq_len_cnn=args.seq_len_cnn, seq_len_lstm=args.seq_len_lstm)

    test_dataset = BatteryRULDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print("\n🔍 Evaluating on test set...")
    y_true, y_pred = evaluate(model, test_loader, device)

    plot_predictions(y_true, y_pred, save_path='outputs/figures/test_rul_prediction.png')


