# battery_fno/models/fno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FNO1DBlock

class FNO1D(nn.Module):
    """
    Fourier Neural Operator for 1D Time Series Prediction.
    Based on FNO.py and utils.py/TimeSeriesFNO.
    Predicts the value(s) at the last time step(s).
    """
    def __init__(self, input_dim, output_dim, modes, width, depth, seq_len):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.seq_len = seq_len # Store seq_len if needed internally

        self.input_proj = nn.Linear(input_dim, width) # Project input features to width

        self.fno_blocks = nn.Sequential(
            *[FNO1DBlock(width, modes) for _ in range(depth)]
        )

        # Output projection layers
        self.output_fc1 = nn.Linear(width, 128)
        self.output_fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # print(f"Input shape: {x.shape}")

        # 1. Input Projection
        x = self.input_proj(x) # -> (batch, seq_len, width)
        # print(f"After input_proj: {x.shape}")


        # 2. Permute for FNO blocks: (batch, width, seq_len)
        x = x.permute(0, 2, 1)
        # print(f"After permute 1: {x.shape}")


        # 3. FNO Blocks
        x = self.fno_blocks(x) # -> (batch, width, seq_len)
        # print(f"After fno_blocks: {x.shape}")


        # 4. Permute back: (batch, seq_len, width)
        x = x.permute(0, 2, 1)
        # print(f"After permute 2: {x.shape}")


        # 5. Select last time step for prediction
        x = x[:, -1, :] # -> (batch, width)
        # print(f"After selecting last timestep: {x.shape}")


        # 6. Output Layers
        x = F.gelu(self.output_fc1(x)) # -> (batch, 128)
        x = self.output_fc2(x)         # -> (batch, output_dim)
        # print(f"Final output shape: {x.shape}")


        return x

# --- Optional: FNO_RUL_Hybrid Model ---
# If you need the exact FNO_RUL.py structure, define it here.
# It uses separate FNOBlocks for V, I, T and an LSTM for C.

class FNO_RUL_Hybrid(nn.Module):
    """
    Hybrid FNO + LSTM model based on FNO_RUL.py.
    Requires multi-input data (V, I, T, C).
    """
    def __init__(self, modes, width, seq_len_cnn, seq_len_lstm, input_dims):
        super().__init__()
        # Note: Using FNO1DBlock requires adapting input/output dims or using a simpler SpectralConv1d directly
        # Using FNO1DBlock for consistency in structure - requires input_proj inside block or here
        self.v_proj = nn.Linear(input_dims['v'], width)
        self.i_proj = nn.Linear(input_dims['i'], width)
        self.t_proj = nn.Linear(input_dims['t'], width)

        # Using a single FNO block per input for simplicity, FNO_RUL.py uses FNOBlock which includes lifting
        self.v_fno = FNO1DBlock(width, modes)
        self.i_fno = FNO1DBlock(width, modes)
        self.t_fno = FNO1DBlock(width, modes)

        self.fno_flat_dim = width # After FNO block, we average or take last step

        self.lstm = nn.LSTM(input_size=input_dims['c'], hidden_size=width, batch_first=True, num_layers=1) # Simplified LSTM part

        # Adjust FC layer input size
        # We take the output of FNO blocks (batch, width, seq_len), average over seq_len -> (batch, width)
        # LSTM output is (batch, width)
        self.fc = nn.Sequential(
            nn.Linear(width * 3 + width, 128), # width*3 from FNO mean outputs + width from LSTM
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Predict single RUL value
        )

    def forward(self, inputs):
        # inputs is a dictionary: {'voltage': v, 'current': i, 'temperature': t, 'capacity': c}
        v, i, t, c = inputs['voltage'], inputs['current'], inputs['temperature'], inputs['capacity']
        # Shapes: v, i, t: (batch, seq_len_cnn, input_dim_cnn=1)
        # Shape: c: (batch, seq_len_lstm, input_dim_lstm=1)

        # Project and permute for FNO blocks
        v = self.v_proj(v).permute(0, 2, 1) # (batch, width, seq_len_cnn)
        i = self.i_proj(i).permute(0, 2, 1) # (batch, width, seq_len_cnn)
        t = self.t_proj(t).permute(0, 2, 1) # (batch, width, seq_len_cnn)

        # Apply FNO blocks
        v_feat = self.v_fno(v) # (batch, width, seq_len_cnn)
        i_feat = self.i_fno(i) # (batch, width, seq_len_cnn)
        t_feat = self.t_fno(t) # (batch, width, seq_len_cnn)

        # Aggregate FNO features (e.g., mean pooling over sequence length)
        v_feat_agg = torch.mean(v_feat, dim=2) # (batch, width)
        i_feat_agg = torch.mean(i_feat, dim=2) # (batch, width)
        t_feat_agg = torch.mean(t_feat, dim=2) # (batch, width)

        # Process capacity with LSTM
        # c shape: (batch, seq_len_lstm, input_dim_c=1)
        _, (c_lstm_hn, _) = self.lstm(c)
        c_feat = c_lstm_hn.squeeze(0) # (batch, width) - take last hidden state

        # Concatenate features
        x = torch.cat([v_feat_agg, i_feat_agg, t_feat_agg, c_feat], dim=1) # (batch, width*4)

        # Fully connected layers for final prediction
        out = self.fc(x) # (batch, 1)
        return out