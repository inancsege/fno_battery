# battery_fno/models/fno.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FNO1DBlock, SpectralConv1d

class FNO1D(nn.Module):
    """
    Simplified Fourier Neural Operator for 1D Time Series Prediction.
    This implementation focuses on stable training and effective sequence modeling.
    """
    def __init__(self, input_dim, output_dim, modes=16, width=64, depth=4, seq_len=100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.seq_len = seq_len
        
        # Input projection - simple but effective
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, width, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(width)
        )
        
        # Core FNO blocks with stable architecture
        self.fno_blocks = nn.ModuleList([
            FNO1DBlock(width, modes) for _ in range(depth)
        ])
        
        # Residual connections for stable gradient flow
        self.residual_convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)
        ])
        
        # Output projections - from feature space to prediction
        self.output_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(width),
            nn.Conv1d(width, output_dim, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass of the FNO model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim, seq_len]
                For battery data, typically [batch_size, 3, seq_len]
                where 3 features are voltage, current, temperature
                
        Returns:
            Output tensor of shape [batch_size, output_dim, seq_len]
                For battery capacity, typically [batch_size, 1, seq_len]
        """
        # Keep original input for residual connection
        identity = x
        
        # Project input to feature space
        x = self.input_proj(x)  # [batch, width, seq_len]
        
        # Apply FNO blocks with residual connections
        for i, (block, res_conv) in enumerate(zip(self.fno_blocks, self.residual_convs)):
            # Store input to this block for residual
            block_input = x
            
            # Apply FNO block
            x = block(x)
            
            # Add residual connection
            x = x + res_conv(block_input)
            
            # Add a long skip connection every 2 blocks
            if i % 2 == 1 and i > 0:
                x = x * 0.8 + block_input * 0.2  # Weighted residual to stabilize training
        
        # Project to output space
        x = self.output_proj(x)  # [batch, output_dim, seq_len]
        
        return x


# Simple alias for backward compatibility
class FNO(FNO1D):
    """FNO is an alias for FNO1D for compatibility."""
    pass


# Advanced model with sequence-to-sequence prediction capabilities
class EnhancedFNO(nn.Module):
    """
    Enhanced FNO model for battery capacity prediction with additional features:
    - Handles nonlinear degradation patterns
    - Captures both short-term and long-term dependencies
    - Processes multi-modal data effectively
    """
    def __init__(self, input_dim, output_dim, modes=32, width=128, depth=4, seq_len=100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.seq_len = seq_len
        
        # Feature extraction paths for different timescales
        self.local_conv = nn.Sequential(
            nn.Conv1d(input_dim, width//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(width//2)
        )
        
        self.global_conv = nn.Sequential(
            nn.Conv1d(input_dim, width//2, kernel_size=9, padding=4),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(width//2)
        )
        
        # Spectral convolution blocks
        self.spectral_blocks = nn.ModuleList([
            nn.Sequential(
                SpectralConv1d(width, width, modes),
                nn.GroupNorm(min(8, width), width)
            ) for _ in range(depth)
        ])
        
        # Regular convolution for local patterns
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1),
                nn.GroupNorm(min(8, width), width)
            ) for _ in range(depth)
        ])
        
        # Residual connections
        self.residual_convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)
        ])
        
        # Attention mechanism for sequence modeling
        self.self_attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(width),
            nn.Dropout(0.1),
            nn.Conv1d(width, output_dim, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass with multi-path feature extraction.
        
        Args:
            x: Input tensor [batch_size, input_dim, seq_len]
        
        Returns:
            Output tensor [batch_size, output_dim, seq_len]
        """
        batch_size = x.shape[0]
        
        # Multi-scale feature extraction
        x_local = self.local_conv(x)
        x_global = self.global_conv(x)
        
        # Combine features
        x = torch.cat([x_local, x_global], dim=1)  # [batch, width, seq_len]
        
        # Apply FNO blocks with dual-path processing
        for i, (spectral_block, conv_block, res_conv) in enumerate(
            zip(self.spectral_blocks, self.conv_blocks, self.residual_convs)
        ):
            # Store for residual
            residual = x
            
            # Parallel paths: spectral and spatial
            x_spectral = spectral_block(x)
            x_conv = conv_block(x)
            
            # Weighted combination (learn to balance spectral and spatial)
            alpha = 0.5  # Can be made learnable
            x = alpha * x_spectral + (1 - alpha) * x_conv
            
            # Add residual connection
            x = x + res_conv(residual)
        
        # Apply self-attention for long-range dependencies
        # Reshape: [batch, width, seq_len] -> [batch, seq_len, width]
        x_perm = x.permute(0, 2, 1)
        x_attn, _ = self.self_attention(x_perm, x_perm, x_perm)
        x_attn = x_attn.permute(0, 2, 1)  # Back to [batch, width, seq_len]
        
        # Combine with residual
        x = x + 0.1 * x_attn
        
        # Output projection
        x = self.output_proj(x)
        
        return x


# Create more specialized models targeting specific tasks

class FNO_BatteryCapacity(nn.Module):
    """
    FNO model specifically optimized for battery capacity prediction,
    with domain-specific design choices.
    """
    def __init__(self, input_dim=3, output_dim=1, modes=32, width=64, depth=4, seq_len=100):
        super().__init__()
        self.input_dim = input_dim  # voltage, current, temperature
        self.output_dim = output_dim  # capacity
        self.modes = modes
        self.width = width
        self.depth = depth
        self.seq_len = seq_len
        
        # Feature normalization
        self.feature_norm = nn.InstanceNorm1d(input_dim, affine=True)
        
        # Main model
        self.fno = EnhancedFNO(
            input_dim=input_dim,
            output_dim=output_dim,
            modes=modes,
            width=width,
            depth=depth,
            seq_len=seq_len
        )
        
        # Optional monotonicity enforcement - helps for capacity prediction
        self.enforce_monotonic = True
    
    def forward(self, x):
        # Normalize input features for stability
        x = self.feature_norm(x)
        
        # Apply FNO
        output = self.fno(x)
        
        # Optionally enforce monotonicity - useful for capacity degradation
        if self.enforce_monotonic and self.training:
            # Calculate forward differences
            diff = output[:, :, 1:] - output[:, :, :-1]
            # Apply soft constraint to encourage non-increasing values
            # (for capacity that typically decreases over time)
            monotonic_loss = F.relu(diff).mean()
            # Store for later access during training
            self.monotonic_loss = monotonic_loss * 0.01
        
        return output

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
        
        # Create bottleneck for reduced capacity
        self.bottleneck_width = max(12, width // 3)
        
        # Projection layers with bottleneck
        self.v_proj = nn.Sequential(
            nn.Linear(input_dims['v'], self.bottleneck_width),
            nn.Dropout(0.3),
            nn.Linear(self.bottleneck_width, width)
        )
        
        self.i_proj = nn.Sequential(
            nn.Linear(input_dims['i'], self.bottleneck_width), 
            nn.Dropout(0.3),
            nn.Linear(self.bottleneck_width, width)
        )
        
        self.t_proj = nn.Sequential(
            nn.Linear(input_dims['t'], self.bottleneck_width),
            nn.Dropout(0.3),
            nn.Linear(self.bottleneck_width, width)
        )
        
        # Add input dropouts
        self.input_dropout = nn.Dropout(0.3)

        # Use smaller modes values to limit capacity
        small_modes = max(2, modes // 2)

        # Using a single FNO block per input for simplicity, FNO_RUL.py uses FNOBlock which includes lifting
        self.v_fno = FNO1DBlock(width, small_modes)
        self.i_fno = FNO1DBlock(width, small_modes)
        self.t_fno = FNO1DBlock(width, small_modes)

        self.fno_flat_dim = width # After FNO block, we average or take last step

        # Add dropout to LSTM and reduce LSTM capacity
        self.lstm_dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            input_size=input_dims['c'], 
            hidden_size=max(12, width // 2),  # Reduced LSTM capacity 
            batch_first=True, 
            num_layers=1, 
            dropout=0.2
        )

        # Add intermediate dropouts
        self.feat_dropout = nn.Dropout(0.4)

        # Adjust FC layer with more regularization and smaller size
        reduced_width = max(12, width // 2)
        self.fc = nn.Sequential(
            nn.Linear(width * 3 + reduced_width, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)  # Predict single RUL value
        )

    def forward(self, inputs):
        # inputs is a dictionary: {'voltage': v, 'current': i, 'temperature': t, 'capacity': c}
        v, i, t, c = inputs['voltage'], inputs['current'], inputs['temperature'], inputs['capacity']
        batch_size = v.shape[0]
        
        # Apply input dropout
        v = self.input_dropout(v)
        i = self.input_dropout(i)
        t = self.input_dropout(t)
        c = self.input_dropout(c)

        # Project and permute for FNO blocks
        v = self.v_proj(v).permute(0, 2, 1)  # (batch, width, seq_len_cnn)
        i = self.i_proj(i).permute(0, 2, 1)  # (batch, width, seq_len_cnn)
        t = self.t_proj(t).permute(0, 2, 1)  # (batch, width, seq_len_cnn)

        # Apply FNO blocks
        v_feat = self.v_fno(v)  # (batch, width, seq_len_cnn)
        i_feat = self.i_fno(i)  # (batch, width, seq_len_cnn)
        t_feat = self.t_fno(t)  # (batch, width, seq_len_cnn)

        # Use consistent aggregation method
        v_feat_agg = torch.mean(v_feat, dim=2)  # (batch, width)
        i_feat_agg = torch.mean(i_feat, dim=2)  # (batch, width)
        t_feat_agg = torch.mean(t_feat, dim=2)  # (batch, width)

        # Process capacity with LSTM
        # c shape: (batch, seq_len_lstm, input_dim_c=1)
        _, (c_lstm_hn, _) = self.lstm(c)
        c_feat = c_lstm_hn.squeeze(0)  # (batch, width) - take last hidden state
        c_feat = self.lstm_dropout(c_feat)

        # Concatenate features with moderate dropout
        x = torch.cat([v_feat_agg, i_feat_agg, t_feat_agg, c_feat], dim=1)  # (batch, width*4)
        x = self.feat_dropout(x)

        # Fully connected layers for final prediction
        out = self.fc(x)  # (batch, 1)
        
        return out

# Hybrid model combining FNO and TCN strengths
class HybridFNO_TCN(nn.Module):
    """
    Hybrid model combining the spectral capabilities of FNO with the efficiency of TCN.
    Specifically designed for battery capacity prediction tasks.
    """
    def __init__(self, input_dim=3, output_dim=1, modes=16, width=64, depth=4, seq_len=100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.seq_len = seq_len
        
        # Input feature normalization
        self.feature_norm = nn.InstanceNorm1d(input_dim, affine=True)
        
        # First half of channels processed with FNO (frequency domain)
        fno_width = width // 2
        self.fno_input = nn.Conv1d(input_dim, fno_width, kernel_size=1)
        self.fno_blocks = nn.ModuleList([
            nn.Sequential(
                SpectralConv1d(fno_width, fno_width, modes),
                nn.GroupNorm(min(8, fno_width), fno_width),
                nn.LeakyReLU(0.1)
            ) for _ in range(depth//2)
        ])
        
        # Second half of channels processed with TCN (time domain)
        tcn_width = width - fno_width
        self.tcn_input = nn.Conv1d(input_dim, tcn_width, kernel_size=1)
        self.tcn_blocks = nn.ModuleList()
        for i in range(depth//2):
            dilation = 2**i
            padding = (3 - 1) * dilation // 2  # For kernel_size=3
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(
                    tcn_width, tcn_width, 
                    kernel_size=3, 
                    padding=padding,
                    dilation=dilation
                ),
                nn.GroupNorm(min(8, tcn_width), tcn_width),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            ))
        
        # Merge layer to combine FNO and TCN paths
        self.merge = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
        # Final fusion blocks with channel attention
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, width), width),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1)
            ) for _ in range(2)
        ])
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(width, width // 4, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(width // 4, width, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Conv1d(width, output_dim, kernel_size=1)
        
        # Optional monotonicity enforcement for battery capacity
        self.enforce_monotonic = True
    
    def forward(self, x):
        """
        Forward pass with dual-path processing.
        
        Args:
            x: Input tensor [batch_size, input_dim, seq_len]
        
        Returns:
            Output tensor [batch_size, output_dim, seq_len]
        """
        # Normalize input features
        x = self.feature_norm(x)
        
        # Split processing into frequency domain (FNO) and time domain (TCN)
        x_fno = self.fno_input(x)
        x_tcn = self.tcn_input(x)
        
        # Process with FNO blocks
        for block in self.fno_blocks:
            x_fno = block(x_fno)
        
        # Process with TCN blocks
        for block in self.tcn_blocks:
            x_tcn_res = x_tcn
            x_tcn = block(x_tcn)
            # Add residual if shapes match
            if x_tcn.shape == x_tcn_res.shape:
                x_tcn = x_tcn + x_tcn_res
        
        # Merge FNO and TCN paths
        x = torch.cat([x_fno, x_tcn], dim=1)
        x = self.merge(x)
        
        # Apply fusion blocks with residual connections
        for block in self.fusion_blocks:
            res = x
            x = block(x)
            x = x + res
        
        # Apply channel attention
        att = self.channel_attention(x)
        x = x * att
        
        # Output projection
        out = self.output_proj(x)
        
        # Apply monotonicity constraint in training mode
        if self.enforce_monotonic and self.training:
            # Calculate forward differences
            diff = out[:, :, 1:] - out[:, :, :-1]
            # Apply soft constraint to encourage non-increasing values for capacity
            self.monotonic_loss = torch.nn.functional.relu(diff).mean() * 0.01
        
        return out