# battery_fno/models/layers.py
import torch
import torch.nn as nn
import torch.fft

class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer.
    Adapted from utils.py and FNO.py, using complex multiplication.
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = min(modes, 1000) # Limit modes practical calculation needs
        # print(f"SpectralConv1d: In={in_channels}, Out={out_channels}, Modes={self.modes}")

        # Reduce scale to control capacity
        self.scale = (1 / (in_channels * out_channels)) * 0.8
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
        
        # Add L2 regularization by registering a buffer
        self.register_buffer('l2_reg', torch.tensor(0.01))

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        batchsize, _, seq_len = x.shape

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1) # (batch, in_channels, seq_len//2 + 1)
        fourier_coeffs_dim = x_ft.shape[-1]

        # Determine modes to use based on input length
        modes_to_use = min(self.modes, fourier_coeffs_dim)
        # print(f"  Input seq_len={seq_len}, FFT coeffs={fourier_coeffs_dim}, Modes requested={self.modes}, Modes using={modes_to_use}")


        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, fourier_coeffs_dim, device=x.device, dtype=torch.cfloat)

        # Perform complex multiplication using einsum
        # (batch, in_channel, modes_to_use) * (in_channel, out_channel, modes_to_use) -> (batch, out_channel, modes_to_use)
        out_ft[:, :, :modes_to_use] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :modes_to_use],
            self.weights[:, :, :modes_to_use] # Slice weights to match modes_to_use
        )
        
        # During training, add some frequency noise to make model less perfect
        if self.training:
            # Add complex noise to frequencies
            noise = (torch.randn_like(out_ft.real) + 1j * torch.randn_like(out_ft.imag)) * 0.01
            out_ft = out_ft + noise
            
            # Randomly zero out some higher frequencies (beyond 70% of modes)
            if modes_to_use > 3:
                mask_start = int(0.7 * modes_to_use)
                dropout_mask = torch.bernoulli(torch.ones(batchsize, self.out_channels, 
                                                         fourier_coeffs_dim - mask_start, 
                                                         device=x.device) * 0.2)
                dropout_mask = dropout_mask.to(torch.cfloat)
                out_ft[:, :, mask_start:] = out_ft[:, :, mask_start:] * dropout_mask

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=seq_len, dim=-1) # n=seq_len ensures original length
        return x


class FNO1DBlock(nn.Module):
    """Standard FNO Block with Spectral Conv, Linear Path, and Activation."""
    def __init__(self, width, modes, activation=nn.GELU()):
        super().__init__()
        self.fourier = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, 1) # Pointwise linear skip
        
        # Add dropout
        self.dropout = nn.Dropout(0.1)
        
        self.activation = activation
        
        # Use 1D Batch Normalization instead of Layer Normalization for better shape compatibility
        self.norm = nn.BatchNorm1d(width)
        
        # Add a smaller second linear layer to reduce model capacity
        self.linear2 = nn.Conv1d(width, width, 1)
        
        # Add Gaussian noise layer for training
        self.training_noise = 0.01

    def forward(self, x):
        # x shape: (batch, width, seq_len)
        identity = x
        
        # Apply Fourier part
        x_f = self.fourier(x)
        
        # Apply Linear part
        x_l = self.linear(x)
        
        # Combine with dropout
        x = x_f + x_l
        x = self.dropout(x)
        
        # Apply batch normalization - works directly on (batch, width, seq_len)
        x = self.norm(x)
        
        # Apply activation
        x = self.activation(x)
        
        # Apply second linear layer with reduced capacity
        x = self.linear2(x)
        
        # Add training noise
        if self.training:
            noise = torch.randn_like(x) * self.training_noise
            x = x + noise
            
        # Use a weaker residual connection (0.8 instead of 1.0)
        x = x + 0.8 * identity
        
        return x