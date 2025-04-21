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

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

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

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=seq_len, dim=-1) # n=seq_len ensures original length
        return x


class FNO1DBlock(nn.Module):
    """Standard FNO Block with Spectral Conv, Linear Path, and Activation."""
    def __init__(self, width, modes, activation=nn.GELU()):
        super().__init__()
        self.fourier = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, 1) # Pointwise linear skip
        self.activation = activation
        self.norm = nn.BatchNorm1d(width) # Add normalization

    def forward(self, x):
        # x shape: (batch, width, seq_len)
        identity = x
        x_f = self.fourier(x)
        x_l = self.linear(x)
        x = self.norm(x_f + x_l) # Apply normalization before activation
        x = self.activation(x)
        return x + identity # Add residual connection