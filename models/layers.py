# battery_fno/models/layers.py
import torch
import torch.nn as nn
import torch.fft

class SpectralConv1d(nn.Module):
    """
    Enhanced 1D Spectral Convolution Layer with learnable frequency filtering.
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Improved initialization for better training stability
        self.scale = 1.0 / (in_channels * out_channels)**0.5
        
        # Complex weights for Fourier space transformation
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
        
        # Learnable frequency importance weighting
        self.freq_weight = nn.Parameter(torch.ones(1, 1, self.modes))
        
        # Add a small nonlinear projection in frequency domain
        self.freq_mix = nn.Parameter(
            0.01 * torch.randn(min(32, modes), min(32, modes), dtype=torch.cfloat)
        )

    def forward(self, x):
        # x shape: (batch, in_channels, seq_len)
        batchsize, _, seq_len = x.shape

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        fourier_coeffs_dim = x_ft.shape[-1]

        # Determine modes to use based on input length
        modes_to_use = min(self.modes, fourier_coeffs_dim)

        # Prepare output Fourier coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, fourier_coeffs_dim, 
                             device=x.device, dtype=torch.cfloat)

        # Apply frequency domain transformation using complex multiplication
        if modes_to_use > 0:
            # Step 1: Complex multiplication for modes we want to keep
            x_ft_modes = x_ft[:, :, :modes_to_use]  # Extract the modes
            weights_modes = self.weights[:, :, :modes_to_use]  # Extract corresponding weights
            
            # Perform complex multiplication
            out_ft_modes = torch.einsum("bim,iom->bom", x_ft_modes, weights_modes)
            
            # Step 2: Apply learned frequency weighting
            freq_weights = torch.sigmoid(self.freq_weight[:, :, :modes_to_use]) * 2.0
            out_ft_weighted = out_ft_modes * freq_weights
            
            # Step 3: Apply frequency mixing for important modes
            if modes_to_use >= 4:
                mix_modes = min(out_ft_weighted.shape[1], min(32, modes_to_use))
                
                if mix_modes >= 2:
                    # Get lowest modes
                    lowest_modes = out_ft_weighted[:, :mix_modes, :mix_modes]
                    
                    # Mix using einsum
                    mix_matrix = self.freq_mix[:mix_modes, :mix_modes]
                    mixed = torch.einsum('bim,mn->bin', lowest_modes, mix_matrix)
                    
                    # Create new tensor with mixed components
                    mixed_lowest_modes = lowest_modes + 0.01 * mixed
                    
                    # Create a new output tensor with mixed modes
                    new_out_ft_weighted = out_ft_weighted.clone()
                    new_out_ft_weighted[:, :mix_modes, :mix_modes] = mixed_lowest_modes
                    out_ft_weighted = new_out_ft_weighted
            
            # Step 4: Copy to output tensor
            out_ft[:, :, :modes_to_use] = out_ft_weighted

        # Convert back to physical space
        x = torch.fft.irfft(out_ft, n=seq_len, dim=-1)
        return x


class FNO1DBlock(nn.Module):
    """Enhanced FNO Block with advanced spectral conv, nonlinear path, and normalization."""
    def __init__(self, width, modes, activation=nn.LeakyReLU(0.1)):
        super().__init__()
        # Spectral convolution in Fourier space
        self.fourier = SpectralConv1d(width, width, modes)
        
        # Multiple linear paths with different kernel sizes
        self.linear_pointwise = nn.Conv1d(width, width//2, kernel_size=1)
        self.linear_local = nn.Conv1d(width, width//2, kernel_size=3, padding=1)
        
        # Combine channels after processing
        self.combine = nn.Conv1d(width, width, kernel_size=1)
        
        # Normalization and activation
        self.norm1 = nn.GroupNorm(min(8, width), width)  # More stable than BatchNorm
        self.norm2 = nn.GroupNorm(min(8, width), width)
        
        # Activation function
        self.activation = activation
        
        # Second nonlinear transformation
        self.linear2 = nn.Conv1d(width, width, kernel_size=1)
        
        # Optional channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(width, width//4, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(width//4, width, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch, width, seq_len)
        identity = x
        
        # Apply normalization first (pre-norm)
        x = self.norm1(x)
        
        # Split into two paths: Fourier and linear
        x_f = self.fourier(x)
        
        # Linear path with two sub-paths
        x_p = self.linear_pointwise(x)  # Point-wise for global mixing
        x_l = self.linear_local(x)      # Local for neighboring information
        
        # Combine the linear paths
        x_linear = torch.cat([x_p, x_l], dim=1)
        
        # Combine Fourier and linear paths
        x = x_f + x_linear
        
        # Apply activation
        x = self.activation(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply second normalization
        x = self.norm2(x)
        
        # Apply second linear layer
        x = self.linear2(x)
        
        # Apply channel attention
        attention = self.channel_attention(x)
        x = x * attention
        
        # Add residual connection
        x = x + identity
        
        return x