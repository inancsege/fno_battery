import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """
    Removes excess padding from the end of the sequence
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Basic temporal block for TCN
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Temporal Convolutional Network for time series forecasting
    
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        num_channels (list): Number of channels in each layer of the TCN
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout rate
        seq_len (int): Length of the input sequence
    """
    def __init__(self, input_dim, output_dim=1, num_channels=[32, 64, 128, 64], 
                 kernel_size=3, dropout=0.2, seq_len=40):
        super(TCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                    dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                    dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Reshape if necessary: TCN expects (batch, channels, seq_len)
        if x.shape[-1] == self.input_dim:  # (batch, seq, features)
            x = x.transpose(1, 2)  # -> (batch, features, seq)
            
        # Apply TCN layers
        y = self.network(x)
        
        # Global average pooling over the sequence dimension or take the last time step
        y = y[:, :, -1]  # Take the last time step
        
        # Apply the final fully connected layer
        output = self.fc(y)
        
        return output 