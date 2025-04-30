import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM model for time series prediction of battery capacity.
    
    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Size of hidden state
        num_layers (int): Number of LSTM layers
        output_dim (int): Number of output features
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional LSTM
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2, bidirectional=False):
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Reshape input if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # We take the output of the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Pass through the fully connected layer
        output = self.fc(last_time_step)
        
        return output

class LSTMAttention(nn.Module):
    """
    LSTM model with attention mechanism for time series prediction.
    
    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Size of hidden state
        num_layers (int): Number of LSTM layers
        output_dim (int): Number of output features
        dropout (float): Dropout rate
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass with attention
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Reshape input if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, seq_len, hidden_dim)
        
        # Calculate attention weights
        attn_weights = self.attention(lstm_out)  # shape: (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Apply softmax over sequence length
        
        # Apply attention weights to LSTM outputs
        context = torch.sum(attn_weights * lstm_out, dim=1)  # shape: (batch_size, hidden_dim)
        
        # Final prediction
        output = self.fc(context)
        
        return output 