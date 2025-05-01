import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    """
    Enhanced LSTM model for time series prediction of battery capacity.
    
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
        
        # Input normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Feature projection to improve learning
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # LSTM layers with higher capacity
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism for better focus on relevant parts of the sequence
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # Deeper output network with residual connections
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.fc2 = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4)
        self.fc3 = nn.Linear(lstm_output_dim // 4, output_dim)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.75)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(lstm_output_dim // 2)
        self.bn2 = nn.BatchNorm1d(lstm_output_dim // 4)
    
    def forward(self, x):
        """
        Forward pass with attention and residual connections
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Reshape input if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Input normalization
        x = self.layer_norm(x)
        
        # Project features
        x_projected = self.feature_projection(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_projected)  # shape: (batch_size, seq_len, hidden_dim)
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)  # shape: (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Apply softmax over sequence length
        
        # Apply attention weights to get context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)  # shape: (batch_size, hidden_dim)
        
        # Residual network with batch normalization
        out1 = F.gelu(self.fc1(context))
        out1 = self.bn1(out1)
        out1 = self.dropout1(out1)
        
        out2 = F.gelu(self.fc2(out1))
        out2 = self.bn2(out2)
        out2 = self.dropout2(out2)
        
        # Final output
        output = self.fc3(out2)
        
        return output

class LSTMAttention(nn.Module):
    """
    Enhanced LSTM model with multi-head attention for time series prediction.
    
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
        
        # Input normalization
        self.layer_norm_input = nn.LayerNorm(input_dim)
        
        # Feature embedding with positional encoding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # LSTM layers with higher capacity
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output of LSTM will be bidirectional, so double the hidden dim
        lstm_output_dim = hidden_dim * 2
        
        # Layer normalization for LSTM output
        self.layer_norm_lstm = nn.LayerNorm(lstm_output_dim)
        
        # Multi-head attention (using PyTorch's implementation)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Residual feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim * 2, lstm_output_dim)
        )
        
        # Layer normalization for feed-forward output
        self.layer_norm_ff = nn.LayerNorm(lstm_output_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass with multi-head attention and residual connections
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Reshape input if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Input normalization
        x = self.layer_norm_input(x)
        
        # Project features
        x_embedded = self.feature_embedding(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_embedded)  # shape: (batch_size, seq_len, hidden_dim*2)
        
        # Apply layer normalization
        lstm_out = self.layer_norm_lstm(lstm_out)
        
        # Multi-head attention with residual connection
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output + lstm_out  # Residual connection
        
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(attn_output)
        ff_output = ff_output + attn_output  # Residual connection
        
        # Apply layer normalization
        ff_output = self.layer_norm_ff(ff_output)
        
        # Aggregate sequence features (global average pooling)
        avg_pooled = torch.mean(ff_output, dim=1)
        
        # Output projection
        output = self.output_projection(avg_pooled)
        
        return output 