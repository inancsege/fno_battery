import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import extract_VIT_capacity, plot_loss, plot_pred
import os

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", 
            x_ft[:, :, :self.modes], self.weights)
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, input_dim, output_dim):
        super(FNO1d, self).__init__()
        
        self.modes = modes
        self.width = width
        
        # Input layer
        self.fc0 = nn.Linear(input_dim, self.width)
        
        # Fourier layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        
        # Output layer
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Fourier layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Fourier layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Fourier layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        # Fourier layer 4
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        # Output layers
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x[:, -1, :]  # Return only the last prediction

class RULPredictor:
    def __init__(self, seq_len=50, modes=16, width=64, learning_rate=1e-3):
        self.seq_len = seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.modes = modes  # Number of Fourier modes to multiply
        self.width = width  # Hidden dimension size
        
        # Initialize model
        self.model = FNO1d(
            modes=self.modes,
            width=self.width,
            input_dim=4,  # Voltage, Current, Temperature, Capacity
            output_dim=1  # Predicted RUL
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, val_loader, epochs, save_dir):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    pred = self.model(batch_x)
                    loss = self.criterion(pred, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.cpu().numpy())
        return np.concatenate(predictions, axis=0)

def create_data_loaders(x_train, y_train, x_val, y_val, batch_size=32):
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    SEQ_LEN = 50
    BATCH_SIZE = 32
    EPOCHS = 100
    MODES = 16
    WIDTH = 64
    LEARNING_RATE = 1e-3
    
    # Create save directory
    save_dir = "results/FNO"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess data
    x_datasets = ["data/NASA/B0005.csv", "data/NASA/B0006.csv", "data/NASA/B0007.csv"]
    y_datasets = ["data/NASA/B0005.csv", "data/NASA/B0006.csv", "data/NASA/B0007.csv"]
    
    X, y, scalers = extract_VIT_capacity(
        x_datasets=x_datasets,
        y_datasets=y_datasets,
        seq_len=SEQ_LEN,
        extract_all=True
    )
    
    # Split data
    train_size = int(0.8 * len(X))
    x_train, x_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, BATCH_SIZE
    )
    
    # Initialize and train model
    predictor = RULPredictor(
        seq_len=SEQ_LEN,
        modes=MODES,
        width=WIDTH,
        learning_rate=LEARNING_RATE
    )
    
    train_losses, val_losses = predictor.train(
        train_loader, val_loader, EPOCHS, save_dir
    )
    
    # Plot training results
    history = type('History', (), {'history': {'loss': train_losses, 'val_loss': val_losses}})()
    plot_loss(history, "results", "FNO")
    
    # Make predictions on validation set
    predictions = predictor.predict(val_loader)
    plot_pred(predictions, y_val, "results", "FNO", "validation_predictions") 