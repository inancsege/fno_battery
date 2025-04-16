# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fno_battery.utils import load_and_proc_data_FC, evaluate_model
import os

# Define SpectralConv1d class for spectral convolution
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

# Define FNO1d class for Fourier Neural Operator
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

# Define RULPredictor class for Remaining Useful Life prediction
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
            input_dim=2,  # Adjusted to match the number of features in the dataset
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

# Function to create data loaders for training and validation
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
    # Define file paths and parameters
    file_list = ["C:/Users/serha/PycharmProjects/Temp/fno_battery/data/data/ieee1/FC1_test_filtered.csv"]  # Example file paths
    seq_len = 50
    batch_size = 32
    epochs = 20
    save_dir = "models"
    model_save_file = os.path.join(save_dir, 'best_model.pth')
    output_save_file = "outputs/test_results.txt"

    # Load and process data
    X, y, train_loader, val_loader, test_loader, scaler_data = load_and_proc_data_FC(
        file_list=file_list,
        SEQ_LEN=seq_len,
        BATCH_SIZE=batch_size
    )

    # Initialize RULPredictor
    rul_predictor = RULPredictor(seq_len=seq_len)

    # Train the model
    train_losses, val_losses = rul_predictor.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir
    )

    # Evaluate the model
    evaluate_model(
        model=rul_predictor.model,
        test_loader=test_loader,
        model_save_file=model_save_file,
        output_save_file=output_save_file,
        device=rul_predictor.device
    )
    