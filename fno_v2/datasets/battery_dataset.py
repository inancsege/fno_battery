# battery_fno/datasets/battery_dataset.py
import torch
from torch.utils.data import Dataset

class BatteryDataset(Dataset):
    """Standard PyTorch Dataset for battery time series."""
    def __init__(self, X, y):
        # Ensure data is torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ensure target is [batch, 1] for MSELoss compatibility
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            self.y = y_tensor.unsqueeze(1)
        elif y_tensor.ndim == 2 and y_tensor.shape[1] == 1:
             self.y = y_tensor
        else:
             # Attempt to select the last element if target is sequential
             # This might need adjustment based on specific target definition
             if y_tensor.ndim == 2 and y_tensor.shape[1] > 1:
                 print("Warning: Target y has multiple features, selecting the last one.")
                 self.y = y_tensor[:, -1].unsqueeze(1)
             else:
                raise ValueError(f"Unsupported target shape: {y_tensor.shape}")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiInputBatteryDataset(Dataset):
    """PyTorch Dataset for models requiring multiple input tensors (like FNO_RUL)."""
    def __init__(self, X_dict, y):
        self.X_dict = {key: torch.tensor(val, dtype=torch.float32) for key, val in X_dict.items()}
        # Ensure target is [batch, 1]
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            self.y = y_tensor.unsqueeze(1)
        elif y_tensor.ndim == 2 and y_tensor.shape[1] == 1:
             self.y = y_tensor
        else:
             raise ValueError(f"Unsupported target shape for MultiInput: {y_tensor.shape}")


        # Assume all input tensors have the same first dimension (batch size)
        self.length = len(next(iter(self.X_dict.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a tuple of inputs and the target
        inputs = {key: tensor[idx] for key, tensor in self.X_dict.items()}
        return inputs, self.y[idx]