# battery_fno/datasets/battery_dataset.py
import torch
from torch.utils.data import Dataset

class BatteryDataset(Dataset):
    """
    Dataset class for battery data with single input and single output
    """
    
    def __init__(self, X, y):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features with shape (n_samples, seq_len, n_features)
        y : numpy.ndarray
            Target values with shape (n_samples, n_outputs)
        """
        # Convert to PyTorch tensors if they're not already
        if not isinstance(X, torch.Tensor):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X
            
        if not isinstance(y, torch.Tensor):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = y
    
    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Parameters:
        -----------
        idx : int
            Index of the sample
            
        Returns:
        --------
        x : torch.Tensor
            Input features for the sample
        y : torch.Tensor
            Target value for the sample
        """
        return self.X[idx], self.y[idx]


class MultiInputBatteryDataset(Dataset):
    """
    Dataset class for battery data with multiple inputs (for hybrid models)
    """
    
    def __init__(self, X_dict, y):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        X_dict : dict of numpy.ndarray
            Dictionary of input features, e.g., {'voltage': X_v, 'current': X_i, ...}
        y : numpy.ndarray
            Target values with shape (n_samples, n_outputs)
        """
        # Convert all inputs to PyTorch tensors
        self.X_dict = {}
        for key, value in X_dict.items():
            if not isinstance(value, torch.Tensor):
                self.X_dict[key] = torch.tensor(value, dtype=torch.float32)
            else:
                self.X_dict[key] = value
                
        if not isinstance(y, torch.Tensor):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = y
    
    def __len__(self):
        """
        Return the number of samples in the dataset
        """
        # Use the first input's length
        first_key = list(self.X_dict.keys())[0]
        return len(self.X_dict[first_key])
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Parameters:
        -----------
        idx : int
            Index of the sample
            
        Returns:
        --------
        x1, x2, ..., y : tuple of torch.Tensor
            Input features and target value for the sample
        """
        # Return all inputs and the target as a tuple
        return tuple(x[idx] for x in self.X_dict.values()) + (self.y[idx],)