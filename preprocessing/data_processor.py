import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import logging

from preprocessing.nasa_preprocessor import NASAPreprocessor, NASARULPreprocessor
from preprocessing.ieee_fc_preprocessor import IEEEFCPreprocessor
from preprocessing.xjtu_preprocessor import XJTUPreprocessor
from preprocessing.golf_car_preprocessor import GolfCarPreprocessor

class DataProcessor:
    """
    DataProcessor class that handles all data preprocessing tasks.
    It initializes the appropriate dataset-specific preprocessor based on dataset_name.
    """
    
    def __init__(self, dataset_name, **kwargs):
        """
        Initialize the data processor
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to process (NASA_VIT, NASA_RUL, IEEE_FC, XJTU, GOLF_CAR)
        **kwargs : dict
            Additional keyword arguments specific to the dataset preprocessor
        """
        self.dataset_name = dataset_name
        self.logger = logging.getLogger("fno_battery")
        
        # Create the appropriate preprocessor based on dataset_name
        if dataset_name == 'NASA_VIT':
            self.preprocessor = NASAPreprocessor(**kwargs)
        elif dataset_name == 'NASA_RUL':
            self.preprocessor = NASARULPreprocessor(**kwargs)
        elif dataset_name == 'IEEE_FC':
            self.preprocessor = IEEEFCPreprocessor(**kwargs)
        elif dataset_name == 'XJTU':
            self.preprocessor = XJTUPreprocessor(**kwargs)
        elif dataset_name == 'GOLF_CAR':
            self.preprocessor = GolfCarPreprocessor(**kwargs)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        self.batch_size = kwargs.get('batch_size', 32)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.test_split = kwargs.get('test_split', 0.2)
        
        # Process the data
        self.X, self.y, self.dataset_info = self._process_data()
        
    def _process_data(self):
        """
        Process the data using the appropriate preprocessor
        
        Returns:
        --------
        X : numpy array or dict of numpy arrays
            Input features
        y : numpy array
            Target values
        dataset_info : dict
            Information about the dataset
        """
        self.logger.info(f"Processing {self.dataset_name} dataset")
        
        # Load and preprocess data
        data = self.preprocessor.load_data()
        preprocessed_data = self.preprocessor.preprocess(data)
        X, y = self.preprocessor.create_sequences(preprocessed_data)
        
        # Get dataset info
        dataset_info = self.preprocessor.get_dataset_info()
        
        return X, y, dataset_info
        
    def get_data_loaders(self):
        """
        Create PyTorch DataLoaders for training, validation, and testing
        
        Returns:
        --------
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data
        val_loader : torch.utils.data.DataLoader
            DataLoader for validation data
        test_loader : torch.utils.data.DataLoader
            DataLoader for test data
        dataset_info : dict
            Information about the dataset
        """
        # Convert to PyTorch tensors
        if isinstance(self.X, dict):
            # For multi-input models (like FNO_RUL)
            X_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.X.items()}
            y_tensor = torch.tensor(self.y, dtype=torch.float32)
            
            # Create dataset
            dataset = TensorDataset(*list(X_tensors.values()), y_tensor)
        else:
            # For single-input models
            X_tensor = torch.tensor(self.X, dtype=torch.float32)
            y_tensor = torch.tensor(self.y, dtype=torch.float32)
            
            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor)
        
        # Calculate split sizes
        total_size = len(dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size - test_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        self.logger.info(f"Created data loaders: Train ({len(train_dataset)} samples), "
                        f"Validation ({len(val_dataset)} samples), "
                        f"Test ({len(test_dataset)} samples)")
        
        return train_loader, val_loader, test_loader, self.dataset_info 