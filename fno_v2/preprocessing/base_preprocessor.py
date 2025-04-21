# battery_fno/preprocessing/base_preprocessor.py
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import scale_data, scale_data_with_existing, split_data
from ..datasets.battery_dataset import BatteryDataset, MultiInputBatteryDataset

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessing."""

    def __init__(self, config):
        self.config = config
        self.scalers = {} # To store scalers for features and target

    @abstractmethod
    def load_data(self):
        """Loads data specific to the dataset type."""
        pass

    @abstractmethod
    def preprocess(self):
        """Performs preprocessing steps like cleaning, feature engineering."""
        pass

    def sequence_and_split(self, X, y):
        """Creates sequences and splits into train/val/test."""
        X_seq, y_seq = self.create_sequences(X, y)

        # Ensure y_seq is 2D (batch, features) for consistency with scaling
        if y_seq.ndim == 1:
            y_seq = y_seq.reshape(-1, 1)

        # Scale features (X_seq)
        X_seq_scaled, feature_scaler = scale_data(X_seq, scaler_type='standard')
        self.scalers['features'] = feature_scaler

        # Scale target (y_seq) - often beneficial
        y_seq_scaled, target_scaler = scale_data(y_seq, scaler_type='standard')
        self.scalers['target'] = target_scaler

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X_seq_scaled,
            y_seq_scaled,
            val_split=self.config.VALIDATION_SPLIT,
            test_split=self.config.TEST_SPLIT,
            shuffle=True,
            seed=self.config.SEED
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    @abstractmethod
    def create_sequences(self, X, y):
        """Abstract method for sequence creation logic."""
        pass


    def get_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Creates PyTorch DataLoaders."""
        if isinstance(X_train, dict): # Multi-input case
            train_dataset = MultiInputBatteryDataset(X_train, y_train)
            val_dataset = MultiInputBatteryDataset(X_val, y_val)
            test_dataset = MultiInputBatteryDataset(X_test, y_test)
        else: # Standard case
            train_dataset = BatteryDataset(X_train, y_train)
            val_dataset = BatteryDataset(X_val, y_val)
            test_dataset = BatteryDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4
        )

        return train_loader, val_loader, test_loader

    def inverse_transform_target(self, y_scaled):
        """Applies inverse scaling to the target variable."""
        if 'target' not in self.scalers:
            raise ValueError("Target scaler not found. Run preprocessing first.")
        # Ensure y_scaled is 2D for the scaler
        original_shape = y_scaled.shape
        if y_scaled.ndim == 1:
             y_scaled = y_scaled.reshape(-1, 1)
        elif y_scaled.ndim == 0:
             y_scaled = y_scaled.reshape(1, 1)


        y_original = self.scalers['target'].inverse_transform(y_scaled)

        # Return in the original shape (likely 1D or scalar)
        return y_original.reshape(original_shape)


    def run(self):
        """Executes the full preprocessing pipeline."""
        print(f"Running preprocessing for {self.config.DATASET_TYPE}...")
        X, y = self.preprocess()
        X_train, y_train, X_val, y_val, X_test, y_test = self.sequence_and_split(X, y)
        loaders = self.get_loaders(X_train, y_train, X_val, y_val, X_test, y_test)
        print("Preprocessing finished.")
        return loaders, self.scalers