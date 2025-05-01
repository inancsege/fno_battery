# battery_fno/preprocessing/golf_car_preprocessor.py
import pandas as pd
import numpy as np
import os
import glob
import re
from .base_preprocessor import BasePreprocessor
from .utils import create_sequences

class GolfCarPreprocessor(BasePreprocessor):
    """
    Preprocessor for Golf Car battery dataset
    """
    
    def load_data(self):
        """
        Load Golf Car battery dataset
        """
        self.logger.info("Loading Golf Car dataset")
        
        if not self.data_dir or not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all CSV files in the data directory
        data_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        self.logger.info(f"Found {len(data_files)} data files")
        
        if not data_files:
            self.logger.error(f"No CSV files found in {self.data_dir}")
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        # Get sample size from kwargs to limit data loading
        sample_size = self.kwargs.get('sample_size', None)
        if sample_size:
            self.logger.info(f"Using sample size: {sample_size} records per file")
        
        all_data = []
        for file in data_files:
            battery_id = os.path.basename(file).split('.')[0]  # Use filename as battery ID
            self.logger.info(f"Loading file: {file}")
            
            try:
                # Load sample of data if sample_size is specified
                if sample_size:
                    # First count rows to determine if we need to sample
                    total_rows = sum(1 for _ in open(file)) - 1  # Subtract header
                    
                    if total_rows > sample_size:
                        # Use a random sample
                        skip_rows = np.sort(np.random.choice(
                            range(1, total_rows + 1),  # Skip header (row 0)
                            total_rows - sample_size,
                            replace=False
                        ))
                        df = pd.read_csv(file, skiprows=skip_rows)
                    else:
                        df = pd.read_csv(file)
                else:
                    df = pd.read_csv(file)
                
                df['battery_id'] = battery_id
                all_data.append(df)
            except Exception as e:
                self.logger.warning(f"Error loading {file}: {str(e)}")
                continue
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_df)} records from {len(all_data)} files")
        
        return combined_df
    
    def preprocess(self, data):
        """
        Preprocess Golf Car battery data
        """
        self.logger.info("Preprocessing Golf Car data")
        
        # Handle missing values
        for col in self.input_features + [self.target_feature]:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        # Remove outliers (simple z-score method)
        for col in self.input_features:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data[col] = np.where(z_scores < 3, data[col], data[col].median())
        
        return data
    
    def create_sequences(self, preprocessed_data):
        """
        Create sequences for Golf Car battery data
        """
        self.logger.info(f"Creating sequences with length {self.seq_len} and stride {self.stride}")
        
        X_sequences = []
        y_sequences = []
        
        # Group by battery to avoid crossing battery boundaries
        for battery_id, battery_df in preprocessed_data.groupby('battery_id'):
            # Sort by time if available
            if 'timestamp' in battery_df.columns:
                battery_df = battery_df.sort_values('timestamp')
            
            feature_data = battery_df[self.input_features].values
            target_data = battery_df[self.target_feature].values
            
            # Create sequences
            for i in range(0, len(battery_df) - self.seq_len, self.stride):
                X_sequences.append(feature_data[i:i+self.seq_len])
                y_sequences.append(target_data[i+self.seq_len-1])  # Target is the last value
        
        # Convert to numpy arrays
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Ensure y is 2D for consistency
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def get_dataset_info(self):
        """
        Get information about the dataset
        """
        return {
            'input_dim': len(self.input_features),
            'output_dim': 1,
            'seq_len': self.seq_len,
            'features': self.input_features,
            'target': self.target_feature
        } 