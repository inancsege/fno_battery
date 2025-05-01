# battery_fno/preprocessing/ieee_fc_preprocessor.py
import pandas as pd
import numpy as np
import os
from .base_preprocessor import BasePreprocessor
from .utils import create_sequences

class IeeeFcPreprocessor(BasePreprocessor):
    """Preprocessor for IEEE FC dataset."""

    def load_data(self):
        """Loads data from IEEE FC CSV files."""
        all_df = []
        data_config = self.config.ACTIVE_DATA_CONFIG
        for file_path in data_config['files']:
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"Data file not found: {file_path}")
            df = pd.read_csv(file_path)
            required_cols = data_config['features'] + [data_config['target']] # Check features and target
            if not all(col in df.columns for col in required_cols):
                 raise ValueError(f"File {file_path} missing required columns. Found: {df.columns}. Required: {required_cols}")
            all_df.append(df)
        combined_df = pd.concat(all_df, ignore_index=True)
        return combined_df

    def preprocess(self):
        """Basic preprocessing for IEEE FC data."""
        df = self.load_data()
        data_config = self.config.ACTIVE_DATA_CONFIG
        features = data_config['features']
        target = data_config['target']

        # Handle NaNs/Infs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0) # Simple fill

        X = df[features].values
        # Assuming target is RUL or similar, needs to be shaped correctly
        y = df[target].values

        # Clean potential inf values after processing
        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        y = np.nan_to_num(y, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)


        return X, y

    def create_sequences(self, X, y):
        """Uses the standard sequence creation utility."""
        return create_sequences(X, y, seq_len=self.config.SEQ_LEN, stride=1)

# --- battery_fno/preprocessing/xjtu_preprocessor.py ---
# Similar structure to IeeeFcPreprocessor, just change class name
# and potentially add XJTU-specific cleaning/feature engineering logic.
import pandas as pd
import numpy as np
import os
import glob
# from .base_preprocessor import BasePreprocessor (already imported indirectly)
from .utils import create_sequences

class XjtuPreprocessor(BasePreprocessor):
    """Preprocessor for XJTU dataset."""

    def load_data(self):
        """Loads data from XJTU CSV files."""
        all_df = []
        data_config = self.config.ACTIVE_DATA_CONFIG
        data_dir = data_config['dir']
        file_list = glob.glob(os.path.join(data_dir, "*.csv"))
        if not file_list:
             raise FileNotFoundError(f"No CSV files found in {data_dir}")

        for file_path in file_list:
            df = pd.read_csv(file_path)
            required_cols = data_config['features'] # Target might be derived or implicitly present
            if not all(col in df.columns for col in required_cols):
                 print(f"Warning: File {file_path} missing some columns. Found: {df.columns}. Required: {required_cols}. Skipping file or handle carefully.")
                 # Decide whether to skip or impute missing columns
                 continue # Skip for now
            all_df.append(df)

        if not all_df:
             raise ValueError("No valid XJTU data files could be loaded.")

        combined_df = pd.concat(all_df, ignore_index=True)
        return combined_df

    def preprocess(self):
        """Basic preprocessing for XJTU data."""
        df = self.load_data()
        data_config = self.config.ACTIVE_DATA_CONFIG
        features = data_config['features']
        target = data_config['target'] # Usually 'capacity'

        # Handle NaNs/Infs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0) # Simple fill

        X = df[features].values
        y = df[target].values

        # Clean potential inf values after processing
        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        y = np.nan_to_num(y, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)


        return X, y

    def create_sequences(self, X, y):
        """Uses the standard sequence creation utility."""
        # The original main.py used SEQ_LEN=10 for XJTU
        # We use the config SEQ_LEN but you could override here if needed
        return create_sequences(X, y, seq_len=self.config.SEQ_LEN, stride=1)

class IEEEFCPreprocessor(BasePreprocessor):
    """
    Preprocessor for IEEE Fuel Cell dataset
    """
    
    def load_data(self):
        """
        Load IEEE FC dataset
        """
        self.logger.info("Loading IEEE FC dataset")
        all_df = []
        
        for file_path in self.data_paths:
            if not os.path.exists(file_path):
                self.logger.error(f"Data file not found: {file_path}")
                raise FileNotFoundError(f"Data file not found: {file_path}")
                
            self.logger.info(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Check for required columns
            if not all(col in df.columns for col in self.input_features):
                self.logger.error(f"File {file_path} missing required columns.")
                raise ValueError(f"File {file_path} missing required columns. Found: {df.columns}. Required: {self.input_features}")
                
            all_df.append(df)
            
        if not all_df:
            self.logger.error("No data loaded")
            raise ValueError("No data files were loaded")
            
        combined_df = pd.concat(all_df, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_df)} records from {len(all_df)} files")
        
        return combined_df
    
    def preprocess(self, data):
        """
        Preprocess IEEE FC data
        """
        self.logger.info("Preprocessing IEEE FC data")
        
        # Handle missing values
        for col in self.input_features + [self.target_feature]:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        # Remove outliers (simple z-score method)
        for col in self.input_features:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            data[col] = np.where(z_scores < 3, data[col], data[col].median())
        
        return data
    
    def create_sequences(self, preprocessed_data):
        """
        Create sequences for IEEE FC data
        """
        self.logger.info(f"Creating sequences with length {self.seq_len} and stride {self.stride}")
        
        # Get feature data and target data
        feature_data = preprocessed_data[self.input_features].values
        target_data = preprocessed_data[self.target_feature].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(feature_data) - self.seq_len, self.stride):
            X_sequences.append(feature_data[i:i+self.seq_len])
            y_sequences.append(target_data[i+self.seq_len-1])  # Target is the last value in sequence
        
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