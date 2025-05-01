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

class XJTUPreprocessor(BasePreprocessor):
    """
    Preprocessor for XJTU battery dataset
    """
    
    def load_data(self):
        """
        Load XJTU battery dataset
        """
        self.logger.info("Loading XJTU dataset")
        
        if not self.data_dir or not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all CSV files in the data directory
        data_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        self.logger.info(f"Found {len(data_files)} data files")
        
        if not data_files:
            self.logger.error(f"No CSV files found in {self.data_dir}")
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        all_data = []
        for file in data_files:
            battery_id = os.path.basename(file).split('.')[0]  # Use filename as battery ID
            self.logger.info(f"Loading file: {file}")
            
            try:
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
        Preprocess XJTU battery data
        """
        self.logger.info("Preprocessing XJTU data")
        
        # Handle missing values
        for col in self.input_features + [self.target_feature]:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        # Normalize data if needed
        # Remove outliers (simple z-score method)
        for col in self.input_features:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data[col] = np.where(z_scores < 3, data[col], data[col].median())
        
        return data
    
    def create_sequences(self, preprocessed_data):
        """
        Create sequences for XJTU battery data
        """
        self.logger.info(f"Creating sequences with length {self.seq_len} and stride {self.stride}")
        
        X_sequences = []
        y_sequences = []
        
        # Group by battery to avoid crossing battery boundaries
        for battery_id, battery_df in preprocessed_data.groupby('battery_id'):
            # Sort by cycle if available, otherwise by index
            if 'cycle' in battery_df.columns:
                battery_df = battery_df.sort_values('cycle')
            
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