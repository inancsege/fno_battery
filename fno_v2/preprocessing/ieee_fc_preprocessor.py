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