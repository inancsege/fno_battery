# battery_fno/preprocessing/nasa_preprocessor.py
import pandas as pd
import numpy as np
import os
import glob
from .base_preprocessor import BasePreprocessor
from .utils import create_sequences, create_multi_input_sequences, scale_data, split_data
# Can import more advanced functions from preproc.py if needed and adapt them

class NasaVitPreprocessor(BasePreprocessor):
    """Preprocessor for NASA datasets formatted like B0005/6/7 (VIT)."""

    def load_data(self):
        """Loads data from multiple NASA CSV files."""
        all_df = []
        data_config = self.config.ACTIVE_DATA_CONFIG
        for file_path in data_config['files']:
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"Data file not found: {file_path}")
            df = pd.read_csv(file_path)
            # Basic check for required columns - adjust based on actual headers
            required_cols = data_config['features']
            if not all(col in df.columns for col in required_cols):
                 raise ValueError(f"File {file_path} missing required columns. Found: {df.columns}. Required: {required_cols}")
            all_df.append(df)
        combined_df = pd.concat(all_df, ignore_index=True)
        return combined_df

    def preprocess(self):
        """Basic preprocessing for NASA VIT data."""
        df = self.load_data()
        data_config = self.config.ACTIVE_DATA_CONFIG
        features = data_config['features']
        target = data_config['target']

        # --- Add more sophisticated preprocessing here ---
        # Example: Handle NaNs (simple imputation)
        df[features] = df[features].fillna(df[features].median())
        df[target] = df[target].fillna(df[target].median())

        # Example: Calculate RUL if needed (assuming 'cycle' and 'max_cycle' exist or can be derived)
        # if target == 'RUL':
        #     # Find max cycle per battery if multiple batteries are combined
        #     # df['max_cycle'] = df.groupby('battery_id')['cycle'].transform('max')
        #     # df['RUL'] = (df['max_cycle'] - df['cycle']) / df['max_cycle']
        #     pass # Implement RUL calculation logic

        X = df[features].values
        y = df[target].values

        # Clean potential inf values
        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        y = np.nan_to_num(y, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)


        return X, y

    def create_sequences(self, X, y):
        """Uses the standard sequence creation utility."""
        # Use stride=1 for maximum overlap, adjust if needed
        return create_sequences(X, y, seq_len=self.config.SEQ_LEN, stride=1)

# --- Add NasaRulPreprocessor here ---
# This would be more complex, likely involving globbing files,
# calculating RUL per file, and potentially using create_multi_input_sequences
# It might heavily borrow from FNO_RUL.py's load_and_preprocess_data and preproc.py

class NasaRulPreprocessor(BasePreprocessor):
    """Preprocessor for NASA discharge cycle data targeting RUL prediction."""

    def load_data(self):
        """Loads and combines discharge data from multiple files."""
        data_dir = self.config.ACTIVE_DATA_CONFIG['dir']
        # Assumes structure like .../discharge/train/*.csv and .../discharge/test/*.csv
        train_files = glob.glob(os.path.join(data_dir, 'discharge/train/*.csv'))
        test_files = glob.glob(os.path.join(data_dir, 'discharge/test/*.csv')) # Load test files during initial processing too
        discharge_files = train_files + test_files
        print(f"Found {len(discharge_files)} discharge files.")

        all_data = []
        for file in discharge_files:
            battery_id = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file)
            # Add battery ID for potential grouping later
            df['battery_id'] = battery_id
            all_data.append(df)

        if not all_data:
            raise FileNotFoundError(f"No discharge CSV files found in {os.path.join(data_dir, 'discharge')}")

        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    def preprocess(self):
        """Preprocessing inspired by FNO_RUL.py and preproc.py."""
        df = self.load_data()
        data_config = self.config.ACTIVE_DATA_CONFIG
        features = data_config['features'] # ['voltage_battery', 'current_battery', 'temp_battery', 'capacity']
        target = data_config['target'] # 'RUL'

        # --- Add Advanced Preprocessing ---
        # (Outlier removal, filtering, augmentation - adapted from preproc.py)
        # This part needs careful implementation based on preproc.py's functions
        # For simplicity here, we'll just calculate RUL and extract features.

        # Calculate RUL per battery
        df['max_cycle'] = df.groupby('battery_id')['cycle'].transform('max')
        df['RUL'] = (df['max_cycle'] - df['cycle']) / df['max_cycle']
        # Clip RUL between 0 and 1
        df['RUL'] = df['RUL'].clip(0, 1)

        # Handle NaNs/Infs that might arise
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in features + [target]:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0) # Simple fill

        # Prepare data dictionary for multi-input sequencing
        data_dict = {
            'voltage': df['voltage_battery'].values.reshape(-1, 1),
            'current': df['current_battery'].values.reshape(-1, 1),
            'temperature': df['temp_battery'].values.reshape(-1, 1),
            'capacity': df['capacity'].values.reshape(-1, 1)
        }
        target_rul = df['RUL'].values

        return data_dict, target_rul # Return dictionary of features and target array


    def create_sequences(self, X_dict, y):
        """Creates sequences for multi-input FNO+LSTM structure."""
        seq_lens = {
            'cnn': self.config.SEQ_LEN_CNN,
            'lstm': self.config.SEQ_LEN_LSTM
        }
        # Stride can be adjusted, e.g., stride = int(self.config.SEQ_LEN_CNN * 0.5) for 50% overlap
        return create_multi_input_sequences(X_dict, seq_lens, y, stride=1)

    def sequence_and_split(self, X_dict, y):
        """Overrides base method for dictionary input X."""
        X_seq_dict, y_seq = self.create_sequences(X_dict, y)

        # Ensure y_seq is 2D for scaling
        if y_seq.ndim == 1:
            y_seq = y_seq.reshape(-1, 1)

        # Scale features (each entry in X_seq_dict)
        X_seq_scaled_dict = {}
        for key, data in X_seq_dict.items():
             # Use RobustScaler as in preproc.py example, or choose another
             scaled_data, scaler = scale_data(data, scaler_type='robust')
             X_seq_scaled_dict[key] = scaled_data
             self.scalers[key] = scaler # Store scaler per feature type

        # Scale target (y_seq)
        y_seq_scaled, target_scaler = scale_data(y_seq, scaler_type='robust') # Use robust for target too
        self.scalers['target'] = target_scaler

        # Split data (utils.split_data handles dicts)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X_seq_scaled_dict,
            y_seq_scaled,
            val_split=self.config.VALIDATION_SPLIT,
            test_split=self.config.TEST_SPLIT,
            shuffle=True, # Shuffle sequences
            seed=self.config.SEED
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # get_loaders is inherited and works with the MultiInputBatteryDataset
    # run is inherited