# battery_fno/preprocessing/nasa_preprocessor.py
import pandas as pd
import numpy as np
import os
import glob
from preprocessing.base_preprocessor import BasePreprocessor
from .utils import create_sequences, create_multi_input_sequences, scale_data, split_data
# Can import more advanced functions from preproc.py if needed and adapt them

class NASAPreprocessor(BasePreprocessor):
    """
    Preprocessor for NASA battery datasets (VIT format).
    """
    
    def load_data(self):
        """
        Load data from NASA battery discharge CSV files
        """
        self.logger.info("Loading NASA VIT dataset")
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
                
            # Add battery ID from filename
            battery_id = os.path.basename(file_path).split('_')[0]
            df['battery_id'] = battery_id
            
            all_df.append(df)
            
        if not all_df:
            self.logger.error("No data loaded")
            raise ValueError("No data files were loaded")
            
        combined_df = pd.concat(all_df, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_df)} records from {len(all_df)} files")
        
        return combined_df
    
    def preprocess(self, data):
        """
        Preprocess NASA VIT data
        """
        self.logger.info("Preprocessing NASA VIT data")
        
        # Handle missing values
        for col in self.input_features + [self.target_feature]:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        # Remove outliers (simple z-score method)
        for col in self.input_features:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            data[col] = np.where(z_scores < 3, data[col], data[col].median())
        
        # Apply smoothing if needed
        # data[self.input_features] = data[self.input_features].rolling(window=3, min_periods=1).mean()
        
        return data
    
    def create_sequences(self, preprocessed_data):
        """
        Create sequences for time series modeling
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


class NASARULPreprocessor(BasePreprocessor):
    """
    Preprocessor for NASA RUL dataset using multi-input approach (FNO+LSTM)
    """
    
    def load_data(self):
        """
        Load data from NASA battery discharge CSV files
        """
        self.logger.info("Loading NASA RUL dataset")
        
        if not self.data_dir or not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all discharge files
        discharge_files = glob.glob(os.path.join(self.data_dir, 'discharge/train/*.csv'))
        discharge_files += glob.glob(os.path.join(self.data_dir, 'discharge/test/*.csv'))
        
        self.logger.info(f"Found {len(discharge_files)} discharge files")
        
        all_data = []
        for file in discharge_files:
            battery_id = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file)
            df['battery_id'] = battery_id
            all_data.append(df)
            
        if not all_data:
            self.logger.error("No discharge files found")
            raise FileNotFoundError(f"No discharge CSV files found in {self.data_dir}")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_df)} records from {len(all_data)} files")
        
        return combined_df
    
    def preprocess(self, data):
        """
        Preprocess NASA RUL data
        """
        self.logger.info("Preprocessing NASA RUL data")
        
        # Calculate RUL per battery
        data['max_cycle'] = data.groupby('battery_id')['cycle'].transform('max')
        data['RUL'] = (data['max_cycle'] - data['cycle']) / data['max_cycle']
        # Clip RUL between 0 and 1
        data['RUL'] = data['RUL'].clip(0, 1)
        
        # Handle missing values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in self.input_features + [self.target_feature]:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove outliers (simple z-score method)
        for col in self.input_features:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data[col] = np.where(z_scores < 3, data[col], data[col].median())
        
        return data
    
    def create_sequences(self, preprocessed_data):
        """
        Create multi-input sequences for hybrid FNO+LSTM model
        """
        self.logger.info("Creating multi-input sequences for RUL prediction")
        
        # Extract CNN and LSTM sequence lengths from kwargs
        seq_len_cnn = self.kwargs.get('seq_len_cnn', 64)
        seq_len_lstm = self.kwargs.get('seq_len_lstm', 10)
        self.logger.info(f"Using sequence lengths: CNN={seq_len_cnn}, LSTM={seq_len_lstm}")
        
        # Prepare containers for sequences
        voltage_seqs = []
        current_seqs = []
        temp_seqs = []
        capacity_seqs = []
        rul_targets = []
        
        # Group by battery to maintain integrity
        for battery_id, battery_df in preprocessed_data.groupby('battery_id'):
            # Sort by cycle
            battery_df = battery_df.sort_values('cycle')
            
            # Calculate stride (use smaller stride for more sequences)
            stride = max(1, seq_len_cnn // 4)  # 75% overlap
            
            # Create sequences for each battery
            for i in range(0, len(battery_df) - max(seq_len_cnn, seq_len_lstm), stride):
                voltage_seq = battery_df['voltage_battery'].values[i:i+seq_len_cnn]
                current_seq = battery_df['current_battery'].values[i:i+seq_len_cnn]
                temp_seq = battery_df['temp_battery'].values[i:i+seq_len_cnn]
                capacity_seq = battery_df['capacity'].values[i:i+seq_len_lstm]
                
                # Target is RUL at end of sequence
                rul = battery_df['RUL'].values[i+seq_len_cnn-1]
                
                # Store sequences
                voltage_seqs.append(voltage_seq.reshape(-1, 1))
                current_seqs.append(current_seq.reshape(-1, 1))
                temp_seqs.append(temp_seq.reshape(-1, 1))
                capacity_seqs.append(capacity_seq.reshape(-1, 1))
                rul_targets.append(rul)
        
        # Create dictionary of input features
        X = {
            'voltage': np.array(voltage_seqs),
            'current': np.array(current_seqs),
            'temperature': np.array(temp_seqs),
            'capacity': np.array(capacity_seqs)
        }
        
        # Convert targets to numpy array
        y = np.array(rul_targets).reshape(-1, 1)
        
        self.logger.info(f"Created {len(y)} multi-input sequences")
        
        return X, y
    
    def get_dataset_info(self):
        """
        Get information about the dataset
        """
        # Extract CNN and LSTM sequence lengths from kwargs
        seq_len_cnn = self.kwargs.get('seq_len_cnn', 64)
        seq_len_lstm = self.kwargs.get('seq_len_lstm', 10)
        
        return {
            'input_dim_cnn': 1,  # Single feature per input (V, I, T processed separately)
            'input_dim_lstm': 1,  # Capacity processed by LSTM
            'output_dim': 1,  # RUL prediction
            'seq_len_cnn': seq_len_cnn,
            'seq_len_lstm': seq_len_lstm,
            'features': self.input_features,
            'target': self.target_feature
        }

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