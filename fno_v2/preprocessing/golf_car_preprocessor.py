# battery_fno/preprocessing/golf_car_preprocessor.py
import pandas as pd
import numpy as np
import os
import glob
import re
from .base_preprocessor import BasePreprocessor
from .utils import create_sequences

class GolfCarPreprocessor(BasePreprocessor):
    """Preprocessor for Golf Car battery dataset from scaledData directory."""

    def load_data(self):
        """Loads data from scaledData CSV files."""
        all_df = []
        data_config = self.config.ACTIVE_DATA_CONFIG
        data_dir = data_config['dir']
        file_pattern = os.path.join(data_dir, "scaledData*.csv")
        file_list = glob.glob(file_pattern)
        
        if not file_list:
            raise FileNotFoundError(f"No CSV files found with pattern {file_pattern}")

        print(f"Found {len(file_list)} golf car battery data files")
        
        # Load each file, sample them to keep size manageable if needed
        sample_size = data_config.get('sample_size', 10000)  # Default sample size if not specified
        
        for i, file_path in enumerate(file_list):
            print(f"Loading file {i+1}/{len(file_list)}: {os.path.basename(file_path)}")
            
            try:
                # Try different encodings
                try:
                    df = pd.read_csv(file_path, nrows=sample_size)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, nrows=sample_size, encoding='latin1')
                    except Exception:
                        df = pd.read_csv(file_path, nrows=sample_size, encoding='utf-8-sig')
                
                # Get expected column names from the config
                expected_features = data_config['features']
                expected_target = data_config['target']
                
                # Check actual column names in the file
                actual_columns = df.columns.tolist()
                print(f"Actual columns: {actual_columns}")
                
                # Check if all expected columns exist, accounting for encoding issues
                all_columns_present = True
                
                # Normalize column names to handle encoding issues
                normalized_cols = {}
                for col in actual_columns:
                    # Remove non-ASCII characters and normalize spaces
                    normalized = re.sub(r'[^\x00-\x7F]+', '', col).strip()
                    normalized_cols[normalized] = col
                
                # Map expected feature names to actual column names
                feature_mapping = {}
                for expected_feature in expected_features:
                    normalized_expected = re.sub(r'[^\x00-\x7F]+', '', expected_feature).strip()
                    
                    # Try to find a match in the normalized columns
                    found = False
                    for norm_col, actual_col in normalized_cols.items():
                        if normalized_expected in norm_col or norm_col in normalized_expected:
                            feature_mapping[expected_feature] = actual_col
                            found = True
                            break
                    
                    if not found:
                        all_columns_present = False
                        print(f"  Missing expected feature: {expected_feature}")
                
                # Check target column
                normalized_target = re.sub(r'[^\x00-\x7F]+', '', expected_target).strip()
                target_found = False
                for norm_col, actual_col in normalized_cols.items():
                    if normalized_target in norm_col or norm_col in normalized_target:
                        feature_mapping[expected_target] = actual_col
                        target_found = True
                        break
                
                if not target_found:
                    all_columns_present = False
                    print(f"  Missing expected target: {expected_target}")
                
                if not all_columns_present:
                    print(f"Skipping file {file_path} due to missing columns")
                    continue
                
                # Rename columns to expected names
                df_renamed = df.rename(columns={v: k for k, v in feature_mapping.items()})
                
                # Add the file to our dataset
                all_df.append(df_renamed)
                
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
                continue
                
        if not all_df:
            raise ValueError("No valid golf car battery data files could be loaded.")

        combined_df = pd.concat(all_df, ignore_index=True)
        print(f"Total samples after loading: {len(combined_df)}")
        return combined_df

    def preprocess(self):
        """Basic preprocessing for golf car battery data."""
        df = self.load_data()
        data_config = self.config.ACTIVE_DATA_CONFIG
        features = data_config['features']
        target = data_config['target']
        
        # Handle NaNs/Infs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)  # Simple fill
        
        # Normalize data 
        X = df[features].values
        y = df[target].values
        
        # Clean potential inf values after processing
        X = np.nan_to_num(X, nan=0.0, posinf=1e30, neginf=-1e30)
        y = np.nan_to_num(y, nan=0.0, posinf=1e30, neginf=-1e30)
        
        return X, y

    def create_sequences(self, X, y):
        """Uses the standard sequence creation utility."""
        return create_sequences(X, y, seq_len=self.config.SEQ_LEN, stride=1) 