import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
import tensorflow as tf

def load_and_preprocess_data(data_dir='../data/NASA', 
                           seq_len_cnn=64, 
                           seq_len_lstm=10, 
                           test_size=0.2, 
                           random_state=42,
                           overlap_ratio=0.5):  # 50% overlap between sequences
    """
    Enhanced data preprocessing pipeline for RUL prediction
    
    Parameters:
    -----------
    data_dir : str
        Root directory of NASA battery data
    seq_len_cnn : int
        Sequence length for CNN inputs (voltage, current, temperature)
    seq_len_lstm : int
        Sequence length for LSTM input (capacity)
    test_size : float
        Proportion of data to use for validation
    random_state : int
        Random seed for reproducibility
    overlap_ratio : float
        Ratio of overlap between consecutive sequences (0 to 1)
    """
    
    def normalize_sequence(seq, scaler=None, fit=False):
        """
        Normalize sequence data using robust scaling
        
        Parameters:
        -----------
        seq : numpy array
            Input sequence to normalize
        scaler : sklearn.preprocessing.RobustScaler
            Scaler to use (if None and fit=True, creates new scaler)
        fit : bool
            Whether to fit the scaler on this data
        """
        original_shape = seq.shape
        seq_reshaped = seq.reshape(-1, 1)
        
        if scaler is None and fit:
            scaler = RobustScaler()
            seq_normalized = scaler.fit_transform(seq_reshaped)
            return seq_normalized.reshape(original_shape), scaler
        elif scaler is not None:
            seq_normalized = scaler.transform(seq_reshaped)
            return seq_normalized.reshape(original_shape), scaler
        else:
            raise ValueError("Either scaler must be provided or fit must be True")

    def remove_outliers(seq, threshold=3):
        """
        Remove outliers using z-score method
        
        Parameters:
        -----------
        seq : numpy array
            Input sequence
        threshold : float
            Z-score threshold for outlier detection
        """
        z_scores = np.abs((seq - np.mean(seq)) / np.std(seq))
        seq_clean = np.where(z_scores < threshold, seq, np.nan)
        # Interpolate NaN values
        mask = np.isnan(seq_clean)
        seq_clean[mask] = np.interp(np.flatnonzero(mask), 
                                  np.flatnonzero(~mask), 
                                  seq_clean[~mask])
        return seq_clean

    def filter_noise(seq, window_length=5, polyorder=2):
        """
        Apply Savitzky-Golay filter to remove noise
        
        Parameters:
        -----------
        seq : numpy array
            Input sequence
        window_length : int
            Length of the filter window
        polyorder : int
            Order of the polynomial used to fit the samples
        """
        return signal.savgol_filter(seq, window_length, polyorder)

    def augment_sequence(seq, noise_level=0.01, num_augmentations=1):
        """
        Perform data augmentation on sequences
        
        Parameters:
        -----------
        seq : numpy array
            Input sequence
        noise_level : float
            Standard deviation of Gaussian noise
        num_augmentations : int
            Number of augmented sequences to generate
        """
        augmented_seqs = []
        for _ in range(num_augmentations):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, seq.shape)
            aug_seq = seq + noise
            
            # Random scaling (±5%)
            scale_factor = np.random.uniform(0.95, 1.05)
            aug_seq *= scale_factor
            
            # Random time warping (slight stretching/compression)
            time_warp_factor = np.random.uniform(0.95, 1.05)
            orig_len = len(aug_seq)
            warped_len = int(orig_len * time_warp_factor)
            warped_seq = signal.resample(aug_seq, warped_len)
            # Resample back to original length
            aug_seq = signal.resample(warped_seq, orig_len)
            
            augmented_seqs.append(aug_seq)
        return augmented_seqs

    def create_sequences(df, battery_id, max_cycle):
        """
        Create sequences with overlap for a single battery
        
        Parameters:
        -----------
        df : pandas DataFrame
            Battery data
        battery_id : str
            Battery identifier
        max_cycle : int
            Maximum cycle number for this battery
        """
        sequences = {
            'voltage': [], 'current': [], 'temperature': [], 
            'capacity': [], 'rul': [], 'battery_id': []
        }
        
        # Calculate stride for overlapping sequences
        stride = int(seq_len_cnn * (1 - overlap_ratio))
        
        for i in range(0, len(df) - max(seq_len_cnn, seq_len_lstm), stride):
            # Extract sequences
            v_seq = df['voltage_battery'].iloc[i:i+seq_len_cnn].values
            i_seq = df['current_battery'].iloc[i:i+seq_len_cnn].values
            t_seq = df['temp_battery'].iloc[i:i+seq_len_cnn].values
            c_seq = df['capacity'].iloc[i:i+seq_len_lstm].values
            
            # Clean and preprocess sequences
            v_seq = remove_outliers(v_seq)
            i_seq = remove_outliers(i_seq)
            t_seq = remove_outliers(t_seq)
            
            # Apply noise filtering
            v_seq = filter_noise(v_seq)
            i_seq = filter_noise(i_seq)
            t_seq = filter_noise(t_seq)
            
            # Calculate RUL
            current_cycle = df['cycle'].iloc[i+seq_len_cnn-1]
            rul = (max_cycle - current_cycle) / max_cycle
            
            # Store sequences
            sequences['voltage'].append(v_seq.reshape(-1, 1))
            sequences['current'].append(i_seq.reshape(-1, 1))
            sequences['temperature'].append(t_seq.reshape(-1, 1))
            sequences['capacity'].append(c_seq.reshape(-1, 1))
            sequences['rul'].append(rul)
            sequences['battery_id'].append(battery_id)
            
            # Add augmented sequences for training data
            if rul > 0.2:  # Only augment early/middle lifecycle data
                aug_v_seqs = augment_sequence(v_seq)
                aug_i_seqs = augment_sequence(i_seq)
                aug_t_seqs = augment_sequence(t_seq)
                
                for aug_v, aug_i, aug_t in zip(aug_v_seqs, aug_i_seqs, aug_t_seqs):
                    sequences['voltage'].append(aug_v.reshape(-1, 1))
                    sequences['current'].append(aug_i.reshape(-1, 1))
                    sequences['temperature'].append(aug_t.reshape(-1, 1))
                    sequences['capacity'].append(c_seq.reshape(-1, 1))
                    sequences['rul'].append(rul)
                    sequences['battery_id'].append(battery_id)
        
        return sequences

    # Main preprocessing pipeline
    try:
        print("Starting data preprocessing pipeline...")
        
        # 1. Load data
        discharge_files = glob.glob(os.path.join(data_dir, 'discharge/train/*.csv'))
        print(f"Found {len(discharge_files)} discharge files")
        
        # Initialize scalers
        scalers = {
            'voltage': None,
            'current': None,
            'temperature': None,
            'capacity': None
        }
        
        # 2. Process each battery
        all_sequences = {
            'voltage': [], 'current': [], 'temperature': [], 
            'capacity': [], 'rul': [], 'battery_id': []
        }
        
        for discharge_file in discharge_files:
            battery_id = os.path.basename(discharge_file).split('_')[0]
            print(f"Processing battery {battery_id}...")
            
            # Load discharge data
            df = pd.read_csv(discharge_file)
            max_cycle = df['cycle'].max()
            
            # Create sequences for this battery
            sequences = create_sequences(df, battery_id, max_cycle)
            
            # Extend all sequences
            for key in all_sequences:
                all_sequences[key].extend(sequences[key])
        
        # 3. Convert to numpy arrays
        for key in all_sequences:
            if key != 'battery_id':
                all_sequences[key] = np.array(all_sequences[key])
        
        # 4. Normalize sequences
        for key in ['voltage', 'current', 'temperature', 'capacity']:
            all_sequences[key], scalers[key] = normalize_sequence(
                all_sequences[key], 
                fit=True
            )
        
        # 5. Split into train/validation sets
        # Use time-based split (earlier batteries for training, later for validation)
        unique_batteries = np.unique(all_sequences['battery_id'])
        split_idx = int(len(unique_batteries) * (1 - test_size))
        train_batteries = unique_batteries[:split_idx]
        
        train_mask = np.isin(all_sequences['battery_id'], train_batteries)
        
        X_train = {
            'voltage': all_sequences['voltage'][train_mask],
            'current': all_sequences['current'][train_mask],
            'temperature': all_sequences['temperature'][train_mask],
            'capacity': all_sequences['capacity'][train_mask]
        }
        
        X_val = {
            'voltage': all_sequences['voltage'][~train_mask],
            'current': all_sequences['current'][~train_mask],
            'temperature': all_sequences['temperature'][~train_mask],
            'capacity': all_sequences['capacity'][~train_mask]
        }
        
        y_train = all_sequences['rul'][train_mask]
        y_val = all_sequences['rul'][~train_mask]
        
        print("Data preprocessing completed successfully!")
        print(f"Training samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        
        return X_train, X_val, y_train, y_val, scalers
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise