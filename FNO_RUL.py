import tensorflow as tf
# Enable device placement logging for debugging
tf.debugging.set_log_device_placement(True)
# Limit memory growth to prevent OOM issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Error setting memory growth for GPU")

import numpy as np
import os
import pandas as pd
import glob
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import Keras components
from tensorflow.keras.layers import LayerNormalization, Input, Dense, Dropout, LSTM, Reshape, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("Error setting memory growth for GPU")

class SpectralConv1D(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes, **kwargs):
        super(SpectralConv1D, self).__init__(**kwargs)
        # Enable debugging for this layer
        print(f"Initializing SpectralConv1D with in_channels={in_channels}, out_channels={out_channels}, modes={modes}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Initialize spectral kernel as a Variable for better memory management
        self.spectral_kernel = self.add_weight(
            name='spectral_kernel',
            shape=(modes, in_channels, out_channels),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        
        # Create a buffer Variable for zeros padding to avoid repeated allocation
        self.zeros_buffer = None
    def call(self, x):
        # Debug input tensor
        print(f"SpectralConv1D input shape: {x.shape}, dtype: {x.dtype}")
        
        # Define spectral convolution without experimental_compile
        # which can cause memory management issues
        def spectral_convolution(input_x):
            # Print tensor info for debugging
            print(f"Starting spectral_convolution with input shape: {input_x.shape}")
            
            # input_x shape: [batch, seq_len, in_channels]
            batch_size = tf.shape(input_x)[0]
            seq_len = tf.shape(input_x)[1]
            
            # Always ensure we're working with float32 for input
            # Use convert_to_tensor to get a fresh tensor without reference issues
            input_x = tf.convert_to_tensor(input_x, dtype=tf.float32)
            
            # Validate input channels only
            tf.debugging.assert_equal(
                tf.shape(input_x)[-1], 
                self.in_channels,
                message="Input channels mismatch in SpectralConv1D"
            )
            
            try:
                # Use a more controlled scope for FFT operations
                with tf.name_scope("fft_operations"):
                    # Transpose for channel-first format (required for rfft)
                    # Use convert_to_tensor to avoid memory leaks
                    x_t = tf.convert_to_tensor(
                        tf.transpose(input_x, [0, 2, 1]),  # [batch, in_channels, seq_len]
                        dtype=tf.float32
                    )
                    
                    print(f"Pre-FFT shape: {x_t.shape}, dtype: {x_t.dtype}")
                    
                    # Compute real FFT with safe handling - use CPU for better memory management
                    with tf.device('/CPU:0'):
                        x_ft = tf.signal.rfft(x_t)  # [batch, in_channels, seq_len//2 + 1]
                        
                    print(f"Post-FFT shape: {x_ft.shape}, dtype: {x_ft.dtype}")
                    
                    # Clear the intermediate tensor to save memory
                    x_t = None
                    
                    # Transpose for spectral convolution
                    # Create new tensor to avoid memory leaks
                    x_ft_transposed = tf.convert_to_tensor(
                        tf.transpose(x_ft, [0, 2, 1]),  # [batch, seq_len//2 + 1, in_channels]
                        dtype=tf.complex64
                    )
                    
                    # Clear the intermediate tensor to save memory
                    x_ft = None
                    
                    # Calculate frequency modes to use
                    n_freqs = tf.shape(x_ft_transposed)[1]
                    actual_modes = tf.minimum(self.modes, n_freqs)
                    
                    # Get the low-frequency modes only - avoid in-place operations
                    x_ft_modes = tf.slice(
                        x_ft_transposed, 
                        [0, 0, 0], 
                        [batch_size, actual_modes, self.in_channels]
                    )
                    
                    # Create complex kernel for spectral convolution
                    kernel_complex = tf.cast(
                        self.spectral_kernel[:actual_modes],
                        dtype=tf.complex64
                    )
                    
                    print(f"x_ft_modes shape: {x_ft_modes.shape}, kernel shape: {kernel_complex.shape}")
                    
                    # Perform spectral convolution using einsum
                    modes_product = tf.einsum('bmi,mio->bmo', x_ft_modes, kernel_complex)
                    
                    # Clear intermediate tensors
                    x_ft_modes = None
                    x_ft_transposed = None
                    
                    # Create zero padding for higher frequencies
                    padding_shape = [batch_size, n_freqs - actual_modes, self.out_channels]
                    zeros_padding = tf.zeros(padding_shape, dtype=tf.complex64)
                    
                    print(f"modes_product shape: {modes_product.shape}, zeros shape: {zeros_padding.shape}")
                    
                    # Concatenate the computed modes with zeros for higher frequencies
                    # Use convert_to_tensor to ensure we have a clean tensor
                    out_ft = tf.convert_to_tensor(
                        tf.concat([modes_product, zeros_padding], axis=1),
                        dtype=tf.complex64
                    )
                    
                    # Clear intermediate tensors
                    modes_product = None
                    zeros_padding = None
                    
                    # Transpose for inverse FFT
                    out_ft_transposed = tf.convert_to_tensor(
                        tf.transpose(out_ft, [0, 2, 1]),  # [batch, out_channels, n_freqs]
                        dtype=tf.complex64
                    )
                    
                    # Clear intermediate tensor
                    out_ft = None
                    
                    print(f"Pre-IFFT shape: {out_ft_transposed.shape}")
                    
                    # Apply inverse real FFT with safe handling - use CPU for better memory management
                    with tf.device('/CPU:0'):
                        x_out = tf.signal.irfft(out_ft_transposed, fft_length=[seq_len])
                    
                    # Clear intermediate tensor
                    out_ft_transposed = None
                    
                    print(f"Post-IFFT shape: {x_out.shape}")
                    
                    # Transpose back to channel-last format - avoid in-place operations
                    output = tf.convert_to_tensor(
                        tf.transpose(x_out, [0, 2, 1]),  # [batch, seq_len, out_channels]
                        dtype=tf.float32
                    )
                    
                    # Clear intermediate tensor
                    x_out = None
                    
                    print(f"Final output shape: {output.shape}")
                    
                    # Explicitly cast to float32 and create a new tensor
                    result = tf.cast(output, tf.float32)
                    return result
                
            except Exception as e:
                print(f"Error in spectral_convolution: {e}")
                # Return zeros with the correct shape as a fallback
                return tf.zeros([batch_size, seq_len, self.out_channels], dtype=tf.float32)
        
        try:
            # Call the spectral convolution function
            result = spectral_convolution(x)
            
            # Add final shape validation (only essential dimensions)
            tf.debugging.assert_equal(
                tf.shape(result)[-1], 
                self.out_channels,
                message="Output channels mismatch in final result"
            )
            
            # Debug final result
            print(f"Final result shape: {result.shape}, dtype: {result.dtype}")
            
            # Return the result with clean memory handling
            return tf.identity(result)
            
        except Exception as e:
            # Print debugging information
            print(f"Error in SpectralConv1D.call: {e}")
            # Return zeros with the correct shape as a fallback
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            return tf.zeros([batch_size, seq_len, self.out_channels], dtype=tf.float32)

class FNOBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, hidden_channels, modes, seq_length=64, activation='relu', **kwargs):
        super(FNOBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.seq_length = seq_length
        
        # Activation function
        if isinstance(activation, str):
            self.activation = tf.keras.activations.get(activation)
        else:
            self.activation = activation
        
        # Initial channel projection
        self.lifting = tf.keras.layers.Dense(hidden_channels)
        
        # Spectral convolution
        self.conv = SpectralConv1D(hidden_channels, hidden_channels, modes)
        
        # MLP branch
        self.w1 = tf.keras.layers.Dense(hidden_channels, activation=self.activation)
        self.w2 = tf.keras.layers.Dense(hidden_channels)
        
        # Layer normalization
        self.norm = LayerNormalization()
    
    def call(self, x):
        # Shape validation for specific dimensions only
        tf.debugging.assert_equal(
            tf.shape(x)[1], 
            self.seq_length,
            message="Sequence length mismatch in FNOBlock input"
        )
        tf.debugging.assert_equal(
            tf.shape(x)[2], 
            self.in_channels,
            message="Input channels mismatch in FNOBlock"
        )
        
        # Project input to hidden channels
        x = self.lifting(x)  # [batch, seq_length, hidden_channels]
        identity = x
        
        # Spectral convolution branch
        x1 = self.conv(x)  # [batch, seq_length, hidden_channels]
        
        # MLP branch
        x2 = self.w1(x)  # [batch, seq_length, hidden_channels]
        x2 = self.w2(x2)  # [batch, seq_length, hidden_channels]
        
        # Combine branches
        x = x1 + x2
        x = self.norm(x)
        x = x + identity
        
        # Shape validation for essential dimensions only
        tf.debugging.assert_equal(
            tf.shape(x)[1], 
            self.seq_length,
            message="Sequence length mismatch in FNOBlock output"
        )
        tf.debugging.assert_equal(
            tf.shape(x)[2], 
            self.hidden_channels,
            message="Hidden channels mismatch in FNOBlock output"
        )
        
        return x

def build_fno_model(seq_len_lstm, seq_len_cnn, input_dims):
    # Input layers
    v_input = Input(shape=(seq_len_cnn, input_dims['v']), name='voltage_input')
    i_input = Input(shape=(seq_len_cnn, input_dims['i']), name='current_input')
    t_input = Input(shape=(seq_len_cnn, input_dims['t']), name='temperature_input')
    c_input = Input(shape=(seq_len_lstm, input_dims['c']), name='capacity_input')
    
    # FNO processing with explicit dimensions
    hidden_channels = 32
    v_fno = FNOBlock(in_channels=1, hidden_channels=hidden_channels, modes=4, seq_length=seq_len_cnn)(v_input)
    i_fno = FNOBlock(in_channels=1, hidden_channels=hidden_channels, modes=4, seq_length=seq_len_cnn)(i_input)
    t_fno = FNOBlock(in_channels=1, hidden_channels=hidden_channels, modes=4, seq_length=seq_len_cnn)(t_input)
    
    # Calculate flattened dimension
    fno_flat_dim = seq_len_cnn * hidden_channels
    
    # Flatten FNO outputs
    v_flat = Reshape((fno_flat_dim,))(v_fno)
    i_flat = Reshape((fno_flat_dim,))(i_fno)
    t_flat = Reshape((fno_flat_dim,))(t_fno)
    
    # LSTM processing with fixed output
    c_lstm = LSTM(hidden_channels, return_sequences=False)(c_input)
    
    # Concatenate features with known dimensions
    concat = concatenate([v_flat, i_flat, t_flat, c_lstm])
    
    # Dense layers with explicit dimensions
    x = Dense(64, activation='relu')(concat)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, name='rul_output')(x)
    
    # Create model
    model = Model(inputs=[v_input, i_input, t_input, c_input], outputs=output)
    return model

def load_and_preprocess_data(data_dir='../data/NASA', seq_len_cnn=64, seq_len_lstm=10, test_size=0.2, random_state=42):
    """
    Load and preprocess the NASA battery data for RUL prediction.
    
    Args:
        data_dir: Root directory of NASA data
        seq_len_cnn: Sequence length for CNN inputs (voltage, current, temperature)
        seq_len_lstm: Sequence length for LSTM input (capacity)
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val: Dictionary of training and validation inputs
        y_train, y_val: Training and validation targets (RUL)
    """
    # Load discharge data
    discharge_files = glob.glob(os.path.join(data_dir, 'discharge/train/*.csv'))
    # Load charge data
    charge_files = glob.glob(os.path.join(data_dir, 'charge/train/*.csv'))
    
    print(f"Found {len(discharge_files)} discharge files and {len(charge_files)} charge files")
    
    # Lists to store sequences and targets
    voltage_seqs = []
    current_seqs = []
    temp_seqs = []
    capacity_seqs = []
    rul_targets = []
    
    # Process each battery
    for discharge_file in discharge_files:
        battery_id = os.path.basename(discharge_file).split('_')[0]
        print(f"Processing {battery_id}...")
        
        # Load discharge data
        discharge_df = pd.read_csv(discharge_file)
        
        # Get matching charge file if it exists
        charge_file = os.path.join(data_dir, f'charge/train/{battery_id}_charge.csv')
        
        # Calculate the maximum cycle (for RUL calculation)
        max_cycle = discharge_df['cycle'].max()
        
        # Generate sequences
        for i in range(len(discharge_df) - max(seq_len_cnn, seq_len_lstm)):
            # Get data for CNN sequences
            v_seq = discharge_df['voltage_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            i_seq = discharge_df['current_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            t_seq = discharge_df['temp_battery'].iloc[i:i+seq_len_cnn].values.reshape(-1, 1)
            
            # Get data for LSTM sequence
            c_seq = discharge_df['capacity'].iloc[i:i+seq_len_lstm].values.reshape(-1, 1)
            
            # Get current cycle for target calculation
            current_cycle = discharge_df['cycle'].iloc[i+seq_len_cnn-1]
            
            # Calculate RUL: remaining cycles / total cycles (normalized between 0 and 1)
            rul = (max_cycle - current_cycle) / max_cycle
            
            # Store sequences and target
            voltage_seqs.append(v_seq)
            current_seqs.append(i_seq)
            temp_seqs.append(t_seq)
            capacity_seqs.append(c_seq)
            rul_targets.append(rul)
    
    # Convert lists to numpy arrays
    voltage_seqs = np.array(voltage_seqs)
    current_seqs = np.array(current_seqs)
    temp_seqs = np.array(temp_seqs)
    capacity_seqs = np.array(capacity_seqs)
    rul_targets = np.array(rul_targets)
    
    print(f"Generated {len(rul_targets)} sequences with shapes:")
    print(f"Voltage: {voltage_seqs.shape}")
    print(f"Current: {current_seqs.shape}")
    print(f"Temperature: {temp_seqs.shape}")
    print(f"Capacity: {capacity_seqs.shape}")
    print(f"RUL targets: {rul_targets.shape}")
    
    # Split into training and validation sets
    indices = np.arange(len(rul_targets))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    # Create dictionaries for model inputs
    X_train = {
        'voltage_input': voltage_seqs[train_indices],
        'current_input': current_seqs[train_indices],
        'temperature_input': temp_seqs[train_indices],
        'capacity_input': capacity_seqs[train_indices]
    }
    
    X_val = {
        'voltage_input': voltage_seqs[val_indices],
        'current_input': current_seqs[val_indices],
        'temperature_input': temp_seqs[val_indices],
        'capacity_input': capacity_seqs[val_indices]
    }
    
    y_train = rul_targets[train_indices]
    y_val = rul_targets[val_indices]
    
    return X_train, X_val, y_train, y_val

def create_callbacks(model_dir='../models'):
    """
    Create callbacks for model training.
    
    Args:
        model_dir: Directory to save model checkpoints
    
    Returns:
        List of callback objects
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create timestamp for unique model save path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_dir, f'fno_rul_model_{timestamp}.h5')
    log_dir = os.path.join(model_dir, 'logs', timestamp)
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    return [early_stopping, model_checkpoint, reduce_lr, tensorboard]

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train the FNO model.
    
    Args:
        model: Compiled FNO model
        X_train, y_train: Training data and targets
        X_val, y_val: Validation data and targets
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Training history
    """
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation loss.
    
    Args:
        history: Training history from model.fit()
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('../plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'../plots/training_history_{timestamp}.png')
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train FNO model for RUL prediction')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../data/NASA',
                       help='Directory containing NASA battery data')
    
    # Sequence parameters
    parser.add_argument('--seq_len_cnn', type=int, default=64,
                       help='Sequence length for CNN inputs (voltage, current, temperature)')
    parser.add_argument('--seq_len_lstm', type=int, default=10,
                       help='Sequence length for LSTM input (capacity)')
    
    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=32,
                       help='Number of hidden channels in FNO blocks')
    parser.add_argument('--modes', type=int, default=4,
                       help='Number of Fourier modes to use in FNO blocks')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Print arguments
    print("Training with the following parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X_train, X_val, y_train, y_val = load_and_preprocess_data(
        data_dir=args.data_dir,
        seq_len_cnn=args.seq_len_cnn,
        seq_len_lstm=args.seq_len_lstm,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Get input dimensions
    input_dims = {
        'v': X_train['voltage_input'].shape[2],
        'i': X_train['current_input'].shape[2],
        't': X_train['temperature_input'].shape[2],
        'c': X_train['capacity_input'].shape[2]
    }
    
    # Build model
    print("\nBuilding FNO model...")
    model = build_fno_model(
        seq_len_lstm=args.seq_len_lstm,
        seq_len_cnn=args.seq_len_cnn,
        input_dims=input_dims
    )
    
    # Set learning rate and compile model with appropriate loss and metrics
    optimizer = Adam(learning_rate=args.lr, clipnorm=1.0)  # Add gradient clipping for stability
    model.compile(
        optimizer=optimizer, 
        loss='mse',  # Mean squared error loss
        metrics=['mae', 'mape']  # Mean absolute error and mean absolute percentage error
    )
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on validation set
    print("\nEvaluating model on validation set...")
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation Loss (MSE): {val_loss:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    
    print("\nTraining complete!")
