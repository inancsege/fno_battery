import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from ..utils import (
    create_sequences,
    create_multi_input_sequences,
    scale_data,
    scale_data_with_existing,
    split_data,
    EXPECTED_KEYS
)

class TestCreateSequences:
    def test_create_sequences_basic(self, sample_data):
        # Test with simple 1D array
        X = sample_data
        y = np.array([10, 20, 30, 40, 50])
        seq_len = 2
        
        X_seq, y_seq = create_sequences(X, y, seq_len)
        
        assert X_seq.shape == (4, 2)
        assert y_seq.shape == (4,)
        assert np.array_equal(X_seq[0], [1, 2])
        assert np.array_equal(y_seq, np.array([20, 30, 40, 50], dtype=np.float32))
    
    def test_create_sequences_stride(self):
        # Test with stride > 1
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        seq_len = 3
        stride = 2
        
        X_seq, y_seq = create_sequences(X, y, seq_len, stride)
        
        assert X_seq.shape == (3, 3)
        assert y_seq.shape == (3,)
        assert np.array_equal(X_seq[0], [1, 2, 3])
        assert np.array_equal(X_seq[1], [3, 4, 5])
        assert np.array_equal(X_seq[2], [5, 6, 7])
        assert np.array_equal(y_seq, np.array([30, 50, 70], dtype=np.float32))
    
    def test_create_sequences_2d_input(self, sample_2d_data):
        # Test with 2D input
        X = sample_2d_data
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        seq_len = 2
        
        X_seq, y_seq = create_sequences(X, y, seq_len)
        
        assert X_seq.shape == (4, 2, 2)
        assert y_seq.shape == (4,)
        assert np.array_equal(X_seq[0], [[1, 10], [2, 20]])
        assert np.array_equal(y_seq, np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32))


class TestCreateMultiInputSequences:
    def test_create_multi_input_sequences_basic(self, sample_multi_input_data):
        # Create test data
        data_dict, target_rul = sample_multi_input_data
        seq_lens = {'cnn': 3, 'lstm': 2}
        
        result, rul = create_multi_input_sequences(data_dict, seq_lens, target_rul)
        
        # Check shapes
        assert result['voltage'].shape == (8, 3, 1)
        assert result['current'].shape == (8, 3, 1)
        assert result['temperature'].shape == (8, 3, 1)
        assert result['capacity'].shape == (8, 2, 1)
        assert rul.shape == (8,)
        
        # Check values (using maximum sequence length for RUL)
        assert np.array_equal(rul, np.array([2, 3, 4, 5, 6, 7, 8, 9]))
    
    def test_create_multi_input_sequences_missing_key(self):
        # Test with missing key
        data_dict = {
            'voltage': np.ones((10, 1)),
            'current': np.ones((10, 1)),
            # Missing 'temperature'
            'capacity': np.ones((10, 1))
        }
        seq_lens = {'cnn': 3, 'lstm': 2}
        target_rul = np.arange(10)
        
        with pytest.raises(ValueError, match="Missing required keys"):
            create_multi_input_sequences(data_dict, seq_lens, target_rul)
    
    def test_create_multi_input_sequences_different_lengths(self):
        # Test with arrays of different lengths
        data_dict = {
            'voltage': np.ones((10, 1)),
            'current': np.ones((10, 1)),
            'temperature': np.ones((10, 1)),
            'capacity': np.ones((11, 1))  # Different length
        }
        seq_lens = {'cnn': 3, 'lstm': 2}
        target_rul = np.arange(10)
        
        with pytest.raises(ValueError, match="have different lengths"):
            create_multi_input_sequences(data_dict, seq_lens, target_rul)


class TestScaleData:
    def test_scale_data_1d(self, sample_data):
        # Test 1D data
        data = sample_data
        scaled, scaler = scale_data(data)
        
        # The function reshapes 1D data to (n, 1)
        assert scaled.shape == (5, 1)
        assert np.allclose(scaled.mean(), 0, atol=1e-10)  # For StandardScaler
        assert np.allclose(scaled.std(), 1, atol=1e-10)
    
    def test_scale_data_2d(self, sample_2d_data):
        # Test 2D data
        data = sample_2d_data
        scaled, scaler = scale_data(data)
        
        assert scaled.shape == data.shape
        assert np.allclose(scaled.mean(axis=0), [0, 0], atol=1e-10)
        assert np.allclose(scaled.std(axis=0), [1, 1], atol=1e-10)
    
    def test_scale_data_3d(self, sample_3d_data):
        # Test 3D data
        data = sample_3d_data
        scaled, scaler = scale_data(data)
        
        assert scaled.shape == data.shape
        # Reshape to check stats
        flattened = scaled.reshape(-1, 1)
        assert np.allclose(flattened.mean(), 0, atol=1e-10)
        assert np.allclose(flattened.std(), 1, atol=1e-10)
    
    def test_scale_data_minmax(self, sample_data):
        # Test MinMaxScaler
        data = sample_data
        scaled, scaler = scale_data(data, scaler_type='minmax')
        
        # The function reshapes 1D data to (n, 1)
        assert scaled.shape == (5, 1)
        assert np.min(scaled) == 0
        assert np.max(scaled) == 1
    
    def test_scale_data_robust(self):
        # Test RobustScaler
        data = np.array([1, 2, 3, 100, 5])  # With outlier
        scaled, scaler = scale_data(data, scaler_type='robust')
        
        # The function reshapes 1D data to (n, 1)
        assert scaled.shape == (5, 1)
        # RobustScaler should make scaling less sensitive to outliers
        
    def test_scale_data_invalid_type(self):
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown scaler_type"):
            scale_data(data, scaler_type='unknown')
    
    def test_scale_data_invalid_dims(self):
        # Test data with invalid dimensions
        data = np.array([[[[1, 2]]]])  # 4D
        with pytest.raises(ValueError, match="Data must be 1D, 2D or 3D"):
            scale_data(data)


class TestScaleDataWithExisting:
    def test_scale_data_with_existing_1d(self, sample_data):
        # First fit a scaler
        data = sample_data
        _, scaler = scale_data(data)
        
        # Test with new data
        new_data = np.array([6, 7, 8])
        scaled = scale_data_with_existing(new_data, scaler)
        
        # The function reshapes 1D data to (n, 1)
        assert scaled.shape == (3, 1)
    
    def test_scale_data_with_existing_2d(self, sample_2d_data):
        # First fit a scaler on 2D data
        data = sample_2d_data
        _, scaler = scale_data(data)
        
        # Test with new data
        new_data = np.array([[4, 40], [5, 50]])
        scaled = scale_data_with_existing(new_data, scaler)
        
        assert scaled.shape == new_data.shape
    
    def test_scale_data_with_existing_3d(self, sample_3d_data):
        # First fit a scaler on 3D data
        data = sample_3d_data
        _, scaler = scale_data(data)
        
        # Test with new data
        new_data = np.array([[[5], [6]]])
        scaled = scale_data_with_existing(new_data, scaler)
        
        assert scaled.shape == new_data.shape


class TestSplitData:
    def test_split_data_array(self):
        # Test with numpy array
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X, y, val_split=0.2, test_split=0.1, shuffle=False
        )
        
        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 20
        assert X_test.shape[0] == 10
        assert y_train.shape[0] == 70
        assert y_val.shape[0] == 20
        assert y_test.shape[0] == 10
    
    def test_split_data_dict(self, sample_multi_input_data):
        # Test with dictionary input
        X, y = sample_multi_input_data
        # Extend y to 100 for this test
        y = np.arange(100)
        
        # Make data dictionary of size 100
        X = {
            'voltage': np.ones((100, 1)),
            'current': np.ones((100, 1)) * 2,
            'temperature': np.ones((100, 1)) * 3,
            'capacity': np.ones((100, 1)) * 4
        }
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X, y, val_split=0.2, test_split=0.1, shuffle=False
        )
        
        assert X_train['voltage'].shape[0] == 70
        assert X_val['current'].shape[0] == 20
        assert X_test['temperature'].shape[0] == 10
        assert y_train.shape[0] == 70
    
    def test_split_data_shuffle(self):
        # Test with shuffling
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X, y, val_split=0.2, test_split=0.1, shuffle=True, seed=42
        )
        
        # With shuffling, indices aren't sequential, but shapes should be the same
        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 20
        assert X_test.shape[0] == 10
        # With shuffling and a fixed seed, results should be deterministic
        
    def test_split_data_invalid_split(self):
        X = np.arange(10).reshape(-1, 1)
        y = np.arange(10)
        
        # Test invalid splits
        with pytest.raises(AssertionError):
            split_data(X, y, val_split=-0.1, test_split=0.1)
        
        with pytest.raises(AssertionError):
            split_data(X, y, val_split=0.5, test_split=0.6)  # Sum > 1
    
    def test_split_data_invalid_type(self):
        # Test with invalid X type that will cause specific errors
        # that are different from the expected TypeError
        X = "not a valid input"
        y = np.arange(10)
        
        # The actual error will be TypeError: string indices must be integers
        # We'll catch any TypeError
        with pytest.raises(TypeError):
            split_data(X, y, val_split=0.2, test_split=0.1) 