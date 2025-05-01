import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

class MLModelWrapper:
    """
    Base wrapper class for sklearn-style ML models with Pytorch compatibility.
    This allows seamless integration with our existing Trainer/Evaluator classes.
    """
    def __init__(self, model, seq_len=40, input_dim=None, output_dim=1, use_cuda=False):
        """
        Initialize the ML model wrapper
        
        Args:
            model: The scikit-learn compatible model instance
            seq_len (int): The sequence length from the input data
            input_dim (int): Number of input features
            output_dim (int): Number of output features (always 1 for regression)
            use_cuda (bool): Whether to use cuda (only affects the device for input/output)
        """
        self.model = model
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_fitted = False
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
    def preprocess_input(self, x):
        """
        Preprocess the PyTorch tensor input for scikit-learn models
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                            or (batch_size, input_dim)
                            
        Returns:
            numpy.ndarray: Processed input for scikit-learn model
        """
        # Move to CPU and convert to numpy
        x = x.detach().cpu().numpy()
        
        # Reshape if needed
        if len(x.shape) == 3:  # (batch, seq, features)
            # Flatten the sequence dimension into the feature dimension
            batch_size, seq_len, feat_dim = x.shape
            x = x.reshape(batch_size, seq_len * feat_dim)
        
        return x
    
    def postprocess_output(self, y_pred):
        """
        Convert numpy predictions back to PyTorch tensors
        
        Args:
            y_pred (numpy.ndarray): Model predictions
            
        Returns:
            torch.Tensor: Predictions as tensor
        """
        # Ensure y_pred is 2D
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
            
        # Convert back to torch tensor
        return torch.tensor(y_pred, dtype=torch.float32, device=self.device)
    
    def fit(self, X, y):
        """
        Fit the model to training data
        
        Args:
            X (torch.Tensor): Training inputs
            y (torch.Tensor): Training targets
            
        Returns:
            self: The fitted model wrapper
        """
        X_np = self.preprocess_input(X)
        y_np = y.detach().cpu().numpy()
        
        if len(y_np.shape) > 1 and y_np.shape[1] == 1:
            y_np = y_np.ravel()
            
        self.model.fit(X_np, y_np)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Generate predictions with the model
        
        Args:
            X (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Predictions
        """
        X_np = self.preprocess_input(X)
        y_pred = self.model.predict(X_np)
        return self.postprocess_output(y_pred)
    
    def forward(self, X):
        """
        PyTorch-style forward pass for compatibility with existing code
        
        Args:
            X (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Predictions
        """
        return self.predict(X)
    
    def to(self, device):
        """
        Simulate the PyTorch to() method for compatibility
        
        Args:
            device: The device to move the model to
            
        Returns:
            self: The model wrapper
        """
        self.device = device
        return self
    
    def eval(self):
        """
        Set model to evaluation mode (no-op for sklearn models, for compatibility)
        
        Returns:
            self: The model wrapper
        """
        return self
    
    def train(self):
        """
        Set model to training mode (no-op for sklearn models, for compatibility)
        
        Returns:
            self: The model wrapper
        """
        return self
    
    def parameters(self):
        """
        Dummy method for compatibility with PyTorch optimizers
        
        Returns:
            list: Empty list (no parameters to optimize with PyTorch)
        """
        return []


class XGBoostModel(MLModelWrapper):
    """
    XGBoost model wrapper for battery capacity prediction.
    
    Args:
        seq_len (int): The sequence length from the input data
        input_dim (int): Number of input features
        n_estimators (int): Number of trees
        max_depth (int): Maximum tree depth
        learning_rate (float): Learning rate
        use_cuda (bool): Whether to use GPU acceleration (if available)
    """
    def __init__(self, seq_len=40, input_dim=None, n_estimators=100, max_depth=5, 
                 learning_rate=0.1, use_cuda=False):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            tree_method='gpu_hist' if use_cuda and torch.cuda.is_available() else 'auto',
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        super(XGBoostModel, self).__init__(model, seq_len, input_dim, use_cuda=use_cuda)


class RandomForestModel(MLModelWrapper):
    """
    Random Forest model wrapper for battery capacity prediction.
    
    Args:
        seq_len (int): The sequence length from the input data
        input_dim (int): Number of input features
        n_estimators (int): Number of trees
        max_depth (int): Maximum tree depth
    """
    def __init__(self, seq_len=40, input_dim=None, n_estimators=100, max_depth=10):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        super(RandomForestModel, self).__init__(model, seq_len, input_dim)


class LinearRegressionModel(MLModelWrapper):
    """
    Linear Regression model wrapper for battery capacity prediction.
    
    Args:
        seq_len (int): The sequence length from the input data
        input_dim (int): Number of input features
    """
    def __init__(self, seq_len=40, input_dim=None):
        model = LinearRegression(n_jobs=-1)  # Use all available cores
        super(LinearRegressionModel, self).__init__(model, seq_len, input_dim)


class SVRModel(MLModelWrapper):
    """
    Support Vector Regression model wrapper for battery capacity prediction.
    
    Args:
        seq_len (int): The sequence length from the input data
        input_dim (int): Number of input features
        kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C (float): Regularization parameter
        epsilon (float): Epsilon in the epsilon-SVR model
    """
    def __init__(self, seq_len=40, input_dim=None, kernel='rbf', C=1.0, epsilon=0.1):
        model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon
        )
        super(SVRModel, self).__init__(model, seq_len, input_dim) 