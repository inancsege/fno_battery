# battery_fno/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm

class Trainer:
    """Handles the model training loop, validation, and saving."""

    def __init__(self, model, train_loader, val_loader, config, scaler=None):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.scaler = scaler # Pass scaler for potential inverse transform during logging if needed
        self.device = config.DEVICE
        
        # Improved check for scikit-learn and other ML models
        model_type = getattr(config, 'MODEL_TYPE', None)
        ml_models = ["XGBoost", "RandomForest", "LinearRegression", "SVR"]
        
        # Check both ways: by model characteristics and by model_type in config
        self.is_sklearn_model = (
            (hasattr(model, 'model') and hasattr(model, 'fit')) or 
            (hasattr(model, 'fit') and not hasattr(model, 'forward')) or
            (model_type in ml_models)
        )
        
        print(f"Model detected as {'ML/sklearn model' if self.is_sklearn_model else 'PyTorch model'}")
        
        # Only create optimizer for PyTorch models
        if not self.is_sklearn_model:
            # Add weight decay for L2 regularization
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=config.LEARNING_RATE,
                weight_decay=0.05  # Increased L2 regularization
            )
            
            # Add learning rate scheduler that changes more drastically
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor=0.4, patience=3, verbose=True
            )
        else:
            # For sklearn models, create placeholder attributes
            self.optimizer = None
            self.scheduler = None
            print("Skipping optimizer/scheduler creation for ML model")
        
        # Use combined loss: MSE + L1 + Huber for better generalization
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        self.huber_criterion = nn.SmoothL1Loss()
        
        # Define stronger data augmentation strength
        self.aug_noise_std = 0.15  # Triple the noise level
        self.aug_prob = 0.9  # Higher probability of applying augmentation
        
        # Define feature dropout parameters
        self.feature_dropout_prob = 0.2  # Probability of zeroing out features
        
        # Define scaling and shifting parameters
        self.scale_range = (0.7, 1.3)  # Random scaling
        self.shift_range = (-0.2, 0.2)  # Random shifting

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Implement aggressive early stopping
        self.patience = 7  # Reduced patience
        self.patience_counter = 0
        
        # Add curriculum learning to make model focus on different aspects
        self.epoch_count = 0
        
    def _apply_data_augmentation(self, inputs):
        """Apply aggressive data augmentation to the inputs during training."""
        # If inputs is a dictionary (for multi-input model)
        if isinstance(inputs, dict):
            augmented_inputs = {}
            
            for key, tensor in inputs.items():
                # Add much stronger Gaussian noise (always apply)
                noise = torch.randn_like(tensor) * self.aug_noise_std * tensor.std()
                
                # Random scaling (90% probability)
                if torch.rand(1).item() < 0.9:
                    scale_factor = torch.tensor(
                        np.random.uniform(self.scale_range[0], self.scale_range[1]),
                        device=tensor.device
                    )
                    tensor = tensor * scale_factor
                
                # Random shifting (80% probability)
                if torch.rand(1).item() < 0.8:
                    shift_amount = torch.tensor(
                        np.random.uniform(self.shift_range[0], self.shift_range[1]) * tensor.std(),
                        device=tensor.device
                    )
                    tensor = tensor + shift_amount
                
                # Add trends and seasonality (70% probability)
                if torch.rand(1).item() < 0.7 and tensor.size(1) > 10:
                    # Add trend
                    batch_size, seq_len = tensor.size(0), tensor.size(1)
                    trend_slope = np.random.uniform(-0.1, 0.1) * tensor.std().item()
                    trend = torch.arange(seq_len, device=tensor.device).float() * trend_slope
                    trend = trend.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
                    tensor = tensor + trend
                    
                    # Add seasonality
                    if torch.rand(1).item() < 0.5:
                        frequency = np.random.uniform(0.1, 0.5)
                        amplitude = np.random.uniform(0.05, 0.15) * tensor.std().item()
                        time_steps = torch.arange(seq_len, device=tensor.device).float()
                        seasonality = torch.sin(time_steps * frequency * 2 * np.pi) * amplitude
                        seasonality = seasonality.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
                        tensor = tensor + seasonality
                
                # Feature dropout - randomly zero out some dimensions (40% probability)
                if torch.rand(1).item() < 0.4 and tensor.size(-1) > 1:
                    feature_mask = torch.rand(tensor.size(-1), device=tensor.device) > self.feature_dropout_prob
                    tensor = tensor * feature_mask.float()
                
                # Add extreme outliers (10% probability)
                if torch.rand(1).item() < 0.1:
                    outlier_mask = torch.rand_like(tensor) < 0.01  # 1% of values become outliers
                    outlier_values = torch.randn_like(tensor) * tensor.std() * 5  # 5x std outliers
                    tensor = torch.where(outlier_mask, outlier_values, tensor)
                
                # Add final noise after all transformations
                tensor = tensor + noise
                
                augmented_inputs[key] = tensor
                
            return augmented_inputs
        
        # If inputs is a tensor (for standard model)
        else:
            # Similar augmentations for tensor case
            # Add Gaussian noise (always apply)
            noise = torch.randn_like(inputs) * self.aug_noise_std * inputs.std()
            
            # Random scaling (90% probability)
            if torch.rand(1).item() < 0.9:
                scale_factor = torch.tensor(
                    np.random.uniform(self.scale_range[0], self.scale_range[1]),
                    device=inputs.device
                )
                inputs = inputs * scale_factor
            
            # Random shifting (80% probability)
            if torch.rand(1).item() < 0.8:
                shift_amount = torch.tensor(
                    np.random.uniform(self.shift_range[0], self.shift_range[1]) * inputs.std(),
                    device=inputs.device
                )
                inputs = inputs + shift_amount
            
            # Add trends and seasonality (70% probability)
            if torch.rand(1).item() < 0.7 and inputs.size(1) > 10:
                # Add trend
                batch_size, seq_len = inputs.size(0), inputs.size(1)
                trend_slope = np.random.uniform(-0.1, 0.1) * inputs.std().item()
                trend = torch.arange(seq_len, device=inputs.device).float() * trend_slope
                trend = trend.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
                inputs = inputs + trend
                
                # Add seasonality
                if torch.rand(1).item() < 0.5:
                    frequency = np.random.uniform(0.1, 0.5)
                    amplitude = np.random.uniform(0.05, 0.15) * inputs.std().item()
                    time_steps = torch.arange(seq_len, device=inputs.device).float()
                    seasonality = torch.sin(time_steps * frequency * 2 * np.pi) * amplitude
                    seasonality = seasonality.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
                    inputs = inputs + seasonality
            
            # Feature dropout - randomly zero out some dimensions (40% probability)
            if torch.rand(1).item() < 0.4 and inputs.size(-1) > 1:
                feature_mask = torch.rand(inputs.size(-1), device=inputs.device) > self.feature_dropout_prob
                inputs = inputs * feature_mask.float()
            
            # Add extreme outliers (10% probability)
            if torch.rand(1).item() < 0.1:
                outlier_mask = torch.rand_like(inputs) < 0.01  # 1% of values become outliers
                outlier_values = torch.randn_like(inputs) * inputs.std() * 5  # 5x std outliers
                inputs = torch.where(outlier_mask, outlier_values, inputs)
            
            # Add final noise after all transformations
            inputs = inputs + noise
            
            return inputs

    def _run_epoch(self, epoch, is_training):
        """Runs a single epoch of training or validation."""
        if is_training:
            self.model.train()
            loader = self.train_loader
            desc = f"Epoch {epoch+1}/{self.config.EPOCHS} - Training"
        else:
            self.model.eval()
            loader = self.val_loader
            desc = f"Epoch {epoch+1}/{self.config.EPOCHS} - Validation"

        total_loss = 0.0
        total_l1_loss = 0.0
        total_mse_loss = 0.0
        total_huber_loss = 0.0
        num_batches = len(loader)
        
        # For scikit-learn models, we collect all data for batch training
        if self.is_sklearn_model and is_training:
            all_inputs = []
            all_targets = []
            
            for batch_data in tqdm(loader, desc=desc, leave=False):
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Skip augmentation if configured that way
                if is_training and self.config.AUGMENTATION and np.random.random() < self.aug_prob:
                    inputs = self._apply_data_augmentation(inputs)
                
                all_inputs.append(inputs)
                all_targets.append(targets)
            
            # Concatenate all batches
            X = torch.cat(all_inputs, dim=0)
            y = torch.cat(all_targets, dim=0)
            
            # Train the scikit-learn model
            self.model.fit(X, y)
            
            # Calculate training loss
            with torch.no_grad():
                y_pred = self.model.predict(X)
                mse_loss = self.mse_criterion(y_pred, y)
                l1_loss = self.l1_criterion(y_pred, y)
                huber_loss = self.huber_criterion(y_pred, y)
                
                # Combined loss
                loss = (self.config.MSE_WEIGHT * mse_loss + 
                        self.config.L1_WEIGHT * l1_loss + 
                        (1 - self.config.MSE_WEIGHT - self.config.L1_WEIGHT) * huber_loss)
                
                total_loss = loss.item()
                total_mse_loss = mse_loss.item()
                total_l1_loss = l1_loss.item()
                total_huber_loss = huber_loss.item()
            
            return total_loss, total_mse_loss, total_l1_loss, total_huber_loss
        
        # For scikit-learn models in validation mode
        elif self.is_sklearn_model and not is_training:
            all_inputs = []
            all_targets = []
            
            for batch_data in tqdm(loader, desc=desc, leave=False):
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                all_inputs.append(inputs)
                all_targets.append(targets)
            
            # Concatenate all batches
            X = torch.cat(all_inputs, dim=0)
            y = torch.cat(all_targets, dim=0)
            
            # Calculate validation loss
            with torch.no_grad():
                y_pred = self.model.predict(X)
                mse_loss = self.mse_criterion(y_pred, y)
                l1_loss = self.l1_criterion(y_pred, y)
                huber_loss = self.huber_criterion(y_pred, y)
                
                # Combined loss
                loss = (self.config.MSE_WEIGHT * mse_loss + 
                        self.config.L1_WEIGHT * l1_loss + 
                        (1 - self.config.MSE_WEIGHT - self.config.L1_WEIGHT) * huber_loss)
                
                total_loss = loss.item()
                total_mse_loss = mse_loss.item()
                total_l1_loss = l1_loss.item()
                total_huber_loss = huber_loss.item()
            
            return total_loss, total_mse_loss, total_l1_loss, total_huber_loss

        # For PyTorch models
        with torch.set_grad_enabled(is_training):
            for batch_data in tqdm(loader, desc=desc, leave=False):
                # Handle standard vs multi-input datasets
                if isinstance(batch_data[0], dict): # Multi-input
                     inputs = {key: tensor.to(self.device) for key, tensor in batch_data[0].items()}
                     targets = batch_data[1].to(self.device)
                else: # Standard input
                     inputs = batch_data[0].to(self.device)
                     targets = batch_data[1].to(self.device)

                # Apply data augmentation during training with specified probability
                if is_training and self.config.AUGMENTATION and np.random.random() < self.aug_prob:
                    inputs = self._apply_data_augmentation(inputs)

                # Reset gradients in training mode
                if is_training:
                    self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate multi-component loss for better generalization
                mse_loss = self.mse_criterion(outputs, targets)
                l1_loss = self.l1_criterion(outputs, targets)
                huber_loss = self.huber_criterion(outputs, targets)
                
                # Dynamically change loss focus based on epoch (curriculum learning)
                if self.epoch_count < self.config.EPOCHS // 3:
                    # Early epochs: Focus more on MSE to get general shape
                    mse_weight = 0.7
                    l1_weight = 0.2
                elif self.epoch_count < 2 * self.config.EPOCHS // 3:
                    # Middle epochs: Balance between MSE and L1
                    mse_weight = 0.5
                    l1_weight = 0.3
                else:
                    # Later epochs: Focus more on L1 for robust prediction
                    mse_weight = 0.4
                    l1_weight = 0.4
                
                # Calculate combined loss
                loss = (mse_weight * mse_loss + 
                        l1_weight * l1_loss + 
                        (1 - mse_weight - l1_weight) * huber_loss)
                
                # Add spectral regularization for FNO models to prevent high-frequency noise
                if hasattr(self.model, 'spectral_regularization'):
                    spec_reg = self.model.spectral_regularization()
                    loss = loss + self.config.SPECTRAL_REG_WEIGHT * spec_reg

                # Backward pass and optimization in training mode
                if is_training:
                    loss.backward()
                    # Add gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Accumulate batch loss
                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
                total_l1_loss += l1_loss.item()
                total_huber_loss += huber_loss.item()

        # Calculate average loss
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_l1_loss = total_l1_loss / num_batches
        avg_huber_loss = total_huber_loss / num_batches

        return avg_loss, avg_mse_loss, avg_l1_loss, avg_huber_loss
    
    def train(self):
        """Runs the training and validation loops for the specified number of epochs."""
        start_time = time.time()
        print(f"Starting training on {self.device}...")

        # Keep track of best loss for early stopping and model saving
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            self.epoch_count = epoch
            
            # Training phase
            train_loss, train_mse, train_l1, train_huber = self._run_epoch(epoch, is_training=True)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_mse, val_l1, val_huber = self._run_epoch(epoch, is_training=False)
            self.val_losses.append(val_loss)
            
            # Print detailed losses
            print(f"  Val losses - Combined: {val_loss:.6f}, MSE: {val_mse:.6f}, L1: {val_l1:.6f}, Huber: {val_huber:.6f}")
            
            # Update learning rate based on validation loss
            if not self.is_sklearn_model:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = 0  # Sklearn models don't have learning rate
                
            # Print epoch results
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}")
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save model for pytorch models only
                if not self.is_sklearn_model:
                    torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                    print(f"  ✨ New best model saved with Val Loss: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if self.config.EARLY_STOP and patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
                    break
                    
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"Training finished in {elapsed_time:.2f} seconds.")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        if not self.is_sklearn_model:
            print(f"Best model saved to: {self.config.BEST_MODEL_PATH}")
        
        print(f"Loss plot saved to {self.config.LOSS_PLOT_PATH}")
        
        return self.train_losses, self.val_losses