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
        self.config = config
        self.device = config['device']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.model_save_path = config['model_save_path']
        self.logger = config.get('logger', None)
        self.augmentation = config.get('augmentation', True)  # Default to True if not provided
        self.early_stop = config.get('early_stop', True)  # Default to True if not provided
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler  # Pass scaler for potential inverse transform during logging if needed
        
        # Improved check for scikit-learn and other ML models
        model_type = self.config.get('model_type', None)
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
            # Enhanced optimizer with weight decay and parameter grouping
            # Different learning rates for different parameter groups
            decay_params = []
            no_decay_params = []
            
            # Separate parameters with and without weight decay
            for name, param in self.model.named_parameters():
                if any(nd in name for nd in ['bias', 'layer_norm', 'LayerNorm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            # Configure parameter groups with different learning rates
            param_groups = [
                {'params': decay_params, 'weight_decay': 0.1},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            
            # Adam with weight decay (AdamW) for better regularization
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # OneCycleLR scheduler for super-convergence
            # Faster convergence with cyclical learning rates
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,  # Peak LR 10x the base LR
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,  # Spend 30% of time warming up
                div_factor=25,  # Initial LR = max_lr/25
                final_div_factor=10000,  # Final LR = max_lr/10000
                anneal_strategy='cos'
            )
        else:
            # For sklearn models, create placeholder attributes
            self.optimizer = None
            self.scheduler = None
            print("Skipping optimizer/scheduler creation for ML model")
        
        # Use combined loss with focal components for hard examples
        self.mse_criterion = nn.MSELoss(reduction='none')  # For focal weighting
        self.l1_criterion = nn.L1Loss(reduction='none')    # For focal weighting
        self.huber_criterion = nn.SmoothL1Loss(beta=0.25)  # More sensitive Huber loss
        
        # Loss weighting coefficients - prioritize different loss components
        self.mse_weight = 0.4
        self.l1_weight = 0.4
        self.huber_weight = 0.2
        self.focal_gamma = 2.0  # Focal loss parameter for hard examples
        
        # Define stronger data augmentation strength
        self.aug_noise_std = 0.05  # Standard noise level
        self.aug_prob = 0.8       # Probability of applying augmentation
        
        # Gradient clipping value to prevent explosions
        self.grad_clip_value = 1.0
        
        # Exponential Moving Average (EMA) model for better generalization
        self.use_ema = False  # Disable by default for now
        self.ema_decay = 0.999
        if not self.is_sklearn_model and self.use_ema:
            self.ema_model = self._create_ema_model()
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Implement sophisticated early stopping
        self.patience = 10
        self.patience_counter = 0
        self.min_improvement = 0.001  # Minimum improvement to reset patience
        
        # Learning rate warmup and curriculum
        self.epoch_count = 0
        self.warmup_epochs = 2
        
    def _create_ema_model(self):
        """Create and initialize EMA model as a copy of the base model."""
        # Create a new instance of the same model class
        ema_model = type(self.model)(**self.model.__dict__.get('_modules', {}))
        ema_model.to(self.device)
        
        # Load state dict directly for a more reliable copy
        ema_model.load_state_dict(self.model.state_dict())
        
        # No gradient tracking needed for EMA params
        for param in ema_model.parameters():
            param.requires_grad_(False)
        
        return ema_model
        
    def _update_ema_model(self):
        """Update Exponential Moving Average model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def _apply_data_augmentation(self, inputs):
        """Apply strategic data augmentation to the inputs during training."""
        # If inputs is a dictionary (for multi-input model)
        if isinstance(inputs, dict):
            augmented_inputs = {}
            
            for key, tensor in inputs.items():
                # Add Gaussian noise
                noise = torch.randn_like(tensor) * self.aug_noise_std * tensor.std()
                
                # Random scaling (70% probability)
                if torch.rand(1).item() < 0.7:
                    scale_factor = torch.tensor(
                        np.random.uniform(0.9, 1.1),  # Moderate scaling
                        device=tensor.device
                    )
                    tensor = tensor * scale_factor
                
                # Random shifting (50% probability)
                if torch.rand(1).item() < 0.5:
                    shift_amount = torch.tensor(
                        np.random.uniform(-0.1, 0.1) * tensor.std(),
                        device=tensor.device
                    )
                    tensor = tensor + shift_amount
                
                # Add small amount of noise
                tensor = tensor + noise
                
                augmented_inputs[key] = tensor
                
            return augmented_inputs
        
        # If inputs is a tensor (for standard model)
        else:
            batch_size = inputs.size(0)
            
            # Add Gaussian noise
            noise = torch.randn_like(inputs) * self.aug_noise_std * inputs.std()
            
            # Random scaling (70% probability)
            if torch.rand(1).item() < 0.7:
                scale_factor = torch.tensor(
                    np.random.uniform(0.9, 1.1),  # Moderate scaling
                    device=inputs.device
                )
                inputs = inputs * scale_factor
            
            # Random shifting (50% probability)
            if torch.rand(1).item() < 0.5:
                shift_amount = torch.tensor(
                    np.random.uniform(-0.1, 0.1) * inputs.std(),
                    device=inputs.device
                )
                inputs = inputs + shift_amount
            
            # Add noise
            inputs = inputs + noise
            
            return inputs

    def _compute_focal_loss(self, pred, target, criterion, gamma=2.0):
        """Compute focal loss to focus on hard examples."""
        loss = criterion(pred, target)  # Elementwise loss
        
        # Calculate the focal weight: (1 - p_t)^gamma
        with torch.no_grad():
            diff = torch.abs(pred - target)
            norm_diff = diff / (diff.max() + 1e-7)  # Normalize to [0, 1]
            focal_weight = (norm_diff + 0.1) ** gamma  # Add small constant to avoid zero
        
        # Apply focal weighting
        weighted_loss = loss * focal_weight
        
        return weighted_loss.mean()

    def _run_epoch(self, epoch, is_training):
        """Runs a single epoch of training or validation with improved techniques."""
        if is_training:
            self.model.train()
            loader = self.train_loader
            desc = f"Epoch {epoch+1}/{self.epochs} - Training"
        else:
            self.model.eval()
            loader = self.val_loader
            desc = f"Epoch {epoch+1}/{self.epochs} - Validation"

        total_loss = 0.0
        num_batches = len(loader)
        epoch_fraction = epoch / self.epochs
        
        # For scikit-learn models, we collect all data for batch training
        if self.is_sklearn_model and is_training:
            all_inputs = []
            all_targets = []
            
            for batch_data in tqdm(loader, desc=desc, leave=False):
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Skip augmentation if configured that way
                if is_training and self.augmentation and np.random.random() < self.aug_prob:
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
                loss = self.huber_criterion(y_pred, y)
                
            total_loss = loss.item()
            
            return total_loss / num_batches
        
        # For PyTorch models
        else:
            # Progress bar
            progress_bar = tqdm(loader, desc=desc, leave=False)
            
            # Set up batch cutoff (curriculum learning - gradually use more data)
            batch_cutoff = None
            if is_training and epoch < self.warmup_epochs:
                # During early epochs, focus on easier examples
                batch_cutoff = int(len(loader) * (0.5 + 0.5 * epoch / self.warmup_epochs))
            
            for batch_idx, batch_data in enumerate(progress_bar):
                # Curriculum learning - skip later batches in early epochs
                if batch_cutoff is not None and batch_idx >= batch_cutoff:
                    break
                    
                # Extract inputs and targets
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Apply augmentation during training
                if is_training and self.augmentation and np.random.random() < self.aug_prob:
                    inputs = self._apply_data_augmentation(inputs)
                
                # Forward pass
                with torch.set_grad_enabled(is_training):
                    predictions = self.model(inputs)
                    
                    # Compute losses with proper shapes
                    if targets.shape != predictions.shape:
                        targets = targets.view(predictions.shape)
                    
                    # Calculate combined loss with focal weighting
                    mse_loss = self._compute_focal_loss(predictions, targets, self.mse_criterion, self.focal_gamma)
                    l1_loss = self._compute_focal_loss(predictions, targets, self.l1_criterion, self.focal_gamma)
                    huber_loss = self.huber_criterion(predictions, targets)
                    
                    # Weighted sum of losses
                    loss = (self.mse_weight * mse_loss + 
                            self.l1_weight * l1_loss + 
                            self.huber_weight * huber_loss)
                    
                    # Adjust loss weights gradually based on epoch
                    # As training progresses, we focus more on L1 and Huber
                    if is_training and epoch > self.epochs // 2:
                        progress = (epoch - self.epochs // 2) / (self.epochs - self.epochs // 2)
                        huber_boost = 0.1 * progress  # Gradually boost Huber loss weight
                        loss = loss + huber_boost * huber_loss
                    
                # Backward pass and optimization
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping to prevent explosions
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                    
                    self.optimizer.step()
                    
                    # Update learning rate with OneCycleLR scheduler
                    self.scheduler.step()
                    
                    # Update EMA model
                    if self.use_ema:
                        self._update_ema_model()
                
                # Update running loss
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': total_loss / (batch_idx + 1)
                })
            
            # Return average loss
            return total_loss / num_batches

    def _validate(self, epoch):
        """Validates the model after training epoch."""
        # Use EMA model for validation if available
        original_model = None
        if self.use_ema and not self.is_sklearn_model:
            original_model = self.model
            self.model = self.ema_model
        
        val_loss = self._run_epoch(epoch, is_training=False)
        self.val_losses.append(val_loss)
        
        # Log validation loss
        if self.logger:
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} - Validation Loss: {val_loss:.6f}")
        
        # Check for improvement
        is_best = val_loss < self.best_val_loss
        
        # Calculate relative improvement
        rel_improvement = 0
        if self.best_val_loss != float('inf'):
            rel_improvement = (self.best_val_loss - val_loss) / self.best_val_loss
            
        # Update best validation loss if improved
        if is_best:
            self.best_val_loss = val_loss
            
            # Save best model
            if self.model_save_path:
                if isinstance(self.model, torch.nn.Module):
                    torch.save(self.model.state_dict(), self.model_save_path)
                else:
                    # For scikit-learn models
                    import joblib
                    joblib.dump(self.model, self.model_save_path)
                    
                if self.logger:
                    self.logger.info(f"Model saved to {self.model_save_path}")
        
        # Early stopping logic
        if is_best or rel_improvement > self.min_improvement:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Restore original model
        if original_model is not None:
            self.model = original_model
        
        return val_loss, is_best

    def train(self):
        """Main training loop with sophisticated monitoring and techniques."""
        if self.logger:
            self.logger.info(f"Starting training for {self.epochs} epochs")
        
        # Track training start time
        start_time = time.time()
        
        # Initialize best model parameters
        best_model_params = None
        
        for epoch in range(self.epochs):
            self.epoch_count = epoch
            
            # Train one epoch
            train_loss = self._run_epoch(epoch, is_training=True)
            self.train_losses.append(train_loss)
            
            # Log training loss
            if self.logger:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs} - Training Loss: {train_loss:.6f}")
            
            # Validate
            val_loss, is_best = self._validate(epoch)
            
            # Save the best model parameters
            if is_best and isinstance(self.model, nn.Module):
                best_model_params = {name: param.clone().detach() 
                                    for name, param in self.model.state_dict().items()}
            
            # Early stopping check
            if self.early_stop and self.patience_counter >= self.patience:
                if self.logger:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load the best model parameters
        if best_model_params is not None and isinstance(self.model, nn.Module):
            self.model.load_state_dict(best_model_params)
            if self.logger:
                self.logger.info("Loaded best model parameters")
        
        # Training complete
        total_time = time.time() - start_time
        if self.logger:
            self.logger.info(f"Training complete in {total_time:.2f} seconds")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses