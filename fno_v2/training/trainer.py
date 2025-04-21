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

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5) # Optional LR scheduler
        self.criterion = nn.MSELoss()

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

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
        num_batches = len(loader)

        with torch.set_grad_enabled(is_training):
            for batch_data in tqdm(loader, desc=desc, leave=False):
                # Handle standard vs multi-input datasets
                if isinstance(batch_data[0], dict): # Multi-input
                     inputs = {key: tensor.to(self.device) for key, tensor in batch_data[0].items()}
                     targets = batch_data[1].to(self.device)
                else: # Standard input
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                if is_training:
                    self.optimizer.zero_grad()

                # Forward pass
                predictions = self.model(inputs)

                # Ensure predictions and targets have compatible shapes for loss
                # Model output is usually (batch, output_dim=1)
                # Target from loader should be (batch, 1)
                loss = self.criterion(predictions, targets)

                if is_training:
                    loss.backward()
                    # Optional: Gradient clipping
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self):
        """Executes the full training process."""
        print(f"Starting training on {self.device}...")
        start_time = time.time()

        for epoch in range(self.config.EPOCHS):
            train_loss = self._run_epoch(epoch, is_training=True)
            val_loss = self._run_epoch(epoch, is_training=False)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Optional: Learning rate scheduling
            # self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.config.EPOCHS} - "
                  f"Train Loss: {train_loss:.6f} - "
                  f"Val Loss: {val_loss:.6f}")
                  # f"LR: {self.optimizer.param_groups[0]['lr']:.6f}") # If using scheduler

            # Save the best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                print(f"  ✨ New best model saved with Val Loss: {self.best_val_loss:.6f}")

        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Best model saved to: {self.config.BEST_MODEL_PATH}")

        return self.train_losses, self.val_losses