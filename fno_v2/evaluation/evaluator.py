# battery_fno/evaluation/evaluator.py
import torch
import numpy as np
from tqdm import tqdm
from .metrics import compute_all_metrics

class Evaluator:
    """Handles model evaluation on a given dataset."""

    def __init__(self, model, data_loader, config, scaler=None):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.scaler = scaler # The preprocessor object or just the target scaler
        self.device = config.DEVICE

    def evaluate(self):
        """Performs evaluation and returns predictions and metrics."""
        self.model.eval()
        all_preds_scaled = []
        all_targets_scaled = []

        with torch.no_grad():
            for batch_data in tqdm(self.data_loader, desc="Evaluating", leave=False):
                 # Handle standard vs multi-input datasets
                if isinstance(batch_data[0], dict): # Multi-input
                     inputs = {key: tensor.to(self.device) for key, tensor in batch_data[0].items()}
                     targets = batch_data[1].to(self.device)
                else: # Standard input
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                predictions = self.model(inputs)

                all_preds_scaled.append(predictions.cpu().numpy())
                all_targets_scaled.append(targets.cpu().numpy())

        # Concatenate all batches
        all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
        all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)

        # Inverse transform using the scaler's method
        if self.scaler and hasattr(self.scaler, 'inverse_transform_target'):
            print("Inverse transforming predictions and targets...")
            # Ensure shapes are correct for inverse transform (usually needs 2D)
            preds_shape = all_preds_scaled.shape
            targets_shape = all_targets_scaled.shape

            # Make sure they are at least 1D before potentially reshaping to 2D
            all_preds_scaled_flat = all_preds_scaled.reshape(-1, preds_shape[-1] if len(preds_shape)>1 else 1)
            all_targets_scaled_flat = all_targets_scaled.reshape(-1, targets_shape[-1] if len(targets_shape)>1 else 1)


            all_preds = self.scaler.inverse_transform_target(all_preds_scaled_flat).reshape(preds_shape)
            all_targets = self.scaler.inverse_transform_target(all_targets_scaled_flat).reshape(targets_shape)
        else:
            print("Warning: Scaler or inverse_transform_target method not available. Reporting metrics on scaled values.")
            all_preds = all_preds_scaled
            all_targets = all_targets_scaled

        # Compute metrics on original scale
        metrics = compute_all_metrics(all_targets, all_preds)

        print("\n📊 Evaluation Metrics:")
        for name, value in metrics.items():
            print(f"  ▪️ {name:<5} = {value:.4f}")

        return all_targets, all_preds, metrics

    def save_results(self, metrics):
        """Saves evaluation metrics to a text file."""
        save_path = self.config.RESULTS_PATH
        print(f"Saving evaluation results to {save_path}")
        with open(save_path, "w") as f:
            f.write(f"Evaluation Results for Model: {self.config.MODEL_NAME}\n")
            f.write(f"Dataset Type: {self.config.DATASET_TYPE}\n")
            f.write("-" * 30 + "\n")
            for name, value in metrics.items():
                f.write(f"{name}: {value:.6f}\n")
            f.write("-" * 30 + "\n")