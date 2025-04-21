# battery_fno/utils/plotting.py
import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid') # Use a visually appealing style

def plot_loss_curves(train_losses, val_losses, save_path):
    """Plots training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--', markersize=5)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.close() # Close the plot to free memory


def plot_predictions(y_true, y_pred, title, save_path, num_samples_to_plot=500):
    """Plots true vs. predicted values for a subset of samples."""
    plt.figure(figsize=(12, 6))

    # Limit the number of samples to plot for clarity
    indices = np.arange(len(y_true))
    if len(y_true) > num_samples_to_plot:
        # Plot a contiguous chunk or random samples
        plot_indices = np.linspace(0, len(y_true) - 1, num_samples_to_plot, dtype=int)
        # plot_indices = np.random.choice(indices, num_samples_to_plot, replace=False)
        # plot_indices.sort()
    else:
        plot_indices = indices

    plt.plot(indices[plot_indices], y_true.flatten()[plot_indices], label='True Values', marker='.', linestyle='-', markersize=5, alpha=0.8)
    plt.plot(indices[plot_indices], y_pred.flatten()[plot_indices], label='Predicted Values', marker='x', linestyle='--', markersize=5, alpha=0.8)

    plt.title(f'True vs. Predicted Values ({title}) - Sampled')
    plt.xlabel('Sample Index (Subset)')
    plt.ylabel('Target Value (Original Scale)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Prediction plot saved to {save_path}")
    plt.close()