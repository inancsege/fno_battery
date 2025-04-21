# battery_fno/utils/helpers.py
import torch
import numpy as np
import random
import os

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)