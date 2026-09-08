"""
Seed management utilities for reproducibility.
"""

import random
import numpy as np
import torch
from src.config import PROJECT_SEED


def set_seed(seed: int = PROJECT_SEED):
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make deterministic where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
