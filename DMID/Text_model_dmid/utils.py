"""Utility functions."""

import torch
import numpy as np
import random
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List, Tuple, Dict, Set

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
