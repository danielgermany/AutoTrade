# random_signals.py
import numpy as np

def gen_signals(close):
    np.random.seed(42)  # For reproducibility
    return np.random.choice([1, 2], size=len(close))