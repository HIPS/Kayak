import numpy as np

def close_float(A, B):
    return np.abs(A-B) < 1e-9
