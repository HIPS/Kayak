import numpy as np

MAX_FLOAT_DIFF = 1e-9
MAX_GRAD_DIFF  = 1e-7
NUM_TRIALS     = 10

def close_float(A, B):
    return np.abs(A-B) < MAX_FLOAT_DIFF
