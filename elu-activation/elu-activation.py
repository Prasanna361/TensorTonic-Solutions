import numpy as np

def elu(x, alpha):
    x = np.array(x, dtype=float)
    result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return result.tolist()