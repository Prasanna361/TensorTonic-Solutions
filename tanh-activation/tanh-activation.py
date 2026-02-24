import numpy as np

def tanh(x):
    x = np.array(x)  # ensure array for element-wise operations
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))