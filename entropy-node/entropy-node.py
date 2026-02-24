import numpy as np

def entropy_node(y, eps=1e-12):
    y = np.asarray(y)

    # Empty node → entropy = 0.0
    if y.size == 0:
        return 0.0

    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    probs = np.clip(probs, eps, 1.0)

    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)