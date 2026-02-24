import numpy as np

def label_smoothing_loss(predictions, target, epsilon=0.1, eps=1e-12):
    p = np.asarray(predictions, dtype=float)

    # Numerical stability
    p = np.clip(p, eps, 1.0)

    K = p.shape[0]

    # Build smoothed target distribution
    q = np.full(K, epsilon / K, dtype=float)
    q[target] = (1.0 - epsilon) + (epsilon / K)

    # Cross-entropy loss
    loss = -np.sum(q * np.log(p))

    return float(loss)