import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    # Ensure batch shape
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        y = np.array([y])

    # Euclidean distance
    d = np.linalg.norm(a - b, axis=1)

    # Per-sample loss
    losses = y * (d ** 2) + (1 - y) * (np.maximum(0, margin - d) ** 2)

    # Reduction handling
    if reduction == "mean":
        return np.mean(losses)
    elif reduction == "sum":
        return np.sum(losses)
    elif reduction == "none":
        return losses
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")