import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean"):
    
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("Shapes of y_true and y_score must match")

    # Validate label set {-1, +1}
    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("y_true must contain only -1 or +1")

    # Vectorized hinge loss
    loss = np.maximum(0, margin - y_true * y_score)

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")