import numpy as np

def binary_focal_loss(predictions, targets, alpha=0.25, gamma=2.0, eps=1e-12):
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    # Numerical stability
    predictions = np.clip(predictions, eps, 1.0 - eps)

    # p_t: probability assigned to true class
    p_t = np.where(targets == 1, predictions, 1.0 - predictions)

    # Focal loss
    loss = -alpha * ((1.0 - p_t) ** gamma) * np.log(p_t)

    return float(np.mean(loss))