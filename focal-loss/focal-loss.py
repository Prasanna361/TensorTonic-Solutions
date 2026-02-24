import numpy as np

def focal_loss(p: np.ndarray, y: np.ndarray, gamma: float = 2.0) -> float:
    # Convert inputs to numpy arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Numerical stability to avoid log(0)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    # Focal Loss computation
    loss = -((1 - p) ** gamma) * y * np.log(p) \
           - (p ** gamma) * (1 - y) * np.log(1 - p)

    # Return mean loss
    return float(np.mean(loss))