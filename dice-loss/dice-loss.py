import numpy as np

def dice_loss(p, y, eps=1e-8):
    # Convert to numpy arrays
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Flatten in case of 2D/3D inputs (segmentation masks)
    p = p.reshape(-1)
    y = y.reshape(-1)

    # Compute Dice coefficient
    intersection = np.sum(p * y)
    dice = (2 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)

    # Dice loss
    return float(1.0 - dice)