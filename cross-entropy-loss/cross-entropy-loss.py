import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    N = y_true.shape[0]
    
    # Add small epsilon for numerical stability (avoid log(0))
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Select predicted probability of the correct class
    correct_class_probs = y_pred[np.arange(N), y_true]
    
    # Compute average negative log likelihood
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss