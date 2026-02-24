import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)
    
    N, d = X.shape
    
    # Initialize parameters
    w = np.zeros(d)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass
        z = X @ w + b
        p = _sigmoid(z)
        
        # Gradients
        dw = (1 / N) * (X.T @ (p - y))
        db = (1 / N) * np.sum(p - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    return w, b
    