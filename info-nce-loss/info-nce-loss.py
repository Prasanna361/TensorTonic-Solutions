import numpy as np

def info_nce_loss(Z1, Z2, tau=0.1):
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)
    
    N = Z1.shape[0]
    
    # Similarity matrix
    S = (Z1 @ Z2.T) / tau
    
    # Numerical stability trick (subtract row max)
    S = S - np.max(S, axis=1, keepdims=True)
    
    # Exponentiate
    exp_S = np.exp(S)
    
    # Positive pairs are diagonal elements
    positives = np.diag(exp_S)
    
    # Denominator: sum over each row
    denominators = np.sum(exp_S, axis=1)
    
    # Compute loss
    loss = -np.mean(np.log(positives / denominators))
    
    return loss