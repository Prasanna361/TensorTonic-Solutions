import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # Cosine similarity
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)

    # Numerical safety (avoid division by zero)
    if norm1 == 0 or norm2 == 0:
        cos_sim = 0.0
    else:
        cos_sim = dot / (norm1 * norm2)

    # Loss based on label
    if label == 1:
        loss = 1.0 - cos_sim
    else:  # label == -1
        loss = max(0.0, cos_sim - margin)

    return float(loss)