import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import load_iris

def multisurf(X, y):
    """
    MultiSURF feature selection algorithm.

    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Target vector.

    Returns:
    dict: Feature weights.
    list: Best features (with positive weights).
    """
    # Initialize weights
    w = np.zeros(X.shape[1])

    # Compute distance matrix
    d = euclidean_distances(X, X)

    # Compute thresholds and deviations
    T = np.mean(d, axis=1)
    tau = np.std(d, axis=1)
    D = tau / 2

    # MultiSURF Algorithm
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i == j:
                continue
            if d[i, j] < T[i] - D[i]:
                for a in range(X.shape[1]):
                    if X[i, a] != X[j, a]:
                        if y[i] == y[j]:
                            w[a] -= 1
                        else:
                            w[a] += 1
            elif d[i, j] > T[i] + D[i]:
                for a in range(X.shape[1]):
                    if X[i, a] == X[j, a]:
                        if y[i] == y[j]:
                            w[a] -= 1
                        else:
                            w[a] += 1

    return w
