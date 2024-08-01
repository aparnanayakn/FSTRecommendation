import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def multisurf(X, y, subset_size=1000):
    """
    MultiSURF feature selection algorithm.

    Parameters:
    X (pandas.DataFrame): Feature matrix.
    y (numpy.ndarray or pandas.Series): Target vector.
    subset_size (int): Size of each subset.

    Returns:
    dict: Feature weights.
    list: Best features (with positive weights).
    """
    # Initialize weights
    w = np.zeros(X.shape[1])
    
    # Split the dataset into subsets
    num_samples = X.shape[0]
    num_subsets = (num_samples + subset_size - 1) // subset_size
    
    for subset_index in range(num_subsets):
        print(subset_index)
        start_index = subset_index * subset_size
        end_index = min((subset_index + 1) * subset_size, num_samples)
        
        # Compute distance matrix for the subset
        X_subset = X.iloc[start_index:end_index]
        d_subset = euclidean_distances(X_subset, X)
        
        # Compute thresholds and deviations for the subset
        T = np.mean(d_subset, axis=1)
        tau = np.std(d_subset, axis=1)
        D = tau / 2
        
        # MultiSURF Algorithm on the subset
        for i in range(start_index, end_index):
            for j in range(start_index, end_index):
                if i == j:
                    continue
                if d_subset[i - start_index, j] < T[i - start_index] - D[i - start_index]:
                    for a in range(X.shape[1]):
                        if X.iloc[i, a] != X.iloc[j, a]:
                            if y[i] == y[j]:
                                w[a] -= 1
                            else:
                                w[a] += 1
                elif d_subset[i - start_index, j] > T[i - start_index] + D[i - start_index]:
                    for a in range(X.shape[1]):
                        if X.iloc[i, a] == X.iloc[j, a]:
                            if y[i] == y[j]:
                                w[a] -= 1
                            else:
                                w[a] += 1
    
    return w

