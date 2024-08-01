import numpy as np
from sklearn.datasets import load_iris

def set_cover_feature_selection(X, y):
    n_samples, n_features = X.shape
    all_features = set(range(n_features))
    selected_features = set()
    
    # Initialize weights for feature importance
    feature_importances = np.zeros(n_features)
    
    # Create a list of all pairs of examples with different classes
    P = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples) if y[i] != y[j]]
    print(P)
    while P:
        # Dictionary to count the number of pairs distinguished by each feature
        feature_counts = {f: 0 for f in all_features - selected_features}
        
        for (i, j) in P:
            for f in feature_counts:
                if X[i, f] != X[j, f]:
                    feature_counts[f] += 1
        
        # Select the feature that maximizes the number of distinguished pairs
        best_feature = max(feature_counts, key=feature_counts.get)
        selected_features.add(best_feature)
        
        # Update the importance of the selected feature
        feature_importances[best_feature] += feature_counts[best_feature]
        
        # Remove all pairs that are distinguished by the selected feature
        P = [(i, j) for (i, j) in P if X[i, best_feature] == X[j, best_feature]]
    
    # Normalize the feature importances
    if np.max(feature_importances) > 0:
        feature_importances /= np.max(feature_importances)
    
    return feature_importances


