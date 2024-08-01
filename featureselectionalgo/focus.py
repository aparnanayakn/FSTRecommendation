import numpy as np
from itertools import combinations

def is_consistent(X, y, feature_subset):
    """Check if the feature subset is consistent with the class labels."""
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if np.all(X[i, feature_subset] == X[j, feature_subset]) and y[i] != y[j]:
                return False
    return True

def focus(X, y):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights for feature importance

    # Iterate through all subset sizes
    for i in range(1, n_features + 1):
        # Iterate through all subsets of the current size
        for feature_subset in combinations(range(n_features), i):
            if is_consistent(X, y, feature_subset):
                for feature in feature_subset:
                    weights[feature] += 1  # Increment weight for each feature in the subset

    # Normalize the weights by the maximum weight to get a relative importance score
    if np.max(weights) > 0:
        weights /= np.max(weights)

    return weights


# Initialize estimator
#estimator = RandomForestClassifier(random_state=42)

# Evaluate the selected features using cross-validation
#if selected_feature_subset is not None:
 #   X_selected = X[:, selected_feature_subset]
  #  scores = cross_val_score(estimator, X_selected, y, cv=5, scoring='accuracy')
   # mean_score = scores.mean()
    
    # Print the results
    #print("Selected features:", selected_feature_subset)
   # print("Best cross-validation score:", mean_score)
#else:
#    print("No consistent subset of features found.")
