import numpy as np
from featureselectionalgo.mutual_information import su_calculation

def merit_calculation(X, y):
    """
    This function calculates the merit of X given class labels y, where
    merits = (k * rcf) / sqrt (k + k*(k-1)*rff)
    rcf = (1/k)*sum(su(fi, y)) for all fi in X
    rff = (1/(k*(k-1)))*sum(su(fi, fj)) for all fi and fj in X

    :param X:  {numpy array}, shape (n_samples, n_features) input data
    :param y:  {numpy array}, shape (n_samples) input class labels
    :return merits: {float}  merit of a feature subset X
    """

    n_samples, n_features = X.shape
    rff = 0
    rcf = 0
    for i in range(n_features):
        fi = X[:, i]
        rcf += su_calculation(fi, y)  # su is the symmetrical uncertainty of fi and y
        for j in range(n_features):
            if j > i:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
    rff *= 2 / (n_features * (n_features - 1))
    rcf /= n_features
    merits = rcf / np.sqrt(1 + rff)
    return merits

def cfs(X, y):
    """
    This function uses a correlation based heuristic to evaluate the worth of features which is called CFS

    :param X: {numpy array}, shape (n_samples, n_features) input data
    :param y: {numpy array}, shape (n_samples) input class labels
    :return F: {numpy array}, index of selected features
    :return importances: {numpy array}, importance weights of all features
    """

    n_samples, n_features = X.shape
    F = []
    M = []  # M stores the merit values
    importances = np.zeros(n_features)  # To store the importance of each feature

    while True:
        merit = -100000000000
        idx = -1
        for i in range(n_features):
            if i not in F:
                F.append(i)
                # calculate the merit of current selected features
                t = merit_calculation(X[:, F], y)
                if t > merit:
                    merit = t
                    idx = i
                F.pop()
        F.append(idx)
        M.append(merit)
        importances[idx] = merit

        if len(M) > 5:
            if M[-1] <= M[-2] and M[-2] <= M[-3] and M[-3] <= M[-4] and M[-4] <= M[-5]:
                break

    return importances

# Example usage:
# X = np.array(...)  # Your feature matrix
# y = np.array(...)  # Your labels
# feature_importances = cfs(X, y)
# print("Selected features:", selected_features)
# print("Feature importances:", feature_importances)

