import numpy as np
import pandas as pd
from mutual_information import su_calculation

def cfs(X, y):
  
    n_samples, n_features = X.shape
    merit_values = []
    if isinstance(X, pd.DataFrame):
        X = X.values  # Convert to NumPy array if it's a DataFrame

    for i in range(n_features):
        fi = X[:, i]
        rcf = su_calculation(fi, y)
        rff = 0
        
        for j in range(n_features):
            if i != j:
                fj = X[:, j]
                rff += su_calculation(fi, fj)
        
        rff *= 2
        merit = rcf / np.sqrt(1 + rff)  # The denominator is adjusted to avoid zero division and properly normalize the merit
        merit_values.append(merit)
    print(merit_values)
    return merit_values
