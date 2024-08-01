import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

def transform_to_non_negative(X):
    df_transformed = X.copy()
    for col_index in range(df_transformed.shape[1]):
        min_value = df_transformed.iloc[:, col_index].min()
        if min_value < 0:
            df_transformed.iloc[:, col_index] += abs(min_value)
    return df_transformed


def chi_square_feature_importance(X, y, k='all'):
    # Ensure y is a Series
    X = transform_to_non_negative(X)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    chi2_selector = SelectKBest(chi2, k=k)
    X_new = chi2_selector.fit_transform(X, y)

    chi2_scores = chi2_selector.scores_
    
    selected_indices = chi2_selector.get_support(indices=True)
    
    selected_features = X.iloc[:, selected_indices]
    selected_scores = pd.Series(chi2_scores[selected_indices], index=selected_features.columns)

    # Create a Series with the Chi-Square scores of all features
    all_scores = pd.Series(chi2_scores, index=X.columns)
    
    return  all_scores

