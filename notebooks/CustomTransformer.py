import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def custom_scaler(mean, average):
    scaler = StandardScaler().fit(np.zeros((1, 1)))  # required to fit data
    scaler.mean_ = np.array([mean])
    scaler.scale_ = np.array([average])
    return scaler


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, mean, scale):
        self.mean = np.array(mean)
        self.scale = np.array(scale)

    def fit(self, X, y=None):
        # Do nothing — this prevents leakage
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale


class TopCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=30, min_count=2):
        self.max_features = max_features
        self.min_count = min_count

    def fit(self, X, y=None):
        # X is a DataFrame with one column
        X = X.iloc[:, 0]
        counts = X.value_counts()
        self.top_categories_ = counts[counts >= self.min_count]\
            .index[:self.max_features]
        return self

    def transform(self, X):
        name = X.columns[0]
        X = X.iloc[:, 0]
        filtered = X.where(X.isin(self.top_categories_), other="other")
        dummies = pd.get_dummies(filtered, prefix=name)
        for category in self.top_categories_:
            col = f"{name}_{category}"
        if col not in dummies.columns:
            dummies[col] = 0

        # Ensure 'other' column is also always there
        other_col = f"{name}_other"
        if other_col not in dummies.columns:
            dummies[other_col] = 0
        return dummies


score_scaler = CustomScaler(6.39041096, 0.96877844)
