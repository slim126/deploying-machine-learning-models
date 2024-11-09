from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = [s[0] if not pd.isna(s) else '' for s in X[feature]]
        return X
