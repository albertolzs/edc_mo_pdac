import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RemoveFeaturesWithZeros(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: float = 0.2, verbose: bool = False):
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        self.columns_ = X.columns[(X == 0).sum(axis=0) / len(X) < self.threshold]
        if self.verbose:
            print(f"{self.__class__.__name__} keeping {len(self.columns_)} features")
        return self

    def transform(self, X, y=None):
        transformed_X = X[self.columns_]
        return transformed_X


class RemoveFeaturesWithNaN(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: float = 0.2, verbose: bool = False):
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        self.columns_ = X.columns[(X.isna()).sum(axis=0) / len(X) < self.threshold]
        if self.verbose:
            print(f"{self.__class__.__name__} keeping {len(self.columns_)} features")
        return self

    def transform(self, X, y=None):
        transformed_X = X[self.columns_]
        return transformed_X


class FeatureSelectionNMF(BaseEstimator, TransformerMixin):

    def __init__(self, nmf: int, n_largest: int = 1, verbose: bool = False):
        self.nmf = nmf
        self.n_largest = n_largest
        self.verbose = verbose

    def fit(self, X, y=None):
        components = self.nmf.fit(X).components_
        components = pd.DataFrame(components, columns= X.columns).abs()
        columns = []
        for idx, row in components.iterrows():
            cols = row.drop(index=columns).nlargest(self.n_largest).index.to_list()
            columns.extend(cols)
        self.columns_ = X.columns.intersection(pd.Index(columns))
        if self.verbose:
            print(f"{self.__class__.__name__} keeping {len(self.columns_)} features")
        return self

    def transform(self, X, y=None):
        transformed_X = X[self.columns_]
        return transformed_X

class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, threshold: float = 0.2, verbose: bool = False):
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        corr_mat = np.abs(np.corrcoef(X.T.values))
        np.fill_diagonal(corr_mat, 0)
        upper = pd.DataFrame(corr_mat, index= X.columns, columns= X.columns)
        self.columns_to_drop_ = upper.columns[(upper > 0.85).any()]
        if self.verbose:
            print(f"{self.__class__.__name__} keeping {len(X.columns) - len(self.columns_to_drop_)} features")
        return self

    def transform(self, X, y=None):
        transformed_X = X.drop(columns= self.columns_to_drop_)
        return transformed_X


class RemoveFeaturesLowMAE(BaseEstimator, TransformerMixin):

    def __init__(self, percentage_to_keep: float, verbose: bool = False):
        self.percentage_to_keep = percentage_to_keep
        self.verbose = verbose

    def fit(self, X, y=None):
        var = np.abs(X - np.mean(X, axis=0))
        var = np.mean(var, axis= 0)
        columns = var.nlargest(n= int(X.shape[1] * self.percentage_to_keep)).index
        self.columns_ = X.columns.intersection(columns)
        if self.verbose:
            print(f"{self.__class__.__name__} keeping {len(self.columns_)} features")
        return self

    def transform(self, X, y=None):
        transformed_X = X[self.columns_]
        return transformed_X
