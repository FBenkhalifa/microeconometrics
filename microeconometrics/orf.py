from typing import Optional, Union


from sklearn.ensemble import ExtraTreesClassifier

from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import numpy as np
import pandas as pd


class OrderedExtraTreesClassifier(BaseEstimator, MultiOutputMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forest_kwargs = kwargs

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]  # noqa(N806)
    ) -> "OrderedExtraTreesClassifier":

        # Check input and register outcomes
        X, y = check_X_y(X, y)  # noqa(N806)
        self.classes_ = unique_labels(y)

        # Transform data so we can perform MCMC on it
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()  # noqa(N806)

        # We need column vectors for y
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Set up the target dummies
        self.encoder_ = OrdinalEncoder()
        encoded_y = self.encoder_.fit_transform(y)

        target_vars = {}
        for cutoff in range(len(self.encoder_.categories_[0])-1):
            target_vars[cutoff] = (encoded_y <= cutoff).astype(int)

        # For each dummy, fit a random forrest
        self.forests_ = {}
        for cutoff, target_var in target_vars.items():
            self.forests_[cutoff] = ExtraTreesClassifier(
                **self.forest_kwargs
            )
            self.forests_[cutoff].fit(X, target_var)




data = pd.read_csv("data/data_group4.csv", nrows=1_000)
y = (data["POINTS_A"] - data["POINTS_H"])
x = data.copy().select_dtypes("float")
x = x.apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis=0)

x.pop("SCORE_A")
x.pop("SCORE_H")
x.pop("POINTS_A")
x.pop("POINTS_H")
x = x.to_numpy()[:, :5]
X=x


self = OrderedExtraTreesClassifier()
