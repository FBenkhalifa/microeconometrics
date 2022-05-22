import dataclasses
from typing import Optional, Union, List
import logging
import dataclasses

import pandas as pd
import numpy as np
import arviz as az

from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OrdinalEncoder

import pymc3 as pm

log = logging.getLogger(__name__)

pd.options.display.max_columns = 10
pd.options.display.width = 1000


@dataclasses.dataclass
class Prediction:
    """
    Stores the prediction results of a bayesian ordered choice model.
    The class contains the following attributes:
    - `eta`: The values of the latent predictor with shape (n_observations, n_draws)
    - `predictions`: The predicted classes with shape (n_observations, n_draws)
    """
    eta: np.ndarray
    cutoffs: np.ndarray


class BayesianOrderedRegression(BaseEstimator, MultiOutputMixin):

    def __init__(self):
        super(BayesianOrderedRegression, self).__init__()

        self.y_encoder_ = OrdinalEncoder()

        self.classes_: Optional[np.ndarray] = None
        self.columns_: Optional[List[str]] = None
        self.trace_: Optional[az.InferenceData] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],  # noqa(N806)
    ) -> "BayesianOrderedRegression":
        """
        Fit the model on the provided data. The values in y are encoded using an OrdinalEncoder.
        If the order of the categories is not given, the order of appearance is used.
        To fix the order of the variables, use an ordered pd.Categorical ore use integer or
         float values.
        :param X:
        :param y:
        :return:
        """

        # Store the variable names if a dataframe is passed
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.to_list()

        # Check input and register outcomes
        X, y = check_X_y(X, y)  # noqa(N806)
        self.classes_ = unique_labels(y)

        # We need column vectors for y
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Encode y
        y_ordered = self.y_encoder_.fit_transform(y)

        # Fit the model
        self._fit_pymc_model(
            x=X, y_ordered=y_ordered, num_outcome_classes=len(self.classes_)
        )

        return self

    def _fit_pymc_model(
        self, x: np.ndarray, y_ordered: np.ndarray, num_outcome_classes: int
    ) -> None:
        """
        Fits the model and stores the sampling trace internally.
        """

        # Check shapes
        if not y_ordered.shape == (x.shape[0], 1):
            raise ValueError("y.shape != (x.shape[0], 1)")

        with pm.Model() as model:

            # Priors for the regression coefficients
            beta = pm.Laplace("beta", mu=0, b=1/np.sqrt(2), shape=(1, x.shape[1]))
            # beta = np.ones(shape=(1, x.shape[1]))

            # Priors for the offset of each class
            class_cutoffs = pm.Normal(
                "class_cutoffs",
                mu=[-1, 1],
                sigma=10,
                shape=num_outcome_classes - 1,
                transform=pm.distributions.transforms.ordered,
            )

            # Set up the slope of the regression line
            slope = pm.math.sum(beta * x, axis=1, keepdims=True)

            # Define likelihood
            pm.OrderedLogistic(
                "y", eta=slope, cutpoints=class_cutoffs, observed=y_ordered
            )

            # Use the MAP as the starting point
            initvals = pm.find_MAP()
            # Scale the step using the initial values (i.e. the MAP from the Laplace priors)
            step = pm.NUTS(scaling=initvals)
            # Sample the model
            self.trace_ = pm.sample(
                draws=1_000,
                tune=1_000,
                step=step,
                initvals=initvals,
                return_inferencedata=True,
                progressbar=True,
            )

    def predict(self, X: Union[pd.DataFrame, np.ndarray], max_draws: int = None) -> pd.DataFrame:  # noqa(N806)
        """
        Predict the outcome of the provided data. This method will return draws from the posterior
        predictive. The return value is an array with shape (n_observations, n_draws).
        :param X: The data to predict.
        :param max_draws: Limit the number of draws from the posterior predictive.
        :return: The predicted outcomes with shape (n_observations, n_draws).
        """
        check_is_fitted(self, "classes_")
        X = check_array(X)  # noqa(N806)
        # get the parameter draws from the posterior distribution
        posterior_coefs = self.trace_.posterior.beta.squeeze().to_numpy().reshape((-1, 10))
        posterior_cutoffs = self.trace_.posterior.class_cutoffs.to_numpy().reshape((-1, 2))

        # Limit the number of draws if desired
        if max_draws:
            posterior_coefs = posterior_coefs[:max_draws]

        # Broadcast all data to the shape (n_obs, n_draws, n_features)
        broadcasted_coefs = np.tile(posterior_coefs[np.newaxis, ...], (X.shape[0], 1, 1))
        broadcasted_x = np.tile(X[:, np.newaxis, :], (1, posterior_coefs.shape[0], 1))

        # Take the sum of all coefficients
        eta_preds = np.sum(broadcasted_coefs * broadcasted_x, axis=-1)
        eta_preds.shape


        class_preds = np.argmax(eta_preds, axis=-1)

        return class_preds

    def summary(self, **kwargs) -> pd.DataFrame:
        """
        Returns a summary table of the model.
        :param kwargs: Arguments to pass to az.summary
        :return: A pd.DataFrame containing the model summary.
        """
        summary_table: pd.DataFrame = az.summary(self.trace_, kind="stats", **kwargs)
        summary_table.reset_index(inplace=True)
        summary_table.rename(
            columns={"index": "variable"},
            inplace=True,
        )
        if self.columns_ is not None:
            summary_table.iloc[:len(self.columns_), 0] = self.columns_

        return summary_table
