from typing import Optional
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class OrdinaryLeastSquaresRegressor(BaseModel):
    """
    Ordinary Least Squares (OLS) Regressor for linear regression tasks.

    This class implements a linear regression model using the Ordinary Least Squares method
    to estimate the relationship between input features and continuous target variables.
    It fits a linear model by minimizing the residual sum of squares between observed
    and predicted targets.

    Attributes:
        coefficients (np.ndarray): Estimated coefficients for the linear model.
        intercept (float): Intercept term of the linear model.
        is_fitted (bool): Flag indicating if the model has been fitted.

    Methods:
        fit: Fits the linear model using OLS estimation.
        predict: Makes predictions using the fitted linear model.
        score: Calculates the coefficient of determination (R^2) of the prediction.
    """

    def __init__(self, fit_intercept: bool=True, name: Optional[str]=None):
        """
        Initialize the OrdinaryLeastSquaresRegressor.

        Args:
            fit_intercept (bool): Whether to calculate the intercept for this model.
                                  If False, no intercept will be used in calculations.
            name (Optional[str]): Name of the model instance.
        """
        super().__init__(name=name)
        self.fit_intercept = fit_intercept
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: float = 0.0

    def fit(self, X: FeatureSet, y: np.ndarray, **kwargs) -> 'OrdinaryLeastSquaresRegressor':
        """
        Fit the linear regression model using Ordinary Least Squares.

        This method estimates the coefficients of the linear model by minimizing
        the residual sum of squares between the observed targets and the predictions
        of the linear approximation.

        Args:
            X (FeatureSet): Training data containing features.
            y (np.ndarray): Target values corresponding to the training data.
            **kwargs: Additional fitting parameters (not used in this implementation).

        Returns:
            OrdinaryLeastSquaresRegressor: Returns self for method chaining.

        Raises:
            ValueError: If dimensions of X and y do not match.
        """
        X_features = X.features
        if X_features.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X ({X_features.shape[0]}) does not match number of samples in y ({y.shape[0]})')
        (n_samples, n_features) = X_features.shape
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(n_samples), X_features])
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y
        else:
            XtX = X_features.T @ X_features
            Xty = X_features.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtX) @ Xty
        if self.fit_intercept:
            self.intercept = float(beta[0])
            self.coefficients = beta[1:]
        else:
            self.intercept = 0.0
            self.coefficients = beta
        self.is_fitted = True
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted linear model.

        This method uses the estimated coefficients and intercept to compute
        predictions for new input data.

        Args:
            X (FeatureSet): Input data for which predictions are to be made.
            **kwargs: Additional prediction parameters (not used in this implementation).

        Returns:
            np.ndarray: Predicted values for the input data.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("This OrdinaryLeastSquaresRegressor instance is not fitted yet. Call 'fit' before using this estimator.")
        X_features = X.features
        predictions = X_features @ self.coefficients + self.intercept
        return predictions

    def score(self, X: FeatureSet, y: np.ndarray, **kwargs) -> float:
        """
        Calculate the coefficient of determination (R^2) of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0.

        Args:
            X (FeatureSet): Test samples.
            y (np.ndarray): True values for X.
            **kwargs: Additional scoring parameters (not used in this implementation).

        Returns:
            float: R^2 score of the prediction.
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        r2_score = 1 - ss_res / ss_tot
        return float(r2_score)