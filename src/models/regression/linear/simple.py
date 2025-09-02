from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class LinearRegressionModel(BaseModel):

    def __init__(self, fit_intercept: bool=True, name: Optional[str]=None):
        """
        Initialize the LinearRegressionModel.
        
        Parameters
        ----------
        fit_intercept : bool, default=True
            Whether to calculate the intercept term
        name : Optional[str], default=None
            Name identifier for the model
        """
        super().__init__(name)
        self.fit_intercept = fit_intercept
        self.coefficients: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> 'LinearRegressionModel':
        """
        Fit the linear regression model using ordinary least squares.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training features of shape (n_samples, n_features)
        y : Union[np.ndarray, list]
            Target values of shape (n_samples,)
        **kwargs : dict
            Additional fitting parameters (ignored in this implementation)
            
        Returns
        -------
        LinearRegressionModel
            Fitted model instance
            
        Raises
        ------
        ValueError
            If dimensions of X and y do not match
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError(f'X must be a 2D array, got {X_array.ndim}D')
        if y_array.ndim != 1:
            if y_array.ndim == 2 and y_array.shape[1] == 1:
                y_array = y_array.ravel()
            else:
                raise ValueError(f'y must be a 1D array, got shape {y_array.shape}')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f'Number of samples in X ({X_array.shape[0]}) does not match number of samples in y ({y_array.shape[0]})')
        if self.fit_intercept:
            X_processed = np.column_stack([np.ones(X_array.shape[0]), X_array])
        else:
            X_processed = X_array
        try:
            coefficients = np.linalg.lstsq(X_processed, y_array, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            raise ValueError(f'Could not compute linear regression coefficients: {str(e)}')
        if self.fit_intercept:
            self.intercept_ = float(coefficients[0])
            self.coefficients = coefficients[1:]
        else:
            self.intercept_ = None
            self.coefficients = coefficients
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Make predictions using the fitted linear model.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Features to predict on of shape (n_samples, n_features)
        **kwargs : dict
            Additional prediction parameters (ignored in this implementation)
            
        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,)
            
        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before making predictions.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError(f'X must be a 2D array, got {X_array.ndim}D')
        if self.coefficients is None:
            raise ValueError('Model coefficients are not available.')
        if X_array.shape[1] != len(self.coefficients):
            raise ValueError(f'Number of features in X ({X_array.shape[1]}) does not match the number of coefficients ({len(self.coefficients)})')
        predictions = X_array @ self.coefficients
        if self.intercept_ is not None:
            predictions += self.intercept_
        return predictions

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Calculate the R² score (coefficient of determination) of the prediction.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Test features of shape (n_samples, n_features)
        y : Union[np.ndarray, list]
            True target values of shape (n_samples,)
        **kwargs : dict
            Additional scoring parameters (ignored in this implementation)
            
        Returns
        -------
        float
            R² score where best possible score is 1.0
            
        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before scoring.")
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape')
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        r2_score = 1 - ss_res / ss_tot
        return float(r2_score)