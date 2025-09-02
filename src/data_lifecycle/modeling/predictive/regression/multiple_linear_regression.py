from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import Optional
import numpy as np

class MultipleLinearRegressionModel(BaseModel):
    """
    A multiple linear regression model for predicting continuous target variables.
    
    This class implements a linear regression model that learns relationships between
    multiple features and a continuous target variable. It supports fitting the model
    to training data and making predictions on new data.
    
    Attributes:
        coefficients (Optional[list]): The learned coefficients for each feature after fitting.
        intercept (Optional[float]): The learned intercept term after fitting.
        is_fitted (bool): Whether the model has been fitted to training data.
        
    Methods:
        fit: Train the model on provided features and targets.
        predict: Make predictions on new feature data.
        score: Evaluate model performance using R-squared metric.
    """

    def __init__(self, fit_intercept: bool=True, normalize: bool=False, name: Optional[str]=None):
        """
        Initialize the Multiple Linear Regression model.
        
        Args:
            fit_intercept (bool): Whether to calculate the intercept for this model.
                If set to False, no intercept will be used in calculations.
            normalize (bool): Whether to normalize the features before regression.
                This parameter is ignored when fit_intercept is False.
            name (Optional[str]): Name identifier for the model instance.
        """
        super().__init__(name=name)
        self.fit_intercept = fit_intercept
        self.normalize = normalize and fit_intercept
        self.coefficients: Optional[list] = None
        self.intercept: Optional[float] = None
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: Optional[float] = None

    def fit(self, X: FeatureSet, y: DataBatch, **kwargs) -> 'MultipleLinearRegressionModel':
        """
        Fit the linear regression model to the training data.
        
        Args:
            X (FeatureSet): The input features for training. Must contain a 2D array
                of shape (n_samples, n_features).
            y (DataBatch): The target values for training. Must contain a 1D array
                of shape (n_samples,) with continuous target values.
            **kwargs: Additional fitting parameters (ignored in this implementation).
                
        Returns:
            MultipleLinearRegressionModel: The fitted model instance.
            
        Raises:
            ValueError: If the dimensions of X and y do not match.
            RuntimeError: If the model fitting fails due to numerical issues.
        """
        X_data = X.features
        y_data = y.data
        if X_data.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y_data.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X_data.shape[0] != y_data.shape[0]:
            raise ValueError('Number of samples in X and y must match')
        X_processed = X_data.copy().astype(np.float64)
        y_processed = y_data.copy().astype(np.float64)
        if self.normalize:
            self._X_mean = np.mean(X_processed, axis=0)
            self._X_std = np.std(X_processed, axis=0)
            self._X_std = np.where(self._X_std == 0, 1.0, self._X_std)
            X_processed = (X_processed - self._X_mean) / self._X_std
        if self.fit_intercept and self.normalize:
            self._y_mean = np.mean(y_processed)
            y_processed = y_processed - self._y_mean
        else:
            self._y_mean = 0.0
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(X_processed.shape[0]), X_processed])
        else:
            X_with_intercept = X_processed
            self.intercept = 0.0
        try:
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            Xty = np.dot(X_with_intercept.T, y_processed)
            theta = np.linalg.solve(XtX, Xty)
            if self.fit_intercept:
                self.intercept = float(theta[0])
                self.coefficients = theta[1:].tolist()
            else:
                self.coefficients = theta.tolist()
            if self.fit_intercept and self.normalize and (self._y_mean != 0):
                self.intercept += self._y_mean
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f'Numerical error in fitting model: {str(e)}')
        self.is_fitted = True
        return self

    def predict(self, X: FeatureSet, **kwargs) -> list:
        """
        Make predictions using the fitted linear regression model.
        
        Args:
            X (FeatureSet): The input features for prediction. Must contain a 2D array
                of shape (n_samples, n_features).
            **kwargs: Additional prediction parameters (ignored in this implementation).
                
        Returns:
            list: Predicted target values for the input samples.
            
        Raises:
            ValueError: If the model has not been fitted yet.
            ValueError: If the number of features in X does not match the fitted model.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if self.coefficients is None:
            raise ValueError('Model coefficients are not available.')
        X_data = X.features
        if X_data.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_data.shape[1] != len(self.coefficients):
            raise ValueError(f'Number of features in X ({X_data.shape[1]}) does not match the number of coefficients ({len(self.coefficients)})')
        X_processed = X_data.copy().astype(np.float64)
        if self.normalize and self._X_mean is not None and (self._X_std is not None):
            X_processed = (X_processed - self._X_mean) / self._X_std
        predictions = np.dot(X_processed, np.array(self.coefficients))
        if self.intercept is not None:
            predictions += self.intercept
        return predictions.tolist()

    def score(self, X: FeatureSet, y: DataBatch, **kwargs) -> float:
        """
        Calculate the R-squared score of the model on the given test data.
        
        Args:
            X (FeatureSet): The input features for evaluation.
            y (DataBatch): The true target values for evaluation.
            **kwargs: Additional scoring parameters (ignored in this implementation).
                
        Returns:
            float: The R-squared score, where best possible score is 1.0 and it can be
                negative (because the model can be arbitrarily worse).
                
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        y_pred = np.array(self.predict(X))
        y_true = y.data.astype(np.float64)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError('Mismatch in number of samples between predictions and true values')
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        r_squared = 1 - ss_res / ss_tot
        return float(r_squared)