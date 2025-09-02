import numpy as np
from typing import Optional, Union, Callable
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from src.models.regression.linear.ordinary_least_squares import OrdinaryLeastSquaresRegressor

class RANSACRegressor(BaseModel):
    """
    Robust regression model using Random Sample Consensus (RANSAC) algorithm.
    
    RANSAC is an iterative method to estimate parameters of a mathematical model
    from observed data that contains outliers. It's particularly useful when
    dealing with datasets where a significant portion of the data may be outliers.
    
    This implementation supports both linear and non-linear regression models
    through customizable base estimators.
    
    Attributes
    ----------
    base_estimator : object, optional
        Base estimator to fit on the inlier data. If None, uses LinearRegression.
    min_samples : int, optional
        Minimum number of samples chosen randomly from original data.
    residual_threshold : float, optional
        Maximum residual for a data sample to be classified as an inlier.
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_probability : float, optional
        Desired probability of obtaining a correct result.
    stop_score : float, optional
        Stop iteration if the score reaches this value.
    loss : str or callable, optional
        Loss function to compute residuals.
    random_state : int, optional
        Seed for random number generator.
        
    Methods
    -------
    fit(X, y) : Fit the model using RANSAC algorithm.
    predict(X) : Predict using the RANSAC model.
    score(X, y) : Return the coefficient of determination R^2 of the prediction.
    get_support_mask() : Get the mask of inliers found by RANSAC.
    """

    def __init__(self, base_estimator: Optional[object]=None, min_samples: Optional[int]=None, residual_threshold: Optional[float]=None, max_trials: int=100, stop_probability: float=0.99, stop_score: float=np.inf, loss: Union[str, Callable]='absolute_error', random_state: Optional[int]=None):
        """
        Initialize the RANSACRegressor.
        
        Parameters
        ----------
        base_estimator : object, optional
            Base estimator to fit on the inlier data. If None, uses LinearRegression.
        min_samples : int, optional
            Minimum number of samples chosen randomly from original data.
            Defaults to the number of features plus 1 for linear regression.
        residual_threshold : float, optional
            Maximum residual for a data sample to be classified as an inlier.
            By default, it is set to the MAD (median absolute deviation) of the
            target values times 1.4826.
        max_trials : int, optional
            Maximum number of iterations for random sample selection.
        stop_probability : float, optional
            Desired probability of obtaining a correct result.
        stop_score : float, optional
            Stop iteration if the score reaches this value.
        loss : str or callable, optional
            Loss function to compute residuals. Supported string values are
            'absolute_error' and 'squared_error'. A callable must take two
            arguments (y_true, y_pred) and return a 1D array of residuals.
        random_state : int, optional
            Seed for random number generator.
        """
        super().__init__(name='RANSACRegressor')
        self.base_estimator = base_estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.stop_probability = stop_probability
        self.stop_score = stop_score
        self.loss = loss
        self.random_state = random_state
        self.estimator_ = None
        self.support_ = None

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'RANSACRegressor':
        """
        Fit the RANSAC regression model.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        RANSACRegressor
            Fitted estimator.
            
        Raises
        ------
        ValueError
            If the input data dimensions don't match.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        elif y_array.ndim != 2:
            raise ValueError('y must be a 1D or 2D array')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f'Number of samples in X ({X_array.shape[0]}) does not match number of samples in y ({y_array.shape[0]})')
        (n_samples, n_features) = X_array.shape
        if self.base_estimator is None:
            self.base_estimator = OrdinaryLeastSquaresRegressor()
        if self.min_samples is None:
            self.min_samples = n_features + 1
        if self.min_samples < 1 or self.min_samples > n_samples:
            raise ValueError(f'min_samples must be >= 1 and <= n_samples ({n_samples}), got {self.min_samples}')
        if self.residual_threshold is None:
            median_y = np.median(y_array)
            self.residual_threshold = 1.4826 * np.median(np.abs(y_array - median_y))
        if self.residual_threshold <= 0:
            raise ValueError('residual_threshold must be positive')
        if isinstance(self.loss, str):
            if self.loss == 'absolute_error':
                loss_function = lambda y_true, y_pred: np.abs(y_true - y_pred)
            elif self.loss == 'squared_error':
                loss_function = lambda y_true, y_pred: (y_true - y_pred) ** 2
            else:
                raise ValueError(f"Supported loss functions are 'absolute_error' and 'squared_error', but got '{self.loss}'")
        elif callable(self.loss):
            loss_function = self.loss
        else:
            raise ValueError('loss must be a string or callable')
        rng = np.random.default_rng(self.random_state)
        best_score = -np.inf
        best_inlier_mask = None
        best_estimator = None
        if self.min_samples < n_samples:
            inlier_ratio = 0.5
            min_trials = np.ceil(np.log(1 - self.stop_probability) / np.log(1 - inlier_ratio ** self.min_samples))
            min_trials = max(1, min_trials)
        else:
            min_trials = 1
        for trial in range(self.max_trials):
            sample_indices = rng.choice(n_samples, size=self.min_samples, replace=False)
            X_subset = X_array[sample_indices]
            y_subset = y_array[sample_indices]
            temp_estimator = self._clone_estimator(self.base_estimator)
            try:
                if isinstance(temp_estimator, OrdinaryLeastSquaresRegressor):
                    temp_feature_set = FeatureSet(features=X_subset)
                    temp_estimator.fit(temp_feature_set, y_subset.ravel())
                else:
                    temp_estimator.fit(X_subset, y_subset.ravel())
                if isinstance(temp_estimator, OrdinaryLeastSquaresRegressor):
                    all_feature_set = FeatureSet(features=X_array)
                    y_pred = temp_estimator.predict(all_feature_set)
                else:
                    y_pred = temp_estimator.predict(X_array)
                residuals = loss_function(y_array.ravel(), y_pred)
                inlier_mask = residuals <= self.residual_threshold
                n_inliers = np.sum(inlier_mask)
                if n_inliers == 0:
                    continue
                X_inliers = X_array[inlier_mask]
                y_inliers = y_array[inlier_mask].ravel()
                refit_estimator = self._clone_estimator(self.base_estimator)
                if isinstance(refit_estimator, OrdinaryLeastSquaresRegressor):
                    inlier_feature_set = FeatureSet(features=X_inliers)
                    refit_estimator.fit(inlier_feature_set, y_inliers)
                else:
                    refit_estimator.fit(X_inliers, y_inliers)
                if isinstance(refit_estimator, OrdinaryLeastSquaresRegressor):
                    inlier_feature_set_all = FeatureSet(features=X_array)
                    score = refit_estimator.score(inlier_feature_set_all, y_array.ravel())
                else:
                    score = refit_estimator.score(X_array, y_array.ravel())
                if score > best_score:
                    best_score = score
                    best_inlier_mask = inlier_mask.copy()
                    best_estimator = refit_estimator
                    inlier_ratio = n_inliers / n_samples
                    if inlier_ratio < 1.0 and inlier_ratio > 0:
                        min_trials = np.ceil(np.log(1 - self.stop_probability) / np.log(1 - inlier_ratio ** self.min_samples))
                        min_trials = max(1, min_trials)
                if score >= self.stop_score:
                    break
                if trial >= min_trials and best_score > -np.inf:
                    break
            except Exception:
                continue
        if best_estimator is None:
            raise ValueError('RANSAC could not find a valid consensus set. Try decreasing the residual_threshold or increasing max_trials.')
        self.estimator_ = best_estimator
        self.support_ = best_inlier_mask
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict using the RANSAC regression model.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            Samples.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        if not self.is_fitted:
            raise ValueError("This RANSACRegressor instance is not fitted yet. Call 'fit' before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        if isinstance(self.estimator_, OrdinaryLeastSquaresRegressor):
            feature_set = FeatureSet(features=X_array)
            return self.estimator_.predict(feature_set)
        else:
            return self.estimator_.predict(X_array)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for X.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            R^2 of self.predict(X) wrt. y.
        """
        if not self.is_fitted:
            raise ValueError("This RANSACRegressor instance is not fitted yet. Call 'fit' before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f'Number of samples in X ({X_array.shape[0]}) does not match number of samples in y ({y_array.shape[0]})')
        y_pred = self.predict(X)
        y_true = y_array.ravel()
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - ss_res / ss_tot

    def get_support_mask(self) -> np.ndarray:
        """
        Get the mask of inliers found by RANSAC.
        
        Returns
        -------
        ndarray of shape (n_samples,)
            Boolean mask of inliers classified as ``True``.
        """
        if not self.is_fitted:
            raise ValueError("This RANSACRegressor instance is not fitted yet. Call 'fit' before using this estimator.")
        return self.support_.copy()

    def _clone_estimator(self, estimator):
        """
        Create a fresh copy of the estimator.
        
        Parameters
        ----------
        estimator : object
            Estimator to clone.
            
        Returns
        -------
        object
            Cloned estimator.
        """
        return estimator.__class__()