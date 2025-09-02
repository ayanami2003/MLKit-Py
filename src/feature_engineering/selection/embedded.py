from typing import Optional, Union, Dict, Any
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ElasticNetFeatureSelector(BaseTransformer):

    def __init__(self, alpha: float=1.0, l1_ratio: float=0.5, fit_intercept: bool=True, max_iter: int=1000, tol: float=0.0001, warm_start: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self._coef = None
        self._selected_features = None
        self._feature_names = None
        self._is_fitted = False
        self._support_mask = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'ElasticNetFeatureSelector':
        """
        Fit the Elastic Net feature selector to the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Training data containing features. If FeatureSet, uses feature names for tracking.
        y : np.ndarray
            Target values.
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        ElasticNetFeatureSelector
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If data and target dimensions don't match.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        else:
            X = data
            self._feature_names = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if not isinstance(y, np.ndarray):
            raise TypeError('Target values must be a numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if y.ndim != 1:
            raise ValueError('Target values must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in data and target must match')
        n_features = X.shape[1]
        if self._feature_names is None:
            self._feature_names = [f'x{i}' for i in range(n_features)]
        elif len(self._feature_names) != n_features:
            raise ValueError('Number of feature names must match number of features')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._coef = self._fit_elastic_net(X, y)
        self._selected_features = np.abs(self._coef) > 1e-10
        self._support_mask = self._selected_features.copy()
        self._is_fitted = True
        return self

    def _fit_elastic_net(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit Elastic Net using coordinate descent algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        np.ndarray
            Coefficients of shape (n_features,)
        """
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1
        X = X / X_std
        y_mean = np.mean(y)
        y_centered = y - y_mean
        (n_samples, n_features) = X.shape
        if self.warm_start and self._coef is not None and (len(self._coef) == n_features):
            coef = self._coef.copy().astype(np.float64)
            coef = coef * X_std
        else:
            coef = np.zeros(n_features, dtype=np.float64)
        y_mean_offset = np.mean(y) if self.fit_intercept else 0
        y_centered = y - y_mean_offset
        X_norms = np.sum(X ** 2, axis=0)
        X_norms = np.maximum(X_norms, 1e-12)
        for iteration in range(self.max_iter):
            coef_old = coef.copy()
            for j in range(n_features):
                residual = y_centered - X @ coef + coef[j] * X[:, j]
                corr = np.dot(X[:, j], residual)
                if self.l1_ratio == 1:
                    if corr > self.alpha:
                        coef[j] = (corr - self.alpha) / X_norms[j]
                    elif corr < -self.alpha:
                        coef[j] = (corr + self.alpha) / X_norms[j]
                    else:
                        coef[j] = 0.0
                else:
                    rho = corr + X_norms[j] * coef[j]
                    threshold = self.alpha * self.l1_ratio
                    l2_reg = self.alpha * (1 - self.l1_ratio)
                    denominator = X_norms[j] + l2_reg
                    if rho > threshold:
                        coef[j] = (rho - threshold) / denominator
                    elif rho < -threshold:
                        coef[j] = (rho + threshold) / denominator
                    else:
                        coef[j] = 0.0
            coef_diff = coef - coef_old
            max_diff = np.max(np.abs(coef_diff))
            if max_diff < self.tol:
                break
        coef = coef / X_std
        return coef

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply feature selection to transform the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have same number of features as training data.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with only selected features.
            
        Raises
        ------
        ValueError
            If transformer hasn't been fitted yet or if data dimensions don't match.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != len(self._selected_features):
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted with {len(self._selected_features)} features.')
        X_selected = X[:, self._selected_features]
        if is_feature_set:
            if feature_names is not None:
                selected_feature_names = [name for (i, name) in enumerate(feature_names) if self._selected_features[i]]
            else:
                selected_feature_names = [name for (i, name) in enumerate(self._feature_names) if self._selected_features[i]]
            feature_types = ['numeric'] * X_selected.shape[1]
            return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return X_selected

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Reverse the transformation (not supported for feature selection).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to inverse transform (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Original data with zeros padded for removed features.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection.
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection.')

    def get_support(self, indices: bool=False) -> Union[np.ndarray, list]:
        """
        Get a mask or indices of selected features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return indices of selected features instead of boolean mask.
            
        Returns
        -------
        Union[np.ndarray, list]
            Boolean mask or list of indices indicating selected features.
            
        Raises
        ------
        ValueError
            If transformer hasn't been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if indices:
            return np.where(self._selected_features)[0]
        else:
            return self._selected_features.copy()

    def get_feature_names_out(self) -> Optional[list]:
        """
        Get names of selected features.
        
        Returns
        -------
        Optional[list]
            Names of selected features, or None if not available.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if self._feature_names is not None:
            return [name for (i, name) in enumerate(self._feature_names) if self._selected_features[i]]
        else:
            return None

    @property
    def coef_(self) -> np.ndarray:
        """
        Coefficients of the features in the decision function.
        
        Returns
        -------
        np.ndarray
            Coefficient vector.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        return self._coef.copy()