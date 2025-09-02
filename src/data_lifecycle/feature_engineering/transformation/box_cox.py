import numpy as np
from scipy import stats
from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class BoxCoxTransformer(BaseTransformer):

    def __init__(self, optimize_lambda: bool=True, fixed_lambda: Optional[float]=None, min_value: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the Box-Cox transformer.
        
        Parameters
        ----------
        optimize_lambda : bool, default=True
            Whether to estimate the optimal lambda parameter during fitting.
            If False, uses the fixed_lambda value.
        fixed_lambda : float, optional
            Fixed lambda value to use when optimize_lambda is False.
            Required if optimize_lambda is False.
        min_value : float, optional
            Minimum value to shift data to ensure positivity.
            If None, automatically calculated during fitting.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.optimize_lambda = optimize_lambda
        self.fixed_lambda = fixed_lambda
        self.min_value = min_value
        self.lambda_: Optional[Union[float, np.ndarray]] = None
        self.shift_: Optional[Union[float, np.ndarray]] = None
        self.is_fitted_ = False

    def _extract_array(self, data):
        """Extract numpy array from various input types."""
        if isinstance(data, DataBatch):
            return np.asarray(data.data)
        elif isinstance(data, FeatureSet):
            return np.asarray(data.features)
        else:
            return np.asarray(data)

    def fit(self, data, **kwargs):
        """
        Fit the Box-Cox transformer to the data.
        
        Estimates the optimal lambda parameter if optimize_lambda is True,
        and determines the minimum shift value if needed.
        
        Parameters
        ----------
        data : np.ndarray or FeatureSet or DataBatch
            Input data to fit the transformer on. Should contain only positive values.
            If FeatureSet, uses the features attribute.
            If DataBatch, uses the data attribute.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        BoxCoxTransformer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If data contains non-positive values or if fixed_lambda is not provided
            when optimize_lambda is False.
        """
        X = self._extract_array(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if not np.isfinite(X).all():
            raise ValueError('Input data contains non-finite values')
        if self.min_value is not None:
            self.shift_ = self.min_value
        else:
            min_vals = np.min(X, axis=0)
            if np.all(min_vals > 0):
                self.shift_ = 0.0
            else:
                required_shift = 1.0 - min_vals
                self.shift_ = np.maximum(required_shift, 0)
            if X.shape[1] == 1:
                self.shift_ = float(self.shift_)
        X_shifted = X + (self.shift_ if np.isscalar(self.shift_) else self.shift_)
        if np.any(X_shifted <= 0):
            raise ValueError('Data contains non-positive values even after shifting')
        if X.shape[0] == 1:
            if self.optimize_lambda:
                self.lambda_ = 1.0 if X.shape[1] == 1 else np.ones(X.shape[1])
            else:
                if self.fixed_lambda is None:
                    raise ValueError('fixed_lambda must be provided when optimize_lambda is False')
                self.lambda_ = self.fixed_lambda if X.shape[1] == 1 else np.full(X.shape[1], self.fixed_lambda)
        else:
            constant_columns = np.all(X_shifted == X_shifted[0], axis=0)
            if np.any(constant_columns):
                if X.shape[1] == 1:
                    self.lambda_ = 1.0
                else:
                    self.lambda_ = np.ones(X.shape[1])
                    non_constant_indices = ~constant_columns
                    if np.any(non_constant_indices) and self.optimize_lambda:
                        for i in np.where(non_constant_indices)[0]:
                            try:
                                (_, lambda_opt) = stats.boxcox(X_shifted[:, i])
                                self.lambda_[i] = lambda_opt
                            except Exception:
                                self.lambda_[i] = 1.0
                    elif not self.optimize_lambda:
                        if self.fixed_lambda is None:
                            raise ValueError('fixed_lambda must be provided when optimize_lambda is False')
                        self.lambda_ = np.full(X.shape[1], self.fixed_lambda)
            elif self.optimize_lambda:
                if X.shape[1] == 1:
                    try:
                        (_, lambda_opt) = stats.boxcox(X_shifted[:, 0])
                        self.lambda_ = lambda_opt
                    except Exception:
                        self.lambda_ = 1.0
                else:
                    self.lambda_ = np.empty(X.shape[1])
                    for i in range(X.shape[1]):
                        try:
                            (_, lambda_opt) = stats.boxcox(X_shifted[:, i])
                            self.lambda_[i] = lambda_opt
                        except Exception:
                            self.lambda_[i] = 1.0
            else:
                if self.fixed_lambda is None:
                    raise ValueError('fixed_lambda must be provided when optimize_lambda is False')
                self.lambda_ = self.fixed_lambda if X.shape[1] == 1 else np.full(X.shape[1], self.fixed_lambda)
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[np.ndarray, FeatureSet, DataBatch], **kwargs) -> Union[np.ndarray, FeatureSet, DataBatch]:
        """
        Apply the Box-Cox transformation to the data.
        
        Parameters
        ----------
        data : np.ndarray or FeatureSet or DataBatch
            Input data to transform. Must have the same number of features as the fitted data.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        np.ndarray or FeatureSet or DataBatch
            Transformed data in the same format as input.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or if data contains invalid values.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        input_is_data_batch = isinstance(data, DataBatch)
        input_is_feature_set = isinstance(data, FeatureSet)
        if input_is_data_batch:
            X = np.asarray(data.data)
            original_data_batch = data
        elif input_is_feature_set:
            X = np.asarray(data.features)
            feature_names = getattr(data, 'feature_names', None)
            feature_types = getattr(data, 'feature_types', None)
        else:
            X = np.asarray(data)
        original_shape_1d = X.ndim == 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if not np.isfinite(X).all():
            raise ValueError('Input data contains non-finite values')
        X_shifted = X + (self.shift_ if np.isscalar(self.shift_) else self.shift_)
        if np.any(X_shifted <= 0):
            raise ValueError('Data contains non-positive values even after shifting')
        if X.shape[1] == 1:
            lambda_val = self.lambda_ if np.isscalar(self.lambda_) else self.lambda_[0]
            if lambda_val == 0:
                X_transformed = np.log(X_shifted)
            else:
                X_transformed = (np.power(X_shifted, lambda_val) - 1) / lambda_val
        else:
            X_transformed = np.zeros_like(X_shifted)
            for i in range(X.shape[1]):
                lambda_val = self.lambda_[i] if hasattr(self.lambda_, '__len__') else self.lambda_
                if lambda_val == 0:
                    X_transformed[:, i] = np.log(X_shifted[:, i])
                else:
                    X_transformed[:, i] = (np.power(X_shifted[:, i], lambda_val) - 1) / lambda_val
        if original_shape_1d and X_transformed.ndim > 1:
            X_transformed = X_transformed.flatten()
        if input_is_data_batch:
            return DataBatch(data=X_transformed, labels=original_data_batch.labels, metadata=original_data_batch.metadata.copy() if original_data_batch.metadata else None, sample_ids=original_data_batch.sample_ids, feature_names=original_data_batch.feature_names, batch_id=original_data_batch.batch_id)
        elif input_is_feature_set:
            if X_transformed.ndim == 1:
                X_transformed = X_transformed.reshape(-1, 1)
            return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types)
        else:
            return X_transformed

    def inverse_transform(self, data: Union[np.ndarray, FeatureSet, DataBatch], **kwargs) -> Union[np.ndarray, FeatureSet, DataBatch]:
        """
        Apply the inverse Box-Cox transformation.
        
        Parameters
        ----------
        data : np.ndarray or FeatureSet or DataBatch
            Transformed data to invert.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        np.ndarray or FeatureSet or DataBatch
            Original scale data in the same format as input.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        input_is_data_batch = isinstance(data, DataBatch)
        input_is_feature_set = isinstance(data, FeatureSet)
        if input_is_data_batch:
            X = np.asarray(data.data)
            original_data_batch = data
        elif input_is_feature_set:
            X = np.asarray(data.features)
            feature_names = getattr(data, 'feature_names', None)
            feature_types = getattr(data, 'feature_types', None)
        else:
            X = np.asarray(data)
        original_shape_1d = X.ndim == 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if X.shape[1] == 1:
            lambda_val = self.lambda_ if np.isscalar(self.lambda_) else self.lambda_[0]
            if lambda_val == 0:
                X_inverted = np.exp(X)
            else:
                X_inverted = np.power(lambda_val * X + 1, 1 / lambda_val)
        else:
            X_inverted = np.zeros_like(X)
            for i in range(X.shape[1]):
                lambda_val = self.lambda_[i] if hasattr(self.lambda_, '__len__') else self.lambda_
                if lambda_val == 0:
                    X_inverted[:, i] = np.exp(X[:, i])
                else:
                    X_inverted[:, i] = np.power(lambda_val * X[:, i] + 1, 1 / lambda_val)
        if X.shape[1] == 1:
            shift_val = self.shift_ if np.isscalar(self.shift_) else self.shift_[0]
            X_inverted = X_inverted - shift_val
        else:
            X_inverted = X_inverted - self.shift_
        if original_shape_1d and X_inverted.ndim > 1:
            X_inverted = X_inverted.flatten()
        if input_is_data_batch:
            return DataBatch(data=X_inverted, labels=original_data_batch.labels, metadata=original_data_batch.metadata.copy() if original_data_batch.metadata else None, sample_ids=original_data_batch.sample_ids, feature_names=original_data_batch.feature_names, batch_id=original_data_batch.batch_id)
        elif input_is_feature_set:
            if X_inverted.ndim == 1:
                X_inverted = X_inverted.reshape(-1, 1)
            return FeatureSet(features=X_inverted, feature_names=feature_names, feature_types=feature_types)
        else:
            return X_inverted