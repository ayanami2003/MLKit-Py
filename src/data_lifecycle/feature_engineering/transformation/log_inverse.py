from typing import Union, Optional
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class LogInverseTransformer(BaseTransformer):

    def __init__(self, base: str='e', feature_indices: Optional[Union[list, str]]='all'):
        """
        Initialize the LogInverseTransformer.

        Args:
            base (str): The base of the logarithm to invert ('e', '10', or '2'). Defaults to 'e'.
            feature_indices (Optional[Union[list, str]]): Specific features to transform.
                Can be a list of indices/names or 'all' for all features. Defaults to 'all'.
                
        Raises:
            ValueError: If base is not one of 'e', '10', or '2'.
        """
        super().__init__(name='LogInverseTransformer')
        if base not in ['e', '10', '2']:
            raise ValueError("Base must be one of 'e', '10', or '2'")
        self.base = base
        self.feature_indices = feature_indices
        self._fitted_features = None

    def fit(self, data: Union[np.ndarray, FeatureSet], **kwargs) -> 'LogInverseTransformer':
        """
        Fit the transformer to the input data.

        This method identifies which features will be transformed and validates
        that they contain only positive values (required for log transformations).
        
        Args:
            data (Union[np.ndarray, FeatureSet]): Input data to fit on.
            **kwargs: Additional fitting parameters.
            
        Returns:
            LogInverseTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._input_is_feature_set = True
            self._feature_names = data.feature_names
        else:
            X = data
            self._input_is_feature_set = False
            self._feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        self.n_features_in_ = X.shape[1]
        if self.feature_indices is None or self.feature_indices == 'all':
            self._fitted_features = list(range(X.shape[1]))
        elif isinstance(self.feature_indices, str):
            if self._feature_names and self.feature_indices in self._feature_names:
                self._fitted_features = [self._feature_names.index(self.feature_indices)]
            else:
                raise ValueError(f"Feature '{self.feature_indices}' not found in feature names")
        elif all((isinstance(idx, int) for idx in self.feature_indices)):
            if any((idx < 0 or idx >= X.shape[1] for idx in self.feature_indices)):
                raise IndexError('Feature index out of range')
            self._fitted_features = self.feature_indices
        elif all((isinstance(idx, str) for idx in self.feature_indices)):
            if not self._feature_names:
                raise ValueError('Cannot use feature names when input is numpy array')
            try:
                self._fitted_features = [self._feature_names.index(name) for name in self.feature_indices]
            except ValueError as e:
                raise ValueError(f'Feature name not found: {str(e)}')
        else:
            raise ValueError('feature_indices must contain either all integers or all strings')
        for idx in self._fitted_features:
            feature_col = X[:, idx]
            if not np.all(feature_col > 0):
                raise ValueError(f'All values in feature at index {idx} must be positive for log transformation')
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[np.ndarray, FeatureSet], **kwargs) -> Union[np.ndarray, FeatureSet]:
        """
        Apply the inverse logarithmic transformation to the input data.

        Applies the exponential function (inverse of log) to specified features.
        The exact function depends on the base:
        - For base 'e': exp(x)
        - For base '10': 10^x
        - For base '2': 2^x

        Args:
            data (Union[np.ndarray, FeatureSet]): Input data to transform.
            **kwargs: Additional transformation parameters.
            
        Returns:
            Union[np.ndarray, FeatureSet]: Transformed data with the same type as input.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            input_is_feature_set = True
            feature_names = data.feature_names
            feature_types = data.feature_types
        else:
            X = data.copy() if isinstance(data, np.ndarray) else np.array(data)
            input_is_feature_set = False
            feature_names = getattr(self, '_feature_names', None)
            feature_types = getattr(self, '_feature_types', None)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Number of features in transform data ({X.shape[1]}) does not match fitted data ({self.n_features_in_})')
        X_transformed = X.copy()
        for idx in self._fitted_features:
            if self.base == 'e':
                X_transformed[:, idx] = np.exp(X[:, idx])
            elif self.base == '10':
                X_transformed[:, idx] = np.power(10, X[:, idx])
            elif self.base == '2':
                X_transformed[:, idx] = np.power(2, X[:, idx])
        if input_is_feature_set:
            result = FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types)
        else:
            result = X_transformed if data.ndim == 2 else X_transformed.ravel()
        return result

    def inverse_transform(self, data: Union[np.ndarray, FeatureSet], **kwargs) -> Union[np.ndarray, FeatureSet]:
        """
        Apply the forward logarithmic transformation (inverse of this transformer).

        This method applies the actual logarithmic transformation, effectively
        undoing the inverse transform operation.

        Args:
            data (Union[np.ndarray, FeatureSet]): Data to apply forward log transform to.
            **kwargs: Additional transformation parameters.
            
        Returns:
            Union[np.ndarray, FeatureSet]: Log-transformed data with the same type as input.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            input_is_feature_set = True
            feature_names = data.feature_names
            feature_types = data.feature_types
        else:
            X = data.copy() if isinstance(data, np.ndarray) else np.array(data)
            input_is_feature_set = False
            feature_names = getattr(self, '_feature_names', None)
            feature_types = getattr(self, '_feature_types', None)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be a 1D or 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Number of features in inverse transform data ({X.shape[1]}) does not match fitted data ({self.n_features_in_})')
        X_transformed = X.copy()
        for idx in self._fitted_features:
            if self.base == 'e':
                X_transformed[:, idx] = np.log(X[:, idx])
            elif self.base == '10':
                X_transformed[:, idx] = np.log10(X[:, idx])
            elif self.base == '2':
                X_transformed[:, idx] = np.log2(X[:, idx])
        if input_is_feature_set:
            result = FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types)
        else:
            result = X_transformed if data.ndim == 2 else X_transformed.ravel()
        return result