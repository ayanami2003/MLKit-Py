from typing import Optional, List, Union
import numpy as np
import pandas as pd
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class WoEEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', handle_missing: str='error', regularization: float=0.01, default_woe: float=0.0, smooth_factor: float=1.0, name: Optional[str]=None):
        """
        Initialize the WoE encoder.
        
        Parameters
        ----------
        handle_unknown : str, default='error'
            Strategy for handling unknown categories during transform
        handle_missing : str, default='error'
            Strategy for handling missing values during transform
        regularization : float, default=0.01
            Regularization parameter to prevent division by zero
        default_woe : float, default=0.0
            Default WoE value for unknown/missing categories
        smooth_factor : float, default=1.0
            Smoothing factor for probability estimates
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.regularization = regularization
        self.default_woe = default_woe
        self.smooth_factor = smooth_factor
        self._woe_mappings = {}
        self._categories = {}
        self._fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'WoEEncoder':
        """
        Fit the WoE encoder to the input data and target variable.
        
        Calculates the WoE values for each category in each categorical feature based on
        the relationship with the target variable.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode
        y : Optional[np.ndarray]
            Target variable (binary) used to calculate WoE values
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        WoEEncoder
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If target variable is not provided or is not binary
        """
        if y is None:
            raise ValueError("Target variable 'y' must be provided for WoE encoding")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        y = np.asarray(y)
        if len(np.unique(y)) != 2:
            raise ValueError('Target variable must be binary for WoE encoding')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if len(y) != n_samples:
            raise ValueError("Number of samples in 'data' and 'y' must match")
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        self._feature_names_in = feature_names
        self._feature_names_out = feature_names.copy()
        total_events = np.sum(y == 1)
        total_non_events = np.sum(y == 0)
        for i in range(n_features):
            feature_values = X[:, i]
            feature_name = feature_names[i]
            non_nan_mask = ~pd.isna(feature_values)
            categories = np.unique(feature_values[non_nan_mask])
            self._categories[feature_name] = set(categories)
            woe_mapping = {}
            for category in categories:
                mask = feature_values == category
                if np.sum(mask) == 0:
                    continue
                events = np.sum(y[mask] == 1)
                non_events = np.sum(y[mask] == 0)
                events_smooth = events + self.regularization * self.smooth_factor
                non_events_smooth = non_events + self.regularization * self.smooth_factor
                total_events_smooth = total_events + 2 * self.regularization * self.smooth_factor
                total_non_events_smooth = total_non_events + 2 * self.regularization * self.smooth_factor
                event_rate = events_smooth / total_events_smooth
                non_event_rate = non_events_smooth / total_non_events_smooth
                woe = np.log(non_event_rate / event_rate)
                woe_mapping[category] = woe
            if np.any(~non_nan_mask):
                missing_mask = ~non_nan_mask
                missing_events = np.sum(y[missing_mask] == 1)
                missing_non_events = np.sum(y[missing_mask] == 0)
                if missing_events + missing_non_events > 0:
                    missing_events_smooth = missing_events + self.regularization * self.smooth_factor
                    missing_non_events_smooth = missing_non_events + self.regularization * self.smooth_factor
                    missing_event_rate = missing_events_smooth / total_events_smooth
                    missing_non_event_rate = missing_non_events_smooth / total_non_events_smooth
                    missing_woe = np.log(missing_non_event_rate / missing_event_rate)
                    woe_mapping[np.nan] = missing_woe
                else:
                    woe_mapping[np.nan] = 0.0
            self._woe_mappings[feature_name] = woe_mapping
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using fitted WoE mappings.
        
        Applies the previously calculated WoE values to encode categorical features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with WoE-encoded features
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or unknown categories encountered with 'error' handling
        """
        if not self._fitted:
            raise ValueError("This WoEEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this transformer.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        if feature_names != self._feature_names_in:
            raise ValueError('Feature names in transform do not match those in fit')
        transformed_features = np.full(X.shape, np.nan, dtype=float)
        for i in range(n_features):
            feature_values = X[:, i]
            feature_name = feature_names[i]
            woe_mapping = self._woe_mappings[feature_name]
            for j in range(n_samples):
                value = feature_values[j]
                if pd.isna(value):
                    if self.handle_missing == 'error':
                        raise ValueError(f"Missing value encountered in feature '{feature_name}' with handle_missing='error'")
                    elif self.handle_missing == 'return_nan':
                        transformed_features[j, i] = np.nan
                    elif self.handle_missing == 'use_default':
                        transformed_features[j, i] = self.default_woe
                    else:
                        raise ValueError(f'Invalid handle_missing value: {self.handle_missing}')
                elif value in woe_mapping:
                    transformed_features[j, i] = woe_mapping[value]
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{value}' encountered in feature '{feature_name}' with handle_unknown='error'")
                elif self.handle_unknown == 'return_nan':
                    transformed_features[j, i] = np.nan
                elif self.handle_unknown == 'use_default':
                    transformed_features[j, i] = self.default_woe
                else:
                    raise ValueError(f'Invalid handle_unknown value: {self.handle_unknown}')
        return FeatureSet(features=transformed_features, feature_names=feature_names, feature_types=['numeric'] * n_features, metadata=data.metadata if isinstance(data, FeatureSet) else None)

    def fit_transform(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> FeatureSet:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode
        y : Optional[np.ndarray]
            Target variable (binary) used to calculate WoE values
        **kwargs : dict
            Additional parameters for fitting and transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with WoE-encoded features
        """
        return self.fit(data, y, **kwargs).transform(data, **kwargs)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reverse the WoE encoding back to original categorical values.
        
        Maps WoE-encoded values back to their original categorical representations.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            WoE-encoded data to convert back to categorical
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Data with categorical values restored
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported for current configuration
        """
        raise NotImplementedError('Inverse transformation is not implemented for WoEEncoder as WoE values are not necessarily unique to categories')

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after WoE encoding.
        
        Parameters
        ----------
        input_features : Optional[List[str]]
            Input feature names. If None, indices will be used.
            
        Returns
        -------
        List[str]
            Output feature names after encoding
        """
        if not self._fitted:
            raise ValueError("This WoEEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if input_features is None:
            return self._feature_names_out
        else:
            return input_features

    @property
    def woe_mappings_(self):
        """Public accessor for WoE mappings."""
        return self._woe_mappings