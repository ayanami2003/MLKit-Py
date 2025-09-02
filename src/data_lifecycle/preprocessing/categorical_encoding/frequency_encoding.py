from general.base_classes.transformer_base import BaseTransformer
from typing import Optional, List, Union, Dict, Any
from general.structures.feature_set import FeatureSet
import numpy as np


# ...(code omitted)...


class CyclicalFrequencyEncoder(BaseTransformer):
    """
    Encode cyclical categorical features using frequency-based representations.
    
    This transformer encodes categorical features considering both their cyclical nature
    and frequency of occurrence. It supports three encoding modes:
    - 'sin_cos': Encodes using sine and cosine transformations of positions
    - 'one_hot': Encodes using one-hot representation with frequency weighting
    - 'frequency_only': Encodes using only the frequency of categories
    
    Attributes
    ----------
    encode_type : str
        Type of encoding to apply ('sin_cos', 'one_hot', or 'frequency_only')
    cycle_length : int
        Length of the cycle for cyclical encoding
    frequency_smoothing : float
        Smoothing factor to add to frequency counts to avoid zero values
    category_frequencies_ : dict
        Dictionary mapping categories to their smoothed frequencies
    feature_names_in_ : list
        Names of input features used during fitting
        
    Methods
    -------
    fit(data, **kwargs)
        Learn the frequencies of categories from the input data.
    transform(data, **kwargs)
        Apply cyclical frequency encoding to the input data.
    inverse_transform(data, **kwargs)
        Not supported for cyclical frequency encoding.
    """

    def __init__(self, encode_type: str='sin_cos', cycle_length: int=12, frequency_smoothing: float=1e-06, name: Optional[str]=None):
        super().__init__(name=name)
        self.encode_type = encode_type
        self.cycle_length = cycle_length
        self.frequency_smoothing = frequency_smoothing
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.encode_type not in ['sin_cos', 'one_hot', 'frequency_only']:
            raise ValueError("encode_type must be one of 'sin_cos', 'one_hot', 'frequency_only'")
        if not isinstance(self.cycle_length, int) or self.cycle_length <= 0:
            raise ValueError('cycle_length must be a positive integer')
        if not isinstance(self.frequency_smoothing, (int, float)) or self.frequency_smoothing < 0:
            raise ValueError('frequency_smoothing must be a non-negative number')

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CyclicalFrequencyEncoder':
        """
        Learn the frequency of each category in the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        CyclicalFrequencyEncoder
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.feature_names_in_ = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        else:
            X = data
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
        n_features = X.shape[1]
        self.category_frequencies_ = {}
        for i in range(n_features):
            feature_values = X[:, i]
            if feature_values.dtype.kind in ['f', 'i']:
                valid_mask = ~np.isnan(feature_values.astype(float))
            else:
                valid_mask = np.ones(len(feature_values), dtype=bool)
            valid_values = feature_values[valid_mask]
            (unique_vals, counts) = np.unique(valid_values, return_counts=True)
            freq_map = {}
            for (val, count) in zip(unique_vals, counts):
                str_val = str(val)
                freq_map[str_val] = count + self.frequency_smoothing
            if not np.all(valid_mask):
                nan_count = np.sum(~valid_mask)
                freq_map['nan'] = nan_count + self.frequency_smoothing
            self.category_frequencies_[self.feature_names_in_[i]] = freq_map
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply cyclical frequency encoding to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed data with cyclical frequency-encoded features.
        """
        if not hasattr(self, 'category_frequencies_'):
            raise ValueError("This CyclicalFrequencyEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(data, FeatureSet):
            X = data.features
            sample_ids = data.sample_ids
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        else:
            X = data
            sample_ids = [f'sample_{i}' for i in range(X.shape[0])]
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        transformed_features = []
        new_feature_names = []
        for i in range(n_features):
            feature_name = self.feature_names_in_[i]
            freq_map = self.category_frequencies_[feature_name]
            feature_values = X[:, i]
            str_values = []
            for val in feature_values:
                if isinstance(val, float) and np.isnan(val):
                    str_values.append('nan')
                else:
                    str_values.append(str(val))
            frequencies = np.array([freq_map.get(val, self.frequency_smoothing) for val in str_values])
            if self.encode_type == 'frequency_only':
                transformed_features.append(frequencies.reshape(-1, 1))
                new_feature_names.append(f'{feature_name}_frequency')
            elif self.encode_type == 'one_hot':
                unique_cats = list(freq_map.keys())
                for cat in unique_cats:
                    cat_mask = np.array([1.0 if val == cat else 0.0 for val in str_values])
                    weighted_cat = cat_mask * frequencies
                    transformed_features.append(weighted_cat.reshape(-1, 1))
                    new_feature_names.append(f'{feature_name}_{cat}')
            elif self.encode_type == 'sin_cos':
                unique_cats = list(freq_map.keys())
                cat_to_pos = {cat: idx % self.cycle_length for (idx, cat) in enumerate(unique_cats)}
                positions = np.array([cat_to_pos.get(val, 0) for val in str_values])
                sin_values = np.sin(2 * np.pi * positions / self.cycle_length) * frequencies
                cos_values = np.cos(2 * np.pi * positions / self.cycle_length) * frequencies
                transformed_features.append(sin_values.reshape(-1, 1))
                transformed_features.append(cos_values.reshape(-1, 1))
                new_feature_names.append(f'{feature_name}_sin')
                new_feature_names.append(f'{feature_name}_cos')
        if transformed_features:
            transformed_X = np.hstack(transformed_features)
        else:
            transformed_X = np.empty((n_samples, 0))
        return FeatureSet(features=transformed_X, feature_names=new_feature_names, sample_ids=sample_ids)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for cyclical frequency encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to inverse transform (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Original data (unchanged).
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not supported.
        """
        raise NotImplementedError('inverse_transform is not implemented for CyclicalFrequencyEncoder')