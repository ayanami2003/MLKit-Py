from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union, List

class CircularEncoder(BaseTransformer):

    def __init__(self, period: int, feature_indices: Optional[List[int]]=None, handle_unknown: str='ignore', name: Optional[str]=None):
        """
        Initialize the CircularEncoder.
        
        Parameters
        ----------
        period : int
            The period of the cyclical feature (e.g., 12 for months, 24 for hours).
            Must be a positive integer greater than 1.
        feature_indices : Optional[List[int]], optional
            Indices of features to encode. If None, all features will be considered.
        handle_unknown : str, default='ignore'
            How to handle unknown categories at transform time. Options are:
            - 'ignore': Ignore unknown categories (will result in NaN values)
            - 'error': Raise an error for unknown categories
        name : Optional[str], optional
            Name of the transformer instance.
            
        Raises
        ------
        ValueError
            If period is not a positive integer greater than 1.
        """
        super().__init__(name=name)
        if not isinstance(period, int) or period <= 1:
            raise ValueError('Period must be a positive integer greater than 1.')
        self.period = period
        self.feature_indices = feature_indices
        self.handle_unknown = handle_unknown
        self._sin_features_indices = []
        self._cos_features_indices = []
        self._n_features_out = 0
        self._fitted_feature_indices = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CircularEncoder':
        """
        Fit the encoder to the input data.
        
        This method identifies which features to encode and stores any necessary
        parameters for the transformation. For circular encoding, the main parameter
        is the period which is set at initialization.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to be encoded.
            If FeatureSet, features should be in the `features` attribute.
            If ndarray, should be a 2D array where each column is a feature.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        CircularEncoder
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        if self.feature_indices is None:
            self._fitted_feature_indices = list(range(n_features))
        else:
            for idx in self.feature_indices:
                if idx < 0 or idx >= n_features:
                    raise ValueError(f'Feature index {idx} is out of bounds for data with {n_features} features')
            self._fitted_feature_indices = self.feature_indices
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply circular encoding to the input data.
        
        Transforms selected categorical features into two continuous features
        using sine and cosine transformations:
        - sin_feature = sin(2 * pi * category / period)
        - cos_feature = cos(2 * pi * category / period)
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have the same number of features as
            the data used for fitting.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            Transformed data with circular encoding applied. The number of features
            will be increased by the number of encoded features (each becomes 2).
            
        Raises
        ------
        ValueError
            If handle_unknown is 'error' and unknown categories are encountered.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if self._fitted_feature_indices is None:
            features_to_encode = list(range(n_features))
        else:
            features_to_encode = self._fitted_feature_indices
        n_encoded_features = len(features_to_encode)
        n_output_features = n_features + n_encoded_features
        X_out = np.zeros((n_samples, n_output_features))
        out_feature_names = []
        out_feature_types = []
        output_col_idx = 0
        for feature_idx in range(n_features):
            if feature_idx in features_to_encode:
                feature_col = X[:, feature_idx]
                try:
                    numeric_feature = feature_col.astype(float)
                except ValueError:
                    unique_vals = np.unique(feature_col)
                    val_to_int = {val: i for (i, val) in enumerate(unique_vals)}
                    numeric_feature = np.array([val_to_int[val] for val in feature_col])
                if self.handle_unknown == 'error':
                    invalid_mask = (numeric_feature < 0) | (numeric_feature >= self.period)
                    if np.any(invalid_mask):
                        raise ValueError('Unknown categories encountered during transform')
                normalized = numeric_feature % self.period / self.period
                sin_values = np.sin(2 * np.pi * normalized)
                cos_values = np.cos(2 * np.pi * normalized)
                if self.handle_unknown == 'ignore':
                    invalid_mask = (numeric_feature < 0) | (numeric_feature >= self.period)
                    sin_values[invalid_mask] = np.nan
                    cos_values[invalid_mask] = np.nan
                X_out[:, output_col_idx] = sin_values
                X_out[:, output_col_idx + 1] = cos_values
                base_name = feature_names[feature_idx] if feature_names and feature_idx < len(feature_names) else f'feature_{feature_idx}'
                out_feature_names.extend([f'{base_name}_sin', f'{base_name}_cos'])
                out_feature_types.extend(['numeric', 'numeric'])
                output_col_idx += 2
            else:
                X_out[:, output_col_idx] = X[:, feature_idx]
                if feature_names and feature_idx < len(feature_names):
                    out_feature_names.append(feature_names[feature_idx])
                else:
                    out_feature_names.append(f'feature_{feature_idx}')
                if feature_types and feature_idx < len(feature_types):
                    out_feature_types.append(feature_types[feature_idx])
                else:
                    out_feature_types.append('numeric')
                output_col_idx += 1
        return FeatureSet(features=X_out, feature_names=out_feature_names, feature_types=out_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert encoded features back to original representation (approximate).
        
        Uses arc tangent function to recover the original categorical values from
        the sine and cosine representations. Note that this is an approximate
        reconstruction.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Encoded data to convert back to original representation.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            Data with circular encoding reversed (approximately).
            
        Raises
        ------
        ValueError
            If the data does not have the expected number of features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if self._fitted_feature_indices is None:
            if n_features % 2 != 0:
                raise ValueError('Invalid number of features for inverse transform when all features were encoded')
            n_original_features = n_features // 2
            features_that_were_encoded = list(range(n_original_features))
        else:
            n_encoded_features = len(self._fitted_feature_indices)
            n_original_features = n_features - n_encoded_features
            features_that_were_encoded = self._fitted_feature_indices
            expected_features = n_original_features + n_encoded_features * 2
            if n_features != expected_features:
                raise ValueError(f'Data has {n_features} features, expected {expected_features}')
        X_out = np.zeros((n_samples, n_original_features))
        out_feature_names = []
        out_feature_types = []
        input_col_idx = 0
        output_col_idx = 0
        for original_feature_idx in range(n_original_features):
            if original_feature_idx in features_that_were_encoded:
                sin_col_idx = input_col_idx
                cos_col_idx = input_col_idx + 1
                if cos_col_idx >= n_features:
                    raise ValueError('Not enough columns for inverse transform')
                sin_col = X[:, sin_col_idx]
                cos_col = X[:, cos_col_idx]
                angles = np.arctan2(sin_col, cos_col)
                angles = (angles + 2 * np.pi) % (2 * np.pi)
                original_values = angles / (2 * np.pi) * self.period
                original_values = np.round(original_values).astype(int)
                original_values = original_values % self.period
                X_out[:, output_col_idx] = original_values
                base_name = feature_names[sin_col_idx] if feature_names and sin_col_idx < len(feature_names) else f'feature_{original_feature_idx}'
                if base_name.endswith('_sin'):
                    base_name = base_name[:-4]
                out_feature_names.append(base_name)
                out_feature_types.append('categorical')
                input_col_idx += 2
            else:
                X_out[:, output_col_idx] = X[:, input_col_idx]
                if feature_names and input_col_idx < len(feature_names):
                    out_feature_names.append(feature_names[input_col_idx])
                else:
                    out_feature_names.append(f'feature_{original_feature_idx}')
                if feature_types and input_col_idx < len(feature_types):
                    out_feature_types.append(feature_types[input_col_idx])
                else:
                    out_feature_types.append('numeric')
                input_col_idx += 1
            output_col_idx += 1
        return FeatureSet(features=X_out, feature_names=out_feature_names, feature_types=out_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)