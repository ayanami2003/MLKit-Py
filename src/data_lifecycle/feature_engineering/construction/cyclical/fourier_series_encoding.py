import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class FourierSeriesEncoder(BaseTransformer):

    def __init__(self, periods: dict, feature_names: Optional[List[str]]=None, n_harmonics: int=1, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the FourierSeriesEncoder.
        
        Parameters
        ----------
        periods : dict
            Dictionary mapping feature names to their cycle periods
        feature_names : Optional[List[str]], optional
            Names of features to encode; if None, uses all features in periods
        n_harmonics : int, default=1
            Number of harmonics to include in the Fourier expansion
        drop_original : bool, default=True
            Whether to remove the original cyclical features from the output
        name : Optional[str], optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.periods = periods
        self.feature_names = feature_names
        self.n_harmonics = n_harmonics
        self.drop_original = drop_original

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'FourierSeriesEncoder':
        """
        Fit the encoder to the input data.
        
        This method validates that all specified features exist in the data
        and have corresponding periods defined.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing cyclical features to encode
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FourierSeriesEncoder
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If specified features are not found in the data or if periods
            are not defined for all features
        """
        if isinstance(data, FeatureSet):
            if data.feature_names is None:
                raise ValueError('FeatureSet must have feature_names defined')
            all_feature_names = data.feature_names
        else:
            if self.feature_names is None:
                raise ValueError('feature_names must be provided when using numpy arrays')
            all_feature_names = self.feature_names
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(all_feature_names)
            if missing_features:
                raise ValueError(f'Specified features not found in data: {missing_features}')
            features_to_encode = [name for name in self.feature_names if name in all_feature_names]
        else:
            features_to_encode = [name for name in self.periods.keys() if name in all_feature_names]
            if not features_to_encode:
                raise ValueError('No features to encode: none of the period keys match available features')
        missing_periods = set(features_to_encode) - set(self.periods.keys())
        if missing_periods:
            raise ValueError(f'Periods not defined for features: {missing_periods}')
        self._feature_names = features_to_encode
        self._periods = {k: self.periods[k] for k in features_to_encode}
        self.all_feature_names_ = all_feature_names
        self.feature_indices_ = {name: i for (i, name) in enumerate(all_feature_names)}
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply Fourier series encoding to the input data.
        
        For each cyclical feature, generates sin and cos components for
        each harmonic up to n_harmonics.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed data with Fourier-encoded cyclical features
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or if input data
            doesn't match the fitted schema
        """
        if not hasattr(self, '_feature_names'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        else:
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if X.shape[1] != len(self.all_feature_names_):
            raise ValueError(f'Input data has {X.shape[1]} features, but expected {len(self.all_feature_names_)}')
        fourier_features = []
        fourier_feature_names = []
        fourier_feature_types = []
        for feature_name in self._feature_names:
            feature_idx = self.feature_indices_[feature_name]
            period = self._periods[feature_name]
            feature_values = X[:, feature_idx]
            for harmonic in range(1, self.n_harmonics + 1):
                angle = 2 * np.pi * harmonic * feature_values / period
                sin_component = np.sin(angle)
                cos_component = np.cos(angle)
                fourier_features.append(sin_component)
                fourier_features.append(cos_component)
                fourier_feature_names.append(f'{feature_name}_sin_{harmonic}')
                fourier_feature_names.append(f'{feature_name}_cos_{harmonic}')
                fourier_feature_types.extend(['numeric', 'numeric'])
        if fourier_features:
            fourier_matrix = np.column_stack(fourier_features)
        else:
            fourier_matrix = np.empty((X.shape[0], 0))
        if self.drop_original:
            remaining_indices = [i for (i, name) in enumerate(self.all_feature_names_) if name not in self._feature_names]
            if remaining_indices:
                remaining_features = X[:, remaining_indices]
                remaining_feature_names = [self.all_feature_names_[i] for i in remaining_indices]
                remaining_feature_types = [feature_types[i] for i in remaining_indices] if feature_types else None
                X_transformed = np.hstack([remaining_features, fourier_matrix])
                final_feature_names = remaining_feature_names + fourier_feature_names
                final_feature_types = remaining_feature_types + fourier_feature_types if remaining_feature_types else None
            else:
                X_transformed = fourier_matrix
                final_feature_names = fourier_feature_names
                final_feature_types = fourier_feature_types
        else:
            X_transformed = np.hstack([X, fourier_matrix])
            final_feature_names = self.all_feature_names_ + fourier_feature_names
            final_feature_types = feature_types + fourier_feature_types if feature_types else None
        return FeatureSet(features=X_transformed, feature_names=final_feature_names, feature_types=final_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for Fourier series encoding
        as information is lost during the transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Original data without any changes
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not possible
        """
        raise NotImplementedError('Inverse transformation is not supported for Fourier series encoding')

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get the names of the output features after transformation.
        
        Parameters
        ----------
        input_features : Optional[List[str]], optional
            Names of input features; if None, uses fitted feature names
            
        Returns
        -------
        List[str]
            Names of output features after Fourier series encoding
        """
        if not hasattr(self, '_feature_names'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if input_features is not None:
            features_to_encode = [name for name in self._feature_names if name in input_features]
            all_feature_names = input_features
        else:
            features_to_encode = self._feature_names
            all_feature_names = self.all_feature_names_
        if self.drop_original:
            remaining_features = [name for name in all_feature_names if name not in self._feature_names]
        else:
            remaining_features = all_feature_names.copy()
        fourier_feature_names = []
        for feature_name in features_to_encode:
            for harmonic in range(1, self.n_harmonics + 1):
                fourier_feature_names.append(f'{feature_name}_sin_{harmonic}')
                fourier_feature_names.append(f'{feature_name}_cos_{harmonic}')
        return remaining_features + fourier_feature_names