from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class MatrixNormalizer(BaseTransformer):

    def __init__(self, norm: str='l2', copy: bool=True, name: Optional[str]=None):
        """
        Initialize the MatrixNormalizer.
        
        Parameters
        ----------
        norm : str, default='l2'
            The norm to use for normalization. Supported norms are 'l1', 'l2', and 'max'.
        copy : bool, default=True
            If False, try to avoid a copy and do inplace normalization.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.norm = norm
        self.copy = copy

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'MatrixNormalizer':
        """
        Fit the transformer to the input data.
        
        For matrix normalization, fitting doesn't require learning any parameters,
        so this method primarily validates the input and stores metadata.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        MatrixNormalizer
            Self instance for method chaining.
        """
        if not isinstance(data, (FeatureSet, np.ndarray)):
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.norm not in ('l1', 'l2', 'max'):
            raise ValueError(f"Unsupported norm: {self.norm}. Supported norms are 'l1', 'l2', and 'max'")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply normalization to the input data.
        
        Normalizes each sample (row) of the input matrix independently according to
        the specified norm.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to normalize. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            Normalized data as a FeatureSet with the same structure as input.
        """
        if not hasattr(self, 'n_features_in_'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        if self.copy:
            X = X.copy()
        if self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported norm: {self.norm}. Supported norms are 'l1', 'l2', and 'max'")
        norms = np.where(norms == 0, 1.0, norms)
        X_normalized = X / norms
        metadata['normalization_method'] = self.norm
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_normalized, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return FeatureSet(features=X_normalized, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply the inverse transformation if possible.
        
        Note: Matrix normalization is generally not invertible as it changes the scale
        of the data in a non-reversible way.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Normalized data to attempt inversion on.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            Returns the input data unchanged as normalization is not invertible.
        """
        return data

class UnitVectorScaler(BaseTransformer):

    def __init__(self, copy: bool=True, name: Optional[str]=None):
        """
        Initialize the UnitVectorScaler.
        
        Parameters
        ----------
        copy : bool, default=True
            If False, try to avoid a copy and do inplace scaling.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.copy = copy

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'UnitVectorScaler':
        """
        Fit the transformer to the input data.
        
        For unit vector scaling, fitting doesn't require learning any parameters,
        so this method primarily validates the input and stores metadata.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        UnitVectorScaler
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Scale input data to unit vectors.
        
        Each sample (row) in the input matrix is scaled to have L2 norm equal to 1.
        Samples with zero norm are left unchanged to avoid division by zero.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to scale. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Unit-scaled data in the same format as the input.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            is_feature_set = True
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            is_feature_set = False
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if not hasattr(self, 'n_features_in_'):
            raise RuntimeError("UnitVectorScaler has not been fitted yet. Call 'fit' first.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_in_} features')
        if self.copy or is_feature_set:
            X = X.copy()
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, 1, norms)
        X_scaled = X / norms_safe
        if is_feature_set:
            metadata['scaling_method'] = 'unit_vector'
            return FeatureSet(features=X_scaled, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif not self.copy:
            data[:] = X_scaled
            return data
        else:
            return X_scaled

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply the inverse transformation if possible.
        
        Note: Unit vector scaling is generally not invertible as it normalizes
        the magnitude of vectors to 1, losing information about their original scale.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Unit-scaled data to attempt inversion on.
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Returns the input data unchanged as unit vector scaling is not invertible.
        """
        return data