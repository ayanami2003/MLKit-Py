from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ParetoScaler(BaseTransformer):

    def __init__(self, name: Optional[str]=None):
        """
        Initialize the ParetoScaler.
        
        Parameters
        ----------
        name : Optional[str]
            Name identifier for the transformer instance
        """
        super().__init__(name=name)
        self.feature_means_: Optional[np.ndarray] = None
        self.feature_sqrt_stds_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ParetoScaler':
        """
        Compute the mean and square root of standard deviation for each feature.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the scaler on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional fitting parameters (ignored)
            
        Returns
        -------
        ParetoScaler
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data is empty or contains non-numeric values
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.size == 0:
            raise ValueError('Input data is empty')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError('Input data contains non-numeric values')
        self.feature_means_ = np.mean(X, axis=0)
        feature_vars = np.var(X, axis=0)
        feature_vars = np.where(feature_vars == 0, 1.0, feature_vars)
        self.feature_sqrt_stds_ = np.sqrt(feature_vars)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply Pareto scaling to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, transforms the features attribute.
        **kwargs : dict
            Additional transformation parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed data with Pareto scaling applied
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted or if data dimensions don't match
        """
        if self.feature_means_ is None or self.feature_sqrt_stds_ is None or self.n_features_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
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
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_} features')
        X_scaled = (X - self.feature_means_) / self.feature_sqrt_stds_
        metadata['scaling_method'] = 'pareto'
        metadata['scaling_params'] = {'means': self.feature_means_, 'sqrt_stds': self.feature_sqrt_stds_}
        return FeatureSet(features=X_scaled, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reverse the Pareto scaling transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Scaled data to inverse transform
        **kwargs : dict
            Additional inverse transformation parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Original scale data
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted or if data dimensions don't match
        """
        if self.feature_means_ is None or self.feature_sqrt_stds_ is None or self.n_features_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
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
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_} features')
        X_original = X * self.feature_sqrt_stds_ + self.feature_means_
        if 'scaling_method' in metadata:
            del metadata['scaling_method']
        if 'scaling_params' in metadata:
            del metadata['scaling_params']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)