from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class MinMaxScaler(BaseTransformer):
    """
    A transformer that scales features to a given range, typically [0, 1], using min-max normalization.
    
    This scaler transforms features by scaling each feature to a specified range, which is useful
    when features have varying scales and you want to normalize their impact on the model.
    The transformation is given by::
    
        X_scaled = (X - X_min) / (X_max - X_min)
    
    It can also scale back to the original representation using the inverse transform.
    
    Parameters
    ----------
    feature_range : tuple, optional (default=(0, 1))
        Desired range of transformed data.
    copy : bool, optional (default=True)
        Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).
    clip : bool, optional (default=False)
        Set to True to clip transformed values of held-out data to provided feature range.
    
    Attributes
    ----------
    min_ : np.ndarray of shape (n_features,)
        Per feature adjustment for minimum.
    scale_ : np.ndarray of shape (n_features,)
        Per feature relative scaling of the data.
    data_min_ : np.ndarray of shape (n_features,)
        Per feature minimum seen in the data.
    data_max_ : np.ndarray of shape (n_features,)
        Per feature maximum seen in the data.
    data_range_ : np.ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self, feature_range: tuple=(0, 1), copy: bool=True, clip: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
            
        Returns
        -------
        self : object
            Fitted scaler.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError('Input data must be either FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = np.where(self.data_max_ == self.data_min_, 1.0, self.data_max_ - self.data_min_)
        (min_val, max_val) = self.feature_range
        self.scale_ = np.where(self.data_range_ == 0, 0.0, (max_val - min_val) / self.data_range_)
        self.min_ = min_val - self.data_min_ * self.scale_
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Scale features according to feature_range.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            The data to be transformed.
            
        Returns
        -------
        FeatureSet
            The scaled features.
        """
        if not hasattr(self, 'scale_') or not hasattr(self, 'min_'):
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' first.")
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
            raise TypeError('Input data must be either FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_in_} features')
        if self.copy:
            X = X.copy()
        X_scaled = X * self.scale_ + self.min_
        if self.clip:
            (min_val, max_val) = self.feature_range
            X_scaled = np.clip(X_scaled, min_val, max_val)
        metadata['scaling_method'] = 'minmax'
        metadata['feature_range'] = self.feature_range
        return FeatureSet(features=X_scaled, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Undo the scaling of X according to feature_range.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            The data to be inverse transformed.
            
        Returns
        -------
        FeatureSet
            The original-scale features.
        """
        if not hasattr(self, 'scale_') or not hasattr(self, 'min_'):
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' first.")
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
            raise TypeError('Input data must be either FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_in_} features')
        if self.copy:
            X = X.copy()
        X_original = np.where(self.scale_ == 0, self.data_min_, (X - self.min_) / self.scale_)
        if 'scaling_method' in metadata:
            del metadata['scaling_method']
        if 'feature_range' in metadata:
            del metadata['feature_range']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)