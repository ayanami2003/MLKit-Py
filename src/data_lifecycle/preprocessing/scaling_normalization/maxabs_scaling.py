from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class MaxAbsScaler(BaseTransformer):

    def __init__(self, name: Optional[str]=None):
        """
        Initialize the MaxAbsScaler.
        
        Parameters
        ----------
        name : Optional[str], default=None
            Name of the transformer instance. If None, uses class name.
        """
        super().__init__(name=name)
        self.max_abs_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'MaxAbsScaler':
        """
        Compute the maximum absolute value for each feature to be used for later scaling.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to compute the scaling parameters from.
            If FeatureSet, uses the features attribute.
            If ndarray, expects shape (n_samples, n_features).
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        MaxAbsScaler
            Fitted scaler instance.
            
        Raises
        ------
        ValueError
            If data has inconsistent number of features compared to previous fit.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if self.n_features_in_ is not None and self.n_features_in_ != n_features:
            raise ValueError(f'Number of features ({n_features}) does not match previous fit ({self.n_features_in_})')
        self.max_abs_ = np.maximum(np.abs(np.max(X, axis=0)), np.abs(np.min(X, axis=0)))
        self.max_abs_ = np.where(self.max_abs_ == 0, 1.0, self.max_abs_)
        self.scale_ = self.max_abs_
        self.n_features_in_ = n_features
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Scale features according to the previously computed maximum absolute values.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
            If FeatureSet, transforms the features attribute and returns a new FeatureSet.
            If ndarray, expects shape (n_samples, n_features) and returns ndarray.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Scaled data in the same format as input.
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        ValueError
            If data has different number of features than during fitting.
        """
        if self.scale_ is None or self.n_features_in_ is None:
            raise RuntimeError("MaxAbsScaler has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_in_} features')
            X_scaled = X / self.scale_
            return FeatureSet(features=X_scaled, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Input data must be a 2D array')
            if data.shape[1] != self.n_features_in_:
                raise ValueError(f'Input data has {data.shape[1]} features, but scaler was fitted on {self.n_features_in_} features')
            return data / self.scale_
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Undo the scaling of features according to the previously computed maximum absolute values.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Scaled data to inverse transform.
            If FeatureSet, inverse transforms the features attribute and returns a new FeatureSet.
            If ndarray, expects shape (n_samples, n_features) and returns ndarray.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Original scale data in the same format as input.
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        ValueError
            If data has different number of features than during fitting.
        """
        if self.max_abs_ is None or self.n_features_in_ is None:
            raise RuntimeError("MaxAbsScaler has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_in_} features')
            X_original = X * self.max_abs_
            return FeatureSet(features=X_original, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Input data must be a 2D array')
            if data.shape[1] != self.n_features_in_:
                raise ValueError(f'Input data has {data.shape[1]} features, but scaler was fitted on {self.n_features_in_} features')
            return data * self.max_abs_
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')

    def fit_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Fit the scaler to the data and then transform it.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit and transform.
        **kwargs : dict
            Additional keyword arguments passed to fit and transform.
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Scaled data in the same format as input.
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)