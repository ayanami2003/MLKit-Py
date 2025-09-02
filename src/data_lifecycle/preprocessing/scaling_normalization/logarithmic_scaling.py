from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class LogarithmicScaler(BaseTransformer):

    def __init__(self, base: Optional[float]=None, handle_zeros: str='add_constant', add_constant: float=1.0, name: Optional[str]=None):
        """
        Initialize the LogarithmicScaler.
        
        Parameters
        ----------
        base : Optional[float], default=None
            The base of the logarithm. If None, natural logarithm is used.
        handle_zeros : str, default='add_constant'
            Strategy for handling zero values. Options are 'add_constant' or 'skip'.
        add_constant : float, default=1.0
            Constant to add to all values when handling zeros with 'add_constant'.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.base = base
        self.handle_zeros = handle_zeros
        self.add_constant = add_constant

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'LogarithmicScaler':
        """
        Fit the scaler to the data.
        
        For logarithmic scaling, fitting typically just validates the data
        and stores any necessary parameters.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            The input data to fit the scaler on.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        LogarithmicScaler
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
        if self.handle_zeros not in ['add_constant', 'skip']:
            raise ValueError("handle_zeros must be either 'add_constant' or 'skip'")
        if self.base is not None and self.base <= 0:
            raise ValueError('base must be positive or None for natural logarithm')
        elif self.base is not None and self.base == 1:
            raise ValueError('base cannot be equal to 1')
        if self.handle_zeros == 'add_constant' and self.add_constant <= 0:
            raise ValueError('add_constant must be positive when handle_zeros is add_constant')
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply logarithmic scaling to the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            The input data to transform.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        FeatureSet
            The transformed data with logarithmic scaling applied.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
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
        X_transform = X.copy()
        if self.handle_zeros == 'add_constant':
            X_transform = X_transform + self.add_constant
        elif self.handle_zeros == 'skip':
            X_transform = np.where(X_transform == 0, np.nan, X_transform)
        if self.base is None or self.base == np.e:
            X_result = np.log(X_transform)
        else:
            X_result = np.log(X_transform) / np.log(self.base)
        if self.handle_zeros == 'skip':
            X_result = np.where(np.isnan(X_result), 0, X_result)
        metadata['log_scaler_base'] = self.base
        metadata['log_scaler_handle_zeros'] = self.handle_zeros
        if self.handle_zeros == 'add_constant':
            metadata['log_scaler_add_constant'] = self.add_constant
        return FeatureSet(features=X_result, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply the inverse of the logarithmic scaling.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            The scaled data to inverse transform.
        **kwargs : dict
            Additional keyword arguments.
            
        Returns
        -------
        FeatureSet
            The data restored to its original scale.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
            return_feature_set = True
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
            return_feature_set = False
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        if self.base is None or self.base == np.e:
            X_exp = np.exp(X)
        else:
            X_exp = np.power(self.base, X)
        if self.handle_zeros == 'add_constant':
            X_original = X_exp - self.add_constant
        elif self.handle_zeros == 'skip':
            X_original = X_exp
        result = X_original
        if return_feature_set:
            return FeatureSet(features=result, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return result