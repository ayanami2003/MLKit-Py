from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ZScoreNormalizer(BaseTransformer):

    def __init__(self, with_mean: bool=True, with_std: bool=True, handle_outliers: bool=False, outlier_threshold: float=3.0, name: Optional[str]=None):
        """
        Initialize the ZScoreNormalizer.
        
        Args:
            with_mean (bool): If True, center the data before scaling. Default is True.
            with_std (bool): If True, scale the data to unit variance. Default is True.
            handle_outliers (bool): If True, apply outlier detection and treatment during transformation. Default is False.
            outlier_threshold (float): Threshold (in standard deviations) for identifying outliers. Default is 3.0.
            name (Optional[str]): Name of the transformer instance. If None, uses class name.
        """
        super().__init__(name=name)
        self.with_mean = with_mean
        self.with_std = with_std
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ZScoreNormalizer':
        """
        Compute the mean and standard deviation to be used for later scaling.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit the transformer on.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            ZScoreNormalizer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Perform standardization by centering and scaling.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Transformed data with standardized features.
        """
        if self.with_mean and (not hasattr(self, 'mean_')):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.with_std and (not hasattr(self, 'scale_')):
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
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.with_mean and hasattr(self, 'mean_') and (X.shape[1] != len(self.mean_)):
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {len(self.mean_)} features')
        if self.with_std and hasattr(self, 'scale_') and (X.shape[1] != len(self.scale_)):
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {len(self.scale_)} features')
        X_transformed = X.copy()
        if self.with_mean and hasattr(self, 'mean_'):
            X_transformed = X_transformed - self.mean_
        if self.with_std and hasattr(self, 'scale_'):
            X_transformed = X_transformed / self.scale_
        if self.handle_outliers:
            lower_bound = -self.outlier_threshold
            upper_bound = self.outlier_threshold
            X_transformed = np.clip(X_transformed, lower_bound, upper_bound)
            metadata['outlier_handling'] = {'method': 'clipping', 'threshold': self.outlier_threshold}
        metadata['zscore_scaling'] = {'with_mean': self.with_mean, 'with_std': self.with_std}
        if self.with_mean and hasattr(self, 'mean_'):
            metadata['zscore_scaling']['mean'] = self.mean_
        if self.with_std and hasattr(self, 'scale_'):
            metadata['zscore_scaling']['scale'] = self.scale_
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Scale back the data to the original representation.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Transformed data to inverse transform.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Data scaled back to original representation.
        """
        if self.with_mean and (not hasattr(self, 'mean_')):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.with_std and (not hasattr(self, 'scale_')):
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
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.with_mean and hasattr(self, 'mean_') and (X.shape[1] != len(self.mean_)):
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {len(self.mean_)} features')
        if self.with_std and hasattr(self, 'scale_') and (X.shape[1] != len(self.scale_)):
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {len(self.scale_)} features')
        X_original = X.copy()
        if self.with_std and hasattr(self, 'scale_'):
            X_original = X_original * self.scale_
        if self.with_mean and hasattr(self, 'mean_'):
            X_original = X_original + self.mean_
        if 'zscore_scaling' in metadata:
            del metadata['zscore_scaling']
        if 'outlier_handling' in metadata:
            del metadata['outlier_handling']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class ZScoreOutlierDetector(BaseTransformer):
    """
    A transformer that applies z-score normalization with outlier detection and treatment.
    
    This transformer extends standard z-score normalization by identifying and treating outliers
    based on a specified threshold in standard deviations. Outliers can be capped, removed, or
    flagged depending on the configured treatment strategy.
    
    Attributes:
        threshold (float): Threshold (in standard deviations) for identifying outliers.
        treat_method (str): Method for treating outliers ('cap', 'remove', 'flag').
        name (str): Name of the transformer instance.
        
    Methods:
        fit: Compute the mean and standard deviation for later scaling and outlier detection.
        transform: Perform standardization and outlier treatment.
        inverse_transform: Scale back the data (note: may not perfectly reconstruct if outliers were removed).
    """

    def __init__(self, threshold: float=3.0, treat_method: str='cap', name: Optional[str]=None):
        """
        Initialize the ZScoreOutlierDetector.
        
        Args:
            threshold (float): Threshold (in standard deviations) for identifying outliers. Default is 3.0.
            treat_method (str): Method for treating outliers. Options are 'cap' (default), 'remove', or 'flag'.
            name (Optional[str]): Name of the transformer instance. If None, uses class name.
            
        Raises:
            ValueError: If treat_method is not one of 'cap', 'remove', or 'flag'.
        """
        super().__init__(name=name)
        self.threshold = threshold
        if treat_method not in ['cap', 'remove', 'flag']:
            raise ValueError("treat_method must be one of 'cap', 'remove', or 'flag'")
        self.treat_method = treat_method
        self.mean_ = None
        self.scale_ = None
        self._outlier_flags = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ZScoreOutlierDetector':
        """
        Compute the mean and standard deviation to be used for later scaling and outlier detection.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit the transformer on.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            ZScoreOutlierDetector: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Perform standardization and outlier treatment.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Transformed data with standardized features and outliers treated.
        """
        if not hasattr(self, 'mean_') or not hasattr(self, 'scale_'):
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
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != len(self.mean_) or X.shape[1] != len(self.scale_):
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {len(self.mean_)} features')
        X_transformed = (X - self.mean_) / self.scale_
        lower_bound = -self.threshold
        upper_bound = self.threshold
        outlier_mask = (X_transformed < lower_bound) | (X_transformed > upper_bound)
        self._outlier_flags = outlier_mask
        if self.treat_method == 'cap':
            X_transformed = np.clip(X_transformed, lower_bound, upper_bound)
            metadata['outlier_handling'] = {'method': 'cap', 'threshold': self.threshold}
        elif self.treat_method == 'remove':
            rows_with_outliers = np.any(outlier_mask, axis=1)
            X_transformed = X_transformed[~rows_with_outliers]
            if sample_ids is not None:
                sample_ids = [sid for (i, sid) in enumerate(sample_ids) if not rows_with_outliers[i]]
            metadata['outlier_handling'] = {'method': 'remove', 'threshold': self.threshold, 'rows_removed': np.sum(rows_with_outliers)}
        elif self.treat_method == 'flag':
            metadata['outlier_handling'] = {'method': 'flag', 'threshold': self.threshold}
            metadata['outlier_flags'] = outlier_mask
        metadata['zscore_scaling'] = {'mean': self.mean_, 'scale': self.scale_}
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Scale back the data to the original representation.
        
        Note: If outliers were removed, perfect reconstruction is not possible.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Transformed data to inverse transform.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Data scaled back to original representation.
        """
        if not hasattr(self, 'mean_') or not hasattr(self, 'scale_'):
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
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != len(self.mean_) or X.shape[1] != len(self.scale_):
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {len(self.mean_)} features')
        X_original = X * self.scale_ + self.mean_
        if 'zscore_scaling' in metadata:
            del metadata['zscore_scaling']
        if 'outlier_handling' in metadata:
            del metadata['outlier_handling']
        if 'outlier_flags' in metadata:
            del metadata['outlier_flags']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)