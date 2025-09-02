from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import numpy as np

# ...(code omitted)...

class FeatureSet:
    """
    Standardized structure for feature matrices and associated metadata.
    
    Encapsulates feature data along with names, types, and quality metrics
    for use across feature engineering and modeling components.
    
    Attributes
    ----------
    features : np.ndarray
        2D array of shape (n_samples, n_features) containing feature values
    feature_names : Optional[List[str]]
        Names of the features in column order
    feature_types : Optional[List[str]]
        Data types of features (numeric, categorical, datetime, etc.)
    sample_ids : Optional[List[str]]
        Identifiers for samples/rows
    metadata : Optional[Dict[str, Any]]
        Additional feature set information
    quality_scores : Optional[Dict[str, float]]
        Quality metrics for features (variance, missingness, etc.)
    """
    features: np.ndarray
    feature_names: Optional[List[str]] = None
    feature_types: Optional[List[str]] = None
    sample_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    quality_scores: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.quality_scores is None:
            self.quality_scores = {}
        if not isinstance(self.features, np.ndarray):
            raise TypeError('Features must be a numpy array')
        if self.features.ndim != 2:
            raise ValueError('Features must be a 2D array')
        (n_samples, n_features) = self.features.shape
        if self.feature_names is not None:
            if not isinstance(self.feature_names, list):
                raise TypeError('feature_names must be a list')
            if len(self.feature_names) != n_features:
                raise ValueError(f'Number of feature names ({len(self.feature_names)}) must match number of feature columns ({n_features})')
        if self.sample_ids is not None:
            if not isinstance(self.sample_ids, list):
                raise TypeError('sample_ids must be a list')
            if len(self.sample_ids) != n_samples:
                raise ValueError(f'Number of sample IDs ({len(self.sample_ids)}) must match number of samples ({n_samples})')
        if self.feature_types is not None:
            if not isinstance(self.feature_types, list):
                raise TypeError('feature_types must be a list')
            if len(self.feature_types) != n_features:
                raise ValueError(f'Number of feature types ({len(self.feature_types)}) must match number of feature columns ({n_features})')

    def get_feature(self, index: Union[int, str]) -> np.ndarray:
        """
        Get a single feature column by index or name.
        
        Parameters
        ----------
        index : Union[int, str]
            Column index (int) or feature name (str)
            
        Returns
        -------
        np.ndarray
            1D array of feature values
        """
        if isinstance(index, str):
            if self.feature_names is None:
                raise ValueError('Feature names not available')
            try:
                col_idx = self.feature_names.index(index)
            except ValueError:
                raise KeyError(f"Feature '{index}' not found")
            index = col_idx
        elif isinstance(index, int):
            if index < 0 or index >= self.features.shape[1]:
                raise IndexError(f'Index {index} is out of bounds for axis 1 with size {self.features.shape[1]}')
        else:
            raise TypeError('Index must be either an integer or a string')
        return self.features[:, index]

    def get_n_samples(self) -> int:
        """Get the number of samples."""
        return self.features.shape[0]

    def get_n_features(self) -> int:
        """Get the number of features."""
        return self.features.shape[1]