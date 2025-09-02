import numpy as np
from typing import Union, Optional, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet


# ...(code omitted)...


class FeatureDiscretizer(BaseTransformer):
    """
    A transformer for discretizing specific features within a dataset using configurable strategies.
    
    This class allows targeted discretization of selected numerical features, leaving others unchanged.
    It supports multiple binning approaches and maintains mappings for invertible transformations.
    
    Attributes:
        feature_indices (List[int]): Indices of features to discretize.
        strategy (str): Discretization approach ('equal_width', 'equal_frequency').
        n_bins (int): Number of bins for discretization.
    """

    def __init__(self, feature_indices: List[int], strategy: str='equal_width', n_bins: int=5, name: Optional[str]=None):
        super().__init__(name=name)
        if strategy not in ['equal_width', 'equal_frequency']:
            raise ValueError("strategy must be either 'equal_width' or 'equal_frequency'")
        if n_bins <= 0:
            raise ValueError('n_bins must be positive')
        if not isinstance(feature_indices, list):
            raise TypeError('feature_indices must be a list')
        self.feature_indices = feature_indices
        self.strategy = strategy
        self.n_bins = n_bins
        self._bin_mappings = {}
        self._fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'FeatureDiscretizer':
        """
        Fit the discretizer to the input data by computing bin edges for specified features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the discretizer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureDiscretizer
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.size == 0:
            self._fitted = True
            return self
        n_features = X.shape[1]
        for idx in self.feature_indices:
            if idx < 0 or idx >= n_features:
                raise IndexError(f'Feature index {idx} is out of bounds for data with {n_features} features')
        for feature_idx in self.feature_indices:
            feature_data = X[:, feature_idx]
            unique_vals = np.unique(feature_data)
            if len(unique_vals) == 1:
                val = unique_vals[0]
                edges = np.array([val, val])
                midpoints = np.array([val])
            else:
                if self.strategy == 'equal_width':
                    edges = np.linspace(feature_data.min(), feature_data.max(), self.n_bins + 1)
                elif self.strategy == 'equal_frequency':
                    quantiles = np.linspace(0, 100, self.n_bins + 1)
                    edges = np.percentile(feature_data, quantiles)
                edges = np.unique(edges)
                midpoints = (edges[:-1] + edges[1:]) / 2
            self._bin_mappings[feature_idx] = (edges, midpoints)
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply discretization to specified features in the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with discretized features.
        """
        if not self._fitted:
            raise RuntimeError('FeatureDiscretizer must be fitted before transform')
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data.copy()
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.size == 0:
            if is_feature_set:
                return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
            return X
        for feature_idx in self.feature_indices:
            if feature_idx in self._bin_mappings:
                (edges, _) = self._bin_mappings[feature_idx]
                bin_indices = np.digitize(X[:, feature_idx], edges, right=False)
                bin_indices = np.clip(bin_indices, 1, len(edges) - 1) - 1
                X[:, feature_idx] = bin_indices
        if is_feature_set:
            return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        return X

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Map discretized values back to approximate continuous values (bin midpoints).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Discretized input data to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Approximate continuous values mapped from discretized bins.
        """
        if not self._fitted:
            raise RuntimeError('FeatureDiscretizer must be fitted before inverse_transform')
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data.copy()
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.size == 0:
            if is_feature_set:
                return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
            return X
        for feature_idx in self.feature_indices:
            if feature_idx in self._bin_mappings:
                (_, midpoints) = self._bin_mappings[feature_idx]
                discretized_values = X[:, feature_idx].astype(int)
                discretized_values = np.clip(discretized_values, 0, len(midpoints) - 1)
                X[:, feature_idx] = midpoints[discretized_values]
        if is_feature_set:
            return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        return X