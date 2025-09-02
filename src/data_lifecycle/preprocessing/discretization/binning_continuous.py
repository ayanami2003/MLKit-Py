from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, List, Union

class ContinuousBinningTransformer(BaseTransformer):

    def __init__(self, strategy: str='equal_width', n_bins: int=5, custom_bins: Optional[List[float]]=None, feature_indices: Optional[List[int]]=None, name: Optional[str]=None):
        """
        Initialize the ContinuousBinningTransformer.
        
        Args:
            strategy (str): Binning strategy - 'equal_width', 'equal_frequency', or 'custom'
            n_bins (int): Number of bins for equal-width or equal-frequency strategies
            custom_bins (Optional[List[float]]): Custom bin boundaries for 'custom' strategy
            feature_indices (Optional[List[int]]): Specific feature indices to transform
            name (Optional[str]): Name for the transformer instance
        """
        super().__init__(name=name)
        if strategy not in ['equal_width', 'equal_frequency', 'custom']:
            raise ValueError("strategy must be one of 'equal_width', 'equal_frequency', or 'custom'")
        if strategy == 'custom' and custom_bins is None:
            raise ValueError("custom_bins must be provided when strategy is 'custom'")
        if strategy != 'custom' and custom_bins is not None:
            raise ValueError("custom_bins should only be provided when strategy is 'custom'")
        if n_bins <= 0:
            raise ValueError('n_bins must be positive')
        self.strategy = strategy
        self.n_bins = n_bins
        self.custom_bins = custom_bins
        self.feature_indices = feature_indices
        self._bin_edges = {}

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ContinuousBinningTransformer':
        """
        Fit the binning transformer to the data by computing bin boundaries.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data containing continuous features
            **kwargs: Additional fitting parameters
            
        Returns:
            ContinuousBinningTransformer: Returns self for method chaining
            
        Raises:
            ValueError: If strategy is 'custom' but custom_bins is not provided
            ValueError: If custom_bins is provided but strategy is not 'custom'
        """
        if isinstance(data, FeatureSet):
            features = data.features
        else:
            features = data
        if self.feature_indices is None:
            if isinstance(data, FeatureSet) and data.feature_types is not None:
                feature_indices = [i for (i, ft) in enumerate(data.feature_types) if ft == 'numeric']
            else:
                feature_indices = list(range(features.shape[1]))
        else:
            feature_indices = self.feature_indices
        self._bin_edges = {}
        for idx in feature_indices:
            feature_values = features[:, idx]
            valid_values = feature_values[~np.isnan(feature_values)]
            if len(valid_values) == 0:
                self._bin_edges[idx] = np.array([0.0, 1.0])
                continue
            if self.strategy == 'equal_width':
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                if min_val == max_val:
                    bin_edges = np.array([min_val, max_val])
                else:
                    bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            elif self.strategy == 'equal_frequency':
                if len(valid_values) <= self.n_bins:
                    bin_edges = np.unique(valid_values)
                    if len(bin_edges) == 1:
                        bin_edges = np.array([bin_edges[0], bin_edges[0]])
                    else:
                        bin_edges = np.append(bin_edges, bin_edges[-1] + 1e-10)
                else:
                    quantiles = np.linspace(0, 100, self.n_bins + 1)
                    bin_edges = np.percentile(valid_values, quantiles)
                    bin_edges = np.unique(bin_edges)
            elif self.strategy == 'custom':
                bin_edges = np.array(self.custom_bins)
            self._bin_edges[idx] = bin_edges
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply binning transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            FeatureSet: Transformed data with binned features
            
        Raises:
            RuntimeError: If transformer has not been fitted
        """
        if not hasattr(self, '_bin_edges') or len(self._bin_edges) == 0:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            features = data.features.copy()
            result_fs = FeatureSet(features=features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        else:
            features = data.copy()
            result_fs = FeatureSet(features=features)
        for (idx, bin_edges) in self._bin_edges.items():
            if idx < features.shape[1]:
                binned_values = np.digitize(features[:, idx], bin_edges, right=False) - 1
                binned_values = np.clip(binned_values, 0, len(bin_edges) - 2)
                nan_mask = np.isnan(features[:, idx])
                binned_values = binned_values.astype(float)
                binned_values[nan_mask] = np.nan
                features[:, idx] = binned_values
        result_fs.features = features
        return result_fs

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert binned data back to approximate continuous values using bin midpoints.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Binned input data
            **kwargs: Additional parameters
            
        Returns:
            FeatureSet: Data with binned features converted back to continuous values
            
        Raises:
            RuntimeError: If transformer has not been fitted
        """
        if not hasattr(self, '_bin_edges') or len(self._bin_edges) == 0:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            features = data.features.copy()
            result_fs = FeatureSet(features=features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        else:
            features = data.copy()
            result_fs = FeatureSet(features=features)
        for (idx, bin_edges) in self._bin_edges.items():
            if idx < features.shape[1]:
                midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
                binned_vals = features[:, idx]
                continuous_vals = np.full_like(binned_vals, np.nan, dtype=float)
                valid_mask = ~np.isnan(binned_vals)
                if np.any(valid_mask):
                    indices = binned_vals[valid_mask].astype(int)
                    indices = np.clip(indices, 0, len(midpoints) - 1)
                    continuous_vals[valid_mask] = midpoints[indices]
                features[:, idx] = continuous_vals
        result_fs.features = features
        return result_fs

    def get_bin_edges(self, feature_index: int) -> Optional[np.ndarray]:
        """
        Get the bin edges for a specific feature.
        
        Args:
            feature_index (int): Index of the feature
            
        Returns:
            Optional[np.ndarray]: Array of bin edges or None if feature not binned
        """
        return self._bin_edges.get(feature_index, None)

    def get_bin_labels(self, feature_index: int) -> Optional[List[str]]:
        """
        Get descriptive labels for bins of a specific feature.
        
        Args:
            feature_index (int): Index of the feature
            
        Returns:
            Optional[List[str]]: List of bin labels or None if feature not binned
        """
        bin_edges = self.get_bin_edges(feature_index)
        if bin_edges is None:
            return None
        n_bins = len(bin_edges) - 1
        return [f'Bin {i}' for i in range(n_bins)]