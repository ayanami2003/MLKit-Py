import numpy as np
from typing import Union, Optional, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class DataBinner(BaseTransformer):

    def __init__(self, strategy: str='equal_width', n_bins: int=5, feature_indices: Optional[List[int]]=None, name: Optional[str]=None):
        """
        Initialize the DataBinner transformer.
        
        Args:
            strategy (str): Binning strategy to use. Supported values are 'equal_width' and 'equal_frequency'.
            n_bins (int): Number of bins to create for each feature.
            feature_indices (Optional[List[int]]): Specific feature indices to apply binning to. 
                                                  If None, all features will be binned.
            name (Optional[str]): Name for the transformer instance.
        """
        super().__init__(name=name)
        self.strategy = strategy
        self.n_bins = n_bins
        self.feature_indices = feature_indices
        self.bin_edges: List[np.ndarray] = []

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'DataBinner':
        """
        Fit the binning parameters to the input data.
        
        This method computes the bin edges for each feature based on the specified strategy
        and stores them for later use in transformation.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit the binning on.
                                                 If FeatureSet, uses the features attribute.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            DataBinner: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not np.isfinite(X).all():
            raise ValueError('Input data contains non-finite values')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        if any((idx < 0 or idx >= n_features for idx in feature_indices)):
            raise ValueError('Feature indices out of range')
        self.bin_edges = []
        for i in range(n_features):
            if i in feature_indices:
                feature_data = X[:, i]
                unique_values = np.unique(feature_data)
                if len(unique_values) == 1:
                    val = unique_values[0]
                    bin_edges = np.array([val - 0.5, val + 0.5])
                elif self.strategy == 'equal_width':
                    (min_val, max_val) = (np.min(feature_data), np.max(feature_data))
                    if min_val == max_val:
                        bin_edges = np.array([min_val - 0.5, min_val + 0.5])
                    else:
                        bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
                elif self.strategy == 'equal_frequency':
                    if len(unique_values) <= self.n_bins:
                        bin_edges = np.concatenate([[unique_values[0]], unique_values, [unique_values[-1] + 1e-10]])
                    else:
                        percentiles = np.linspace(0, 100, self.n_bins + 1)
                        bin_edges = np.percentile(feature_data, percentiles)
                        bin_edges = np.unique(bin_edges)
                        if len(bin_edges) < self.n_bins + 1:
                            while len(bin_edges) < self.n_bins + 1:
                                bin_edges = np.append(bin_edges, bin_edges[-1] + 1e-10)
                else:
                    raise ValueError(f'Unknown strategy: {self.strategy}')
                self.bin_edges.append(bin_edges)
            else:
                self.bin_edges.append(np.array([]))
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply binning to the input data using the fitted bin edges.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
                                                 If FeatureSet, uses the features attribute.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            FeatureSet: Transformed data with binned features.
        """
        if not hasattr(self, 'bin_edges') or len(self.bin_edges) == 0:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = np.asarray(data).copy()
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.bin_edges):
            raise ValueError(f'Data has {X.shape[1]} features, but transformer was fitted on {len(self.bin_edges)} features')
        X_binned = X.copy()
        for i in range(X.shape[1]):
            if len(self.bin_edges[i]) > 0:
                bin_edges = self.bin_edges[i]
                if len(bin_edges) == 2 and np.isclose(bin_edges[1] - bin_edges[0], 1.0):
                    X_binned[:, i] = 0
                else:
                    bin_indices = np.digitize(X[:, i], bin_edges, right=False)
                    bin_indices = np.clip(bin_indices, 1, len(bin_edges) - 1) - 1
                    X_binned[:, i] = bin_indices
        return FeatureSet(features=X_binned, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert binned data back to approximate continuous values.
        
        This method maps binned values back to the midpoint of their respective bins.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Binned input data to inverse transform.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            FeatureSet: Data with binned features converted back to approximate continuous values.
        """
        if not hasattr(self, 'bin_edges') or len(self.bin_edges) == 0:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = np.asarray(data).copy()
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.bin_edges):
            raise ValueError(f'Data has {X.shape[1]} features, but transformer was fitted on {len(self.bin_edges)} features')
        X_continuous = X.copy().astype(float)
        for i in range(X.shape[1]):
            if len(self.bin_edges[i]) > 0:
                bin_edges = self.bin_edges[i]
                if len(bin_edges) >= 2:
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    for j in range(len(bin_centers)):
                        mask = X[:, i] == j
                        X_continuous[mask, i] = bin_centers[j]
                else:
                    val = bin_edges[0]
                    X_continuous[:, i] = val
        return FeatureSet(features=X_continuous, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def get_bin_edges(self) -> List[np.ndarray]:
        """
        Get the computed bin edges for all features.
        
        Returns:
            List[np.ndarray]: List of bin edges for each feature.
        """
        return self.bin_edges

    def get_bin_centers(self) -> List[np.ndarray]:
        """
        Calculate the center points of all bins for each feature.
        
        Returns:
            List[np.ndarray]: List of bin center points for each feature.
        """
        centers = []
        for edges in self.bin_edges:
            if len(edges) > 0 and len(edges) >= 2:
                bin_centers = (edges[:-1] + edges[1:]) / 2
                centers.append(bin_centers)
            else:
                centers.append(np.array([]))
        return centers