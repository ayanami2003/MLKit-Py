from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans

class AdaptiveBinningTransformer(BaseTransformer):

    def __init__(self, strategy: str='quantile', n_bins: int=5, feature_indices: Optional[List[int]]=None, name: Optional[str]=None):
        """
        Initialize the AdaptiveBinningTransformer.
        
        Parameters
        ----------
        strategy : str, default='quantile'
            The adaptive binning strategy to use. Must be one of ['quantile', 'kmeans', 'optimization'].
        n_bins : int, default=5
            The target number of bins to create. Must be >= 2.
        feature_indices : Optional[List[int]], optional
            Indices of features to transform. If None, transforms all numerical features.
        name : Optional[str], optional
            Name of the transformer instance.
            
        Raises
        ------
        ValueError
            If strategy is not supported or n_bins < 2.
        """
        super().__init__(name=name)
        if strategy not in ['quantile', 'kmeans', 'optimization']:
            raise ValueError("Strategy must be one of ['quantile', 'kmeans', 'optimization']")
        if n_bins < 2:
            raise ValueError('n_bins must be >= 2')
        self.strategy = strategy
        self.n_bins = n_bins
        self.feature_indices = feature_indices
        self.bin_edges_ = {}
        self.is_fitted = False

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'AdaptiveBinningTransformer':
        """
        Fit the transformer to the input data by computing adaptive bin edges.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to fit the transformer on. Numerical features will be binned.
        **kwargs : dict
            Additional parameters for fitting (ignored).
            
        Returns
        -------
        AdaptiveBinningTransformer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If data has inconsistent shapes or contains non-numerical features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = data.data
        else:
            X = data
        if X.size == 0:
            raise ValueError('Input data is empty')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        (n_samples, n_features) = X.shape
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            if any((idx < 0 or idx >= n_features for idx in self.feature_indices)):
                raise ValueError('Feature indices out of bounds')
            feature_indices = self.feature_indices
        self.bin_edges_ = {}
        for feature_idx in feature_indices:
            feature_data = X[:, feature_idx]
            if not np.issubdtype(feature_data.dtype, np.number):
                raise ValueError(f'Feature at index {feature_idx} is not numerical')
            unique_vals = np.unique(feature_data)
            if len(unique_vals) == 1:
                val = unique_vals[0]
                edges = np.full(self.n_bins + 1, val)
                self.bin_edges_[feature_idx] = edges
                continue
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) == 0:
                raise ValueError(f'All values in feature {feature_idx} are NaN')
            if self.strategy == 'quantile':
                edges = self._compute_quantile_edges(valid_data)
            elif self.strategy == 'kmeans':
                edges = self._compute_kmeans_edges(valid_data)
            elif self.strategy == 'optimization':
                edges = self._compute_optimization_edges(valid_data)
            else:
                raise ValueError(f'Unsupported strategy: {self.strategy}')
            self.bin_edges_[feature_idx] = edges
        self.is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Apply adaptive binning to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation (ignored).
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Transformed data with binned features.
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or data shape is inconsistent.
        """
        if not self.is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            input_type = 'FeatureSet'
        elif isinstance(data, DataBatch):
            X = data.data.copy()
            input_type = 'DataBatch'
        else:
            X = data.copy()
            input_type = 'ndarray'
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        (n_samples, n_features) = X.shape
        if self.feature_indices is not None:
            if any((idx < 0 or idx >= n_features for idx in self.feature_indices)):
                raise ValueError('Feature indices out of bounds')
        X_transformed = X.copy()
        for (feature_idx, edges) in self.bin_edges_.items():
            if feature_idx >= n_features:
                raise ValueError(f'Feature index {feature_idx} out of bounds for input data')
            feature_values = X[:, feature_idx]
            digitized = np.digitize(feature_values, edges) - 1
            clipped = np.clip(digitized, 0, len(edges) - 2)
            X_transformed[:, feature_idx] = clipped
        for feature_idx in self.bin_edges_.keys():
            X_transformed[:, feature_idx] = X_transformed[:, feature_idx].astype(int)
        if input_type == 'FeatureSet':
            result = FeatureSet(features=X_transformed)
            return result
        elif input_type == 'DataBatch':
            result = DataBatch(data=X_transformed, labels=data.labels if hasattr(data, 'labels') else None)
            return result
        else:
            return X_transformed

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Reverse the binning transformation by mapping bins back to approximate values.
        
        For each binned feature, values are mapped to the midpoint of their respective bin.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Binned data to inverse transform.
        **kwargs : dict
            Additional parameters for inverse transformation (ignored).
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Data with binned features converted back to approximate continuous values.
            
        Raises
        ------
        ValueError
            If transformer has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            input_type = 'FeatureSet'
        elif isinstance(data, DataBatch):
            X = data.data.copy()
            input_type = 'DataBatch'
        else:
            X = data.copy()
            input_type = 'ndarray'
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        (n_samples, n_features) = X.shape
        if self.feature_indices is not None:
            if any((idx < 0 or idx >= n_features for idx in self.feature_indices)):
                raise ValueError('Feature indices out of bounds')
        for (feature_idx, edges) in self.bin_edges_.items():
            if feature_idx >= n_features:
                raise ValueError(f'Feature index {feature_idx} out of bounds for input data')
            bin_indices = X[:, feature_idx].astype(int)
            bin_indices = np.clip(bin_indices, 0, len(edges) - 2)
            midpoints = (edges[:-1] + edges[1:]) / 2
            X[:, feature_idx] = midpoints[bin_indices]
        if input_type == 'FeatureSet':
            result = FeatureSet(features=X)
            return result
        elif input_type == 'DataBatch':
            result = DataBatch(data=X, labels=data.labels if hasattr(data, 'labels') else None)
            return result
        else:
            return X

    def _compute_quantile_edges(self, data: np.ndarray) -> np.ndarray:
        """Compute bin edges using quantiles."""
        if len(np.unique(data)) == 1:
            val = data[0]
            return np.full(self.n_bins + 1, val)
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(data, percentiles)
        return edges

    def _compute_kmeans_edges(self, data: np.ndarray) -> np.ndarray:
        """Compute bin edges using k-means clustering."""
        if len(np.unique(data)) == 1:
            val = data[0]
            return np.full(self.n_bins + 1, val)
        X = data.reshape(-1, 1)
        n_unique = len(np.unique(data))
        n_clusters = min(self.n_bins, n_unique, len(data))
        if n_clusters <= 1:
            return np.linspace(data.min(), data.max(), self.n_bins + 1)
        if len(data) < n_clusters:
            return np.linspace(data.min(), data.max(), self.n_bins + 1)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        centers = np.sort(kmeans.cluster_centers_.flatten())
        edges = np.empty(len(centers) + 1)
        edges[0] = data.min()
        edges[-1] = data.max()
        for i in range(len(centers) - 1):
            edges[i + 1] = (centers[i] + centers[i + 1]) / 2
        if len(edges) != self.n_bins + 1:
            edges = np.linspace(data.min(), data.max(), self.n_bins + 1)
        return edges

    def _compute_optimization_edges(self, data: np.ndarray) -> np.ndarray:
        """Compute bin edges using optimization to minimize information loss."""
        if len(np.unique(data)) == 1:
            val = data[0]
            return np.full(self.n_bins + 1, val)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        if n < 2 * self.n_bins:
            return np.linspace(sorted_data[0], sorted_data[-1], self.n_bins + 1)

        def objective(split_points):
            if np.isscalar(split_points):
                split_points = np.array([split_points])
            edges = np.concatenate([[sorted_data[0]], split_points, [sorted_data[-1]]])
            edges = np.unique(edges)
            if len(edges) < 2:
                return np.inf
            bin_indices = np.digitize(sorted_data, edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(edges) - 2)
            total_loss = 0
            for i in range(len(edges) - 1):
                mask = bin_indices == i
                if np.sum(mask) > 1:
                    bin_data = sorted_data[mask]
                    total_loss += np.var(bin_data) * len(bin_data)
            return total_loss
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        initial_splits = np.percentile(sorted_data, quantiles)
        if self.n_bins <= 2:
            result = minimize_scalar(objective, bounds=(sorted_data[0], sorted_data[-1]), method='bounded')
            if result.success:
                optimal_split = result.x
                edges = np.array([sorted_data[0], optimal_split, sorted_data[-1]])
                if len(edges) != self.n_bins + 1:
                    edges = np.linspace(sorted_data[0], sorted_data[-1], self.n_bins + 1)
                return edges
            else:
                return np.linspace(sorted_data[0], sorted_data[-1], self.n_bins + 1)
        else:
            return np.linspace(sorted_data[0], sorted_data[-1], self.n_bins + 1)