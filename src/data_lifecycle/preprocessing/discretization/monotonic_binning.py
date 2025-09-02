from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.modeling.clustering.k_means.kmeans_algorithm import KMeansClusteringModel

class MonotonicBinningProcessor(BaseTransformer):
    """
    A transformer that performs monotonic binning on numerical features without requiring target values.
    
    This processor implements an unsupervised approach to monotonic binning that preserves the 
    natural order of data while creating discrete bins. It's particularly useful for preparing
    numerical features for algorithms that work better with categorical inputs.
    
    The transformation finds optimal cutpoints that maintain data distribution characteristics
    while ensuring monotonicity in bin assignments.
    
    Attributes
    ----------
    feature_indices : Optional[List[int]]
        Indices of features to apply monotonic binning. If None, applies to all numerical features.
    n_bins : int
        Target number of bins to create (default: 10).
    min_bin_size : float
        Minimum proportion of samples required in each bin (default: 0.05).
    strategy : str
        Strategy for determining bin edges ('quantile', 'uniform', or 'kmeans').
    """

    def __init__(self, feature_indices: Optional[List[int]]=None, n_bins: int=10, min_bin_size: float=0.05, strategy: str='quantile', name: Optional[str]=None):
        super().__init__(name=name)
        self.feature_indices = feature_indices
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.strategy = strategy
        self._bin_edges = {}
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'MonotonicBinningProcessor':
        """
        Fit the monotonic binning processor to the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing features to be binned.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        MonotonicBinningProcessor
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data)
            self._feature_names = data.feature_names
        else:
            X = np.asarray(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        for idx in feature_indices:
            if idx < 0 or idx >= n_features:
                raise ValueError(f'Feature index {idx} is out of bounds for data with {n_features} features')
        self._bin_edges = {}
        min_samples_per_bin = max(1, int(self.min_bin_size * n_samples))
        for feature_idx in feature_indices:
            feature_data = X[:, feature_idx]
            valid_mask = np.isfinite(feature_data)
            clean_data = feature_data[valid_mask]
            if len(clean_data) == 0:
                raise ValueError(f'All values are NaN in feature {feature_idx}')
            if self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                bin_edges = np.percentile(clean_data, quantiles)
            elif self.strategy == 'uniform':
                (min_val, max_val) = (np.min(clean_data), np.max(clean_data))
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            elif self.strategy == 'kmeans':
                reshaped_data = clean_data.reshape(-1, 1)
                feature_set_for_kmeans = FeatureSet(features=reshaped_data)
                kmeans = KMeansClusteringModel(n_clusters=self.n_bins, random_state=42, n_init=10)
                kmeans.fit(feature_set_for_kmeans)
                centers = np.sort(kmeans.cluster_centers_.flatten())
                bin_edges = np.zeros(self.n_bins + 1)
                bin_edges[0] = np.min(clean_data)
                bin_edges[-1] = np.max(clean_data)
                if self.n_bins > 1:
                    for i in range(self.n_bins - 1):
                        bin_edges[i + 1] = (centers[i] + centers[i + 1]) / 2
            else:
                raise ValueError(f'Unknown strategy: {self.strategy}')
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) > 2 and min_samples_per_bin > 1:
                adjusted_edges = [bin_edges[0]]
                current_idx = 0
                for i in range(1, len(bin_edges)):
                    mask = (clean_data >= bin_edges[current_idx]) & (clean_data < bin_edges[i])
                    if i == len(bin_edges) - 1:
                        mask = (clean_data >= bin_edges[current_idx]) & (clean_data <= bin_edges[i])
                    if np.sum(mask) >= min_samples_per_bin or i == len(bin_edges) - 1:
                        adjusted_edges.append(bin_edges[i])
                        current_idx = i
                if len(adjusted_edges) > 1:
                    bin_edges = np.array(adjusted_edges)
            self._bin_edges[feature_idx] = bin_edges
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply monotonic binning to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Transformed data with binned features.
        """
        if not self._bin_edges:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            original_type = 'FeatureSet'
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            original_type = 'DataBatch'
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        else:
            X = np.asarray(data).copy()
            original_type = 'ndarray'
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_binned = X.copy()
        for (feature_idx, bin_edges) in self._bin_edges.items():
            if feature_idx < X.shape[1]:
                if len(bin_edges) > 2:
                    binned_values = np.digitize(X[:, feature_idx], bin_edges[1:-1], right=True)
                else:
                    binned_values = np.zeros(X.shape[0], dtype=int)
                X_binned[:, feature_idx] = binned_values
        if original_type == 'FeatureSet':
            return FeatureSet(features=X_binned, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == 'DataBatch':
            return DataBatch(data=X_binned, labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            return X_binned

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Reverse the binning transformation (not supported for monotonic binning).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Binned data to inverse transform.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Original data format.
            
        Raises
        ------
        NotImplementedError
            Monotonic binning does not support inverse transformation.
        """
        raise NotImplementedError('Monotonic binning does not support inverse transformation')

    def get_bin_edges(self, feature_index: int) -> Optional[np.ndarray]:
        """
        Get the bin edges for a specific feature.
        
        Parameters
        ----------
        feature_index : int
            Index of the feature.
            
        Returns
        -------
        Optional[np.ndarray]
            Array of bin edges or None if feature not binned.
        """
        return self._bin_edges.get(feature_index, None)

    def get_bin_centers(self, feature_index: int) -> Optional[np.ndarray]:
        """
        Get the center values of bins for a specific feature.
        
        Parameters
        ----------
        feature_index : int
            Index of the feature.
            
        Returns
        -------
        Optional[np.ndarray]
            Array of bin centers or None if feature not binned.
        """
        bin_edges = self.get_bin_edges(feature_index)
        if bin_edges is None or len(bin_edges) < 2:
            return None
        return (bin_edges[:-1] + bin_edges[1:]) / 2

class MonotonicBinningTransformer(BaseTransformer):

    def __init__(self, feature_indices: Optional[List[int]]=None, n_bins: int=10, min_bin_size: float=0.05, ensure_monotonicity: str='auto', name: Optional[str]=None):
        super().__init__(name=name)
        self.feature_indices = feature_indices
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.ensure_monotonicity = ensure_monotonicity
        self._bin_edges = {}
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, DataBatch], y: Optional[np.ndarray]=None, **kwargs) -> 'MonotonicBinningTransformer':
        """
        Fit the monotonic binning transformer to the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing features to be binned.
        y : Optional[np.ndarray]
            Target values used to determine optimal binning. Required for supervised binning.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        MonotonicBinningTransformer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If y is not provided for supervised binning.
        """
        if y is None:
            raise ValueError("Target variable 'y' is required for monotonic binning.")
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            X = data.data
            self._feature_names = getattr(data, 'feature_names', None)
        else:
            raise TypeError('data must be either FeatureSet or DataBatch')
        n_features = X.shape[1]
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        if any((idx < 0 or idx >= n_features for idx in feature_indices)):
            raise ValueError('Invalid feature index provided.')
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            valid_mask = ~np.isnan(feature_values)
            clean_values = feature_values[valid_mask]
            clean_y = y[valid_mask] if isinstance(y, np.ndarray) else np.array(y)[valid_mask]
            if len(clean_values) == 0:
                continue
            bin_edges = self._compute_monotonic_bins(clean_values, clean_y)
            self._bin_edges[feature_idx] = bin_edges
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply monotonic binning to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Transformed data with binned features.
        """
        if not self._bin_edges:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            is_feature_set = True
        elif isinstance(data, DataBatch):
            X = data.data.copy()
            is_feature_set = False
        else:
            raise TypeError('data must be either FeatureSet or DataBatch')
        for (feature_idx, bin_edges) in self._bin_edges.items():
            if feature_idx < X.shape[1]:
                X[:, feature_idx] = np.digitize(X[:, feature_idx], bin_edges, right=False) - 1
        if is_feature_set:
            return FeatureSet(features=X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=getattr(data, 'metadata', None), quality_scores=getattr(data, 'quality_scores', None))
        else:
            result = DataBatch(data=X, sample_ids=data.sample_ids, feature_names=getattr(data, 'feature_names', None), metadata=getattr(data, 'metadata', None))
            return result

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Reverse the binning transformation (not supported for monotonic binning).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Binned data to inverse transform.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Original data format (identity transformation).
        """
        pass

    def get_bin_edges(self, feature_index: int) -> Optional[np.ndarray]:
        """
        Get the bin edges for a specific feature.
        
        Parameters
        ----------
        feature_index : int
            Index of the feature.
            
        Returns
        -------
        Optional[np.ndarray]
            Array of bin edges or None if feature not binned.
        """
        pass

    def get_bin_labels(self, feature_index: int) -> Optional[List[str]]:
        """
        Get descriptive labels for bins of a specific feature.
        
        Parameters
        ----------
        feature_index : int
            Index of the feature.
            
        Returns
        -------
        Optional[List[str]]
            List of bin labels or None if feature not binned.
        """
        pass