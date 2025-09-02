from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from sklearn.cluster import KMeans

class KMeansDiscretizer(BaseTransformer):

    def __init__(self, n_bins: int=5, feature_indices: Optional[List[int]]=None, random_state: Optional[int]=None, max_iter: int=300, tol: float=0.0001, name: Optional[str]=None):
        """
        Initialize the KMeansDiscretizer.
        
        Parameters
        ----------
        n_bins : int, default=5
            Number of bins to create for discretization (number of clusters)
        feature_indices : Optional[List[int]], default=None
            Indices of features to discretize. If None, all features are processed
        random_state : Optional[int], default=None
            Random seed for k-means initialization for reproducibility
        max_iter : int, default=300
            Maximum number of iterations for the k-means algorithm
        tol : float, default=1e-4
            Relative tolerance for declaring convergence in k-means
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.n_bins = n_bins
        self.feature_indices = feature_indices
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self._cluster_centers = {}
        self._bin_edges = {}
        self._n_features_fit = 0

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'KMeansDiscretizer':
        """
        Fit the k-means discretizer to the input data by determining cluster centers.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to fit the discretizer on. Can be a FeatureSet, DataBatch, or numpy array
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        KMeansDiscretizer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If n_bins is less than 2 or if data is empty
        """
        if self.n_bins < 2:
            raise ValueError('n_bins must be at least 2')
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
        else:
            X = np.array(data)
        if X.size == 0:
            raise ValueError('Input data is empty')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (n_samples, n_features) = X.shape
        self._n_features_fit = n_features
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = [idx for idx in self.feature_indices if 0 <= idx < n_features]
        self._cluster_centers.clear()
        self._bin_edges.clear()
        for feature_idx in feature_indices:
            feature_data = X[:, feature_idx].reshape(-1, 1)
            kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, max_iter=self.max_iter, tol=self.tol, n_init=10)
            kmeans.fit(feature_data)
            centers = np.sort(kmeans.cluster_centers_.flatten())
            self._cluster_centers[feature_idx] = centers
            bin_edges = np.empty(len(centers) + 1)
            feature_min = feature_data.min()
            feature_max = feature_data.max()
            bin_edges[0] = feature_min - 0.001 * abs(feature_min) if feature_min != 0 else -0.001
            bin_edges[-1] = feature_max + 0.001 * abs(feature_max) if feature_max != 0 else 0.001
            for i in range(len(centers) - 1):
                bin_edges[i + 1] = (centers[i] + centers[i + 1]) / 2
            self._bin_edges[feature_idx] = bin_edges
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Apply k-means discretization to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to transform. Must have same number of features as fitted data
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Discretized data with same type as input
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        ValueError
            If input data dimensions don't match fitted data
        """
        if not self._cluster_centers:
            raise RuntimeError('Transformer has not been fitted yet.')
        original_type = type(data).__name__
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        else:
            X = np.array(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (n_samples, n_features) = X.shape
        if n_features != self._n_features_fit:
            raise ValueError(f'Data has {n_features} features, but transformer was fitted on {self._n_features_fit} features')
        X_transformed = X.copy()
        for (feature_idx, centers) in self._cluster_centers.items():
            if feature_idx < n_features:
                feature_data = X[:, feature_idx]
                bin_edges = self._bin_edges[feature_idx]
                bin_indices = np.digitize(feature_data, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, len(centers) - 1)
                X_transformed[:, feature_idx] = bin_indices
        for feature_idx in self._cluster_centers.keys():
            if feature_idx < n_features:
                X_transformed[:, feature_idx] = X_transformed[:, feature_idx].astype(int)
        if original_type == 'FeatureSet':
            return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == 'DataBatch':
            return DataBatch(data=X_transformed.tolist(), labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            return X_transformed

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Map discretized values back to continuous space using cluster centers.
        
        Each discretized value (bin index) is mapped to its corresponding cluster center.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Discretized input data to inverse transform
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Continuous reconstructed data with same type as input
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        ValueError
            If input data dimensions don't match fitted data
        """
        if not self._cluster_centers:
            raise RuntimeError('Transformer has not been fitted yet.')
        original_type = type(data).__name__
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        else:
            X = np.array(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (n_samples, n_features) = X.shape
        if n_features != self._n_features_fit:
            raise ValueError(f'Data has {n_features} features, but transformer was fitted on {self._n_features_fit} features')
        X_inverse = X.copy().astype(float)
        for (feature_idx, centers) in self._cluster_centers.items():
            if feature_idx < n_features:
                discretized_values = X[:, feature_idx]
                valid_indices = np.clip(discretized_values.astype(int), 0, len(centers) - 1)
                continuous_values = centers[valid_indices]
                X_inverse[:, feature_idx] = continuous_values
        if original_type == 'FeatureSet':
            return FeatureSet(features=X_inverse, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == 'DataBatch':
            return DataBatch(data=X_inverse.tolist(), labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            return X_inverse