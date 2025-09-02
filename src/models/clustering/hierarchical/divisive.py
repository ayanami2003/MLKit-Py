import numpy as np
from sklearn.metrics import silhouette_score
from general.base_classes.model_base import BaseModel
from src.data_lifecycle.mathematical_foundations.specialized_functions.distance_metrics import manhattan_distance, compute_distance_matrix
from typing import Optional, Tuple, Union
from collections import deque
from general.structures.feature_set import FeatureSet






class DivisiveClusteringModel(BaseModel):
    """
    Divisive clustering model implementing a top-down hierarchical clustering approach.
    
    This model starts with all data points in a single cluster and recursively splits
    clusters into smaller groups based on dissimilarity measures. It is particularly
    effective for identifying clusters of varying sizes and shapes.
    
    The implementation supports various splitting criteria and can handle different
    distance metrics for determining cluster divisions.
    
    Attributes
    ----------
    n_clusters : int
        The number of clusters to form.
    max_depth : Optional[int]
        Maximum depth of the hierarchical tree. If None, continues until n_clusters.
    affinity : str
        Distance metric to use for calculating dissimilarities ('euclidean', 'manhattan', etc.).
    linkage : str
        Linkage criterion for splitting clusters ('ward', 'complete', 'average').
    threshold : float
        Threshold for early stopping cluster splits.
    name : Optional[str]
        Name identifier for the model instance.
    
    Methods
    -------
    fit(X, y=None, **kwargs)
        Fit the divisive clustering model to the data.
    predict(X, **kwargs)
        Predict cluster labels for new data points.
    fit_predict(X, y=None, **kwargs)
        Fit the model and predict cluster labels in one step.
    score(X, y, **kwargs)
        Calculate silhouette score for the clustering.
    get_cluster_hierarchy()
        Retrieve the hierarchical structure of clusters.
    """

    def __init__(self, n_clusters: int=2, max_depth: Optional[int]=None, affinity: str='euclidean', linkage: str='ward', threshold: float=0.0, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.affinity = affinity
        self.linkage = linkage
        self.threshold = threshold
        self._cluster_labels = None
        self._hierarchy = {}
        self._centroids = None
        self._X_fit = None

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.n_clusters < 1:
            raise ValueError('n_clusters must be at least 1')
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError('max_depth must be at least 1 if specified')
        if self.threshold < 0:
            raise ValueError('threshold must be non-negative')

    def _compute_cluster_distance(self, X: np.ndarray, cluster1_indices: np.ndarray, cluster2_indices: np.ndarray) -> float:
        """
        Compute distance between two clusters based on the linkage criterion.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        cluster1_indices : np.ndarray
            Indices of points in first cluster
        cluster2_indices : np.ndarray
            Indices of points in second cluster
            
        Returns
        -------
        float
            Distance between clusters
        """
        if len(cluster1_indices) == 0 or len(cluster2_indices) == 0:
            return np.inf
        distances = compute_distance_matrix(X[cluster1_indices], X[cluster2_indices], metric=self.affinity)
        if self.linkage == 'single':
            return np.min(distances)
        elif self.linkage == 'complete':
            return np.max(distances)
        elif self.linkage == 'average':
            return np.mean(distances)
        else:
            return np.mean(distances)

    def _find_best_split(self, X: np.ndarray, cluster_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Find the best way to split a cluster into two sub-clusters.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        cluster_indices : np.ndarray
            Indices of points in the cluster to split
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Indices of points in first sub-cluster, indices in second sub-cluster, and split quality metric
        """
        if len(cluster_indices) <= 1:
            return (cluster_indices, np.array([]), 0.0)
        cluster_data = X[cluster_indices]
        distances = compute_distance_matrix(cluster_data, cluster_data, metric=self.affinity)
        np.fill_diagonal(distances, 0)
        (i, j) = np.unravel_index(np.argmax(distances), distances.shape)
        cluster1_indices = np.array([cluster_indices[i]])
        cluster2_indices = np.array([cluster_indices[j]])
        remaining_indices = np.setdiff1d(cluster_indices, [cluster_indices[i], cluster_indices[j]])
        for idx in remaining_indices:
            point = X[idx].reshape(1, -1)
            dist_to_cluster1 = self._compute_cluster_distance(X, cluster1_indices, np.array([idx]))
            dist_to_cluster2 = self._compute_cluster_distance(X, cluster2_indices, np.array([idx]))
            if dist_to_cluster1 < dist_to_cluster2:
                cluster1_indices = np.append(cluster1_indices, idx)
            else:
                cluster2_indices = np.append(cluster2_indices, idx)
        split_quality = abs(len(cluster1_indices) - len(cluster2_indices))
        return (cluster1_indices, cluster2_indices, split_quality)

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'DivisiveClusteringModel':
        """
        Fit the divisive clustering model to the input data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training data consisting of n_samples samples and n_features features.
        y : Optional[Union[np.ndarray, list]]
            Ignored in unsupervised learning, included for API compatibility.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        DivisiveClusteringModel
            Fitted model instance.
        """
        self._validate_params()
        if isinstance(X, FeatureSet):
            X_array = X.get_features().values
        else:
            X_array = X
        self._X_fit = X_array.copy()
        n_samples = X_array.shape[0]
        current_clusters = [np.arange(n_samples)]
        cluster_labels = np.zeros(n_samples, dtype=int)
        hierarchy = {'root': {'indices': np.arange(n_samples), 'children': [], 'depth': 0}}
        queue = deque([(np.arange(n_samples), 'root', 0)])
        next_cluster_id = 0
        while len(current_clusters) < self.n_clusters and (self.max_depth is None or any((d < self.max_depth for (_, _, d) in queue))):
            if not queue:
                break
            (cluster_indices, parent_id, depth) = queue.popleft()
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            (cluster1_indices, cluster2_indices, split_quality) = self._find_best_split(X_array, cluster_indices)
            if len(cluster1_indices) == 0 or len(cluster2_indices) == 0:
                continue
            next_cluster_id += 1
            cluster_labels[cluster2_indices] = next_cluster_id
            current_clusters.append(cluster2_indices)
            current_clusters = [c for c in current_clusters if not np.array_equal(c, cluster_indices)]
            current_clusters.append(cluster1_indices)
            child1_id = f'{parent_id}_0'
            child2_id = f'{parent_id}_1'
            hierarchy[child1_id] = {'indices': cluster1_indices, 'children': [], 'depth': depth + 1}
            hierarchy[child2_id] = {'indices': cluster2_indices, 'children': [], 'depth': depth + 1}
            hierarchy[parent_id]['children'] = [child1_id, child2_id]
            if len(cluster1_indices) > 1 and (self.max_depth is None or depth + 1 < self.max_depth):
                queue.append((cluster1_indices, child1_id, depth + 1))
            if len(cluster2_indices) > 1 and (self.max_depth is None or depth + 1 < self.max_depth):
                queue.append((cluster2_indices, child2_id, depth + 1))
        self._cluster_labels = cluster_labels
        self._hierarchy = hierarchy
        self._centroids = []
        for cluster_idx in np.unique(cluster_labels):
            cluster_points = X_array[cluster_labels == cluster_idx]
            self._centroids.append(np.mean(cluster_points, axis=0))
        self._centroids = np.array(self._centroids)
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data to predict cluster labels for.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Array of cluster labels for each data point.
        """
        if self._centroids is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        if isinstance(X, FeatureSet):
            X_array = X.get_features().values
        else:
            X_array = X
        predictions = np.zeros(X_array.shape[0], dtype=int)
        for (i, point) in enumerate(X_array):
            distances = compute_distance_matrix(point.reshape(1, -1), self._centroids, metric=self.affinity).flatten()
            predictions[i] = np.argmin(distances)
        return predictions

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training data.
        y : Optional[Union[np.ndarray, list]]
            Ignored in unsupervised learning.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        np.ndarray
            Cluster labels for the input data.
        """
        self.fit(X, y, **kwargs)
        return self._cluster_labels.copy()

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Calculate silhouette score for the clustering.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data that was clustered.
        y : Union[np.ndarray, list]
            Cluster labels.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Silhouette score ranging from -1 to 1.
        """
        if isinstance(X, FeatureSet):
            X_array = X.get_features().values
        else:
            X_array = X
        if not isinstance(y, np.ndarray):
            y_array = np.array(y)
        else:
            y_array = y
        return silhouette_score(X_array, y_array, metric=self.affinity)

    def get_cluster_hierarchy(self) -> dict:
        """
        Retrieve the hierarchical structure of clusters.
        
        Returns
        -------
        dict
            Nested dictionary representing the cluster hierarchy with split information.
        """
        if not self._hierarchy:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'get_cluster_hierarchy'.")
        return self._hierarchy.copy()