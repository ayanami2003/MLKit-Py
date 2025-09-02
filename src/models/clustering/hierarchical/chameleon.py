from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, Tuple, Dict, List
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

class ChameleonClusteringModel(BaseModel):
    """
    Chameleon clustering model implementation for hierarchical clustering.
    
    Chameleon is a hierarchical clustering algorithm that uses a dynamic modeling approach
    to merge clusters based on both interconnectivity and closeness measures. It dynamically
    determines the similarity between clusters by considering the internal structure of
    clusters rather than just their centroid or representative points.
    
    This implementation follows the two-phase approach:
    1. Graph partitioning phase using K-nearest neighbors
    2. Hierarchical clustering phase with adaptive merge criteria
    
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    k_neighbors : int, default=5
        Number of nearest neighbors to consider for graph construction.
    min_cluster_size : int, default=5
        Minimum size of clusters to consider during merging.
    alpha : float, default=2.0
        Weight parameter for relative interconnectivity in merge criteria.
    beta : float, default=1.0
        Weight parameter for relative closeness in merge criteria.
    metric : str, default='euclidean'
        Distance metric to use for computing similarities.
    name : Optional[str], default=None
        Name of the clustering model.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers (if applicable).
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self, n_clusters: int=2, k_neighbors: int=5, min_cluster_size: int=5, alpha: float=2.0, beta: float=1.0, metric: str='euclidean', name: Optional[str]=None):
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.min_cluster_size = min_cluster_size
        self.alpha = alpha
        self.beta = beta
        self.metric = metric

    def _validate_input(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Validate and convert input data to numpy array."""
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise TypeError('X must be either a FeatureSet or numpy array')
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[0] < 2:
            raise ValueError('At least 2 samples are required')
        return X_array

    def _build_knn_graph(self, X: np.ndarray) -> csr_matrix:
        """Build K-nearest neighbors graph."""
        n_samples = X.shape[0]
        if n_samples <= self.k_neighbors:
            distances = cdist(X, X, metric=self.metric)
            np.fill_diagonal(distances, np.inf)
            return csr_matrix(distances < np.inf)
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, metric=self.metric)
        nbrs.fit(X)
        knn_graph = nbrs.kneighbors_graph(mode='connectivity')
        knn_graph = knn_graph.maximum(knn_graph.T)
        return knn_graph

    def _graph_partitioning(self, X: np.ndarray, knn_graph: csr_matrix) -> np.ndarray:
        """Partition the graph into micro-clusters using connected components."""
        n_samples = X.shape[0]
        labels = -np.ones(n_samples, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        for i in range(n_samples):
            if not visited[i]:
                queue = [i]
                visited[i] = True
                component = []
                while queue:
                    node = queue.pop(0)
                    component.append(node)
                    neighbors = knn_graph[node].nonzero()[1]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                labels[component] = cluster_id
                cluster_id += 1
        return labels

    def _compute_cluster_properties(self, X: np.ndarray, labels: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """Compute properties for each cluster: centroids, sizes, and internal distances."""
        unique_labels = np.unique(labels)
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]
        centroids = {}
        sizes = {}
        internal_distances = {}
        for label in unique_labels:
            mask = labels == label
            cluster_points = X[mask]
            centroids[label] = np.mean(cluster_points, axis=0)
            sizes[label] = cluster_points.shape[0]
            if cluster_points.shape[0] > 1:
                dists = pdist(cluster_points, metric=self.metric)
                internal_distances[label] = np.mean(dists)
            else:
                internal_distances[label] = 0.0
        return (centroids, sizes, internal_distances)

    def _compute_interconnectivity(self, X: np.ndarray, labels: np.ndarray, knn_graph: csr_matrix, cluster_i: int, cluster_j: int) -> float:
        """Compute interconnectivity between two clusters."""
        mask_i = labels == cluster_i
        mask_j = labels == cluster_j
        count = 0
        for idx_i in np.where(mask_i)[0]:
            neighbors = knn_graph[idx_i].nonzero()[1]
            for neighbor in neighbors:
                if mask_j[neighbor]:
                    count += 1
        return count

    def _compute_closeness(self, X: np.ndarray, labels: np.ndarray, cluster_i: int, cluster_j: int) -> float:
        """Compute closeness between two clusters."""
        mask_i = labels == cluster_i
        mask_j = labels == cluster_j
        points_i = X[mask_i]
        points_j = X[mask_j]
        if len(points_i) > 0 and len(points_j) > 0:
            distances = cdist(points_i, points_j, metric=self.metric)
            return np.mean(distances)
        else:
            return np.inf

    def _merge_clusters(self, X: np.ndarray, labels: np.ndarray, knn_graph: csr_matrix, centroids: Dict, sizes: Dict, internal_distances: Dict) -> np.ndarray:
        """Merge clusters hierarchically based on relative interconnectivity and closeness."""
        valid_clusters = [label for (label, size) in sizes.items() if size >= self.min_cluster_size]
        if len(valid_clusters) <= self.n_clusters:
            return labels
        merged_labels = labels.copy()
        current_clusters = valid_clusters[:]
        while len(current_clusters) > self.n_clusters:
            best_merge = None
            best_score = -np.inf
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    cluster_i = current_clusters[i]
                    cluster_j = current_clusters[j]
                    e_ij = self._compute_interconnectivity(X, merged_labels, knn_graph, cluster_i, cluster_j)
                    s_ij = self._compute_closeness(X, merged_labels, cluster_i, cluster_j)
                    if e_ij > 0 and s_ij < np.inf and (s_ij > 0):
                        max_e = max(sum((self._compute_interconnectivity(X, merged_labels, knn_graph, cluster_i, c) for c in current_clusters if c != cluster_i)), sum((self._compute_interconnectivity(X, merged_labels, knn_graph, cluster_j, c) for c in current_clusters if c != cluster_j)))
                        min_s = min(internal_distances.get(cluster_i, 0), internal_distances.get(cluster_j, 0))
                        if max_e > 0 and min_s > 0:
                            ri = e_ij / max_e
                            rc = min_s / s_ij
                            score = ri ** self.alpha * rc ** self.beta
                            if score > best_score:
                                best_score = score
                                best_merge = (cluster_i, cluster_j)
            if best_merge is None:
                break
            (cluster_i, cluster_j) = best_merge
            merged_labels[merged_labels == cluster_j] = cluster_i
            mask_i = merged_labels == cluster_i
            centroids[cluster_i] = np.mean(X[mask_i], axis=0)
            sizes[cluster_i] = np.sum(mask_i)
            if sizes[cluster_i] > 1:
                dists = pdist(X[mask_i], metric=self.metric)
                internal_distances[cluster_i] = np.mean(dists)
            else:
                internal_distances[cluster_i] = 0.0
            current_clusters.remove(cluster_j)
        return merged_labels

    def _assign_to_clusters(self, X_train: np.ndarray, labels_train: np.ndarray, X_predict: np.ndarray) -> np.ndarray:
        """Assign new points to the closest cluster."""
        unique_labels = np.unique(labels_train)
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]
        predictions = np.full(X_predict.shape[0], -1, dtype=int)
        for (i, point) in enumerate(X_predict):
            min_distance = np.inf
            assigned_label = -1
            for label in unique_labels:
                cluster_mask = labels_train == label
                cluster_points = X_train[cluster_mask]
                if len(cluster_points) > 0:
                    distances = cdist([point], cluster_points, metric=self.metric).flatten()
                    avg_distance = np.mean(distances)
                    if avg_distance < min_distance:
                        min_distance = avg_distance
                        assigned_label = label
            predictions[i] = assigned_label
        return predictions

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'ChameleonClusteringModel':
        """
        Fit the Chameleon clustering model to the data.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        self : ChameleonClusteringModel
            Fitted estimator.
        """
        X_array = self._validate_input(X)
        self.n_features_in_ = X_array.shape[1]
        if X_array.shape[0] < self.n_clusters:
            self.labels_ = np.arange(X_array.shape[0])
            return self
        knn_graph = self._build_knn_graph(X_array)
        initial_labels = self._graph_partitioning(X_array, knn_graph)
        (centroids, sizes, internal_distances) = self._compute_cluster_properties(X_array, initial_labels)
        self.labels_ = self._merge_clusters(X_array, initial_labels, knn_graph, centroids, sizes, internal_distances)
        self._X_fit = X_array.copy()
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            New data to predict.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' before using 'predict'.")
        X_array = self._validate_input(X)
        return self._assign_to_clusters(self._X_fit, self.labels_, X_array)

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels for X in one step.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return self.fit(X, y, **kwargs).labels_

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Calculate silhouette score for the clustering.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            New data.
        y : ndarray of shape (n_samples,)
            Cluster labels.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        score : float
            Silhouette score of the clustering.
        """
        from sklearn.metrics import silhouette_score
        X_array = self._validate_input(X)
        if len(np.unique(y)) < 2:
            return 0.0
        try:
            return silhouette_score(X_array, y, metric=self.metric)
        except:
            return 0.0