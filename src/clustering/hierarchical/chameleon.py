from typing import Optional, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN

class ChameleonClusteringModel(BaseModel):
    """
    Chameleon clustering algorithm implementation for hierarchical clustering.
    
    Chameleon is an advanced hierarchical clustering algorithm that dynamically models
    the similarity between clusters based on both interconnectivity and closeness.
    It uses a two-phase approach: first partitioning the data into sub-clusters,
    then applying a hierarchical clustering algorithm that considers both the
    interconnectivity and the closeness of clusters when merging them.
    
    This implementation follows the standard Chameleon methodology:
    1. Partition the dataset into a large number of small sub-clusters
    2. Construct a k-nearest neighbor graph for the sub-clusters
    3. Merge clusters based on a dynamic model of similarity that considers
       both relative interconnectivity and relative closeness
    
    Attributes
    ----------
    n_clusters : int, default=3
        The number of clusters to form.
    k_neighbors : int, default=10
        Number of nearest neighbors to consider when building the kNN graph.
    min_cluster_size : int, default=5
        Minimum size of sub-clusters during the initial partitioning phase.
    alpha : float, default=2.0
        Weight parameter for relative interconnectivity in the merging criterion.
    beta : float, default=1.0
        Weight parameter for relative closeness in the merging criterion.
    
    Examples
    --------
    >>> from general.structures.feature_set import FeatureSet
    >>> import numpy as np
    >>> 
    >>> # Create sample data
    >>> features = np.random.rand(100, 2)
    >>> feature_set = FeatureSet(features=features)
    >>> 
    >>> # Initialize and fit the model
    >>> model = ChameleonClusteringModel(n_clusters=3, k_neighbors=5)
    >>> model.fit(feature_set)
    >>> labels = model.predict(feature_set)
    """

    def __init__(self, n_clusters: int=3, k_neighbors: int=10, min_cluster_size: int=5, alpha: float=2.0, beta: float=1.0, name: Optional[str]=None):
        """
        Initialize the Chameleon clustering model.
        
        Parameters
        ----------
        n_clusters : int, default=3
            The number of clusters to form.
        k_neighbors : int, default=10
            Number of nearest neighbors to consider when building the kNN graph.
        min_cluster_size : int, default=5
            Minimum size of sub-clusters during the initial partitioning phase.
        alpha : float, default=2.0
            Weight parameter for relative interconnectivity in the merging criterion.
        beta : float, default=1.0
            Weight parameter for relative closeness in the merging criterion.
        name : str, optional
            Name of the model instance.
        """
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.min_cluster_size = min_cluster_size
        self.alpha = alpha
        self.beta = beta
        self._is_fitted = False

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'ChameleonClusteringModel':
        """
        Fit the Chameleon clustering model to the data.
        
        This method performs the two-phase clustering approach:
        1. Initial partitioning of data into sub-clusters
        2. Hierarchical merging based on relative interconnectivity and closeness
        
        Parameters
        ----------
        X : FeatureSet
            Input features to cluster. Must contain a 2D array of shape (n_samples, n_features).
        y : Any, optional
            Not used in unsupervised clustering but kept for API consistency.
        **kwargs : dict
            Additional parameters for fitting (currently unused).
            
        Returns
        -------
        ChameleonClusteringModel
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is invalid or incompatible.
        """
        if not isinstance(X, FeatureSet):
            raise ValueError('X must be a FeatureSet instance')
        if X.features is None or len(X.features) == 0:
            raise ValueError('X.features cannot be None or empty')
        self.X_original_ = X.features
        n_samples = len(self.X_original_)
        if n_samples < self.n_clusters:
            raise ValueError(f'Number of samples ({n_samples}) is less than n_clusters ({self.n_clusters})')
        n_initial_clusters = max(n_samples // self.min_cluster_size, self.n_clusters * 2, 10)
        try:
            initial_clusterer = KMeans(n_clusters=n_initial_clusters, random_state=42, n_init=10)
            initial_labels = initial_clusterer.fit_predict(self.X_original_)
        except Exception:
            initial_clusterer = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
            initial_labels = initial_clusterer.fit_predict(self.X_original_)
            unique_labels = np.unique(initial_labels[initial_labels >= 0])
            if len(unique_labels) < self.n_clusters:
                initial_labels = np.arange(n_samples) % n_initial_clusters
        valid_points = initial_labels >= 0
        if not np.all(valid_points):
            self.X_filtered_ = self.X_original_[valid_points]
            initial_labels = initial_labels[valid_points]
        else:
            self.X_filtered_ = self.X_original_.copy()
        unique_labels = np.unique(initial_labels)
        self.subcluster_centroids_ = []
        self.subcluster_sizes_ = []
        self.subcluster_members_ = []
        for label in unique_labels:
            mask = initial_labels == label
            members = np.where(valid_points)[0][mask] if not np.all(valid_points) else np.where(mask)[0]
            centroid = np.mean(self.X_original_[members], axis=0)
            self.subcluster_centroids_.append(centroid)
            self.subcluster_sizes_.append(len(members))
            self.subcluster_members_.append(members)
        self.subcluster_centroids_ = np.array(self.subcluster_centroids_)
        self.subcluster_labels_ = unique_labels
        n_subclusters = len(self.subcluster_centroids_)
        if n_subclusters <= self.n_clusters:
            self.final_labels_ = initial_labels
            self._is_fitted = True
            return self
        distances = cdist(self.subcluster_centroids_, self.subcluster_centroids_)
        np.fill_diagonal(distances, np.inf)
        k = min(self.k_neighbors, n_subclusters - 1)
        knn_indices = np.argpartition(distances, k, axis=1)[:, :k]
        self.graph_ = defaultdict(list)
        for i in range(n_subclusters):
            for j in knn_indices[i]:
                self.graph_[i].append(j)
        self.interconnectivity_ = {}
        self.closeness_ = {}
        for i in range(n_subclusters):
            for j in self.graph_[i]:
                if i < j:
                    (conn, close) = self._calculate_cluster_similarity(i, j, initial_labels)
                    self.interconnectivity_[i, j] = conn
                    self.closeness_[i, j] = close
        self.final_labels_ = self._merge_clusters(initial_labels, unique_labels)
        self._is_fitted = True
        return self

    def _calculate_cluster_similarity(self, cluster_i, cluster_j, labels):
        """
        Calculate interconnectivity and closeness between two clusters.
        
        Parameters
        ----------
        cluster_i : int
            Index of first cluster
        cluster_j : int
            Index of second cluster
        labels : np.ndarray
            Cluster labels for all data points
            
        Returns
        -------
        tuple
            (interconnectivity, closeness) between the two clusters
        """
        members_i = self.subcluster_members_[cluster_i]
        members_j = self.subcluster_members_[cluster_j]
        distances = cdist(self.X_original_[members_i], self.X_original_[members_j])
        avg_distance = np.mean(distances)
        closeness = 1.0 / (1.0 + avg_distance)
        interconnectivity = 0
        for member_i in members_i:
            for member_j in members_j:
                if np.linalg.norm(self.X_original_[member_i] - self.X_original_[member_j]) < avg_distance:
                    interconnectivity += 1
        norm_factor = np.sqrt(len(members_i) * len(members_j))
        interconnectivity = interconnectivity / norm_factor if norm_factor > 0 else 0
        return (interconnectivity, closeness)

    def _merge_clusters(self, initial_labels, unique_labels):
        """
        Perform hierarchical merging of sub-clusters.
        
        Parameters
        ----------
        initial_labels : np.ndarray
            Initial sub-cluster assignments
        unique_labels : np.ndarray
            Unique sub-cluster labels
            
        Returns
        -------
        np.ndarray
            Final cluster labels after merging
        """
        n_subclusters = len(unique_labels)
        if n_subclusters <= self.n_clusters:
            return initial_labels
        cluster_groups = {i: [i] for i in range(n_subclusters)}
        current_n_groups = n_subclusters
        point_to_final_cluster = np.full(len(self.X_original_), -1)
        group_properties = {}
        for i in range(n_subclusters):
            group_properties[i] = {'size': self.subcluster_sizes_[i], 'members': set(self.subcluster_members_[i]), 'centroid': self.subcluster_centroids_[i]}
        while current_n_groups > self.n_clusters:
            best_merge = None
            best_score = -np.inf
            merged_groups = set()
            for i in range(n_subclusters):
                if i in merged_groups:
                    continue
                for j in self.graph_[i]:
                    if j in merged_groups or j <= i:
                        continue
                    group_i_exists = any((i in group for group in cluster_groups.values()))
                    group_j_exists = any((j in group for group in cluster_groups.values()))
                    if not group_i_exists or not group_j_exists:
                        continue
                    group_i_key = None
                    group_j_key = None
                    for (key, group) in cluster_groups.items():
                        if i in group:
                            group_i_key = key
                        if j in group:
                            group_j_key = key
                    if group_i_key is None or group_j_key is None or group_i_key == group_j_key:
                        continue
                    try:
                        interconn = self.interconnectivity_.get((i, j), 0)
                        close = self.closeness_.get((i, j), 0)
                        group_i_size = group_properties[group_i_key]['size']
                        group_j_size = group_properties[group_j_key]['size']
                        rel_interconn = interconn * (1.0 / (1.0 + abs(group_i_size - group_j_size)))
                        rel_close = close * (1.0 / (1.0 + abs(group_i_size - group_j_size)))
                        merge_score = rel_interconn ** self.alpha * rel_close ** self.beta
                        if merge_score > best_score:
                            best_score = merge_score
                            best_merge = (group_i_key, group_j_key)
                    except KeyError:
                        continue
            if best_merge is None:
                break
            (group1_key, group2_key) = best_merge
            cluster_groups[group1_key].extend(cluster_groups[group2_key])
            del cluster_groups[group2_key]
            new_members = group_properties[group1_key]['members'].union(group_properties[group2_key]['members'])
            new_size = len(new_members)
            new_centroid = np.mean(self.X_original_[list(new_members)], axis=0)
            group_properties[group1_key] = {'size': new_size, 'members': new_members, 'centroid': new_centroid}
            del group_properties[group2_key]
            merged_groups.add(group1_key)
            merged_groups.add(group2_key)
            current_n_groups -= 1
        final_label = 0
        for (group_key, subclusters) in cluster_groups.items():
            for subcluster_idx in subclusters:
                members = self.subcluster_members_[subcluster_idx]
                point_to_final_cluster[members] = final_label
            final_label += 1
        unassigned = point_to_final_cluster == -1
        if np.any(unassigned):
            point_to_final_cluster[unassigned] = 0
        return point_to_final_cluster

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for the input data.
        
        Parameters
        ----------
        X : FeatureSet
            Input features to assign to clusters.
        **kwargs : dict
            Additional parameters for prediction (currently unused).
            
        Returns
        -------
        np.ndarray
            Array of cluster labels for each sample in X.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        if not isinstance(X, FeatureSet):
            raise ValueError('X must be a FeatureSet instance')
        if X.features is None:
            raise ValueError('X.features cannot be None')
        if not np.array_equal(X.features, self.X_original_):
            centroids = []
            final_labels = np.unique(self.final_labels_)
            for label in final_labels:
                members = self.final_labels_ == label
                centroid = np.mean(self.X_original_[members], axis=0)
                centroids.append(centroid)
            if len(centroids) > 0:
                centroids = np.array(centroids)
                distances = cdist(X.features, centroids)
                return final_labels[np.argmin(distances, axis=1)]
            else:
                return np.zeros(len(X.features), dtype=int)
        else:
            return self.final_labels_

    def fit_predict(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Parameters
        ----------
        X : FeatureSet
            Input features to cluster.
        y : Any, optional
            Not used in unsupervised clustering but kept for API consistency.
        **kwargs : dict
            Additional parameters for fitting and prediction.
            
        Returns
        -------
        np.ndarray
            Array of cluster labels for each sample in X.
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: FeatureSet, y: Any=None, **kwargs) -> float:
        """
        Calculate silhouette score for the clustering result.
        
        Parameters
        ----------
        X : FeatureSet
            Input features to evaluate.
        y : Any, optional
            Not used in unsupervised clustering but kept for API consistency.
        **kwargs : dict
            Additional parameters for scoring (currently unused).
            
        Returns
        -------
        float
            Silhouette score of the clustering result.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'score'.")
        if not isinstance(X, FeatureSet):
            raise ValueError('X must be a FeatureSet instance')
        if X.features is None:
            raise ValueError('X.features cannot be None')
        labels = self.predict(X)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        if len(unique_labels) > len(X.features) / 2:
            return 0.0
        try:
            return silhouette_score(X.features, labels)
        except Exception:
            return 0.0