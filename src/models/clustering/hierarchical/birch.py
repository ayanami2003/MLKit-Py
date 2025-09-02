from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.modeling.clustering.k_means.kmeans_algorithm import KMeansClusteringModel


# ...(code omitted)...


class BirchClusteringModel(BaseModel):
    """
    Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) clustering model.
    
    BIRCH is a hierarchical clustering algorithm that is designed to handle large datasets
    efficiently by constructing a tree structure called a Clustering Feature (CF) tree.
    This implementation supports incremental clustering and is particularly effective
    for datasets with well-separated clusters.
    
    Parameters
    ----------
    threshold : float, default=0.5
        The radius of the sub-clusters obtained by CF Tree.
    branching_factor : int, default=50
        Maximum number of CF sub-clusters in each node.
    n_clusters : int or None, default=3
        Number of clusters after the final clustering step.
        If None, the final clustering step is skipped.
    compute_labels : bool, default=True
        Whether to compute labels for the input data.
    copy : bool, default=True
        Whether to make a copy of the input data.
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Centroids of the clusters.
    """

    def __init__(self, threshold: float=0.5, branching_factor: int=50, n_clusters: Optional[int]=3, compute_labels: bool=True, copy: bool=True, name: Optional[str]=None):
        super().__init__(name=name)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy
        self._cf_tree = None
        self.labels_ = None
        self.cluster_centers_ = None
        self._final_clusters = None

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'BirchClusteringModel':
        """
        Build a CF Tree for the input data and perform clustering.
        
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
        self : object
            Fitted estimator.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = X
        if self.copy:
            X_array = X_array.copy()
        if X_array.shape[0] == 0:
            self.labels_ = np.array([])
            self.cluster_centers_ = np.array([]).reshape(0, X_array.shape[1] if X_array.ndim > 1 else 0)
            return self
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        self._cf_tree = CFTree(self.threshold, self.branching_factor)
        for point in X_array:
            self._cf_tree.insert(point)
        subcluster_centroids = self._cf_tree.get_subclusters()
        if len(subcluster_centroids) == 0:
            self.labels_ = np.zeros(X_array.shape[0], dtype=int)
            self.cluster_centers_ = np.array([]).reshape(0, X_array.shape[1])
            return self
        subcluster_centroids = np.array(subcluster_centroids)
        if self.n_clusters is not None and len(subcluster_centroids) > 0:
            from src.data_lifecycle.modeling.clustering.k_means.kmeans_algorithm import KMeansAlgorithm
            kmeans = KMeansAlgorithm(n_clusters=min(self.n_clusters, len(subcluster_centroids)))
            kmeans.fit(subcluster_centroids)
            self._final_clusters = kmeans
            self.cluster_centers_ = kmeans.cluster_centers_
            if self.compute_labels:
                self.labels_ = self.predict(X_array)
        else:
            self.cluster_centers_ = subcluster_centroids
            if self.compute_labels:
                self.labels_ = np.zeros(X_array.shape[0], dtype=int)
                for (i, point) in enumerate(X_array):
                    min_dist = float('inf')
                    best_cluster = 0
                    for (j, centroid) in enumerate(subcluster_centroids):
                        dist = np.linalg.norm(point - centroid)
                        if dist < min_dist:
                            min_dist = dist
                            best_cluster = j
                    self.labels_[i] = best_cluster
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict data using the trained BIRCH model.
        
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
        if self._cf_tree is None:
            raise ValueError("This BirchClusteringModel instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = X
        if self.copy:
            X_array = X_array.copy()
        if X_array.shape[0] == 0:
            return np.array([])
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        subcluster_centroids = self._cf_tree.get_subclusters()
        if len(subcluster_centroids) == 0:
            return np.zeros(X_array.shape[0], dtype=int)
        subcluster_centroids = np.array(subcluster_centroids)
        subcluster_labels = np.zeros(X_array.shape[0], dtype=int)
        for (i, point) in enumerate(X_array):
            min_dist = float('inf')
            best_cluster = 0
            for (j, centroid) in enumerate(subcluster_centroids):
                dist = np.linalg.norm(point - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
            subcluster_labels[i] = best_cluster
        if self._final_clusters is not None and self.n_clusters is not None:
            final_labels = np.zeros(X_array.shape[0], dtype=int)
            subcluster_to_final = self._final_clusters.predict(subcluster_centroids)
            for i in range(len(subcluster_labels)):
                final_labels[i] = subcluster_to_final[subcluster_labels[i]]
            return final_labels
        else:
            return subcluster_labels

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Build a CF Tree for the input data and perform clustering, returning cluster labels.
        
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
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y, **kwargs).labels_

    def partial_fit(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> 'BirchClusteringModel':
        """
        Online learning. Prevents rebuilding of the CF Tree structure.
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            Input data.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = X
        if self.copy:
            X_array = X_array.copy()
        if X_array.shape[0] == 0:
            return self
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if self._cf_tree is None:
            self._cf_tree = CFTree(self.threshold, self.branching_factor)
        for point in X_array:
            self._cf_tree.insert(point)
        return self

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Calculate the negative inertia (sum of squared distances to nearest cluster center).
        
        Parameters
        ----------
        X : FeatureSet or ndarray of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        score : float
            Negative inertia (sum of squared distances to nearest cluster center).
        """
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = X
        if self.copy:
            X_array = X_array.copy()
        if X_array.shape[0] == 0:
            return 0.0
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if self._cf_tree is None:
            raise ValueError("This BirchClusteringModel instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        labels = self.predict(X_array)
        inertia = 0.0
        if self.cluster_centers_ is not None and len(self.cluster_centers_) > 0:
            for (i, point) in enumerate(X_array):
                label = labels[i]
                if label < len(self.cluster_centers_):
                    center = self.cluster_centers_[label]
                    inertia += np.sum((point - center) ** 2)
        return -inertia