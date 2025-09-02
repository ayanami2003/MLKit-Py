from typing import Optional, Any
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
import numpy as np


# ...(code omitted)...


class KMediansClusteringModel(BaseModel):
    """
    K-Medians clustering algorithm using Manhattan distance.
    
    This model uses medians instead of means for cluster centers, making it more robust to outliers
    compared to k-means. It minimizes the sum of Manhattan distances between samples and their
    nearest cluster centers.
    
    Attributes:
        n_clusters (int): Number of clusters to form.
        init_method (str): Method for initialization ('k-means++' or 'random').
        max_iter (int): Maximum number of iterations.
        tol (float): Relative tolerance for convergence.
        verbose (int): Verbosity level.
        random_state (Optional[int]): Random seed for reproducibility.
    """

    def __init__(self, n_clusters: int=8, init_method: str='k-means++', max_iter: int=300, tol: float=0.0001, verbose: int=0, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'KMediansClusteringModel':
        """
        Compute k-medians clustering.
        
        Args:
            X (FeatureSet): Input data containing numerical features.
            y (Any, optional): Not used, present for API consistency.
            
        Returns:
            KMediansClusteringModel: Fitted estimator.
        """
        if not isinstance(X, FeatureSet):
            raise TypeError('Input X must be a FeatureSet instance')
        X_array = X.get_feature_matrix()
        if X_array.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.n_clusters <= 0:
            raise ValueError('n_clusters must be a positive integer')
        if self.n_clusters > X_array.shape[0]:
            raise ValueError('n_clusters cannot be larger than the number of samples')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be a positive integer')
        if self.tol < 0:
            raise ValueError('tol must be non-negative')
        rng = np.random.default_rng(self.random_state)
        self.cluster_centers_ = self._initialize_centroids(X_array)
        for iteration in range(self.max_iter):
            (labels, distances) = self._assign_clusters(X_array, self.cluster_centers_)
            new_centroids = np.empty_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    new_centroids[k] = np.median(X_array[labels == k], axis=0)
                else:
                    new_centroids[k] = self.cluster_centers_[k]
            centroid_shift = np.sum(np.abs(new_centroids - self.cluster_centers_))
            if self.verbose > 0:
                print(f'Iteration {iteration + 1}: Centroid shift = {centroid_shift}')
            if centroid_shift <= self.tol:
                if self.verbose > 0:
                    print(f'Converged at iteration {iteration + 1}')
                break
            self.cluster_centers_ = new_centroids
        self.labels_ = labels
        self.inertia_ = np.sum(distances)
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Args:
            X (FeatureSet): New data to predict.
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if not isinstance(X, FeatureSet):
            raise TypeError('Input X must be a FeatureSet instance')
        X_array = X.get_feature_matrix()
        (labels, _) = self._assign_clusters(X_array, self.cluster_centers_)
        return labels

    def fit_predict(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Args:
            X (FeatureSet): Input data.
            y (Any, optional): Not used, present for API consistency.
            
        Returns:
            np.ndarray: Cluster indices.
        """
        return self.fit(X, y, **kwargs).labels_

    def score(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> float:
        """
        Return the clustering cost (total Manhattan distance).
        
        Args:
            X (FeatureSet): Input data.
            y (Any, optional): Not used, present for API consistency.
            
        Returns:
            float: Total Manhattan distance between samples and their closest cluster centers.
        """
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if not isinstance(X, FeatureSet):
            raise TypeError('Input X must be a FeatureSet instance')
        X_array = X.get_feature_matrix()
        (_, distances) = self._assign_clusters(X_array, self.cluster_centers_)
        return np.sum(distances)

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids based on the specified method.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Initial centroids of shape (n_clusters, n_features)
        """
        (n_samples, n_features) = X.shape
        rng = np.random.default_rng(self.random_state)
        if self.init_method == 'random':
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()
        elif self.init_method == 'k-means++':
            return self._kmeans_plus_plus_init(X)
        else:
            raise ValueError("init_method must be 'k-means++' or 'random'")

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the k-means++ method adapted for Manhattan distance.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Initial centroids of shape (n_clusters, n_features)
        """
        (n_samples, n_features) = X.shape
        rng = np.random.default_rng(self.random_state)
        centroids = np.empty((self.n_clusters, n_features))
        centroids[0] = X[rng.integers(n_samples)]
        for c_id in range(1, self.n_clusters):
            distances = np.array([min([np.sum(np.abs(x - c)) for c in centroids[:c_id]]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = rng.random()
            for (j, p) in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[c_id] = X[j]
                    break
        return centroids

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> tuple:
        """
        Assign data points to clusters based on nearest centroid using Manhattan distance.
        
        Args:
            X (np.ndarray): Data points of shape (n_samples, n_features)
            centroids (np.ndarray): Cluster centroids of shape (n_clusters, n_features)
            
        Returns:
            tuple: (labels, distances) where labels is array of cluster indices and 
                   distances is array of Manhattan distances to assigned centroids
        """
        distances = np.abs(X[:, np.newaxis] - centroids).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        point_distances = distances[np.arange(len(X)), labels]
        return (labels, point_distances)