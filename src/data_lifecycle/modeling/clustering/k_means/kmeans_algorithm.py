from typing import Optional, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class KMeansClusteringModel(BaseModel):
    """
    K-Means clustering model for partitioning data into K distinct clusters.
    
    This class implements the K-Means clustering algorithm, which partitions
    data points into K clusters by minimizing the variance within each cluster.
    The algorithm works by iteratively assigning data points to the nearest
    cluster centroid and then updating the centroids based on the mean of
    assigned points.
    
    Attributes:
        n_clusters (int): Number of clusters to form.
        init (str): Method for initialization ('k-means++', 'random', or ndarray).
        n_init (int): Number of times the algorithm runs with different centroid seeds.
        max_iter (int): Maximum number of iterations for a single run.
        tol (float): Relative tolerance for declaring convergence.
        random_state (Optional[int]): Random seed for reproducibility.
        algorithm (str): K-Means algorithm variant to use ('lloyd', 'elkan').
        
    Example:
        >>> model = KMeansClusteringModel(n_clusters=3)
        >>> model.fit(feature_set)
        >>> labels = model.predict(feature_set)
    """

    def __init__(self, n_clusters: int=8, init: str='k-means++', n_init: int=10, max_iter: int=300, tol: float=0.0001, random_state: Optional[int]=None, algorithm: str='lloyd', name: Optional[str]=None):
        """
        Initialize the K-Means clustering model.
        
        Args:
            n_clusters (int): Number of clusters to form. Must be > 0.
            init (str): Method for initialization:
                - 'k-means++': Select initial cluster centers using sampling based on
                  variance of data points (default)
                - 'random': Choose k observations (rows) at random from data for
                  initial centroids
                - ndarray: Custom initial centroids as a (n_clusters, n_features) array
            n_init (int): Number of times the k-means algorithm will be run with
                different centroid seeds. The final results will be the best output
                in terms of inertia.
            max_iter (int): Maximum number of iterations of the k-means algorithm
                for a single run.
            tol (float): Relative tolerance with regards to Frobenius norm of the
                difference in the cluster centers of two consecutive iterations
                to declare convergence.
            random_state (Optional[int]): Determines random number generation for
                centroid initialization. Use an int to make the randomness
                deterministic.
            algorithm (str): K-means algorithm to use:
                - 'lloyd': Standard EM-style algorithm
                - 'elkan': Uses triangle inequality to speed up convergence
            name (Optional[str]): Optional name for the model instance.
        """
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.algorithm = algorithm
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids based on the specified method.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Initial centroids of shape (n_clusters, n_features)
        """
        (n_samples, n_features) = X.shape
        if isinstance(self.init, np.ndarray):
            if self.init.shape != (self.n_clusters, n_features):
                raise ValueError('Custom centroids must have shape (n_clusters, n_features)')
            return self.init.copy()
        elif self.init == 'random':
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()
        elif self.init == 'k-means++':
            return self._kmeans_plus_plus_init(X)
        else:
            raise ValueError("init must be 'k-means++', 'random', or a numpy array")

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the k-means++ method.
        
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
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids[:c_id]]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = rng.random()
            for (j, p) in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[c_id] = X[j]
                    break
        return centroids

    def _lloyd_iteration(self, X: np.ndarray, centroids: np.ndarray) -> tuple:
        """
        Perform one iteration of Lloyd's algorithm.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            centroids (np.ndarray): Current centroids of shape (n_clusters, n_features)
            
        Returns:
            tuple: Updated centroids, labels, and inertia
        """
        (n_samples, n_features) = X.shape
        n_clusters = centroids.shape[0]
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.empty_like(centroids)
        inertia = 0.0
        for k in range(n_clusters):
            if np.sum(labels == k) > 0:
                new_centroids[k] = X[labels == k].mean(axis=0)
                inertia += ((X[labels == k] - new_centroids[k]) ** 2).sum()
            else:
                new_centroids[k] = centroids[k]
        return (new_centroids, labels, inertia)

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'KMeansClusteringModel':
        """
        Compute k-means clustering.
        
        Args:
            X (FeatureSet): Training instances with features to cluster.
            y (Optional[Any]): Not used, present for API consistency.
            **kwargs: Additional fitting parameters.
            
        Returns:
            KMeansClusteringModel: Fitted estimator.
            
        Raises:
            ValueError: If n_clusters is larger than the number of samples.
        """
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_data = X.features
        (n_samples, n_features) = X_data.shape
        if self.n_clusters > n_samples:
            raise ValueError(f'n_clusters ({self.n_clusters}) cannot be larger than number of samples ({n_samples})')
        if self.n_clusters <= 0:
            raise ValueError('n_clusters must be positive')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if self.tol < 0:
            raise ValueError('tol must be non-negative')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X_data)
            for _ in range(self.max_iter):
                old_centroids = centroids.copy()
                (centroids, labels, inertia) = self._lloyd_iteration(X_data, centroids)
                if np.linalg.norm(centroids - old_centroids) < self.tol:
                    break
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.is_fitted = True
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Args:
            X (FeatureSet): New data to predict cluster labels for.
            **kwargs: Additional prediction parameters.
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_data = X.features
        distances = np.sqrt(((X_data - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels

    def fit_predict(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Convenience method that combines fit and predict operations.
        
        Args:
            X (FeatureSet): Training instances with features to cluster.
            y (Optional[Any]): Not used, present for API consistency.
            **kwargs: Additional fitting/prediction parameters.
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> float:
        """
        Calculate negative inertia (opposite of sum of squared distances to nearest cluster).
        
        Used for model evaluation where higher values indicate better clustering.
        
        Args:
            X (FeatureSet): Test samples to evaluate.
            y (Optional[Any]): Not used, present for API consistency.
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: Negative inertia (negative sum of squared distances to nearest cluster).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_data = X.features
        distances = np.sqrt(((X_data - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        inertia = 0.0
        for i in range(len(X_data)):
            inertia += np.linalg.norm(X_data[i] - self.cluster_centers_[labels[i]]) ** 2
        return -inertia