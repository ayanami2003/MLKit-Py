import numpy as np
from typing import Optional, Union, Tuple
from general.base_classes.transformer_base import BaseTransformer
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class CentroidInitializer(BaseTransformer):
    """
    Base class for initializing centroids in centroid-based clustering algorithms.
    
    This transformer provides various methods to initialize centroids before
    running iterative clustering algorithms like K-Means. Proper initialization
    can significantly affect convergence speed and final clustering quality.
    
    Attributes:
        n_clusters (int): Number of centroids to initialize
        method (str): Initialization method to use
        random_state (Optional[int]): Random seed for reproducibility
    """

    def __init__(self, n_clusters: int=8, method: str='k-means++', random_state: Optional[int]=None, **kwargs):
        """
        Initialize the CentroidInitializer.
        
        Args:
            n_clusters (int): Number of centroids to initialize. Must be positive.
            method (str): Initialization method. Options include:
                         'k-means++' (default), 'random', 'forgy', 'macqueen'
            random_state (Optional[int]): Random seed for reproducibility.
        """
        super().__init__(name=kwargs.get('name', 'CentroidInitializer'))
        if n_clusters <= 0:
            raise ValueError('n_clusters must be a positive integer')
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.centroids_ = None
        self._supported_methods = {'k-means++', 'random', 'forgy', 'macqueen'}
        if method not in self._supported_methods:
            raise ValueError(f"Method '{method}' is not supported. Choose from {self._supported_methods}")

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CentroidInitializer':
        """
        Fit the initializer by computing initial centroids based on the specified method.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Training data to initialize centroids from.
              If FeatureSet, uses the features attribute.
              
        Returns:
            CentroidInitializer: Fitted initializer with computed centroids.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.n_clusters > X.shape[0]:
            raise ValueError('n_clusters cannot be larger than the number of data points')
        if self.method == 'random':
            self.centroids_ = self._random_init(X)
        elif self.method == 'k-means++':
            self.centroids_ = self._kmeans_plus_plus_init(X)
        elif self.method == 'forgy':
            self.centroids_ = self._forgy_init(X)
        elif self.method == 'macqueen':
            self.centroids_ = self._macqueen_init(X)
        return self

    def _random_init(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids by randomly selecting data points."""
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _forgy_init(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids by randomly selecting data points (Forgy method)."""
        return self._random_init(X)

    def _macqueen_init(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using MacQueen's method (random selection)."""
        return self._random_init(X)

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using the k-means++ algorithm."""
        rng = np.random.default_rng(self.random_state)
        (n_samples, n_features) = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        centroids[0] = X[rng.integers(n_samples)]
        for c in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - centroids[i]) ** 2 for i in range(c)]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = rng.random()
            for (j, p) in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[c] = X[j]
                    break
        return centroids

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Return the initialized centroids.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data used for reference (not modified).
            
        Returns:
            np.ndarray: Array of initialized centroids with shape (n_clusters, n_features).
        """
        if self.centroids_ is None:
            raise ValueError("CentroidInitializer has not been fitted yet. Call 'fit' first.")
        return self.centroids_.copy()

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Not implemented for centroid initialization.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Data to inverse transform.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Returns input unchanged.
        """
        raise NotImplementedError('inverse_transform is not implemented for CentroidInitializer')

    def get_centroids(self) -> Optional[np.ndarray]:
        """
        Get the computed centroids.
        
        Returns:
            Optional[np.ndarray]: Array of centroids or None if not fitted.
        """
        if self.centroids_ is None:
            return None
        return self.centroids_.copy()

class KMeansClusterer(BaseModel):

    def __init__(self, n_clusters: int=8, init_method: str='k-means++', max_iter: int=300, tol: float=0.0001, random_state: Optional[int]=None, **kwargs):
        """
        Initialize the KMeansClusterer.
        
        Args:
            n_clusters (int): Number of clusters to form. Defaults to 8.
            init_method (str): Method for initialization. Options are:
                              'k-means++' (default), 'random', or 'manual'.
            max_iter (int): Maximum number of iterations. Defaults to 300.
            tol (float): Relative tolerance for convergence. Defaults to 1e-4.
            random_state (Optional[int]): Random seed for reproducibility.
        """
        super().__init__(name=kwargs.get('name', 'KMeansClusterer'))
        if n_clusters <= 0:
            raise ValueError('n_clusters must be a positive integer')
        if max_iter <= 0:
            raise ValueError('max_iter must be a positive integer')
        if tol < 0:
            raise ValueError('tol must be non-negative')
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        valid_init_methods = {'k-means++', 'random', 'manual'}
        if init_method not in valid_init_methods:
            raise ValueError(f'init_method must be one of {valid_init_methods}')

    def _convert_input(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, FeatureSet):
            return X.features
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError('Input must be a FeatureSet or numpy array')

    def _initialize_centroids(self, X: np.ndarray, init_centroids=None) -> np.ndarray:
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
        elif self.init_method == 'manual':
            if init_centroids is not None:
                init_centroids = np.array(init_centroids)
                if init_centroids.shape != (self.n_clusters, n_features):
                    raise ValueError(f'init_centroids must have shape ({self.n_clusters}, {n_features})')
                return init_centroids
            else:
                raise ValueError("init_method 'manual' requires 'init_centroids' parameter")
        else:
            raise ValueError(f'Unsupported init_method: {self.init_method}')

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

    def fit(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> 'KMeansClusterer':
        """
        Compute k-means clustering.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training instances to cluster.
              If FeatureSet, uses the features attribute.
              
        Returns:
            KMeansClusterer: Fitted estimator.
        """
        X_data = self._convert_input(X)
        if X_data.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X_data.shape
        if self.n_clusters > n_samples:
            raise ValueError('n_clusters cannot be larger than the number of data points')
        if self.init_method == 'manual':
            if 'init_centroids' in kwargs:
                centroids = np.array(kwargs['init_centroids'])
                if centroids.shape != (self.n_clusters, n_features):
                    raise ValueError(f'init_centroids must have shape ({self.n_clusters}, {n_features})')
            else:
                raise ValueError("init_method 'manual' requires 'init_centroids' parameter")
        else:
            centroids = self._initialize_centroids(X_data, kwargs.get('init_centroids'))
        for _ in range(self.max_iter):
            distances = np.sqrt(((X_data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    new_centroids[k] = X_data[labels == k].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break
            centroids = new_centroids
        inertia = 0.0
        for i in range(n_samples):
            inertia += np.linalg.norm(X_data[i] - centroids[labels[i]]) ** 2
        self.cluster_centers_ = centroids.copy()
        self.labels_ = labels.copy()
        self.inertia_ = inertia
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): New data to predict.
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        X_data = self._convert_input(X)
        distances = np.sqrt(((X_data - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training instances to cluster.
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        return self.fit(X, **kwargs).predict(X, **kwargs)

    def score(self, X: Union[FeatureSet, np.ndarray], y=None, **kwargs) -> float:
        """
        Opposite of the value of X on the K-means objective.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Data to score.
            y: Ignored (present for API compatibility).
            
        Returns:
            float: Negative intra-cluster sum of squares.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        X_data = self._convert_input(X)
        distances = np.sqrt(((X_data - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        inertia = 0.0
        for i in range(len(X_data)):
            inertia += np.linalg.norm(X_data[i] - self.cluster_centers_[labels[i]]) ** 2
        return -inertia

def compute_gap_statistic(data: Union[FeatureSet, np.ndarray], k_range: Union[int, list, range], n_refs: int=10, clusterer: Optional[object]=None, random_state: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the gap statistic for determining optimal number of clusters.
    
    The gap statistic compares the total within intra-cluster variation
    for different values of k with their expected values under null reference
    distribution of the data. The optimal k is the one that gives the maximum
    gap value or the smallest k such that gap(k) >= gap(k+1) - stderr(k+1).
    
    Args:
        data (Union[FeatureSet, np.ndarray]): Input data for clustering.
          If FeatureSet, uses the features attribute.
        k_range (Union[int, list, range]): Range of k values to test.
          Can be a single integer, list of integers, or range object.
        n_refs (int): Number of reference datasets to generate. Defaults to 10.
        clusterer (Optional[object]): Clustering object with fit method.
          Must support n_clusters parameter. Defaults to KMeansClusterer.
        random_state (Optional[int]): Random seed for reproducibility.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - gaps: Array of gap statistic values for each k
            - stderrs: Array of standard errors for each k
            - k_range: Array of k values tested
            
    Raises:
        ValueError: If k_range contains non-positive values or data is empty.
    """
    if isinstance(data, FeatureSet):
        X = data.features
    elif isinstance(data, np.ndarray):
        X = data
    else:
        raise TypeError('Input data must be a FeatureSet or numpy array')
    if X.size == 0:
        raise ValueError('Input data is empty')
    if X.ndim != 2:
        raise ValueError('Input data must be a 2D array')
    if isinstance(k_range, int):
        if k_range <= 0:
            raise ValueError('k_range values must be positive')
        k_values = np.array([k_range])
    else:
        k_values = np.array(list(k_range))
        if len(k_values) == 0:
            raise ValueError('k_range cannot be empty')
        if np.any(k_values <= 0):
            raise ValueError('k_range values must be positive')
    k_values = np.sort(k_values)
    if clusterer is None:
        clusterer = KMeansClusterer
    (n_samples, n_features) = X.shape
    if np.any(k_values > n_samples):
        raise ValueError('k values cannot be larger than the number of data points')
    gaps = np.zeros(len(k_values))
    stderrs = np.zeros(len(k_values))
    wk_refs = np.zeros((len(k_values), n_refs))
    rng = np.random.default_rng(random_state)
    (xmin, xmax) = (X.min(axis=0), X.max(axis=0))

    def _get_clusterer_instance(base_clusterer, n_clusters, rand_state):
        """Helper to get a clusterer instance with specified parameters."""
        if isinstance(base_clusterer, type):
            return base_clusterer(n_clusters=n_clusters, random_state=rand_state)
        elif hasattr(base_clusterer, 'set_params'):
            import copy
            clf_copy = copy.deepcopy(base_clusterer)
            clf_copy.set_params(n_clusters=n_clusters)
            return clf_copy
        else:
            if hasattr(base_clusterer, '__class__'):
                try:
                    return base_clusterer.__class__(n_clusters=n_clusters, random_state=rand_state)
                except:
                    pass
            return base_clusterer
    for (i, k) in enumerate(k_values):
        clf = _get_clusterer_instance(clusterer, k, random_state)
        clf.fit(X)
        wk = _calculate_wcss(X, clf.cluster_centers_, clf.labels_)
        for b in range(n_refs):
            X_ref = rng.uniform(low=xmin, high=xmax, size=(n_samples, n_features))
            ref_clf = _get_clusterer_instance(clusterer, k, random_state)
            ref_clf.fit(X_ref)
            wk_refs[i, b] = _calculate_wcss(X_ref, ref_clf.cluster_centers_, ref_clf.labels_)
        log_wk_refs = np.log(wk_refs[i])
        gap = np.mean(log_wk_refs) - np.log(wk)
        gaps[i] = gap
        sdk = np.sqrt(np.mean((log_wk_refs - np.mean(log_wk_refs)) ** 2))
        stderrs[i] = sdk * np.sqrt(1 + 1 / n_refs)
    return (gaps, stderrs, k_values)