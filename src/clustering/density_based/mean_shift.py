from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class MeanShiftWithAdaptiveBandwidth(BaseModel):

    def __init__(self, bandwidth: Optional[float]=None, seeds: Optional[np.ndarray]=None, cluster_all: bool=True, max_iterations: int=300, convergence_tolerance: float=0.0001, name: Optional[str]=None):
        """
        Initialize the Mean Shift with Adaptive Bandwidth clustering model.
        
        Parameters
        ----------
        bandwidth : float, optional
            The bandwidth parameter for the Gaussian kernel. If None, it will be
            automatically estimated based on the data using adaptive methods.
        seeds : array-like, optional
            Custom initial points for the algorithm. If None, points are sampled
            from the data.
        cluster_all : bool, default=True
            Whether to cluster all points (True) or only points within bandwidth
            reach of a seed (False).
        max_iterations : int, default=300
            Maximum number of iterations for the algorithm.
        convergence_tolerance : float, default=1e-4
            Tolerance for stopping criterion.
        name : str, optional
            Name of the clustering model instance.
        """
        super().__init__(name=name)
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.cluster_all = cluster_all
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance

    def _extract_features(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Extract features from input data.
        
        Parameters
        ----------
        X : FeatureSet or array-like
            Input data
            
        Returns
        -------
        np.ndarray
            Feature matrix
        """
        if isinstance(X, FeatureSet):
            return X.features
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError('X must be a FeatureSet or numpy array')

    def _estimate_bandwidth(self, X: np.ndarray) -> float:
        """
        Estimate bandwidth using the median of pairwise distances.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
            
        Returns
        -------
        float
            Estimated bandwidth
        """
        (n_samples, n_features) = X.shape
        if n_samples == 1:
            return 1.0
        if n_samples <= 1000:
            sample_indices = np.arange(n_samples)
        else:
            sample_indices = np.random.choice(n_samples, size=1000, replace=False)
        sample_X = X[sample_indices]
        n_sample = sample_X.shape[0]
        distances = np.sqrt(((sample_X[:, np.newaxis, :] - sample_X[np.newaxis, :, :]) ** 2).sum(axis=2))
        triu_indices = np.triu_indices(n_sample, k=1)
        pairwise_distances = distances[triu_indices]
        if len(pairwise_distances) == 0:
            return 1e-06
        if np.all(pairwise_distances == 0):
            return 1e-06
        if n_samples == 2:
            return float(pairwise_distances[0] / 2.0)
        bandwidth = np.median(pairwise_distances)
        if bandwidth <= 0:
            return 1e-06
        return float(bandwidth)

    def _gaussian_kernel(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian kernel values.
        
        Parameters
        ----------
        distances : np.ndarray
            Distances to compute kernel for
            
        Returns
        -------
        np.ndarray
            Kernel values
        """
        return np.exp(-0.5 * (distances / self.bandwidth) ** 2)

    def _mean_shift_step(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Perform one step of mean shift update.
        
        Parameters
        ----------
        X : np.ndarray
            Data points
        centroids : np.ndarray
            Current centroids
            
        Returns
        -------
        np.ndarray
            Updated centroids
        """
        (n_points, n_features) = X.shape
        n_centroids = centroids.shape[0]
        distances = np.sqrt(((centroids[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
        weights = self._gaussian_kernel(distances)
        weighted_sum = np.dot(weights, X)
        weight_sum = weights.sum(axis=1)[:, np.newaxis]
        weight_sum = np.where(weight_sum == 0, 1, weight_sum)
        new_centroids = weighted_sum / weight_sum
        return new_centroids

    def fit(self, X: Union[FeatureSet, np.ndarray], y=None, **kwargs) -> 'MeanShiftWithAdaptiveBandwidth':
        """
        Fit the Mean Shift clustering model according to the given training data.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            Training instances to cluster. If FeatureSet is provided, uses its
            features attribute.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        MeanShiftWithAdaptiveBandwidth
            Fitted estimator.
        """
        X_array = self._extract_features(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[0] == 0:
            raise ValueError('X must contain at least one sample')
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X_array)
        if self.seeds is None:
            seeds = X_array.copy()
        else:
            seeds = self.seeds
        if not isinstance(seeds, np.ndarray):
            seeds = np.array(seeds)
        if seeds.ndim != 2:
            raise ValueError('seeds must be a 2D array')
        if seeds.shape[1] != X_array.shape[1]:
            raise ValueError('seeds must have the same number of features as X')
        centroids = seeds.copy()
        for iteration in range(self.max_iterations):
            new_centroids = self._mean_shift_step(X_array, centroids)
            centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids
            if centroid_shift < self.convergence_tolerance:
                break
        unique_centroids = []
        for centroid in centroids:
            is_duplicate = False
            for unique_centroid in unique_centroids:
                if np.linalg.norm(centroid - unique_centroid) < self.convergence_tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_centroids.append(centroid)
        self.cluster_centers_ = np.array(unique_centroids)
        self.labels_ = self._assign_labels(X_array)
        self.is_fitted = True
        return self

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Assign labels to data points based on nearest cluster centers.
        
        Parameters
        ----------
        X : np.ndarray
            Data points
            
        Returns
        -------
        np.ndarray
            Labels for each point
        """
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        if len(self.cluster_centers_) == 0:
            return labels
        distances = np.sqrt(((X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        if not self.cluster_all:
            min_distances = np.min(distances, axis=1)
            labels[min_distances > self.bandwidth] = -1
        return labels

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            New data to predict. If FeatureSet is provided, uses its features
            attribute.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before making predictions')
        X_array = self._extract_features(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError('X must have the same number of features as the training data')
        labels = self._assign_labels(X_array)
        return labels

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y=None, **kwargs) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Convenience method; equivalent to calling fit(X) followed by predict(X).
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            Training instances to cluster. If FeatureSet is provided, uses its
            features attribute.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional fitting/prediction parameters.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: Union[FeatureSet, np.ndarray], y=None, **kwargs) -> float:
        """
        Opposite of the value of X on the Kullback-Leibler divergence.
        
        This score is useful for selecting the optimal number of clusters
        in Mean Shift clustering.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            Test samples. If FeatureSet is provided, uses its features attribute.
        y : Ignored
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Negative KL divergence between data distribution and clustered distribution.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before scoring')
        X_array = self._extract_features(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError('X must have the same number of features as the training data')
        labels = self.predict(X_array)
        total_distance = 0.0
        n_points = X_array.shape[0]
        if n_points == 0:
            return 0.0
        for i in range(len(self.cluster_centers_)):
            cluster_points = X_array[labels == i]
            if len(cluster_points) > 0:
                distances = np.sqrt(((cluster_points[:, np.newaxis, :] - self.cluster_centers_[i]) ** 2).sum(axis=2)).flatten()
                total_distance += distances.sum()
        return -total_distance / n_points if n_points > 0 else 0.0