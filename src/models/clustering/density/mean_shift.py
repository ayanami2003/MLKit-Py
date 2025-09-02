from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet


# ...(code omitted)...


class MeanShiftWithBandwidthOptimization(BaseModel):

    def __init__(self, bandwidth: Optional[float]=None, seeds: Optional[np.ndarray]=None, bin_seeding: bool=False, min_bin_freq: int=1, cluster_all: bool=True, max_iter: int=300, optimization_method: str='silverman', tolerance: float=0.0001):
        """
        Initialize the Mean Shift clustering with bandwidth optimization.
        
        Parameters
        ----------
        bandwidth : float, optional
            Starting bandwidth for the kernel. If None, it will be estimated
        seeds : array-like, optional
            Seeds used to initialize kernels. If None, calculated by binning
        bin_seeding : bool, default=False
            If true, initial kernel locations are not locations of all points
            but rather the location of the discretized version of points
        min_bin_freq : int, default=1
            To speed up calculations, accept only those bins with at least
            min_bin_freq points as seeds
        cluster_all : bool, default=True
            Cluster all points, including noisy points (points not within any kernel)
        max_iter : int, default=300
            Maximum number of iterations
        optimization_method : str, default='silverman'
            Method to use for bandwidth optimization ('silverman', 'scott', 'cross-validation')
        """
        super().__init__(name='MeanShiftWithBandwidthOptimization')
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.max_iter = max_iter
        self.optimization_method = optimization_method
        self.tolerance = tolerance
        self.cluster_centers = None
        self.labels = None
        self.n_iter = 0

    def _validate_inputs(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array if needed."""
        if isinstance(X, np.ndarray):
            return X
        elif hasattr(X, 'features'):
            if hasattr(X.features, 'values'):
                return X.features.values
            else:
                return np.array(X.features)
        else:
            raise TypeError('Input must be numpy array or FeatureSet with features attribute')

    def _silverman_bandwidth(self, X: np.ndarray) -> float:
        """
        Estimate bandwidth using Silverman's rule of thumb.
        
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
        if n_samples <= 1:
            return 0.0
        stds = np.std(X, axis=0)
        factor = (4.0 / (n_features + 2)) ** (1.0 / (n_features + 4))
        bandwidth = factor * n_samples ** (-1.0 / (n_features + 4)) * np.mean(stds)
        return bandwidth

    def _scott_bandwidth(self, X: np.ndarray) -> float:
        """
        Estimate bandwidth using Scott's rule.
        
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
        if n_samples <= 1:
            return 0.0
        stds = np.std(X, axis=0)
        factor = n_samples ** (-1.0 / (n_features + 4))
        bandwidth = factor * np.mean(stds)
        return bandwidth

    def _cross_validation_bandwidth(self, X: np.ndarray, n_folds: int=5) -> float:
        """
        Estimate bandwidth using cross-validation approach.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        n_folds : int, default=5
            Number of folds for cross-validation
            
        Returns
        -------
        float
            Estimated bandwidth
        """
        n_samples = X.shape[0]
        if n_samples < n_folds:
            return self._silverman_bandwidth(X)
        silverman_bw = self._silverman_bandwidth(X)
        bandwidth_candidates = np.logspace(np.log10(silverman_bw * 0.1), np.log10(silverman_bw * 10), 20)
        best_bandwidth = silverman_bw
        best_score = -np.inf
        for _ in range(3):
            indices = np.random.permutation(n_samples)
            fold_scores = []
            for fold in range(n_folds):
                test_start = fold * (n_samples // n_folds)
                test_end = test_start + n_samples // n_folds
                if fold == n_folds - 1:
                    test_end = n_samples
                test_indices = indices[test_start:test_end]
                train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
                if len(train_indices) == 0 or len(test_indices) == 0:
                    continue
                X_train = X[train_indices]
                X_test = X[test_indices]
                for bw in bandwidth_candidates:
                    try:
                        (centers, _, _) = self._mean_shift_core(X_train, bw)
                        if len(centers) > 0:
                            distances = np.sqrt(((X_test[:, np.newaxis] - centers) ** 2).sum(axis=2))
                            min_distances = np.min(distances, axis=1)
                            score = -np.mean(min_distances)
                            fold_scores.append((bw, score))
                    except:
                        continue
            if fold_scores:
                scores_by_bandwidth = {}
                for (bw, score) in fold_scores:
                    if bw not in scores_by_bandwidth:
                        scores_by_bandwidth[bw] = []
                    scores_by_bandwidth[bw].append(score)
                for bw in scores_by_bandwidth:
                    avg_score = np.mean(scores_by_bandwidth[bw])
                    if avg_score > best_score:
                        best_score = avg_score
                        best_bandwidth = bw
            if best_bandwidth != silverman_bw:
                break
        return best_bandwidth

    def estimate_bandwidth(self, X: Union[FeatureSet, np.ndarray]) -> float:
        """
        Estimate the optimal bandwidth for the Mean Shift algorithm.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input data for bandwidth estimation
            
        Returns
        -------
        float
            Estimated optimal bandwidth
        """
        X_array = self._validate_inputs(X)
        if self.optimization_method == 'silverman':
            return self._silverman_bandwidth(X_array)
        elif self.optimization_method == 'scott':
            return self._scott_bandwidth(X_array)
        elif self.optimization_method == 'cross-validation':
            return self._cross_validation_bandwidth(X_array)
        else:
            raise ValueError(f'Unknown optimization method: {self.optimization_method}')

    def _bin_seeds(self, X: np.ndarray, bin_size: float) -> np.ndarray:
        """
        Bin points to create seed points for mean shift.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        bin_size : float
            Size of bins for seeding
            
        Returns
        -------
        np.ndarray
            Seed points
        """
        if bin_size <= 0:
            return X
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        bin_counts = np.ceil((maxs - mins) / bin_size).astype(int)
        indices = ((X - mins) / bin_size).astype(int)
        indices = np.clip(indices, 0, bin_counts - 1)
        flat_indices = np.ravel_multi_index(indices.T, bin_counts)
        (unique_flat_indices, counts) = np.unique(flat_indices, return_counts=True)
        valid_bins = counts >= self.min_bin_freq
        if not np.any(valid_bins):
            return X
        valid_flat_indices = unique_flat_indices[valid_bins]
        valid_indices = np.column_stack(np.unravel_index(valid_flat_indices, bin_counts))
        seeds = []
        for i in range(len(valid_indices)):
            bin_idx = valid_indices[i]
            mask = np.all(indices == bin_idx, axis=1)
            if np.sum(mask) > 0:
                seeds.append(np.mean(X[mask], axis=0))
        if len(seeds) == 0:
            return X
        return np.array(seeds)

    def _mean_shift_core(self, X: np.ndarray, bandwidth: float, seeds: Optional[np.ndarray]=None) -> tuple:
        """
        Core Mean Shift algorithm implementation.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        bandwidth : float
            Bandwidth parameter
            
        Returns
        -------
        np.ndarray
            Final cluster centers
        """
        if bandwidth <= 0:
            raise ValueError('Bandwidth must be positive')
        (n_samples, n_features) = X.shape
        if seeds is not None:
            current_centers = seeds.copy()
        elif self.seeds is not None:
            current_centers = self.seeds.copy()
        elif self.bin_seeding:
            bin_size = bandwidth / 2.0
            current_centers = self._bin_seeds(X, bin_size)
        else:
            current_centers = X.copy()
        if len(current_centers) == 0:
            return (np.empty((0, n_features)), np.full(n_samples, -1), 0)
        n_iter = 0
        for iter_count in range(self.max_iter):
            new_centers = []
            for center in current_centers:
                distances_squared = np.sum((X - center) ** 2, axis=1)
                weights = np.exp(-distances_squared / (2 * bandwidth ** 2))
                if np.sum(weights) > 0:
                    new_center = np.average(X, weights=weights, axis=0)
                    new_centers.append(new_center)
                else:
                    new_centers.append(center)
            new_centers = np.array(new_centers)
            if len(new_centers) == 0:
                break
            shift = np.linalg.norm(new_centers - current_centers)
            current_centers = new_centers
            n_iter = iter_count + 1
            if shift < self.tolerance:
                break
        labels = self._assign_labels(X, current_centers)
        return (current_centers, labels, n_iter)

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Assign labels to points based on closest cluster center.
        
        Parameters
        ----------
        X : np.ndarray
            Input data points
        centers : np.ndarray
            Cluster centers
            
        Returns
        -------
        np.ndarray
            Labels for each point
        """
        if len(centers) == 0:
            return np.full(X.shape[0], -1)
        distances = np.sqrt(((X[:, np.newaxis] - centers) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def fit(self, X: Union[np.ndarray, FeatureSet], y=None, **kwargs) -> 'MeanShiftWithBandwidthOptimization':
        """
        Perform Mean Shift clustering on the input data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster. If FeatureSet, uses features attribute
        y : Ignored
            Not used, present here for API consistency by convention
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        MeanShiftWithBandwidthOptimization
            Fitted estimator
        """
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[0] == 0:
            raise ValueError('Cannot fit MeanShift on empty dataset')
        self.n_features_in_ = X_array.shape[1]
        if self.bandwidth is None:
            self.bandwidth_ = self.estimate_bandwidth(X_array)
        else:
            self.bandwidth_ = self.bandwidth
        if self.bandwidth_ <= 0:
            raise ValueError('Bandwidth must be positive')
        if self.bin_seeding:
            bin_size = self.bandwidth_ / 2.0
            seeds = self._bin_seeds(X_array, bin_size)
        else:
            seeds = None
        (self.cluster_centers_, self.labels_, self.n_iter_) = self._mean_shift_core(X_array, self.bandwidth_, seeds)
        self._is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            New data to predict. If FeatureSet, uses features attribute
            
        Returns
        -------
        np.ndarray
            Index of the cluster each sample belongs to
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before making predictions')
        X_array = self._validate_inputs(X)
        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            return np.full(X_array.shape[0], -1)
        return self._assign_labels(X_array, self.cluster_centers)

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Compute clustering and predict on the same data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster. If FeatureSet, uses features attribute
        y : Ignored
            Not used, present here for API consistency by convention
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        np.ndarray
            Cluster labels for each point
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> float:
        """
        Return the negative inertia (sum of squared distances to nearest cluster center).
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Test features
        y : np.ndarray
            True target values (ignored for unsupervised learning)
        **kwargs : dict
            Additional evaluation parameters
            
        Returns
        -------
        float
            Negative inertia (higher is better)
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before scoring')
        X_array = self._validate_inputs(X)
        if self.cluster_centers is None or len(self.cluster_centers) == 0:
            return -np.inf
        distances = np.sqrt(((X_array[:, np.newaxis] - self.cluster_centers) ** 2).sum(axis=2))
        min_distances = np.min(distances, axis=1)
        inertia = np.sum(min_distances ** 2)
        return -inertia