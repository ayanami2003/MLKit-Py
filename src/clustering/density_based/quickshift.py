import numpy as np
from typing import Optional, Union
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.mathematical_foundations.specialized_functions.distance_metrics import manhattan_distance


class QuickShiftClusteringModel(BaseModel):
    """
    QuickShift clustering algorithm implementation.
    
    QuickShift is a mode-seeking clustering algorithm that works by identifying modes (cluster centers)
    in the data density and connecting nearby points to these modes. It is particularly effective for
    datasets with irregularly shaped clusters and varying densities.
    
    This implementation follows the standard QuickShift approach where each data point is connected
    to its nearest neighbor with a higher density, forming a forest of trees whose roots are the modes.
    
    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for kernel density estimation. Controls the scale at which density is estimated.
    max_distance : float, default=np.inf
        Maximum distance to consider neighbors. Points farther than this are not considered.
    density_threshold : float, default=0.0
        Minimum density value required for a point to be considered a mode (cluster center).
    metric : str, default='euclidean'
        Distance metric to use for computing distances between points.
    name : Optional[str], default=None
        Name identifier for the model instance.
    
    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point.
    is_fitted : bool
        Whether the model has been fitted.
    """

    def __init__(self, bandwidth: float=1.0, max_distance: float=np.inf, density_threshold: float=0.0, metric: str='euclidean', name: Optional[str]=None):
        super().__init__(name=name)
        if bandwidth <= 0:
            raise ValueError('bandwidth must be positive.')
        self.bandwidth = bandwidth
        self.max_distance = max_distance
        self.density_threshold = density_threshold
        if metric not in ['euclidean']:
            raise ValueError(f'Unsupported metric: {metric}')
        self.metric = metric
        self.cluster_centers_ = None
        self.labels_ = None

    def _validate_data(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Convert input data to numpy array if needed."""
        if isinstance(X, FeatureSet):
            return X.features
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array.')

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'QuickShiftClusteringModel':
        """
        Fit the QuickShift clustering model to the data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training instances to cluster. If FeatureSet, uses features attribute.
        y : Optional[np.ndarray], default=None
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        QuickShiftClusteringModel
            Fitted estimator.
        """
        X = self._validate_data(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        distances = pairwise_distances(X, metric=self.metric)
        density = np.sum(np.exp(-0.5 * (distances / self.bandwidth) ** 2), axis=1)
        parents = -np.ones(n_samples, dtype=int)
        for i in range(n_samples):
            mask_higher_density = density > density[i]
            if not np.any(mask_higher_density):
                continue
            mask_within_distance = distances[i] <= self.max_distance
            combined_mask = mask_higher_density & mask_within_distance
            if np.any(combined_mask):
                candidates = np.where(combined_mask)[0]
                j = candidates[np.argmin(distances[i, candidates])]
                parents[i] = j
        is_mode = density >= self.density_threshold
        labels = -np.ones(n_samples, dtype=int)
        next_label = 0
        point_to_root = np.arange(n_samples)
        for i in range(n_samples):
            current = i
            path = [current]
            while parents[current] != -1 and point_to_root[current] == current:
                current = parents[current]
                path.append(current)
            root = point_to_root[current]
            for p in path:
                point_to_root[p] = root
        unique_roots = np.unique(point_to_root)
        root_to_label = {}
        for root in unique_roots:
            if is_mode[root]:
                root_to_label[root] = next_label
                next_label += 1
            else:
                root_to_label[root] = -1
        for i in range(n_samples):
            root = point_to_root[i]
            labels[i] = root_to_label.get(root, -1)
        if next_label == 0:
            self.cluster_centers_ = np.empty((0, n_features))
        else:
            mode_indices = np.array([i for i in range(n_samples) if is_mode[i] and labels[i] != -1])
            if len(mode_indices) > 0:
                self.cluster_centers_ = X[mode_indices]
            else:
                self.cluster_centers_ = np.empty((0, n_features))
        self.labels_ = labels
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            New data to predict. If FeatureSet, uses features attribute.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        X = self._validate_data(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.cluster_centers_.shape[0] == 0:
            return np.full(X.shape[0], -1)
        distances_to_centers = pairwise_distances(X, self.cluster_centers_, metric=self.metric)
        return np.argmin(distances_to_centers, axis=1)

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels for the input data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training instances to cluster. If FeatureSet, uses features attribute.
        y : Optional[np.ndarray], default=None
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        return self.fit(X, y, **kwargs).labels_

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Return the negative inertia (sum of squared distances to nearest cluster center).
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data to evaluate. If FeatureSet, uses features attribute.
        y : Optional[np.ndarray], default=None
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Negative inertia (higher is better).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        X = self._validate_data(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.cluster_centers_.shape[0] == 0:
            return 0.0
        distances_to_centers = pairwise_distances(X, self.cluster_centers_, metric=self.metric)
        min_distances = np.min(distances_to_centers, axis=1)
        return -np.sum(min_distances ** 2)