from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class DensityPeakClustering(BaseModel):

    def __init__(self, cutoff_distance: float=0.5, gaussian_kernel: bool=False, density_threshold: Optional[float]=None, distance_threshold: Optional[float]=None, max_clusters: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.cutoff_distance = cutoff_distance
        self.gaussian_kernel = gaussian_kernel
        self.density_threshold = density_threshold
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        self._cluster_centers = None
        self._labels = None
        self._densities = None
        self._deltas = None

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

    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all points.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Pairwise distance matrix of shape (n_samples, n_samples)
        """
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _compute_local_densities(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute local density for each point.
        
        Parameters
        ----------
        distances : np.ndarray
            Pairwise distance matrix
        
        Returns
        -------
        np.ndarray
            Local density values for each point
        """
        if self.gaussian_kernel:
            densities = np.sum(np.exp(-(distances / self.cutoff_distance) ** 2), axis=1) - 1
        else:
            densities = np.sum(distances < self.cutoff_distance, axis=1) - 1
        return densities

    def _compute_minimum_distances(self, densities: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance to any point with higher density.
        
        Parameters
        ----------
        densities : np.ndarray
            Local density values for each point
        distances : np.ndarray
            Pairwise distance matrix
            
        Returns
        -------
        np.ndarray
            Minimum distances to higher density points
        """
        n_points = len(densities)
        deltas = np.full(n_points, np.max(distances))
        for i in range(n_points):
            higher_density_indices = np.where(densities > densities[i])[0]
            if len(higher_density_indices) > 0:
                deltas[i] = np.min(distances[i, higher_density_indices])
        max_density_idx = np.argmax(densities)
        deltas[max_density_idx] = np.max(distances[max_density_idx, :])
        return deltas

    def _determine_thresholds(self, densities: np.ndarray, deltas: np.ndarray):
        """
        Determine density and distance thresholds for cluster center selection.
        
        Parameters
        ----------
        densities : np.ndarray
            Local density values
        deltas : np.ndarray
            Minimum distances to higher density points
        """
        if self.density_threshold is None:
            self.density_threshold = np.percentile(densities, 75)
        if self.distance_threshold is None:
            self.distance_threshold = np.percentile(deltas, 75)

    def _find_cluster_centers(self, densities: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        Find cluster centers based on density and distance thresholds.
        
        Parameters
        ----------
        densities : np.ndarray
            Local density values
        deltas : np.ndarray
            Minimum distances to higher density points
            
        Returns
        -------
        np.ndarray
            Indices of cluster centers
        """
        potential_centers = np.where((densities >= self.density_threshold) & (deltas >= self.distance_threshold))[0]
        if self.max_clusters is not None and len(potential_centers) > self.max_clusters:
            scores = densities[potential_centers] * deltas[potential_centers]
            sorted_indices = np.argsort(scores)[::-1]
            potential_centers = potential_centers[sorted_indices[:self.max_clusters]]
        return potential_centers

    def _assign_points_to_clusters(self, X: np.ndarray, centers: np.ndarray, densities: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """
        Assign all points to clusters based on nearest higher-density neighbor.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        centers : np.ndarray
            Indices of cluster centers
        densities : np.ndarray
            Local density values
        distances : np.ndarray
            Pairwise distance matrix
            
        Returns
        -------
        np.ndarray
            Cluster labels for each point
        """
        n_points = X.shape[0]
        labels = np.full(n_points, -1)
        for (i, center_idx) in enumerate(centers):
            labels[center_idx] = i
        nearest_higher_density = np.full(n_points, -1)
        for i in range(n_points):
            higher_density_indices = np.where(densities > densities[i])[0]
            if len(higher_density_indices) > 0:
                nearest_idx = higher_density_indices[np.argmin(distances[i, higher_density_indices])]
                nearest_higher_density[i] = nearest_idx
        for i in range(n_points):
            if labels[i] != -1:
                continue
            current = i
            visited = set()
            while current != -1 and current not in visited:
                visited.add(current)
                if labels[current] != -1:
                    labels[i] = labels[current]
                    break
                current = nearest_higher_density[current]
            if labels[i] == -1 and len(centers) > 0:
                center_distances = distances[i, centers]
                nearest_center = centers[np.argmin(center_distances)]
                labels[i] = labels[nearest_center]
        return labels

    def fit(self, X: Union[FeatureSet, np.ndarray], y=None, **kwargs) -> 'DensityPeakClustering':
        """
        Fit the density peak clustering model to the input data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input data of shape (n_samples, n_features). If FeatureSet is provided,
            only the features attribute will be used.
        y : Ignored
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters (not used in this implementation)
            
        Returns
        -------
        DensityPeakClustering
            Fitted clustering model
            
        Raises
        ------
        ValueError
            If the input data is invalid or incompatible
        """
        X_features = self._extract_features(X)
        if X_features.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet with 2D features')
        if X_features.shape[0] < 2:
            raise ValueError('X must contain at least 2 samples')
        self._X = X_features.copy()
        distances = self._compute_pairwise_distances(X_features)
        self._densities = self._compute_local_densities(distances)
        self._deltas = self._compute_minimum_distances(self._densities, distances)
        self._determine_thresholds(self._densities, self._deltas)
        center_indices = self._find_cluster_centers(self._densities, self._deltas)
        self._cluster_centers = X_features[center_indices].copy()
        self._labels = self._assign_points_to_clusters(X_features, center_indices, self._densities, distances)
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            New data points to assign to clusters
        **kwargs : dict
            Additional prediction parameters (not used in this implementation)
            
        Returns
        -------
        np.ndarray
            Array of cluster labels for each data point
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        X_features = self._extract_features(X)
        if X_features.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet with 2D features')
        if X_features.shape[1] != self._X.shape[1]:
            raise ValueError('Number of features in X must match the training data')
        n_new_points = X_features.shape[0]
        n_training_points = self._X.shape[0]
        distances = np.zeros((n_new_points, n_training_points))
        for i in range(n_new_points):
            diff = X_features[i] - self._X
            distances[i] = np.sqrt(np.sum(diff ** 2, axis=1))
        nearest_training_point = np.argmin(distances, axis=1)
        labels = self._labels[nearest_training_point]
        return labels

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input data to fit and predict
        y : Ignored
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Array of cluster labels for each data point
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: Union[FeatureSet, np.ndarray], y=None, **kwargs) -> float:
        """
        Calculate silhouette score for the clustering result.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input data
        y : Ignored
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional scoring parameters (not used in this implementation)
            
        Returns
        -------
        float
            Silhouette score of the clustering (-1 to 1, higher is better)
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'score'.")
        try:
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError('scikit-learn is required for silhouette score calculation')
        X_features = self._extract_features(X)
        labels = self.predict(X)
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0
        filtered_X = X_features[mask]
        filtered_labels = labels[mask]
        unique_labels = np.unique(filtered_labels)
        if len(unique_labels) < 2:
            return 0.0
        if len(unique_labels) > len(filtered_X) - 1:
            return 0.0
        return silhouette_score(filtered_X, filtered_labels)

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get the coordinates of cluster centers.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of cluster center coordinates, or None if not fitted
        """
        return self._cluster_centers

    def get_density_values(self) -> Optional[np.ndarray]:
        """
        Get local density values for all data points.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of local density values, or None if not fitted
        """
        return self._densities

    def get_distance_values(self) -> Optional[np.ndarray]:
        """
        Get minimum distances to higher density points for all data points.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of minimum distances, or None if not fitted
        """
        return self._deltas