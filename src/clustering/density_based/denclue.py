from typing import Optional, Any
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
import numpy as np

class DenclueClusteringModel(BaseModel):

    def __init__(self, sigma: float=0.5, xi: float=0.1, threshold: float=0.0001, max_iterations: int=100, name: Optional[str]=None):
        """
        Initialize the DENCLUE clustering model.
        
        Parameters
        ----------
        sigma : float, default=0.5
            Width parameter for the Gaussian kernel function (influence radius).
            Larger values capture more global structure, smaller values focus on local density.
        xi : float, default=0.1
            Minimum density threshold for a point to be considered a cluster center.
            Points with density below this value are not considered significant.
        threshold : float, default=1e-4
            Convergence threshold for the hill-climbing procedure.
            Smaller values lead to more precise convergence but may increase computation time.
        max_iterations : int, default=100
            Maximum number of iterations for the hill-climbing procedure.
            Prevents infinite loops in non-converging scenarios.
        name : str, optional
            Name identifier for the model instance
        """
        super().__init__(name=name)
        if sigma <= 0:
            raise ValueError('Sigma must be positive')
        if xi < 0:
            raise ValueError('Xi must be non-negative')
        if threshold <= 0:
            raise ValueError('Threshold must be positive')
        if max_iterations <= 0:
            raise ValueError('Max iterations must be positive')
        self.sigma = sigma
        self.xi = xi
        self.threshold = threshold
        self.max_iterations = max_iterations

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'DenclueClusteringModel':
        """
        Fit the DENCLUE clustering model to the input data.
        
        This method performs density estimation using Gaussian kernels and identifies
        cluster centers through hill-climbing optimization. Each data point is assigned
        to the cluster of its corresponding density attractor.
        
        Parameters
        ----------
        X : FeatureSet
            Input feature set with shape (n_samples, n_features)
        y : Any, optional
            Target values (ignored in unsupervised clustering)
        **kwargs : dict
            Additional fitting parameters (reserved for future extensions)
            
        Returns
        -------
        DenclueClusteringModel
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data is invalid or parameters are out of acceptable ranges
        """
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet')
        X_array = X.features
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        (n_samples, n_features) = X_array.shape
        if n_samples == 0:
            raise ValueError('X cannot be empty')
        self.X_ = X_array
        self.attractors_ = []
        self.attractor_labels_ = []
        for i in range(n_samples):
            point = X_array[i]
            attractor = self._hill_climb(point)
            found = False
            for (j, existing_attractor) in enumerate(self.attractors_):
                if np.linalg.norm(attractor - existing_attractor) < self.threshold:
                    self.attractor_labels_.append(j)
                    found = True
                    break
            if not found:
                self.attractors_.append(attractor)
                self.attractor_labels_.append(len(self.attractors_) - 1)
        self.attractors_ = np.array(self.attractors_)
        attractor_densities = np.array([self._density(attractor) for attractor in self.attractors_])
        valid_attractors = attractor_densities >= self.xi
        index_mapping = {}
        new_index = 0
        for (i, valid) in enumerate(valid_attractors):
            if valid:
                index_mapping[i] = new_index
                new_index += 1
            else:
                index_mapping[i] = -1
        self.labels_ = np.array([index_mapping[label] for label in self.attractor_labels_])
        self.valid_attractors_ = self.attractors_[valid_attractors]
        self.attractor_densities_ = attractor_densities[valid_attractors]
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for the input data.
        
        Assigns each data point to the cluster of its corresponding density attractor
        based on the previously fitted model.
        
        Parameters
        ----------
        X : FeatureSet
            Input feature set with shape (n_samples, n_features)
        **kwargs : dict
            Additional prediction parameters (reserved for future extensions)
            
        Returns
        -------
        np.ndarray
            Array of cluster labels with shape (n_samples,) where -1 indicates noise points
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted before prediction
        """
        if not hasattr(self, 'valid_attractors_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'predict'")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet')
        X_array = X.features
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        n_samples = X_array.shape[0]
        labels = np.full(n_samples, -1)
        if len(self.valid_attractors_) == 0:
            return labels
        for i in range(n_samples):
            point = X_array[i]
            attractor = self._hill_climb(point)
            distances = np.linalg.norm(self.valid_attractors_ - attractor, axis=1)
            nearest_attractor = np.argmin(distances)
            if self.attractor_densities_[nearest_attractor] >= self.xi:
                labels[i] = nearest_attractor
        return labels

    def score(self, X: FeatureSet, y: Any=None, **kwargs) -> float:
        """
        Calculate the silhouette score for the clustering result.
        
        Evaluates the quality of clustering by measuring how similar each point is
        to its own cluster compared to other clusters.
        
        Parameters
        ----------
        X : FeatureSet
            Input feature set with shape (n_samples, n_features)
        y : Any, optional
            True labels for supervised evaluation (not used in unsupervised clustering)
        **kwargs : dict
            Additional scoring parameters (reserved for future extensions)
            
        Returns
        -------
        float
            Silhouette score ranging from -1 to 1, where higher values indicate better clustering
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted before scoring
        """
        if not hasattr(self, 'labels_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'score'")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet')
        X_array = X.features
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        n_samples = X_array.shape[0]
        if n_samples != len(self.labels_):
            raise ValueError('X must have the same number of samples as the fitted data')
        unique_labels = np.unique(self.labels_[self.labels_ != -1])
        if len(unique_labels) < 2:
            return 0.0
        silhouette_scores = np.zeros(n_samples)
        for i in range(n_samples):
            label = self.labels_[i]
            if label == -1:
                silhouette_scores[i] = 0
                continue
            same_cluster = self.labels_ == label
            same_cluster[i] = False
            if np.sum(same_cluster) == 0:
                a_i = 0
            else:
                a_i = np.mean(np.linalg.norm(X_array[same_cluster] - X_array[i], axis=1))
            b_i = np.inf
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_cluster = self.labels_ == other_label
                if np.sum(other_cluster) > 0:
                    avg_distance = np.mean(np.linalg.norm(X_array[other_cluster] - X_array[i], axis=1))
                    b_i = min(b_i, avg_distance)
            if a_i < b_i:
                s_i = 1 - a_i / b_i if b_i > 0 else 0
            elif a_i > b_i:
                s_i = b_i / a_i - 1 if a_i > 0 else 0
            else:
                s_i = 0
            silhouette_scores[i] = s_i
        return float(np.mean(silhouette_scores))

    def _density(self, point: np.ndarray) -> float:
        """
        Calculate the density at a given point using Gaussian kernel density estimation.
        
        Parameters
        ----------
        point : np.ndarray
            Point at which to calculate the density
            
        Returns
        -------
        float
            Density at the given point
        """
        if not hasattr(self, 'X_'):
            raise RuntimeError('Model has not been fitted yet')
        diff = self.X_ - point
        squared_distances = np.sum(diff ** 2, axis=1)
        kernel_values = np.exp(-squared_distances / (2 * self.sigma ** 2))
        d = self.X_.shape[1]
        normalization = (2 * np.pi) ** (d / 2) * self.sigma ** d
        return np.sum(kernel_values) / normalization

    def _hill_climb(self, point: np.ndarray) -> np.ndarray:
        """
        Perform hill-climbing optimization to find the density attractor for a point.
        
        Parameters
        ----------
        point : np.ndarray
            Starting point for hill-climbing
            
        Returns
        -------
        np.ndarray
            Density attractor (local maximum of the density function)
        """
        current_point = point.copy()
        for _ in range(self.max_iterations):
            gradient = self._density_gradient(current_point)
            next_point = current_point + gradient
            if np.linalg.norm(next_point - current_point) < self.threshold:
                break
            current_point = next_point
        return current_point

    def _density_gradient(self, point: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the density function at a given point.
        
        Parameters
        ----------
        point : np.ndarray
            Point at which to calculate the gradient
            
        Returns
        -------
        np.ndarray
            Gradient vector at the given point
        """
        diff = self.X_ - point
        squared_distances = np.sum(diff ** 2, axis=1)
        kernel_values = np.exp(-squared_distances / (2 * self.sigma ** 2))
        d = self.X_.shape[1]
        normalization = (2 * np.pi) ** (d / 2) * self.sigma ** d
        gradient_contributions = kernel_values[:, np.newaxis] * -diff / (self.sigma ** 2 * normalization)
        return np.mean(gradient_contributions, axis=0)