import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet


# ...(code omitted)...


class CLARANSModel(BaseModel):

    def __init__(self, n_clusters: int=3, max_neighbors: int=100, num_local: int=2, distance_func: Optional[Callable[[np.ndarray, np.ndarray], float]]=None, random_state: Optional[int]=None):
        super().__init__(name='CLARANS')
        self.n_clusters = n_clusters
        self.max_neighbors = max_neighbors
        self.num_local = num_local
        self.distance_func = distance_func
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between two points using the specified distance function."""
        if self.distance_func is not None:
            return self.distance_func(a, b)
        return np.sqrt(np.sum((a - b) ** 2))

    def _compute_total_cost(self, X: np.ndarray, medoids: np.ndarray) -> tuple:
        """
        Compute total cost and assignments for given medoids.
        
        Returns
        -------
        tuple
            (total_cost, assignments)
        """
        n_samples = X.shape[0]
        total_cost = 0.0
        assignments = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            min_dist = float('inf')
            best_cluster = 0
            for j in range(len(medoids)):
                medoid_idx = medoids[j]
                dist = self._calculate_distance(X[i], X[medoid_idx])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
            total_cost += min_dist
            assignments[i] = best_cluster
        return (total_cost, assignments)

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'CLARANSModel':
        """
        Compute CLARANS clustering.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster. If FeatureSet, uses its features attribute.
        y : np.ndarray, optional
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters (reserved for future extensions).
            
        Returns
        -------
        CLARANSModel
            Fitted estimator.
        """
        if isinstance(X, FeatureSet):
            X = X.features
        if self.n_clusters > X.shape[0]:
            raise ValueError('n_clusters must be less than or equal to number of samples')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        (n_samples, n_features) = X.shape
        best_medoids = None
        best_cost = float('inf')
        best_assignments = None
        for local_iter in range(self.num_local):
            medoids = np.random.choice(n_samples, size=self.n_clusters, replace=False)
            (current_cost, assignments) = self._compute_total_cost(X, medoids)
            neighbor_count = 0
            while neighbor_count < self.max_neighbors:
                new_medoids = medoids.copy()
                medoid_to_swap = np.random.randint(0, self.n_clusters)
                non_medoids = np.setdiff1d(np.arange(n_samples), medoids)
                if len(non_medoids) == 0:
                    break
                new_medoid = np.random.choice(non_medoids)
                new_medoids[medoid_to_swap] = new_medoid
                (new_cost, new_assignments) = self._compute_total_cost(X, new_medoids)
                neighbor_count += 1
                if new_cost < current_cost:
                    medoids = new_medoids
                    current_cost = new_cost
                    assignments = new_assignments
                    neighbor_count = 0
                elif neighbor_count >= self.max_neighbors:
                    break
            if current_cost < best_cost:
                best_cost = current_cost
                best_medoids = medoids.copy()
                best_assignments = assignments.copy()
        self.cluster_centers_ = X[best_medoids] if best_medoids is not None else None
        self.labels_ = best_assignments
        self.inertia_ = best_cost
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            New data to predict. Must have same number of features as training data.
            
        Returns
        -------
        np.ndarray
            Index of the cluster each sample belongs to.
            
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        if isinstance(X, FeatureSet):
            X = X.features
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            min_dist = float('inf')
            best_cluster = 0
            for j in range(len(self.cluster_centers_)):
                dist = self._calculate_distance(X[i], self.cluster_centers_[j])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
            predictions[i] = best_cluster
        return predictions

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Opposite of the cost function (negative inertia).
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Data to evaluate.
        y : np.ndarray, optional
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        float
            Negative sum of distances to nearest cluster centers.
            
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'score'.")
        if isinstance(X, FeatureSet):
            X = X.features
        total_cost = 0.0
        n_samples = X.shape[0]
        for i in range(n_samples):
            min_dist = float('inf')
            for j in range(len(self.cluster_centers_)):
                dist = self._calculate_distance(X[i], self.cluster_centers_[j])
                if dist < min_dist:
                    min_dist = dist
            total_cost += min_dist
        return -total_cost