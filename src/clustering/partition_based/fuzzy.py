from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class SpatialFuzzyCMeans(BaseModel):

    def __init__(self, n_clusters: int=3, m: float=2.0, spatial_weight: float=0.5, max_iter: int=100, tol: float=0.0001, verbose: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        if n_clusters < 1:
            raise ValueError('n_clusters must be positive')
        if m <= 1:
            raise ValueError('m must be greater than 1')
        if spatial_weight < 0:
            raise ValueError('spatial_weight must be non-negative')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if tol <= 0:
            raise ValueError('tol must be positive')
        self.n_clusters = n_clusters
        self.m = m
        self.spatial_weight = spatial_weight
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X: Union[FeatureSet, np.ndarray], spatial_data: Optional[np.ndarray]=None, **kwargs) -> 'SpatialFuzzyCMeans':
        """
        Compute fuzzy c-means clustering with spatial constraints.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training instances to cluster. If FeatureSet, uses the features attribute.
        spatial_data : Optional[np.ndarray]
            Spatial coordinates or neighborhood information for each sample.
            Shape should match the number of samples in X.
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        SpatialFuzzyCMeans
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If spatial_data shape doesn't match X or if parameters are invalid
        """
        if self.n_clusters <= 0:
            raise ValueError('n_clusters must be positive')
        if self.m <= 1:
            raise ValueError('m must be greater than 1')
        if self.spatial_weight < 0:
            raise ValueError('spatial_weight must be non-negative')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if self.tol <= 0:
            raise ValueError('tol must be positive')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        if spatial_data is not None:
            if len(spatial_data) != len(X_array):
                raise ValueError('spatial_data must have the same number of samples as X')
        (self.n_samples_, self.n_features_) = X_array.shape
        if self.n_clusters > self.n_samples_:
            raise ValueError('n_clusters cannot be larger than number of samples')
        self.membership_ = np.random.rand(self.n_samples_, self.n_clusters)
        self.membership_ = self.membership_ / np.sum(self.membership_, axis=1, keepdims=True)
        self.X_ = X_array.copy()
        self.spatial_data_ = spatial_data.copy() if spatial_data is not None else None
        prev_objective = np.inf
        for iteration in range(self.max_iter):
            self._update_centers()
            objective = self._calculate_objective()
            if abs(prev_objective - objective) < self.tol:
                if self.verbose:
                    print(f'Converged at iteration {iteration}')
                break
            prev_objective = objective
            self._update_memberships()
            if self.verbose and iteration % 10 == 0:
                print(f'Iteration {iteration}: Objective = {objective:.6f}')
        self._update_centers()
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            New data to predict cluster memberships for
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        np.ndarray
            Predicted cluster indices for each sample
        """
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        distances = np.linalg.norm(X_array[:, np.newaxis] - self.cluster_centers_, axis=2)
        if np.any(distances == 0):
            memberships = np.zeros_like(distances)
            memberships[distances == 0] = 1
        else:
            memberships = 1 / distances ** (2 / (self.m - 1))
            memberships = memberships / np.sum(memberships, axis=1, keepdims=True)
        return np.argmax(memberships, axis=1)

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], spatial_data: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Convenience method that combines fit and predict.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training instances to cluster
        spatial_data : Optional[np.ndarray]
            Spatial coordinates or neighborhood information for each sample
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Cluster indices for each sample
        """
        return self.fit(X, spatial_data, **kwargs).predict(X)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Opposite of the value of the objective function.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data to score
        y : Optional[np.ndarray]
            Ignored
        **kwargs : dict
            Additional scoring parameters
            
        Returns
        -------
        float
            Negative of the objective function value
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        distances = np.linalg.norm(X_array[:, np.newaxis] - self.cluster_centers_, axis=2)
        objective = np.sum(self.membership_ ** self.m * distances ** 2)
        if self.spatial_data_ is not None:
            spatial_term = self._calculate_spatial_term()
            objective += self.spatial_weight * spatial_term
        return -objective

    def get_membership_degrees(self) -> np.ndarray:
        """
        Get the fuzzy membership degrees for all samples.
        
        Returns
        -------
        np.ndarray
            Membership degree matrix of shape (n_samples, n_clusters)
        """
        if not hasattr(self, 'membership_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        return self.membership_.copy()

    def _update_centers(self):
        """Update cluster centers based on current membership degrees."""
        memberships_pow = self.membership_ ** self.m
        numerator = memberships_pow.T @ self.X_
        denominator = np.sum(memberships_pow, axis=0, keepdims=True).T
        denominator = np.where(denominator == 0, 1, denominator)
        self.cluster_centers_ = numerator / denominator

    def _update_memberships(self):
        """Update membership degrees based on current cluster centers."""
        distances = np.linalg.norm(self.X_[:, np.newaxis] - self.cluster_centers_, axis=2)
        if self.spatial_data_ is not None:
            spatial_influence = self._calculate_spatial_influence()
        else:
            spatial_influence = 0
        total_distances = distances ** 2 + self.spatial_weight * spatial_influence
        if np.any(total_distances == 0):
            self.membership_[total_distances == 0] = 1
            self.membership_[total_distances != 0] = 0
        else:
            self.membership_ = 1 / total_distances ** (1 / (self.m - 1))
            self.membership_ = self.membership_ / np.sum(self.membership_, axis=1, keepdims=True)

    def _calculate_objective(self) -> float:
        """Calculate the current value of the objective function."""
        distances = np.linalg.norm(self.X_[:, np.newaxis] - self.cluster_centers_, axis=2)
        feature_term = np.sum(self.membership_ ** self.m * distances ** 2)
        if self.spatial_data_ is not None:
            spatial_term = self._calculate_spatial_term()
            return feature_term + self.spatial_weight * spatial_term
        else:
            return feature_term

    def _calculate_spatial_term(self) -> float:
        """Calculate the spatial constraint term of the objective function."""
        if self.spatial_data_ is None:
            return 0
        spatial_influence = self._calculate_spatial_influence()
        return np.sum(self.membership_ ** self.m * spatial_influence)

    def _calculate_spatial_influence(self) -> np.ndarray:
        """Calculate spatial influence matrix."""
        if self.spatial_data_ is None:
            return np.zeros((self.n_samples_, self.n_clusters))
        spatial_distances = np.linalg.norm(self.spatial_data_[:, np.newaxis] - self.spatial_data_[np.newaxis, :], axis=2)
        neighbor_influence = self.membership_[np.newaxis, :, :] * np.exp(-spatial_distances[:, :, np.newaxis])
        spatial_influence = np.sum(neighbor_influence, axis=1)
        return spatial_influence

class FuzzyPossibilisticClustering(BaseModel):
    """
    Fuzzy Possibilistic Clustering algorithm.
    
    Combines concepts from both fuzzy c-means and possibilistic clustering
    to overcome some limitations of each approach. This algorithm is more
    robust to noise and outliers while maintaining the ability to handle
    overlapping clusters.
    
    Unlike traditional fuzzy clustering which focuses on membership degrees
    relative to all clusters, possibilistic clustering considers absolute
    membership degrees, making it less sensitive to distant points.
    
    Attributes
    ----------
    n_clusters : int
        Number of clusters to form
    m : float
        Fuzziness exponent for membership calculation
    eta : float
        Exponent for typicality calculation
    max_iter : int
        Maximum number of iterations
    tol : float
        Relative tolerance for convergence
    verbose : bool
        Whether to print progress information
    """

    def __init__(self, n_clusters: int=3, m: float=2.0, eta: float=2.0, max_iter: int=100, tol: float=0.0001, verbose: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        if n_clusters < 1:
            raise ValueError('n_clusters must be positive')
        if m <= 1:
            raise ValueError('m must be greater than 1')
        if eta <= 1:
            raise ValueError('eta must be greater than 1')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if tol <= 0:
            raise ValueError('tol must be positive')
        self.n_clusters = n_clusters
        self.m = m
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> 'FuzzyPossibilisticClustering':
        """
        Compute fuzzy possibilistic clustering.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training instances to cluster. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        FuzzyPossibilisticClustering
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if self.n_clusters < 1:
            raise ValueError('n_clusters must be positive')
        if self.m <= 1:
            raise ValueError('m must be greater than 1')
        if self.eta <= 1:
            raise ValueError('eta must be greater than 1')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if self.tol <= 0:
            raise ValueError('tol must be positive')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        (self.n_samples_, self.n_features_) = X_array.shape
        if self.n_clusters > self.n_samples_:
            raise ValueError('n_clusters cannot be larger than number of samples')
        self.cluster_centers_ = self._initialize_centers(X_array)
        self.membership_ = np.random.rand(self.n_samples_, self.n_clusters)
        self.membership_ = self.membership_ / np.sum(self.membership_, axis=1, keepdims=True)
        self.typicality_ = np.random.rand(self.n_samples_, self.n_clusters)
        self.typicality_ = self.typicality_ / np.sum(self.typicality_, axis=1, keepdims=True)
        self.X_ = X_array.copy()
        prev_objective = np.inf
        for iteration in range(self.max_iter):
            self._update_centers()
            self._update_memberships()
            self._update_typicalities()
            objective = self._calculate_objective()
            if abs(prev_objective - objective) < self.tol:
                if self.verbose:
                    print(f'Converged at iteration {iteration}')
                break
            prev_objective = objective
            if self.verbose and iteration % 10 == 0:
                print(f'Iteration {iteration}: Objective = {objective:.6f}')
        self._update_centers()
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            New data to predict cluster memberships for
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        np.ndarray
            Predicted cluster indices for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        distances = np.linalg.norm(X_array[:, np.newaxis] - self.cluster_centers_, axis=2)
        if np.any(distances == 0):
            memberships = np.zeros_like(distances)
            memberships[distances == 0] = 1
        else:
            memberships = 1 / distances ** (2 / (self.m - 1))
            memberships = memberships / np.sum(memberships, axis=1, keepdims=True)
        return np.argmax(memberships, axis=1)

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Convenience method that combines fit and predict.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training instances to cluster
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Cluster indices for each sample
        """
        return self.fit(X, **kwargs).predict(X)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Opposite of the value of the objective function.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data to score
        y : Optional[np.ndarray]
            Ignored
        **kwargs : dict
            Additional scoring parameters
            
        Returns
        -------
        float
            Negative of the objective function value
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        objective = self._calculate_objective_for_data(X_array)
        return -objective

    def get_membership_degrees(self) -> np.ndarray:
        """
        Get the fuzzy membership degrees for all samples.
        
        Returns
        -------
        np.ndarray
            Membership degree matrix of shape (n_samples, n_clusters)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        return self.membership_.copy()

    def get_typicality_degrees(self) -> np.ndarray:
        """
        Get the possibilistic typicality degrees for all samples.
        
        Returns
        -------
        np.ndarray
            Typicality degree matrix of shape (n_samples, n_clusters)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        return self.typicality_.copy()

    def _initialize_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers using k-means++ approach.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Initial cluster centers
        """
        (n_samples, n_features) = X.shape
        centers = np.empty((self.n_clusters, n_features))
        centers[0] = X[np.random.randint(n_samples)]
        for c in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - center) ** 2 for center in centers[:c]]) for x in X])
            probabilities = distances / distances.sum()
            cum_probabilities = probabilities.cumsum()
            r = np.random.rand()
            for (i, p) in enumerate(cum_probabilities):
                if r < p:
                    centers[c] = X[i]
                    break
        return centers

    def _update_centers(self):
        """Update cluster centers based on current membership and typicality degrees."""
        membership_pow = self.membership_ ** self.m
        typicality_pow = self.typicality_ ** self.eta
        weights = membership_pow + typicality_pow
        numerator = weights.T @ self.X_
        denominator = np.sum(weights, axis=0, keepdims=True).T
        denominator = np.where(denominator == 0, 1, denominator)
        self.cluster_centers_ = numerator / denominator

    def _update_memberships(self):
        """Update membership degrees based on current cluster centers and typicalities."""
        distances = np.linalg.norm(self.X_[:, np.newaxis] - self.cluster_centers_, axis=2)
        distances = np.where(distances == 0, np.finfo(float).eps, distances)
        denominator = np.sum(self.membership_ ** self.m * distances ** 2 + self.typicality_ ** self.eta * distances ** 2, axis=1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        self.membership_ = 1 / (distances ** 2 + 1e-10) ** (1 / (self.m - 1))
        self.membership_ = self.membership_ / np.sum(self.membership_, axis=1, keepdims=True)

    def _update_typicalities(self):
        """Update typicality degrees based on current cluster centers and memberships."""
        distances = np.linalg.norm(self.X_[:, np.newaxis] - self.cluster_centers_, axis=2)
        distances = np.where(distances == 0, np.finfo(float).eps, distances)
        self.typicality_ = 1 / (distances ** 2 + 1e-10) ** (1 / (self.eta - 1))
        self.typicality_ = self.typicality_ / np.sum(self.typicality_, axis=1, keepdims=True)

    def _calculate_objective(self) -> float:
        """Calculate the current value of the objective function."""
        return self._calculate_objective_for_data(self.X_)

    def _calculate_objective_for_data(self, X: np.ndarray) -> float:
        """Calculate the objective function value for given data."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        membership_term = np.sum(self.membership_ ** self.m * distances ** 2)
        typicality_term = np.sum(self.typicality_ ** self.eta * distances ** 2)
        return membership_term + typicality_term