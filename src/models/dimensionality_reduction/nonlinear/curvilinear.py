from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import pandas as pd

class CurvilinearDistanceAnalyzer(BaseTransformer):

    def __init__(self, n_components: Optional[int]=2, n_neighbors: Optional[int]=None, distance_metric: str='euclidean', embedding_method: str='mds', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.embedding_method = embedding_method
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CurvilinearDistanceAnalyzer':
        """
        Compute curvilinear distances and prepare for transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to analyze. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for fitting (not used).
            
        Returns
        -------
        CurvilinearDistanceAnalyzer
            Self instance for method chaining.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet.')
        (n_samples, n_features) = X.shape
        if self.n_neighbors is None:
            self.n_neighbors_ = min(10, max(n_samples // 10, 2))
        else:
            self.n_neighbors_ = min(self.n_neighbors, n_samples - 1)
        distances = self._compute_pairwise_distances(X, metric=self.distance_metric)
        adjacency = self._build_adjacency_graph(distances, self.n_neighbors_)
        self.curvilinear_distances_ = self._compute_geodesic_distances(adjacency)
        self.n_features_in_ = n_features
        self.training_data_shape_ = X.shape
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Apply dimensionality reduction using computed curvilinear distances.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have same number of features as fitted data.
        **kwargs : dict
            Additional parameters for transformation (not used).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Reduced-dimension representation of the input data.
        """
        if not hasattr(self, 'curvilinear_distances_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' before 'transform'.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet.')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data must have {self.n_features_in_} features, but got {X.shape[1]}.')
        if self.embedding_method == 'mds':
            embedding = self._multidimensional_scaling(self.curvilinear_distances_, n_components=self.n_components)
        else:
            raise ValueError(f'Unsupported embedding method: {self.embedding_method}')
        return embedding

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transformation is not supported for curvilinear distance analysis.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters (not used).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            This method always raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for curvilinear distance analysis.')

    def _compute_pairwise_distances(self, X: np.ndarray, metric: str) -> np.ndarray:
        """
        Compute pairwise distances between all points in the dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Input data array of shape (n_samples, n_features)
        metric : str
            Distance metric to use
            
        Returns
        -------
        np.ndarray
            Pairwise distance matrix of shape (n_samples, n_samples)
        """
        return pairwise_distances(X, metric=metric)

    def _build_adjacency_graph(self, distances: np.ndarray, n_neighbors: int) -> np.ndarray:
        """
        Build adjacency graph based on k-nearest neighbors.
        
        Parameters
        ----------
        distances : np.ndarray
            Pairwise distance matrix of shape (n_samples, n_samples)
        n_neighbors : int
            Number of nearest neighbors to consider
            
        Returns
        -------
        np.ndarray
            Adjacency matrix where connected points have their distances and disconnected points have infinity
        """
        n_samples = distances.shape[0]
        adjacency = np.full_like(distances, np.inf)
        for i in range(n_samples):
            nearest_indices = np.argpartition(distances[i], n_neighbors + 1)[:n_neighbors + 1]
            nearest_indices = nearest_indices[nearest_indices != i][:n_neighbors]
            adjacency[i, nearest_indices] = distances[i, nearest_indices]
            adjacency[nearest_indices, i] = distances[nearest_indices, i]
        return adjacency

    def _compute_geodesic_distances(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Compute geodesic distances using shortest path algorithm (Floyd-Warshall).
        
        Parameters
        ----------
        adjacency : np.ndarray
            Adjacency matrix with direct distances and infinity for non-edges
            
        Returns
        -------
        np.ndarray
            Geodesic distance matrix
        """
        n_samples = adjacency.shape[0]
        geodesic_distances = adjacency.copy()
        np.fill_diagonal(geodesic_distances, 0)
        for k in range(n_samples):
            for i in range(n_samples):
                for j in range(n_samples):
                    if geodesic_distances[i, j] > geodesic_distances[i, k] + geodesic_distances[k, j]:
                        geodesic_distances[i, j] = geodesic_distances[i, k] + geodesic_distances[k, j]
        return geodesic_distances

    def _multidimensional_scaling(self, distance_matrix: np.ndarray, n_components: int) -> np.ndarray:
        """
        Perform classical multidimensional scaling.
        
        Parameters
        ----------
        distance_matrix : np.ndarray
            Distance matrix of shape (n_samples, n_samples)
        n_components : int
            Number of dimensions in the output space
            
        Returns
        -------
        np.ndarray
            Embedded coordinates of shape (n_samples, n_components)
        """
        mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=self.random_state)
        return mds.fit_transform(distance_matrix)