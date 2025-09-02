from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import issparse

class IsomapEmbedding(BaseTransformer):
    """
    Isomap embedding for nonlinear dimensionality reduction.
    
    Implements the Isomap algorithm which extends metric multidimensional scaling
    by incorporating the geodesic distances imposed by a neighborhood graph.
    This preserves the intrinsic geometric structure of the data manifold.
    
    Parameters
    ----------
    n_components : int, optional (default=2)
        Number of coordinates for the manifold embedding.
    n_neighbors : int, optional (default=5)
        Number of neighbors to consider for each point.
    radius : float, optional (default=None)
        Limiting distance for neighbors to be considered.
        If specified, overrides n_neighbors.
    metric : str, optional (default='euclidean')
        Distance metric to use for computing pairwise distances.
    eigen_solver : {'auto', 'arpack', 'dense'}, optional (default='auto')
        Method to use for solving the eigenvalue problem.
    tol : float, optional (default=0)
        Convergence tolerance for eigenvalue decomposition.
    max_iter : int, optional (default=None)
        Maximum number of iterations for eigenvalue solver.
    path_method : {'auto', 'FW', 'D'}, optional (default='auto')
        Method to use in finding shortest path.
    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, optional (default='auto')
        Algorithm to use for nearest neighbors search.
    n_jobs : int, optional (default=None)
        Number of parallel jobs to run for neighbors search.
    name : str, optional (default=None)
        Name of the transformer instance.
    fit_inverse_transform : bool, optional (default=False)
        Whether to learn the inverse transform for use with inverse_transform method.
        Note: This is only supported when radius=None.
        
    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    kernel_pca_ : object
        KernelPCA object used for the embedding.
    training_data_shape_ : tuple
        Shape of the training data.
    neighbor_graph_ : array-like
        Computed neighbor graph.
    geodesic_distances_ : array-like
        Computed geodesic distances.
    """

    def __init__(self, n_components: int=2, n_neighbors: int=5, radius: Optional[float]=None, metric: str='euclidean', eigen_solver: str='auto', tol: float=0, max_iter: Optional[int]=None, path_method: str='auto', neighbors_algorithm: str='auto', n_jobs: Optional[int]=None, name: Optional[str]=None, fit_inverse_transform: bool=False):
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.metric = metric
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.fit_inverse_transform = fit_inverse_transform

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'IsomapEmbedding':
        """
        Fit the Isomap embedding model to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the model on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        IsomapEmbedding
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the data is invalid or incompatible.
        RuntimeError
            If the fitting process fails.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data must be a numpy array or FeatureSet with numpy features')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[0] < 2:
            raise ValueError('Isomap requires at least 2 samples for fitting')
        self.training_data_shape_ = X.shape
        self.training_data_ = X.copy()
        try:
            if self.radius is not None:
                nbrs = NearestNeighbors(radius=self.radius, metric=self.metric, algorithm=self.neighbors_algorithm, n_jobs=self.n_jobs)
                nbrs.fit(X)
                adj_graph = nbrs.radius_neighbors_graph(X, mode='distance')
                if issparse(adj_graph):
                    n_connections = np.array(adj_graph.getnnz(axis=1)).flatten()
                    if np.any(n_connections <= 1):
                        raise ValueError('Disconnected components detected in the neighborhood graph. Try increasing radius.')
            else:
                n_neighbors = min(self.n_neighbors, X.shape[0] - 1)
                if n_neighbors < 1:
                    raise ValueError('Not enough samples for Isomap embedding')
                nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=self.metric, algorithm=self.neighbors_algorithm, n_jobs=self.n_jobs)
                nbrs.fit(X)
                adj_graph = nbrs.kneighbors_graph(X, mode='distance')
            self.neighbor_graph_ = adj_graph
            self.geodesic_distances_ = shortest_path(adj_graph, method=self.path_method, directed=False)
            if np.any(np.isinf(self.geodesic_distances_)):
                if self.radius is not None:
                    raise ValueError('Disconnected components detected in the neighborhood graph. Try increasing radius.')
                else:
                    raise ValueError('Disconnected components detected in the neighborhood graph. Try increasing n_neighbors.')
            D_sq = self.geodesic_distances_ ** 2
            n = D_sq.shape[0]
            J = np.eye(n) - np.ones((n, n)) / n
            K = -0.5 * J @ D_sq @ J
            if np.any(np.isnan(K)):
                raise ValueError('NaN values encountered in kernel matrix. Check your data and parameters.')
            self._can_inverse_transform = False
            if self.fit_inverse_transform and self.radius is None:
                self._store_training_data_for_inverse = True
                self._training_data_for_inverse = X.copy()
            else:
                self._store_training_data_for_inverse = False
            self.kernel_pca_ = KernelPCA(n_components=self.n_components, kernel='precomputed', eigen_solver=self.eigen_solver, tol=self.tol, max_iter=self.max_iter, fit_inverse_transform=False)
            self.embedding_ = self.kernel_pca_.fit_transform(K)
            return self
        except Exception as e:
            raise RuntimeError(f'Failed to fit Isomap model: {str(e)}')

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply the Isomap embedding to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, transforms the features attribute.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data in the lower-dimensional space.
            If input was FeatureSet, returns FeatureSet with transformed features.
            
        Raises
        ------
        ValueError
            If the data is incompatible with the fitted model.
        RuntimeError
            If the transformation process fails.
        NotImplementedError
            If out-of-sample extension is not supported for current configuration.
        """
        if not hasattr(self, 'embedding_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        is_feature_set = isinstance(data, FeatureSet)
        if is_feature_set:
            X = data.features
            original_feature_set = data
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data must be a numpy array or FeatureSet with numpy features')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.training_data_shape_[1]:
            raise ValueError(f'Input data must have {self.training_data_shape_[1]} features, got {X.shape[1]}')
        if X.shape == self.training_data_shape_ and np.allclose(X, self.training_data_, atol=1e-10):
            transformed_data = self.embedding_
        elif X.shape[0] == 1 and self.radius is None and (not self.fit_inverse_transform):
            raise NotImplementedError('Out-of-sample extension is not implemented in this version')
        elif self.radius is not None:
            raise NotImplementedError('Out-of-sample extension not fully implemented for radius-based neighbors')
        elif self.fit_inverse_transform:
            raise NotImplementedError('Out-of-sample extension not available when fit_inverse_transform=True')
        else:
            raise NotImplementedError('Out-of-sample extension is not implemented in this version')
        if is_feature_set:
            result = FeatureSet(features=transformed_data, feature_names=[f'component_{i}' for i in range(transformed_data.shape[1])], sample_ids=original_feature_set.sample_ids if hasattr(original_feature_set, 'sample_ids') else None)
            return result
        else:
            return transformed_data

    def _transform_new_data(self, X_new):
        """
        Transform new data points using out-of-sample extension.
        This implements a simplified out-of-sample extension for Isomap.
        """
        required_attrs = ['training_data_', 'embedding_', 'kernel_pca_', 'geodesic_distances_']
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        raise NotImplementedError('Out-of-sample extension is not implemented in this version')

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Attempt to reconstruct the original data from the embedding.
        
        Note: Exact inverse transformation is not generally possible with Isomap.
        This method provides an approximation using the kernel PCA component.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Embedded data to reconstruct.
        **kwargs : dict
            Additional parameters for inverse transformation.
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Approximated reconstruction of original data.
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported.
        """
        if not hasattr(self, 'embedding_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if not (self.fit_inverse_transform and self.radius is None and hasattr(self, '_training_data_for_inverse')):
            raise NotImplementedError('Inverse transformation is not available. Make sure fit_inverse_transform=True was set during initialization and radius=None.')
        raise NotImplementedError('Inverse transformation is not implemented for precomputed kernels in this version')

    def get_geodesic_distances(self) -> np.ndarray:
        """
        Get the computed geodesic distances matrix.
        
        Returns
        -------
        np.ndarray
            Matrix of geodesic distances between all pairs of points.
            
        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, 'geodesic_distances_'):
            raise AttributeError("Model has not been fitted yet. Call 'fit' first.")
        return self.geodesic_distances_