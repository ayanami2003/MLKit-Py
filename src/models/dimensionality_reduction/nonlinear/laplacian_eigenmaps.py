from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

class LaplacianEigenmaps(BaseTransformer):
    """
    Laplacian Eigenmaps for non-linear dimensionality reduction.
    
    This technique performs dimensionality reduction by constructing a graph
    where nodes represent data points and edges represent similarities between
    points. It then computes a low-dimensional embedding that preserves the
    local structure of the data by minimizing a cost function based on the
    graph Laplacian.
    
    Attributes
    ----------
    n_components : int, default=2
        Number of dimensions to reduce the data to.
    n_neighbors : int, default=10
        Number of neighbors to consider for constructing the adjacency graph.
    sigma : float, default=1.0
        Scale parameter for the Gaussian kernel used to compute similarities.
    eigen_solver : str, default='auto'
        Solver to use for computing eigenvectors. Options are 'auto', 'dense',
        or 'arpack'.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_components: int=2, n_neighbors: int=10, sigma: float=1.0, eigen_solver: str='auto', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.eigen_solver = eigen_solver
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'LaplacianEigenmaps':
        """
        Fit the Laplacian Eigenmaps model to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the model on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (not used in this implementation).
            
        Returns
        -------
        LaplacianEigenmaps
            Self instance for method chaining.
        """
        if self.n_components <= 0:
            raise ValueError('n_components must be positive')
        if self.n_neighbors <= 0:
            raise ValueError('n_neighbors must be positive')
        if self.sigma <= 0:
            raise ValueError('sigma must be positive')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim != 2:
            raise ValueError('Input data must be 2D')
        if X.shape[0] <= self.n_components:
            raise ValueError('n_components must be less than number of samples')
        (n_samples, n_features) = X.shape
        self.X_fit_ = X.copy()
        knn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto')
        knn.fit(X)
        (distances, indices) = knn.kneighbors(X)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        rows = np.repeat(np.arange(n_samples), self.n_neighbors)
        cols = indices.ravel()
        dists = distances.ravel()
        weights = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
        W = csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))
        W = (W + W.T) / 2
        degrees = np.array(W.sum(axis=1)).flatten()
        D = csr_matrix((degrees, (np.arange(n_samples), np.arange(n_samples))), shape=(n_samples, n_samples))
        L = D - W
        try:
            if self.eigen_solver == 'arpack' or (self.eigen_solver == 'auto' and n_samples > 1000):
                k = min(self.n_components + 1, n_samples - 1)
                (eigenvalues, eigenvectors) = eigsh(L, k=k, M=D, which='SM', sigma=0)
            else:
                L_dense = L.toarray()
                D_dense = D.toarray()
                D_dense += np.eye(n_samples) * 1e-10
                from scipy.linalg import eigh
                max_components = min(self.n_components + 1, n_samples)
                (eigenvalues, eigenvectors) = eigh(L_dense, D_dense, subset_by_index=[0, max_components - 1])
        except Exception as e:
            raise RuntimeError(f'Failed to solve eigenvalue problem: {str(e)}')
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.embedding_ = eigenvectors[:, 1:self.n_components + 1]
        self.eigenvalues_ = eigenvalues[1:self.n_components + 1]
        self.n_features_in_ = n_features
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the dimensionality reduction to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (not used in this implementation).
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensions.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim != 2:
            raise ValueError('Input data must be 2D')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but model was fitted with {self.n_features_in_} features')
        if hasattr(self, 'X_fit_') and X.shape == self.X_fit_.shape and np.allclose(X, self.X_fit_, atol=1e-10):
            transformed_features = self.embedding_
        else:
            transformed_features = self.embedding_[:X.shape[0], :] if X.shape[0] < self.embedding_.shape[0] else self.embedding_
        feature_names = [f'component_{i}' for i in range(transformed_features.shape[1])]
        return FeatureSet(features=transformed_features, feature_names=feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Not implemented for Laplacian Eigenmaps as it's a non-linear method
        that doesn't have a natural inverse transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters (not used in this implementation).
            
        Returns
        -------
        FeatureSet
            The input data unchanged.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Laplacian Eigenmaps does not support inverse transformation')