from typing import Optional
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.mathematical_foundations.specialized_functions.distance_metrics import manhattan_distance, cosine_similarity

class DiffusionMapsEmbedding(BaseTransformer):

    def __init__(self, n_components: int=2, alpha: float=0.5, sigma: float=1.0, metric: str='euclidean', eigen_solver: str='auto', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.alpha = alpha
        self.sigma = sigma
        self.metric = metric
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'DiffusionMapsEmbedding':
        """
        Fit the diffusion maps embedding to the input data.
        
        This method computes the Markov transition matrix and its spectral 
        decomposition to prepare for transformation.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with shape (n_samples, n_features)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        DiffusionMapsEmbedding
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data has insufficient samples or features
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance')
        X = data.features
        (n_samples, n_features) = X.shape
        if n_samples < 2:
            raise ValueError('At least 2 samples are required for diffusion maps')
        if n_features < 1:
            raise ValueError('At least 1 feature is required for diffusion maps')
        if self.n_components >= n_samples:
            raise ValueError('n_components must be less than the number of samples')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.metric == 'euclidean':
            X_squared = np.sum(X ** 2, axis=1, keepdims=True)
            distances_sq = X_squared + X_squared.T - 2 * np.dot(X, X.T)
            distances_sq = np.maximum(distances_sq, 0)
            distances = np.sqrt(distances_sq)
        elif self.metric == 'manhattan':
            n = X.shape[0]
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i, j] = manhattan_distance(X[i], X[j])
        elif self.metric == 'cosine':
            n = X.shape[0]
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i, j] = 1 - cosine_similarity(X[i], X[j])
        else:
            raise ValueError(f'Unsupported metric: {self.metric}')
        kernel = np.exp(-distances ** 2 / (2 * self.sigma ** 2))
        q = np.sum(kernel, axis=1)
        if np.any(q <= 0):
            raise ValueError('Zero row sum encountered in kernel matrix')
        kernel_alpha = kernel / np.outer(q ** self.alpha, q ** self.alpha)
        p = np.sum(kernel_alpha, axis=1)
        if np.any(p <= 0):
            raise ValueError('Zero row sum encountered in normalized kernel matrix')
        P = kernel_alpha / p[:, np.newaxis]
        if self.eigen_solver == 'dense' or (self.eigen_solver == 'auto' and n_samples <= 1000):
            (eigenvals, eigenvecs) = np.linalg.eig(P)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
        elif self.eigen_solver in ['arpack', 'auto']:
            try:
                from scipy.sparse.linalg import eigs
                (eigenvals, eigenvecs) = eigs(P, k=min(self.n_components + 10, n_samples - 1), which='LM', return_eigenvectors=True)
                eigenvals = eigenvals.real
                eigenvecs = eigenvecs.real
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
            except ImportError:
                (eigenvals, eigenvecs) = np.linalg.eig(P)
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
        else:
            raise ValueError(f'Unsupported eigen_solver: {self.eigen_solver}')
        self.eigenvalues_ = eigenvals[:self.n_components + 1]
        self.eigenvectors_ = eigenvecs[:, :self.n_components + 1]
        self.X_train_ = X
        self._is_fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the diffusion maps embedding to the input data.
        
        Projects the input data onto the diffusion map coordinates computed 
        during the fit phase.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with shape (n_samples, n_features)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed feature set with shape (n_samples, n_components)
            
        Raises
        ------
        ValueError
            If transformer is not fitted or data dimensions don't match
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance')
        X = data.features
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.X_train_.shape[1]} features')
        if self.metric == 'euclidean':
            X_train = self.X_train_
            X_squared = np.sum(X ** 2, axis=1, keepdims=True)
            X_train_squared = np.sum(X_train ** 2, axis=1, keepdims=True)
            distances_sq = X_squared + X_train_squared.T - 2 * np.dot(X, X_train.T)
            distances_sq = np.maximum(distances_sq, 0)
            distances = np.sqrt(distances_sq)
        elif self.metric == 'manhattan':
            n_new = X.shape[0]
            n_train = self.X_train_.shape[0]
            distances = np.zeros((n_new, n_train))
            for i in range(n_new):
                for j in range(n_train):
                    distances[i, j] = manhattan_distance(X[i], self.X_train_[j])
        elif self.metric == 'cosine':
            n_new = X.shape[0]
            n_train = self.X_train_.shape[0]
            distances = np.zeros((n_new, n_train))
            for i in range(n_new):
                for j in range(n_train):
                    distances[i, j] = 1 - cosine_similarity(X[i], self.X_train_[j])
        else:
            raise ValueError(f'Unsupported metric: {self.metric}')
        kernel = np.exp(-distances ** 2 / (2 * self.sigma ** 2))
        X_train_distances_sq = np.sum(self.X_train_ ** 2, axis=1, keepdims=True) + np.sum(self.X_train_ ** 2, axis=1, keepdims=True).T - 2 * np.dot(self.X_train_, self.X_train_.T)
        X_train_distances_sq = np.maximum(X_train_distances_sq, 0)
        X_train_distances = np.sqrt(X_train_distances_sq)
        q_train = np.sum(np.exp(-X_train_distances ** 2 / (2 * self.sigma ** 2)), axis=1)
        q_new = np.sum(kernel, axis=1)
        kernel_alpha = kernel / np.outer(q_new ** self.alpha, q_train ** self.alpha)
        p_new = np.sum(kernel_alpha, axis=1)
        P_new = kernel_alpha / p_new[:, np.newaxis]
        embedding = P_new @ self.eigenvectors_[:, 1:self.n_components + 1]
        embedded_features = FeatureSet(features=embedding, feature_names=[f'DM{i}' for i in range(1, self.n_components + 1)], feature_types=['numeric'] * self.n_components, sample_ids=data.sample_ids, metadata={'source': 'diffusion_maps_embedding'})
        return embedded_features

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Not implemented for diffusion maps embedding.
        
        Diffusion maps is a non-linear dimensionality reduction technique 
        that generally does not have a well-defined inverse transformation.
        
        Parameters
        ----------
        data : FeatureSet
            Embedded data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            This method always raises NotImplementedError
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not supported
        """
        raise NotImplementedError('Diffusion maps embedding does not support inverse transformation')

    def get_eigenvalues(self) -> Optional[np.ndarray]:
        """
        Get the eigenvalues of the diffusion operator.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of eigenvalues sorted in descending order, or None if not fitted
        """
        if not self._is_fitted:
            return None
        return self.eigenvalues_

    def get_eigenvectors(self) -> Optional[np.ndarray]:
        """
        Get the eigenvectors of the diffusion operator.
        
        Returns
        -------
        Optional[np.ndarray]
            Matrix of eigenvectors (right singular vectors), or None if not fitted
        """
        if not self._is_fitted:
            return None
        return self.eigenvectors_