from typing import Optional, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy.spatial.distance import cdist
from scipy.linalg import null_space
from sklearn.neighbors import NearestNeighbors

class HessianEigenmappingExtractor(BaseTransformer):

    def __init__(self, n_components: int=2, neighborhood_size: int=10, alignment_regularization: float=0.001, name: Optional[str]=None):
        """
        Initialize the HessianEigenmappingExtractor.
        
        Parameters
        ----------
        n_components : int, default=2
            Target dimensionality of the embedding space. Must be less than the
            number of samples and original features.
        neighborhood_size : int, default=10
            Size of local neighborhoods for manifold approximation. Should be
            sufficiently large to capture local geometry (typically > 2*n_components).
        alignment_regularization : float, default=1e-3
            Regularization term to ensure numerical stability during alignment.
        name : Optional[str], default=None
            Name identifier for the transformer instance.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.neighborhood_size = neighborhood_size
        self.alignment_regularization = alignment_regularization
        self.feature_names: Optional[List[str]] = None

    def fit(self, data: FeatureSet, **kwargs) -> 'HessianEigenmappingExtractor':
        """
        Fit the Hessian Eigenmapping extractor to the input data.
        
        Constructs local neighborhoods and computes the Hessian-based embedding
        transformation matrix based on the geometric properties of the data manifold.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with shape (n_samples, n_features).
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        HessianEigenmappingExtractor
            The fitted transformer instance.
        """
        X = data.features
        (n_samples, n_features) = X.shape
        if self.n_components >= n_samples or self.n_components >= n_features:
            raise ValueError(f'n_components ({self.n_components}) must be less than both n_samples ({n_samples}) and n_features ({n_features})')
        min_required_neighborhood = self.n_components * (self.n_components + 1) // 2 + self.n_components + 1
        if self.neighborhood_size < min_required_neighborhood:
            raise ValueError(f'neighborhood_size ({self.neighborhood_size}) should be at least {min_required_neighborhood} to capture local geometry')
        if self.neighborhood_size >= n_samples:
            raise ValueError(f'neighborhood_size ({self.neighborhood_size}) must be less than n_samples ({n_samples})')
        nbrs = NearestNeighbors(n_neighbors=self.neighborhood_size + 1, algorithm='auto').fit(X)
        (_, neighbors_indices) = nbrs.kneighbors(X)
        neighbors_indices = neighbors_indices[:, 1:]
        n_points = n_samples
        dim_null_space = self.n_components * (self.n_components + 1) // 2
        alignment_matrix = np.zeros((dim_null_space * n_points, n_points))
        for i in range(n_points):
            neighbor_idx = neighbors_indices[i]
            neighbors = X[neighbor_idx]
            center = X[i]
            Z = neighbors - center
            (U, _, _) = np.linalg.svd(Z, full_matrices=False)
            effective_n_components = min(self.n_components, U.shape[1])
            P = U[:, :effective_n_components]
            if Z.shape[0] > 0 and P.shape[1] > 0:
                if Z.shape[1] >= P.shape[0]:
                    Y = Z @ P
                else:
                    Y = Z @ P[:Z.shape[1], :] if Z.shape[1] < P.shape[0] else Z @ np.eye(Z.shape[1], min(P.shape[1], Z.shape[1]))
            else:
                Y = np.zeros((Z.shape[0], effective_n_components))
            N = Y.shape[0]
            test_functions = np.ones((N, 1 + effective_n_components + dim_null_space))
            test_functions[:, 1:effective_n_components + 1] = Y
            idx = 1 + effective_n_components
            for k1 in range(effective_n_components):
                for k2 in range(k1, effective_n_components):
                    if idx < test_functions.shape[1]:
                        test_functions[:, idx] = Y[:, k1] * Y[:, k2]
                        idx += 1
            W = test_functions[:, :1 + effective_n_components]
            if 1 + effective_n_components < test_functions.shape[1]:
                V = test_functions[:, 1 + effective_n_components:]
            else:
                V = np.empty((N, 0))
            if W.shape[1] > 0 and W.shape[0] >= W.shape[1]:
                (Q_W, _) = np.linalg.qr(W)
                proj = Q_W @ Q_W.T
                V_orth = V - proj @ V
            else:
                V_orth = V.copy()
            if V_orth.shape[1] > 0:
                V_orth_norm = np.linalg.norm(V_orth, axis=0)
                V_orth_norm[V_orth_norm == 0] = 1
                V_orth = V_orth / V_orth_norm
                V_orth += self.alignment_regularization * np.random.randn(*V_orth.shape)
                for j in range(min(dim_null_space, V_orth.shape[1])):
                    if dim_null_space * i + j < alignment_matrix.shape[0] and j < V_orth.shape[1]:
                        alignment_matrix[dim_null_space * i + j, neighbor_idx] = V_orth[:, j]
        try:
            (_, s, VT) = np.linalg.svd(alignment_matrix, full_matrices=True)
            null_space_dim = len(s) - np.sum(s > 1e-10)
            if null_space_dim >= self.n_components and VT.shape[0] >= self.n_components:
                embedding_vectors = VT[-self.n_components:, :].T
            else:
                actual_components = min(self.n_components, null_space_dim, VT.shape[0])
                if actual_components > 0:
                    embedding_vectors = VT[-actual_components:, :].T
                    if actual_components < self.n_components:
                        padding = np.zeros((n_points, self.n_components - actual_components))
                        embedding_vectors = np.hstack([embedding_vectors, padding])
                else:
                    embedding_vectors = np.zeros((n_points, self.n_components))
        except np.linalg.LinAlgError:
            M = alignment_matrix.T @ alignment_matrix
            (eigenvals, eigenvecs) = np.linalg.eigh(M)
            idx = np.argsort(eigenvals)
            if len(idx) >= self.n_components:
                embedding_vectors = eigenvecs[:, idx[:self.n_components]]
            else:
                embedding_vectors = eigenvecs[:, idx]
                if embedding_vectors.shape[1] < self.n_components:
                    padding = np.zeros((n_points, self.n_components - embedding_vectors.shape[1]))
                    embedding_vectors = np.hstack([embedding_vectors, padding])
        self.embedding_vectors_ = embedding_vectors
        self._training_data = X.copy()
        self.feature_names = [f'hessian_component_{i}' for i in range(self.n_components)]
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the Hessian Eigenmapping transformation to reduce dimensionality.
        
        Projects the input data onto the learned low-dimensional embedding space
        that preserves local geometric structures.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with shape (n_samples, n_features) to transform.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed feature set with shape (n_samples, n_components).
        """
        if not hasattr(self, 'embedding_vectors_'):
            raise ValueError("This HessianEigenmappingExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        X = data.features
        (n_samples, n_features) = X.shape
        if n_features != self._training_data.shape[1]:
            raise ValueError(f'X has {n_features} features, but HessianEigenmappingExtractor is expecting {self._training_data.shape[1]} features as input.')
        distances = cdist(X, self._training_data, metric='euclidean')
        weights = 1.0 / (distances + 1e-08)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        embedded_data = weights @ self.embedding_vectors_
        original_name = data.metadata.get('name', 'data') if data.metadata else 'data'
        new_name = f'{original_name}_hessian_eigenmapping' if original_name else None
        transformed_fs = FeatureSet(features=embedded_data, feature_names=self.get_feature_names(), metadata={'name': new_name} if new_name else None)
        return transformed_fs

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Attempt to reconstruct the original space from the embedding.
        
        Note: Exact reconstruction is generally not possible with nonlinear
        dimensionality reduction techniques like Hessian Eigenmapping.
        
        Parameters
        ----------
        data : FeatureSet
            Embedded feature set with shape (n_samples, n_components).
        **kwargs : dict
            Additional inverse transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Approximated reconstruction in original feature space.
        """
        if not hasattr(self, 'embedding_vectors_'):
            raise ValueError("This HessianEigenmappingExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        embedded_data = data.features
        (n_samples, n_components) = embedded_data.shape
        if n_components != self.n_components:
            raise ValueError(f'Embedded data has {n_components} components, but HessianEigenmappingExtractor is expecting {self.n_components} components.')
        embedding_nbrs = NearestNeighbors(n_neighbors=1).fit(self.embedding_vectors_)
        (_, indices) = embedding_nbrs.kneighbors(embedded_data)
        reconstructed_data = self._training_data[indices.ravel()]
        original_name = data.metadata.get('name', 'data') if data.metadata else 'data'
        new_name = f'{original_name}_reconstructed' if original_name else None
        reconstructed_fs = FeatureSet(features=reconstructed_data, feature_names=None, metadata={'name': new_name, 'reconstruction_method': 'nearest_neighbor_mapping'} if new_name else None)
        return reconstructed_fs

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get names of the extracted features.
        
        Returns
        -------
        Optional[List[str]]
            List of feature names in the embedding space, or None if not available.
        """
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            return None
        return self.feature_names.copy()