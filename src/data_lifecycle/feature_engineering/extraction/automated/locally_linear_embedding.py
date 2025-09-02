from typing import Optional, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class LocallyLinearEmbeddingTransformer(BaseTransformer):

    def __init__(self, n_components: int=2, n_neighbors: int=5, reg: float=0.001, method: str='standard', name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.method = method

    def fit(self, data: FeatureSet, **kwargs) -> 'LocallyLinearEmbeddingTransformer':
        """
        Fit the LLE transformer to the input feature set.
        
        Computes the local reconstruction weights and embedding vectors
        based on the neighborhood relationships in the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with shape (n_samples, n_features).
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        LocallyLinearEmbeddingTransformer
            Fitted transformer instance.
        """
        X = data.features
        (n_samples, n_features) = X.shape
        if n_samples < self.n_neighbors + 1:
            raise ValueError(f'n_samples ({n_samples}) must be greater than n_neighbors ({self.n_neighbors})')
        if self.n_components >= n_features:
            raise ValueError(f'n_components ({self.n_components}) must be less than n_features ({n_features})')
        if self.n_components >= n_samples:
            raise ValueError(f'n_components ({self.n_components}) must be less than n_samples ({n_samples})')
        self.X_train_ = X.copy()
        distances = np.sqrt(((X[:, None] - X[None, :]) ** 2).sum(axis=2))
        np.fill_diagonal(distances, np.inf)
        neighbor_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            neighbors = neighbor_indices[i]
            Z = X[neighbors] - X[i]
            C = Z @ Z.T
            C += np.eye(self.n_neighbors) * self.reg * np.trace(C)
            try:
                w = np.linalg.solve(C, np.ones(self.n_neighbors))
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(C) @ np.ones(self.n_neighbors)
            w = w / w.sum()
            W[i, neighbors] = w
        self.reconstruction_weights_ = W
        I = np.eye(n_samples)
        M = (I - W).T @ (I - W)
        (eigenvals, eigenvecs) = np.linalg.eigh(M)
        idx = np.argsort(eigenvals)[1:self.n_components + 1]
        self.embedded_data_ = eigenvecs[:, idx]
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform the input feature set using the fitted LLE model.
        
        Projects the input data into the lower-dimensional space learned
        during the fitting phase.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with shape (n_samples, n_features).
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed feature set with shape (n_samples, n_components).
        """
        if not hasattr(self, 'reconstruction_weights_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if not hasattr(self, 'X_train_'):
            raise ValueError('Training data not available for out-of-sample transformation.')
        X = data.features
        (n_samples, n_features) = X.shape
        if n_features != self.X_train_.shape[1]:
            raise ValueError(f'Number of features ({n_features}) does not match training data ({self.X_train_.shape[1]})')
        distances = np.sqrt(((X[:, None] - self.X_train_[None, :]) ** 2).sum(axis=2))
        neighbor_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        embedded_new = np.zeros((n_samples, self.n_components))
        for i in range(n_samples):
            neighbors = neighbor_indices[i]
            Z = self.X_train_[neighbors] - X[i]
            C = Z @ Z.T
            C += np.eye(self.n_neighbors) * self.reg * np.trace(C)
            try:
                w = np.linalg.solve(C, np.ones(self.n_neighbors))
            except np.linalg.LinAlgError:
                w = np.linalg.pinv(C) @ np.ones(self.n_neighbors)
            w = w / w.sum()
            embedded_new[i] = self.embedded_data_[neighbors].T @ w
        transformed_features = FeatureSet(features=embedded_new, feature_names=[f'lle_component_{i}' for i in range(self.n_components)], feature_types=['numeric'] * self.n_components, sample_ids=data.sample_ids, metadata={'source': 'LocallyLinearEmbeddingTransformer'})
        return transformed_features

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Attempt to reconstruct the original space from the embedded representation.
        
        Note: Exact inverse transformation is not generally possible with LLE,
        so this method will raise a NotImplementedError.
        
        Parameters
        ----------
        data : FeatureSet
            Embedded feature set with shape (n_samples, n_components).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Reconstructed feature set in original space.
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported.
        """
        raise NotImplementedError('Exact inverse transformation is not supported for LLE.')