from typing import Optional, List
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.mathematical_foundations.optimization_methods.numerical_methods import BFGSOptimizer

class TopologicallyConstrainedIsometricEmbedding(BaseTransformer):

    def __init__(self, n_components: int=2, n_neighbors: int=10, topology_weight: float=0.5, max_iter: int=100, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.topology_weight = topology_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, data: FeatureSet, **kwargs) -> 'TopologicallyConstrainedIsometricEmbedding':
        """
        Fit the transformer to the input data by computing the topologically constrained embedding.

        Parameters
        ----------
        data : FeatureSet
            High-dimensional input data to embed. Must contain a feature matrix of shape
            (n_samples, n_features).
        **kwargs : dict
            Additional parameters for fitting (ignored).

        Returns
        -------
        TopologicallyConstrainedIsometricEmbedding
            Fitted transformer instance.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = data.features
        (n_samples, n_features) = X.shape
        self._training_data_shape = X.shape
        self._original_feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(n_features)]
        self._original_feature_types = data.feature_types if data.feature_types is not None else ['numeric'] * n_features
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto')
        nbrs.fit(X)
        knn_graph = nbrs.kneighbors_graph(mode='distance')
        geodesic_distances = shortest_path(csgraph=csr_matrix(knn_graph), directed=False, method='D')
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        D_squared = geodesic_distances ** 2
        B = -0.5 * H @ D_squared @ H
        (eigenvals, eigenvecs) = np.linalg.eigh(B)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        eigenvals = eigenvals[:self.n_components]
        eigenvecs = eigenvecs[:, :self.n_components]
        initial_embedding = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))

        def objective_function(embedding_flat):
            Y = embedding_flat.reshape(n_samples, self.n_components)
            embedded_distances = cdist(Y, Y, metric='euclidean')
            isometric_loss = np.sum((geodesic_distances - embedded_distances) ** 2)
            knn_indices = nbrs.kneighbors(return_distance=False)
            topological_loss = 0.0
            for i in range(n_samples):
                for j in knn_indices[i]:
                    if i != j:
                        diff = Y[i] - Y[j]
                        topological_loss += np.sum(diff ** 2)
            return isometric_loss + self.topology_weight * topological_loss

        def gradient_function(embedding_flat):
            Y = embedding_flat.reshape(n_samples, self.n_components)
            grad = np.zeros_like(Y)
            embedded_distances = cdist(Y, Y, metric='euclidean')
            np.fill_diagonal(embedded_distances, 1.0)
            ratio = (geodesic_distances - embedded_distances) / (embedded_distances + 1e-12)
            np.fill_diagonal(ratio, 0.0)
            for i in range(n_samples):
                diff = Y[i] - Y
                grad[i] = -2 * np.sum(ratio[i][:, np.newaxis] * diff, axis=0)
            knn_indices = nbrs.kneighbors(return_distance=False)
            for i in range(n_samples):
                for j in knn_indices[i]:
                    if i != j:
                        diff = Y[i] - Y[j]
                        grad[i] += 2 * self.topology_weight * diff
                        grad[j] -= 2 * self.topology_weight * diff
            return grad.ravel()
        optimizer = BFGSOptimizer(max_iterations=self.max_iter, tolerance=1e-06, verbose=False)
        optimizer.fit(objective_function=objective_function, gradient_function=gradient_function, initial_point=initial_embedding.ravel())
        self.embedding_ = optimizer._optimal_x.reshape(n_samples, self.n_components)
        self._training_data = X.copy()
        self._nbrs = nbrs
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the learned embedding transformation to new data.

        Parameters
        ----------
        data : FeatureSet
            Input data to transform. Must have the same number of features as the training data.
        **kwargs : dict
            Additional parameters for transformation (ignored).

        Returns
        -------
        FeatureSet
            Transformed data in the lower-dimensional space.
        """
        if not hasattr(self, '_training_data_shape'):
            raise ValueError('Transformer has not been fitted yet.')
        X = data.features
        if X.shape[1] != self._training_data_shape[1]:
            raise ValueError(f'Input data must have {self._training_data_shape[1]} features, got {X.shape[1]}')
        (distances, indices) = self._nbrs.kneighbors(X, return_distance=True)
        weights = 1.0 / (distances + 1e-08)
        weights = weights / weights.sum(axis=1, keepdims=True)
        transformed_features = np.zeros((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            transformed_features[i] = np.average(self.embedding_[indices[i]], axis=0, weights=weights[i])
        return FeatureSet(features=transformed_features, feature_names=[f'topological_component_{i}' for i in range(self.n_components)], feature_types=['numeric'] * self.n_components, sample_ids=data.sample_ids, metadata={'source': 'TopologicallyConstrainedIsometricEmbedding.transform'})

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Attempt to reconstruct the original high-dimensional data from the embedding.

        Note: Reconstruction may not be exact due to the non-linear and lossy nature of the embedding.

        Parameters
        ----------
        data : FeatureSet
            Embedded data to reconstruct.
        **kwargs : dict
            Additional parameters for inverse transformation (ignored).

        Returns
        -------
        FeatureSet
            Approximated reconstruction of the original high-dimensional data.
        """
        if not hasattr(self, '_training_data_shape'):
            raise ValueError('Transformer has not been fitted yet.')
        X_embedded = data.features
        if X_embedded.shape[1] != self.n_components:
            raise ValueError(f'Input embedded data must have {self.n_components} features, got {X_embedded.shape[1]}')
        nbrs_embedding = NearestNeighbors(n_neighbors=1)
        nbrs_embedding.fit(self.embedding_)
        (distances, indices) = nbrs_embedding.kneighbors(X_embedded, return_distance=True)
        reconstructed_features = self._training_data[indices.ravel()]
        reconstructed_features = reconstructed_features.reshape(X_embedded.shape[0], -1)
        return FeatureSet(features=reconstructed_features, feature_names=self._original_feature_names, feature_types=self._original_feature_types, sample_ids=data.sample_ids, metadata={'source': 'TopologicallyConstrainedIsometricEmbedding.inverse_transform'})