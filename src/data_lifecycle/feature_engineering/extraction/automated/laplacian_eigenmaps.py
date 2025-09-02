import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from typing import Optional, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy.spatial.distance import cdist
from src.data_lifecycle.mathematical_foundations.specialized_functions.distance_metrics import manhattan_distance

class LaplacianEigenmapsExtractor(BaseTransformer):

    def __init__(self, n_components: int=2, n_neighbors: int=10, sigma: float=1.0, metric: str='euclidean', random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Laplacian Eigenmaps transformer.

        Parameters
        ----------
        n_components : int, default=2
            Number of dimensions in the embedded space.
        n_neighbors : int, default=10
            Number of neighbors to consider for constructing the adjacency graph.
        sigma : float, default=1.0
            Scale parameter for the Gaussian kernel used in weight computation.
        metric : str, default='euclidean'
            Distance metric for computing pairwise distances between samples.
        random_state : int or None, default=None
            Random seed for reproducibility.
        name : str or None, default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.metric = metric
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, data: FeatureSet, **kwargs) -> 'LaplacianEigenmapsExtractor':
        """
        Fit the Laplacian Eigenmaps transformer to the input data.
        
        This method constructs the affinity graph, computes the graph Laplacian,
        and finds the eigenvectors corresponding to the smallest eigenvalues
        (excluding the trivial zero eigenvalue).
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing high-dimensional data to be embedded.
        **kwargs : dict
            Additional keyword arguments (ignored in this implementation).
            
        Returns
        -------
        LaplacianEigenmapsExtractor
            Returns self for method chaining.
        """
        self._n_features_in = data.features.shape[1]
        self._n_samples = data.features.shape[0]
        if self.metric == 'manhattan':
            n_samples = data.features.shape[0]
            distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    distances[i, j] = manhattan_distance(data.features[i], data.features[j])
        else:
            distances = cdist(data.features, data.features, metric=self.metric)
        connectivity = kneighbors_graph(data.features, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
        adjacency = connectivity.toarray()
        affinity = np.exp(-distances ** 2 / (2 * self.sigma ** 2))
        affinity = affinity * adjacency
        self.affinity_matrix_ = affinity
        degree = np.sum(affinity, axis=1)
        laplacian = np.diag(degree) - affinity
        degree_sqrt_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
        degree_sqrt_inv_matrix = np.diag(degree_sqrt_inv)
        normalized_laplacian = degree_sqrt_inv_matrix @ laplacian @ degree_sqrt_inv_matrix
        max_components = min(self.n_components, self._n_samples - 2)
        if max_components <= 0:
            max_components = 1
        (eigenvals, eigenvecs) = (None, None)
        try:
            k = min(max_components + 1, self._n_samples - 1)
            if k >= 1:
                (eigenvals, eigenvecs) = eigsh(normalized_laplacian, k=k, which='SM', tol=1e-06, maxiter=5000)
        except Exception:
            try:
                k = min(max_components + 1, self._n_samples - 1)
                if k >= 1:
                    (eigenvals, eigenvecs) = eigsh(normalized_laplacian, k=k, which='SM', tol=0.001, maxiter=10000)
            except Exception:
                (eigenvals, eigenvecs) = np.linalg.eigh(normalized_laplacian)
        if eigenvals is None or eigenvecs is None:
            raise RuntimeError('Failed to compute eigenvalues and eigenvectors')
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        n_embed = min(self.n_components, len(eigenvals) - 1)
        if n_embed <= 0:
            n_embed = 1
        self.embedding_vectors_ = eigenvecs[:, 1:n_embed + 1]
        self.eigenvalues_ = eigenvals[1:n_embed + 1]
        self.components_ = self.embedding_vectors_
        self._training_data = data.features.copy()
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the learned Laplacian Eigenmaps transformation to reduce dimensionality.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to be transformed. Must have the same number of features
            as the data used during fitting.
        **kwargs : dict
            Additional keyword arguments (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            Transformed feature set with reduced dimensionality.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet or if the input data
            does not match the expected feature dimensions.
        """
        if not hasattr(self, '_n_features_in'):
            raise ValueError("This LaplacianEigenmapsExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if data.features.shape[1] != self._n_features_in:
            raise ValueError(f'X has {data.features.shape[1]} features, but LaplacianEigenmapsExtractor is expecting {self._n_features_in} features as input.')
        distances = cdist(data.features, self._training_data, metric=self.metric)
        weights = np.exp(-distances ** 2 / (2 * self.sigma ** 2))
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights_sum[weights_sum == 0] = 1
        weights = weights / weights_sum
        embedded_data = weights @ self.embedding_vectors_
        original_name = data.metadata.get('name', 'data') if data.metadata else 'data'
        new_name = f'{original_name}_laplacian_eigenmaps' if original_name else None
        transformed_fs = FeatureSet(features=embedded_data, feature_names=self.get_feature_names(), metadata={'name': new_name} if new_name else None)
        return transformed_fs

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Attempt to reconstruct the original high-dimensional data from the embedding.
        
        Note: Exact inverse transformation is generally not possible with Laplacian Eigenmaps
        due to the non-linear nature of the mapping. This method will raise a NotImplementedError.
        
        Parameters
        ----------
        data : FeatureSet
            Low-dimensional embedded data to be reconstructed.
        **kwargs : dict
            Additional keyword arguments (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            This method always raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            Laplacian Eigenmaps does not support exact inverse transformation.
        """
        raise NotImplementedError('Laplacian Eigenmaps does not support exact inverse transformation.')

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Generate feature names for the embedded dimensions.
        
        Parameters
        ----------
        input_features : list of str, optional
            Names of the input features. If not provided, generic names will be generated.
            
        Returns
        -------
        list of str
            Names for the embedded feature dimensions.
        """
        return [f'laplacian_component_{i}' for i in range(self.n_components)]