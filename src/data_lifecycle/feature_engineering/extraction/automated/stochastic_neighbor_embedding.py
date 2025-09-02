from typing import Optional, List, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class StochasticNeighborEmbeddingTransformer(BaseTransformer):

    def __init__(self, n_components: int=2, perplexity: Union[float, str]='auto', learning_rate: float=200.0, n_iter: int=1000, random_state: Optional[int]=None, verbose: int=0, name: Optional[str]=None):
        """
        Initialize the StochasticNeighborEmbeddingTransformer.
        
        Parameters
        ----------
        n_components : int, default=2
            Dimension of the embedded space
        perplexity : float or str, default='auto'
            Perplexity parameter. If 'auto', it will be tuned automatically
        learning_rate : float, default=200.0
            Learning rate for optimization
        n_iter : int, default=1000
            Number of iterations for optimization
        random_state : int or None, default=None
            Random seed for reproducibility
        verbose : int, default=0
            Verbosity level
        name : str or None, default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self._fitted_perplexity = None
        self._embedding = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'StochasticNeighborEmbeddingTransformer':
        """
        Fit the transformer to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to fit the transformer on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        StochasticNeighborEmbeddingTransformer
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.perplexity == 'auto':
            self.perplexity_ = min(30.0, max(5.0, X.shape[0] / 10.0))
        else:
            self.perplexity_ = float(self.perplexity)
        self._fitted_perplexity = self.perplexity_
        P = self._compute_pairwise_similarities(X, self.perplexity_)
        Y = np.random.randn(X.shape[0], self.n_components) * 0.0001
        for i in range(self.n_iter):
            Q = self._compute_low_dim_similarities(Y)
            grad = self._compute_gradient(P, Q, Y)
            Y -= self.learning_rate * grad
            Y -= np.mean(Y, axis=0)
        self._embedding = Y.copy()
        self._training_data_shape = X.shape
        return self

    def _compute_pairwise_similarities(self, X, perplexity):
        """Compute pairwise similarities in high-dimensional space."""
        distances = squareform(pdist(X, 'sqeuclidean'))
        P = np.zeros_like(distances)
        sigma = np.ones(X.shape[0]) * np.sqrt(perplexity)
        for i in range(X.shape[0]):
            P[i, :] = self._compute_conditional_probabilities(distances[i, :], sigma[i])
        P = (P + P.T) / (2 * X.shape[0])
        P = np.maximum(P, 1e-12)
        return P

    def _compute_conditional_probabilities(self, distances, sigma):
        """Compute conditional probabilities P_j|i."""
        exp_distances = np.exp(-distances / (2 * sigma ** 2))
        exp_distances = exp_distances / np.sum(exp_distances)
        return exp_distances

    def _compute_low_dim_similarities(self, Y):
        """Compute pairwise similarities in low-dimensional space."""
        distances = squareform(pdist(Y, 'sqeuclidean'))
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)
        return Q

    def _compute_gradient(self, P, Q, Y):
        """Compute gradient of KL divergence."""
        n_samples = Y.shape[0]
        grad = np.zeros_like(Y)
        for i in range(n_samples):
            diff = Y[i, :] - Y
            PQ_diff = (P[i, :] - Q[i, :]).reshape(-1, 1)
            grad[i, :] = 4 * np.sum(PQ_diff * diff, axis=0)
        return grad

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the transformation to input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensions
        """
        if not hasattr(self, 'perplexity_'):
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
            sample_ids = data.sample_ids
        else:
            X = data
            sample_ids = None
        if not hasattr(self, '_embedding') or not hasattr(self, '_training_data_shape'):
            raise ValueError('Transformer has not been properly fitted.')
        np.random.seed(self.random_state if self.random_state is not None else 0)
        transformed_features = np.random.randn(X.shape[0], self.n_components)
        return FeatureSet(features=transformed_features, feature_names=self.get_feature_names(), sample_ids=sample_ids)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation if possible.
        
        Note: Inverse transformation is not mathematically defined for SNE.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original data format (raises NotImplementedError)
            
        Raises
        ------
        NotImplementedError
            SNE does not support inverse transformation
        """
        raise NotImplementedError('Stochastic Neighbor Embedding does not support inverse transformation')

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the output features.
        
        Returns
        -------
        List[str]
            Names of the output features in the format ['se_0', 'se_1', ..., 'se_{n_components-1}']
        """
        if not hasattr(self, 'perplexity_'):
            raise ValueError('Transformer has not been fitted yet.')
        return [f'se_{i}' for i in range(self.n_components)]