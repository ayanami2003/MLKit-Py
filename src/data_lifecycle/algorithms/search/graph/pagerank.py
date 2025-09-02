from general.base_classes.transformer_base import BaseTransformer
import numpy as np
from typing import Union, Optional

class PageRankTransformer(BaseTransformer):

    def __init__(self, damping_factor: float=0.85, max_iterations: int=100, tolerance: float=1e-06, weighted: bool=False, name: Optional[str]=None):
        """
        Initialize the PageRank transformer.
        
        Args:
            damping_factor (float): Probability of following an outgoing link.
            max_iterations (int): Maximum number of iterations to run.
            tolerance (float): Convergence threshold for score changes.
            weighted (bool): Whether to compute weighted PageRank.
            name (Optional[str]): Name for the transformer instance.
        """
        super().__init__(name=name)
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weighted = weighted

    def fit(self, data: Union[np.ndarray, list], **kwargs) -> 'PageRankTransformer':
        """
        Fit the transformer to the adjacency matrix data.
        
        For PageRank, fitting simply validates and stores the adjacency matrix.
        
        Args:
            data (Union[np.ndarray, list]): Square adjacency matrix representing the graph.
                For weighted PageRank, values represent edge weights.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            PageRankTransformer: Self instance for method chaining.
        """
        if isinstance(data, list):
            try:
                data = np.array(data)
            except Exception as e:
                raise TypeError(f'Failed to convert input data to numpy array: {str(e)}')
        if not isinstance(data, np.ndarray):
            raise TypeError('Input data must be a numpy array or list that can be converted to numpy array')
        if data.ndim != 2:
            raise ValueError('Adjacency matrix must be 2-dimensional')
        if data.shape[0] != data.shape[1]:
            raise ValueError('Adjacency matrix must be square')
        self._adj_matrix = data.copy()
        return self

    def transform(self, data: Union[np.ndarray, list], **kwargs) -> np.ndarray:
        """
        Compute PageRank scores for the given adjacency matrix.
        
        Args:
            data (Union[np.ndarray, list]): Square adjacency matrix representing the graph.
                For weighted PageRank, values represent edge weights.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            np.ndarray: Array of PageRank scores for each node in the graph.
        """
        if hasattr(self, '_adj_matrix'):
            adjacency_matrix = self._adj_matrix
        else:
            if isinstance(data, list):
                try:
                    data = np.array(data)
                except Exception as e:
                    raise TypeError(f'Failed to convert input data to numpy array: {str(e)}')
            if not isinstance(data, np.ndarray):
                raise TypeError('Input data must be a numpy array or list that can be converted to numpy array')
            if data.ndim != 2:
                raise ValueError('Adjacency matrix must be 2-dimensional')
            if data.shape[0] != data.shape[1]:
                raise ValueError('Adjacency matrix must be square')
            adjacency_matrix = data
        damping_factor = self.damping_factor
        max_iterations = self.max_iterations
        tolerance = self.tolerance
        weighted = self.weighted
        n = adjacency_matrix.shape[0]
        if n == 0:
            return np.array([])
        M = adjacency_matrix.astype(float)
        if not weighted:
            M = (M > 0).astype(float)
        out_degree = np.sum(M, axis=1)
        M_stochastic = np.zeros_like(M)
        for i in range(n):
            if out_degree[i] != 0:
                M_stochastic[i, :] = M[i, :] / out_degree[i]
        pagerank = np.ones(n) / n
        teleportation_factor = (1 - damping_factor) / n
        for iteration in range(max_iterations):
            dangling_sum = np.sum(pagerank[out_degree == 0])
            new_pagerank = damping_factor * np.dot(M_stochastic.T, pagerank) + teleportation_factor * np.ones(n) + damping_factor * dangling_sum / n
            diff = np.linalg.norm(new_pagerank - pagerank, 1)
            pagerank = new_pagerank
            if diff < tolerance:
                break
        else:
            import warnings
            warnings.warn(f'PageRank did not converge after {max_iterations} iterations. Final difference was {diff}', RuntimeWarning)
        pagerank = pagerank / np.sum(pagerank)
        return pagerank

    def inverse_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Not applicable for PageRank computation.
        
        PageRank is a one-way transformation that cannot be inverted.
        
        Args:
            data (np.ndarray): PageRank scores to "invert".
            **kwargs: Additional parameters (ignored).
            
        Returns:
            np.ndarray: The same data passed in (no-op).
        """
        return data