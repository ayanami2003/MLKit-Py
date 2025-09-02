from typing import Union, Optional, Dict, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch

class PageRankCentralityComputer(BaseModel):

    def __init__(self, damping_factor: float=0.85, max_iterations: int=100, tolerance: float=1e-06, personalization: Optional[Dict[Any, float]]=None, name: Optional[str]=None):
        """
        Initialize the PageRankCentralityComputer.

        Parameters
        ----------
        damping_factor : float, optional
            Damping factor (alpha) for PageRank, between 0 and 1. Default is 0.85.
        max_iterations : int, optional
            Maximum number of iterations for convergence. Default is 100.
        tolerance : float, optional
            Convergence tolerance. Default is 1e-6.
        personalization : Optional[Dict[Any, float]], optional
            Personalization vector indicating the importance of each node.
        name : Optional[str], optional
            Name of the model instance.
        """
        super().__init__(name=name)
        if not 0 < damping_factor < 1:
            raise ValueError('Damping factor must be between 0 and 1.')
        if max_iterations <= 0:
            raise ValueError('Max iterations must be positive.')
        if tolerance <= 0:
            raise ValueError('Tolerance must be positive.')
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.personalization = personalization

    def fit(self, X: Union[np.ndarray, DataBatch], y: Optional[Any]=None, **kwargs) -> 'PageRankCentralityComputer':
        """
        Fit the PageRank centrality model to the graph data.

        Parameters
        ----------
        X : Union[np.ndarray, DataBatch]
            Adjacency matrix representing the graph structure, where X[i, j] represents
            the weight of the edge from node i to node j. If using DataBatch, it should
            contain the adjacency matrix in its data attribute.
        y : Optional[Any], optional
            Not used in PageRank computation.
        **kwargs : dict
            Additional parameters (reserved for future extensions).

        Returns
        -------
        PageRankCentralityComputer
            Self instance for method chaining.
        """
        if isinstance(X, DataBatch):
            adj_matrix = X.data
        else:
            adj_matrix = X
        if not isinstance(adj_matrix, np.ndarray):
            raise TypeError('Adjacency matrix must be a NumPy array.')
        if adj_matrix.ndim != 2:
            raise ValueError('Adjacency matrix must be 2-dimensional.')
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError('Adjacency matrix must be square.')
        self.n_nodes_ = adj_matrix.shape[0]
        if self.personalization is not None and len(self.personalization) > 0:
            pers_array = np.zeros(self.n_nodes_)
            for (node_idx, weight) in self.personalization.items():
                if not isinstance(node_idx, int) or node_idx < 0 or node_idx >= self.n_nodes_:
                    raise ValueError(f'Invalid node index {node_idx} in personalization vector.')
                pers_array[node_idx] = weight
            pers_sum = pers_array.sum()
            if pers_sum == 0:
                self.personalization_ = np.ones(self.n_nodes_) / self.n_nodes_
            else:
                self.personalization_ = pers_array / pers_sum
        else:
            self.personalization_ = np.ones(self.n_nodes_) / self.n_nodes_
        self.adj_matrix_ = adj_matrix.copy()
        return self

    def predict(self, X: Union[np.ndarray, DataBatch], **kwargs) -> np.ndarray:
        """
        Compute PageRank centrality scores for the nodes in the graph.

        Parameters
        ----------
        X : Union[np.ndarray, DataBatch]
            Adjacency matrix representing the graph structure, where X[i, j] represents
            the weight of the edge from node i to node j. If using DataBatch, it should
            contain the adjacency matrix in its data attribute.
        **kwargs : dict
            Additional parameters (reserved for future extensions).

        Returns
        -------
        np.ndarray
            Array of PageRank centrality scores for each node in the graph.
        """
        if not hasattr(self, 'adj_matrix_'):
            raise RuntimeError('Model must be fitted before calling predict.')
        if isinstance(X, DataBatch):
            adj_matrix = X.data
        else:
            adj_matrix = X
        if not np.array_equal(adj_matrix, self.adj_matrix_):
            raise ValueError('Adjacency matrix in predict must match the one used in fit.')
        out_degrees = np.asarray(adj_matrix.sum(axis=1)).flatten()
        trans_matrix = np.zeros_like(adj_matrix, dtype=float)
        for i in range(self.n_nodes_):
            if out_degrees[i] != 0:
                trans_matrix[i, :] = adj_matrix[i, :] / out_degrees[i]
            else:
                trans_matrix[i, :] = 1.0 / self.n_nodes_
        pr_vector = np.ones(self.n_nodes_) / self.n_nodes_
        for _ in range(self.max_iterations):
            prev_pr_vector = pr_vector.copy()
            pr_vector = (1 - self.damping_factor) * self.personalization_ + self.damping_factor * (trans_matrix.T @ prev_pr_vector)
            if np.linalg.norm(pr_vector - prev_pr_vector, ord=1) < self.tolerance:
                break
        pr_vector = pr_vector / pr_vector.sum()
        return pr_vector

    def score(self, X: Union[np.ndarray, DataBatch], y: Any, **kwargs) -> float:
        """
        Evaluate the PageRank model (not typically used for PageRank).

        Parameters
        ----------
        X : Union[np.ndarray, DataBatch]
            Adjacency matrix representing the graph structure.
        y : Any
            Not used in PageRank scoring.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        float
            A dummy score of 1.0 as PageRank doesn't have a standard scoring mechanism.
        """
        return 1.0