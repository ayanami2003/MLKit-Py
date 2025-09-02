from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class MarkovClusteringModel(BaseModel):
    """
    Markov Clustering (MCL) algorithm implementation for graph-based clustering.
    
    This clustering method simulates flow within a graph using Markov chains to identify dense regions 
    (clusters) based on the probability of reaching other nodes. It is particularly effective for 
    identifying clusters in networks where edge weights represent similarities or affinities.
    
    The algorithm alternates between expansion (raising the adjacency matrix to a power) and inflation 
    (raising each column to a power and renormalizing) until convergence.
    
    Attributes
    ----------
    inflation : float
        Inflation parameter controlling cluster granularity (typically between 1.2 and 5.0).
    expansion : int
        Expansion parameter determining the Markov process step size.
    max_iterations : int
        Maximum number of iterations allowed before stopping.
    threshold : float
        Threshold for pruning weak edges during iteration.
    add_self_loops : bool
        Whether to add self-loops to ensure node reachability.
    """

    def __init__(self, inflation: float=2.0, expansion: int=2, max_iterations: int=100, threshold: float=1e-05, add_self_loops: bool=True, name: Optional[str]=None):
        """
        Initialize the Markov Clustering model.
        
        Parameters
        ----------
        inflation : float, default=2.0
            Controls cluster granularity; higher values lead to more fine-grained clusters.
        expansion : int, default=2
            Power to which the transition matrix is raised in each iteration.
        max_iterations : int, default=100
            Maximum number of iterations to run the algorithm.
        threshold : float, default=1e-5
            Minimum value below which matrix entries are pruned to sparsify computation.
        add_self_loops : bool, default=True
            If True, ensures each node has a self-loop for better connectivity.
        name : str, optional
            Name identifier for the model instance.
        """
        super().__init__(name=name)
        self.inflation = inflation
        self.expansion = expansion
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.add_self_loops = add_self_loops

    def _validate_input(self, X: Union[np.ndarray, FeatureSet]) -> np.ndarray:
        """Validate and extract adjacency matrix from input."""
        if isinstance(X, FeatureSet):
            if not hasattr(X, 'adjacency_matrix'):
                raise ValueError("FeatureSet must contain an 'adjacency_matrix' attribute.")
            adj_matrix = X.adjacency_matrix
        elif isinstance(X, np.ndarray):
            adj_matrix = X
        else:
            raise TypeError('Input must be a numpy array or FeatureSet with adjacency matrix.')
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError('Input must be a square adjacency matrix.')
        return adj_matrix.astype(float)

    def _add_self_loops(self, matrix: np.ndarray) -> np.ndarray:
        """Add self-loops to the adjacency matrix."""
        np.fill_diagonal(matrix, np.maximum(matrix.diagonal(), 1))
        return matrix

    def _normalize_columns(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize columns of the matrix to sum to 1."""
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1
        return matrix / col_sums

    def _expand(self, matrix: np.ndarray) -> np.ndarray:
        """Perform expansion step (matrix multiplication)."""
        for _ in range(self.expansion - 1):
            matrix = matrix @ matrix
        return matrix

    def _inflate(self, matrix: np.ndarray) -> np.ndarray:
        """Perform inflation step (element-wise inflation and normalization)."""
        matrix = np.power(matrix, self.inflation)
        matrix = self._normalize_columns(matrix)
        return matrix

    def _prune(self, matrix: np.ndarray) -> np.ndarray:
        """Prune small values below threshold."""
        matrix[matrix < self.threshold] = 0
        return matrix

    def _check_convergence(self, prev_matrix: np.ndarray, curr_matrix: np.ndarray) -> bool:
        """Check if the algorithm has converged."""
        return np.allclose(prev_matrix, curr_matrix, atol=1e-05)

    def _find_clusters(self, matrix: np.ndarray) -> np.ndarray:
        """Identify clusters from the converged matrix."""
        attractors = np.where(matrix.diagonal() > 0)[0]
        labels = -np.ones(matrix.shape[0], dtype=int)
        cluster_id = 0
        for attractor in attractors:
            basin = np.where(matrix[attractor, :] > 0)[0]
            unassigned = basin[labels[basin] == -1]
            if len(unassigned) > 0:
                labels[unassigned] = cluster_id
                cluster_id += 1
        unassigned_nodes = np.where(labels == -1)[0]
        for node in unassigned_nodes:
            labels[node] = cluster_id
            cluster_id += 1
        return labels

    def _compute_modularity(self, adj_matrix: np.ndarray, labels: np.ndarray) -> float:
        """Compute modularity of the clustering."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_nodes = adj_matrix.shape[0]
        m = adj_matrix.sum()
        if m == 0:
            return 0.0
        modularity = 0.0
        for i in range(n_clusters):
            cluster_nodes = np.where(labels == unique_labels[i])[0]
            subgraph = adj_matrix[np.ix_(cluster_nodes, cluster_nodes)]
            within_cluster = subgraph.sum()
            cluster_degree = adj_matrix[cluster_nodes, :].sum()
            modularity += within_cluster / m - (cluster_degree / m) ** 2
        return modularity

    def fit(self, X: Union[np.ndarray, FeatureSet], y: Optional[np.ndarray]=None, **kwargs) -> 'MarkovClusteringModel':
        """
        Fit the Markov Clustering model to the input graph data.
        
        Parameters
        ----------
        X : Union[np.ndarray, FeatureSet]
            Square adjacency matrix representing the graph (n_nodes, n_nodes) or FeatureSet with adjacency data.
        y : np.ndarray, optional
            Ignored in unsupervised setting.
        **kwargs : dict
            Additional fitting parameters (reserved for future extensions).
            
        Returns
        -------
        MarkovClusteringModel
            Fitted model instance.
        """
        adj_matrix = self._validate_input(X)
        matrix = adj_matrix.copy()
        if self.add_self_loops:
            matrix = self._add_self_loops(matrix)
        matrix = self._normalize_columns(matrix)
        for iteration in range(self.max_iterations):
            prev_matrix = matrix.copy()
            matrix = self._expand(matrix)
            matrix = self._inflate(matrix)
            matrix = self._prune(matrix)
            if self._check_convergence(prev_matrix, matrix):
                break
        self.converged_matrix_ = matrix
        self.labels_ = self._find_clusters(matrix)
        return self

    def predict(self, X: Union[np.ndarray, FeatureSet], **kwargs) -> np.ndarray:
        """
        Predict cluster assignments for nodes in the graph.
        
        Parameters
        ----------
        X : Union[np.ndarray, FeatureSet]
            Square adjacency matrix representing the graph (n_nodes, n_nodes) or FeatureSet with adjacency data.
        **kwargs : dict
            Additional prediction parameters (reserved for future extensions).
            
        Returns
        -------
        np.ndarray
            Cluster labels for each node in the graph (n_nodes,).
        """
        if not hasattr(self, 'labels_'):
            raise ValueError('Model must be fitted before predictions can be made.')
        adj_matrix = self._validate_input(X)
        if adj_matrix.shape[0] != len(self.labels_):
            raise ValueError('Input matrix shape must match the training data.')
        return self.labels_.copy()

    def fit_predict(self, X: Union[np.ndarray, FeatureSet], y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster assignments in one step.
        
        Parameters
        ----------
        X : Union[np.ndarray, FeatureSet]
            Square adjacency matrix representing the graph (n_nodes, n_nodes) or FeatureSet with adjacency data.
        y : np.ndarray, optional
            Ignored in unsupervised setting.
        **kwargs : dict
            Additional fitting/prediction parameters (reserved for future extensions).
            
        Returns
        -------
        np.ndarray
            Cluster labels for each node in the graph (n_nodes,).
        """
        return self.fit(X, y, **kwargs).labels_.copy()

    def score(self, X: Union[np.ndarray, FeatureSet], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Compute the clustering score (modularity) of the fitted model.
        
        Parameters
        ----------
        X : Union[np.ndarray, FeatureSet]
            Square adjacency matrix representing the graph (n_nodes, n_nodes) or FeatureSet with adjacency data.
        y : np.ndarray, optional
            Ignored in unsupervised setting.
        **kwargs : dict
            Additional scoring parameters (reserved for future extensions).
            
        Returns
        -------
        float
            Modularity score indicating quality of the clustering.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError('Model must be fitted before scoring.')
        adj_matrix = self._validate_input(X)
        return self._compute_modularity(adj_matrix, self.labels_)