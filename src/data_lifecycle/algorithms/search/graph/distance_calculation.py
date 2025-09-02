import numpy as np
from typing import Union, Optional
from general.base_classes.transformer_base import BaseTransformer
import heapq

class GraphDistanceCalculator(BaseTransformer):

    def __init__(self, method: str='shortest_path', weighted: bool=False, directed: bool=False, name: Optional[str]=None):
        """
        Initialize the GraphDistanceCalculator.
        
        Parameters
        ----------
        method : str, optional
            Distance calculation method (default: 'shortest_path')
        weighted : bool, optional
            Whether to use edge weights (default: False)
        directed : bool, optional
            Whether the graph is directed (default: False)
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name)
        self.method = method
        self.weighted = weighted
        self.directed = directed

    def fit(self, data: Union[np.ndarray, list], **kwargs) -> 'GraphDistanceCalculator':
        """
        Fit the calculator to the graph data.
        
        Parameters
        ----------
        data : Union[np.ndarray, list]
            Graph representation as adjacency matrix or edge list
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        GraphDistanceCalculator
            Self instance for method chaining
        """
        self._original_data = data
        if self.method == 'euclidean':
            if isinstance(data, list):
                self.coordinates_ = np.array(data)
            elif isinstance(data, np.ndarray):
                self.coordinates_ = data.copy()
            else:
                raise TypeError('Coordinate data must be either a numpy array or list')
            return self
        if isinstance(data, list):
            if data:
                max_node = max((max(edge[0], edge[1]) for edge in data))
                num_nodes = max_node + 1
            else:
                num_nodes = 0
            adj_matrix = np.zeros((num_nodes, num_nodes))
            for edge in data:
                if len(edge) < 2:
                    raise ValueError('Edge list entries must contain at least two elements (source, target)')
                (source, target) = (edge[0], edge[1])
                if self.weighted and len(edge) > 2:
                    weight = edge[2]
                else:
                    weight = 1
                adj_matrix[source, target] = weight
                if not self.directed:
                    adj_matrix[target, source] = weight
            self.adjacency_matrix_ = adj_matrix
        elif isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError('Adjacency matrix must be a square 2D array')
            self.adjacency_matrix_ = data.copy()
        else:
            raise TypeError('Data must be either a numpy array (adjacency matrix) or list (edge list)')
        return self

    def transform(self, data: Union[np.ndarray, list]=None, **kwargs) -> np.ndarray:
        """
        Calculate distances between nodes in the graph.
        
        Parameters
        ----------
        data : Union[np.ndarray, list], optional
            Graph representation as adjacency matrix or edge list. If None, uses fitted data.
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        np.ndarray
            Matrix of pairwise distances between nodes
        """
        if self.method == 'euclidean':
            if data is None:
                if hasattr(self, 'coordinates_'):
                    coordinates = self.coordinates_
                else:
                    raise ValueError('No data provided and no fitted data available')
            elif isinstance(data, list):
                coordinates = np.array(data)
            elif isinstance(data, np.ndarray):
                coordinates = data.copy()
            else:
                raise TypeError('Coordinate data must be either a numpy array or list')
            return self._compute_euclidean_distances(coordinates)
        else:
            if data is None:
                if hasattr(self, 'adjacency_matrix_'):
                    adj_matrix = self.adjacency_matrix_
                else:
                    raise ValueError('No data provided and no fitted data available')
            elif hasattr(self, 'adjacency_matrix_') and (isinstance(data, np.ndarray) and np.array_equal(data, self.adjacency_matrix_) or (isinstance(data, list) and data == self._original_data)):
                adj_matrix = self.adjacency_matrix_
            else:
                self.fit(data)
                adj_matrix = self.adjacency_matrix_
            return self._compute_shortest_paths(adj_matrix)

    def _compute_euclidean_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances between all pairs of points.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Array of shape (n_points, n_dimensions) containing coordinates
            
        Returns
        -------
        np.ndarray
            Matrix of Euclidean distances between all pairs of points
        """
        n_points = coordinates.shape[0]
        distance_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = coordinates[i] - coordinates[j]
                dist = np.sqrt(np.sum(diff ** 2))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        return distance_matrix

    def _compute_shortest_paths(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Compute shortest paths between all pairs of nodes using Dijkstra's algorithm.
        
        Parameters
        ----------
        adj_matrix : np.ndarray
            Adjacency matrix representation of the graph
            
        Returns
        -------
        np.ndarray
            Matrix of shortest path distances between all pairs of nodes
        """
        num_nodes = adj_matrix.shape[0]
        distance_matrix = np.full((num_nodes, num_nodes), np.inf)
        adj_list = {}
        for i in range(num_nodes):
            adj_list[i] = []
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    if self.weighted:
                        weight = adj_matrix[i, j]
                    else:
                        weight = 1
                    adj_list[i].append((j, weight))
        for source in range(num_nodes):
            distances = self._dijkstra(adj_list, source, num_nodes)
            distance_matrix[source] = distances
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix

    def _dijkstra(self, adj_list: dict, source: int, num_nodes: int) -> np.ndarray:
        """
        Implementation of Dijkstra's algorithm for single-source shortest paths.
        
        Parameters
        ----------
        adj_list : dict
            Adjacency list representation of the graph
        source : int
            Source node
        num_nodes : int
            Total number of nodes in the graph
            
        Returns
        -------
        np.ndarray
            Array of distances from source to all other nodes
        """
        distances = np.full(num_nodes, np.inf)
        distances[source] = 0
        visited = set()
        pq = [(0, source)]
        while pq:
            (current_distance, current_node) = heapq.heappop(pq)
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node in adj_list:
                for (neighbor, weight) in adj_list[current_node]:
                    if neighbor not in visited:
                        new_distance = current_distance + weight
                        if new_distance < distances[neighbor]:
                            distances[neighbor] = new_distance
                            heapq.heappush(pq, (new_distance, neighbor))
        return distances

    def inverse_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Not applicable for distance calculation.
        
        Parameters
        ----------
        data : np.ndarray
            Distance matrix
            
        Returns
        -------
        np.ndarray
            Original data format (identity operation)
        """
        if hasattr(self, '_original_data'):
            if isinstance(self._original_data, list):
                return self._original_data.copy()
            else:
                return self._original_data.copy()
        else:
            return data.copy()

def calculate_graph_distances(graph: Union[np.ndarray, list], method: str='shortest_path', weighted: bool=False, directed: bool=False) -> np.ndarray:
    """
    Calculate distances between all pairs of nodes in a graph.
    
    This function computes pairwise distances between nodes using various algorithms
    such as shortest path (Dijkstra, Floyd-Warshall), Euclidean distance, etc.
    
    Parameters
    ----------
    graph : Union[np.ndarray, list]
        Graph representation as adjacency matrix or edge list
    method : str, optional
        Distance calculation method (default: 'shortest_path')
    weighted : bool, optional
        Whether to consider edge weights (default: False)
    directed : bool, optional
        Whether the graph is directed (default: False)
        
    Returns
    -------
    np.ndarray
        Matrix of pairwise distances between nodes
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple unweighted graph as adjacency matrix
    >>> adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> distances = calculate_graph_distances(adj_matrix)
    >>> print(distances)
    [[0. 1. 2.]
     [1. 0. 1.]
     [2. 1. 0.]]
    """
    calculator = GraphDistanceCalculator(method=method, weighted=weighted, directed=directed)
    return calculator.fit_transform(graph)