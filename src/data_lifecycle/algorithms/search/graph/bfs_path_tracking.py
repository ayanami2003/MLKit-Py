from typing import Union, List, Optional, Tuple
import numpy as np
from general.base_classes.transformer_base import BaseTransformer

class BFSPathTracker(BaseTransformer):
    """
    A transformer that performs Breadth-First Search (BFS) on a graph while tracking paths.
    
    This class implements BFS traversal on graph data structures, maintaining path information
    from a source node to all reachable nodes. It supports both adjacency matrix and edge list
    representations of graphs.
    
    Attributes
    ----------
    source_node : int
        The starting node for BFS traversal
    track_predecessors : bool
        Whether to track predecessor nodes for path reconstruction
    max_distance : Optional[int]
        Maximum search depth (default: None for unlimited)
    name : Optional[str]
        Name identifier for the transformer
        
    Methods
    -------
    fit() : Fits the transformer to the graph data
    transform() : Performs BFS and returns path information
    get_shortest_path() : Reconstructs path to a target node
    """

    def __init__(self, source_node: int=0, track_predecessors: bool=True, max_distance: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the BFS path tracker.
        
        Parameters
        ----------
        source_node : int, default=0
            The node to start BFS traversal from
        track_predecessors : bool, default=True
            If True, track predecessors to enable path reconstruction
        max_distance : int or None, default=None
            Maximum distance to search (None for unlimited depth)
        name : str or None, default=None
            Optional name for the transformer instance
        """
        super().__init__(name=name)
        self.source_node = source_node
        self.track_predecessors = track_predecessors
        self.max_distance = max_distance
        self._predecessors = None
        self._distances = None
        self._visited = None
        self._graph = None
        self._num_nodes = None
        self._graph_type = None
        self._is_fitted = False

    def fit(self, data: Union[np.ndarray, List[Tuple[int, int]]], **kwargs) -> 'BFSPathTracker':
        """
        Fit the transformer to the graph data.
        
        This method prepares the internal state for BFS traversal but does not perform
        the actual search. The graph can be represented as either an adjacency matrix
        or an edge list.
        
        Parameters
        ----------
        data : numpy.ndarray or List[Tuple[int, int]]
            Graph representation - either adjacency matrix (2D array) or edge list
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        BFSPathTracker
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the graph data format is invalid
        """
        if not isinstance(data, (np.ndarray, list)):
            raise TypeError('Graph must be a numpy array (adjacency matrix) or list of tuples (edge list)')
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError('Adjacency matrix must be a square 2D array')
            if self.source_node < 0 or self.source_node >= data.shape[0]:
                raise ValueError(f'Source node {self.source_node} is out of bounds for graph with {data.shape[0]} nodes')
            self._graph = {}
            self._num_nodes = data.shape[0]
            for i in range(data.shape[0]):
                self._graph[i] = []
                for j in range(data.shape[1]):
                    if data[i, j] != 0:
                        self._graph[i].append(j)
        elif isinstance(data, list):
            if not all((isinstance(edge, tuple) and len(edge) == 2 for edge in data)):
                raise ValueError('Edge list must contain tuples of exactly 2 elements')
            all_nodes = set()
            for edge in data:
                all_nodes.update(edge)
            if all_nodes:
                self._num_nodes = max(all_nodes) + 1
            else:
                self._num_nodes = max(self.source_node + 1, 1)
            if self.source_node < 0 or self.source_node >= self._num_nodes:
                raise ValueError(f'Source node {self.source_node} is out of bounds for graph with {self._num_nodes} nodes')
            self._graph = {i: [] for i in range(self._num_nodes)}
            for (u, v) in data:
                if u < 0 or u >= self._num_nodes or v < 0 or (v >= self._num_nodes):
                    raise ValueError(f'Edge ({u}, {v}) contains out-of-bounds node')
                if v not in self._graph[u]:
                    self._graph[u].append(v)
        self._graph_type = type(data)
        self._is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, List[Tuple[int, int]]], **kwargs) -> dict:
        """
        Perform BFS traversal and track paths.
        
        Executes breadth-first search from the source node, tracking distances
        and optionally predecessors for path reconstruction.
        
        Parameters
        ----------
        data : numpy.ndarray or List[Tuple[int, int]]
            Graph representation - either adjacency matrix (2D array) or edge list
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'distances': Dict mapping node indices to distances from source
            - 'predecessors': Dict mapping node indices to predecessor nodes (if tracked)
            - 'visited_order': List of nodes in order of visit
            - 'reachable_nodes': Set of all nodes reachable from source
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or data format is invalid
        """
        if not self._is_fitted:
            raise ValueError('Transformer must be fitted before calling transform')
        bfs_result = bfs_with_path_tracking_dict(self._graph, source=self.source_node, max_depth=self.max_distance)
        self._distances = bfs_result['distances']
        self._predecessors = bfs_result['predecessors'] if self.track_predecessors else None
        self._visited = set(bfs_result['reachable_nodes'])
        result = {'distances': bfs_result['distances'], 'visited_order': bfs_result['visited_order'], 'reachable_nodes': bfs_result['reachable_nodes']}
        if self.track_predecessors:
            result['predecessors'] = bfs_result['predecessors']
        else:
            result['predecessors'] = None
        return result

    def inverse_transform(self, data: dict, **kwargs) -> dict:
        """
        Reconstruct the original graph representation from BFS results.
        
        This method converts the BFS tracking results back to a graph representation.
        
        Parameters
        ----------
        data : dict
            BFS results dictionary as returned by transform()
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        dict
            Dictionary with reconstructed graph information
        """
        if not isinstance(data, dict):
            raise TypeError('Input data must be a dictionary')
        edges = []
        if 'predecessors' in data and data['predecessors']:
            for (node, pred) in data['predecessors'].items():
                if pred is not None:
                    edges.append((pred, node))
        if 'visited_order' in data:
            visited = data['visited_order']
        return {'edges': edges, 'distances': data.get('distances', {}), 'reachable_nodes': data.get('reachable_nodes', set())}

    def get_shortest_path(self, target_node: int) -> List[int]:
        """
        Reconstruct the shortest path from source to target node.
        
        Uses the predecessor information to build the path from source to target.
        
        Parameters
        ----------
        target_node : int
            Node to find path to
            
        Returns
        -------
        List[int]
            List of nodes representing the shortest path from source to target,
            or empty list if no path exists or predecessors were not tracked
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or target node is invalid
        """
        if not self._is_fitted:
            raise ValueError('Transformer must be fitted before getting shortest path')
        if self._predecessors is None:
            raise ValueError('Predecessors were not tracked. Set track_predecessors=True during initialization.')
        if target_node not in self._distances:
            return []
        path = []
        current = target_node
        while current is not None:
            path.append(current)
            current = self._predecessors.get(current)
        path.reverse()
        if path and path[0] == self.source_node:
            return path
        else:
            return []

def bfs_with_path_tracking(graph: Union[np.ndarray, List[Tuple[int, int]], dict], source: int=0, max_depth: Optional[int]=None) -> dict:
    """
    Perform breadth-first search on a graph while tracking paths.
    
    This function executes BFS traversal from a source node and returns comprehensive
    path tracking information including distances, predecessors, and visit order.
    
    Parameters
    ----------
    graph : numpy.ndarray or List[Tuple[int, int]]
        Graph representation - either adjacency matrix (2D array) or edge list
    source : int, default=0
        Starting node for BFS traversal
    max_depth : int or None, default=None
        Maximum search depth (None for unlimited)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'distances': Dict mapping node indices to distances from source
        - 'predecessors': Dict mapping node indices to predecessor nodes
        - 'visited_order': List of nodes in order of visit
        - 'reachable_nodes': Set of all nodes reachable from source
        
    Raises
    ------
    ValueError
        If graph format is invalid or source node is out of bounds
    TypeError
        If graph is not in a supported format
    """
    if isinstance(graph, dict):
        return bfs_with_path_tracking_dict(graph, source, max_depth)
    elif isinstance(graph, np.ndarray):
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError('Adjacency matrix must be a square 2D array')
        if source < 0 or source >= graph.shape[0]:
            raise ValueError(f'Source node {source} is out of bounds for graph with {graph.shape[0]} nodes')
        adj_matrix = graph
        num_nodes = graph.shape[0]
    elif isinstance(graph, list):
        if not all((isinstance(edge, tuple) and len(edge) == 2 for edge in graph)):
            raise ValueError('Edge list must contain tuples of exactly 2 elements')
        all_nodes = set()
        for edge in graph:
            all_nodes.update(edge)
        if all_nodes:
            num_nodes = max(all_nodes) + 1
        else:
            num_nodes = max(source + 1, 1)
        if source < 0 or source >= num_nodes:
            raise ValueError(f'Source node {source} is out of bounds for graph with {num_nodes} nodes')
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for (u, v) in graph:
            if u < 0 or u >= num_nodes or v < 0 or (v >= num_nodes):
                raise ValueError(f'Edge ({u}, {v}) contains out-of-bounds node')
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
    else:
        raise TypeError('Graph must be a numpy array (adjacency matrix), list of tuples (edge list), or dict (adjacency list)')
    distances = {}
    predecessors = {}
    visited_order = []
    reachable_nodes = set()
    visited = set()
    distances[source] = 0
    predecessors[source] = None
    visited_order.append(source)
    reachable_nodes.add(source)
    visited.add(source)
    queue = [(source, 0)]
    while queue:
        (current_node, current_depth) = queue.pop(0)
        if max_depth is not None and current_depth >= max_depth:
            continue
        for neighbor in range(num_nodes):
            if adj_matrix[current_node, neighbor] == 1 and neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_depth + 1
                predecessors[neighbor] = current_node
                visited_order.append(neighbor)
                reachable_nodes.add(neighbor)
                queue.append((neighbor, current_depth + 1))
    return {'distances': distances, 'predecessors': predecessors, 'visited_order': visited_order, 'reachable_nodes': reachable_nodes}