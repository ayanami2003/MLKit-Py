import numpy as np
from typing import Union, Optional, List, Tuple
from general.base_classes.transformer_base import BaseTransformer
import heapq
from typing import Union, Optional, List, Tuple, Dict, Set

class PriorityQueueOptimizer(BaseTransformer):

    def __init__(self, priority_metric: str='distance', maximize: bool=False, max_iterations: int=1000, tolerance: float=1e-06, name: Optional[str]=None):
        """
        Initialize the PriorityQueueOptimizer.
        
        Parameters
        ----------
        priority_metric : str, optional
            The metric used to determine node priority ('distance', 'cost', 'heuristic')
        maximize : bool, optional
            Whether to maximize (True) or minimize (False) the priority metric
        max_iterations : int, optional
            Maximum number of iterations for the optimization process
        tolerance : float, optional
            Convergence tolerance for stopping criteria
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.priority_metric = priority_metric
        self.maximize = maximize
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._graph: Optional[Dict[int, List[Tuple[int, float]]]] = None
        self._nodes: Optional[Set[int]] = None
        self._is_fitted = False

    def fit(self, data: Union[np.ndarray, List[Tuple]], **kwargs) -> 'PriorityQueueOptimizer':
        """
        Fit the optimizer to the graph data.
        
        Parameters
        ----------
        data : Union[np.ndarray, List[Tuple]]
            Graph representation as adjacency matrix or list of edges
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        PriorityQueueOptimizer
            Self instance for method chaining
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError('Adjacency matrix must be a square 2D array')
            n_nodes = data.shape[0]
            self._graph = {i: [] for i in range(n_nodes)}
            self._nodes = set(range(n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and data[i, j] != 0:
                        self._graph[i].append((j, float(data[i, j])))
        elif isinstance(data, list):
            self._graph = {}
            self._nodes = set()
            for edge in data:
                if len(edge) < 2:
                    raise ValueError('Each edge must have at least source and target nodes')
                (source, target) = (edge[0], edge[1])
                weight = float(edge[2]) if len(edge) > 2 else 1.0
                self._nodes.add(source)
                self._nodes.add(target)
                if source not in self._graph:
                    self._graph[source] = []
                if target not in self._graph:
                    self._graph[target] = []
                self._graph[source].append((target, weight))
        else:
            raise TypeError('Data must be either a numpy array (adjacency matrix) or list of edges')
        self._is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, List[Tuple]], **kwargs) -> np.ndarray:
        """
        Apply priority queue optimization to find optimal paths or solutions.
        
        Parameters
        ----------
        data : Union[np.ndarray, List[Tuple]]
            Graph representation as adjacency matrix or list of edges (ignored if already fitted)
        **kwargs : dict
            Additional transformation parameters (e.g., start_node, end_node)
            
        Returns
        -------
        np.ndarray
            Optimized path or solution based on priority queue optimization
        """
        if not self._is_fitted:
            self.fit(data)
        start_node = kwargs.get('start_node')
        end_node = kwargs.get('end_node')
        if start_node is None:
            raise ValueError('start_node must be provided')
        if start_node not in self._nodes:
            raise ValueError(f'Start node {start_node} not found in graph')
        if end_node is not None and end_node not in self._nodes:
            raise ValueError(f'End node {end_node} not found in graph')
        (distances, previous) = self._dijkstra(start_node, end_node)
        if end_node is not None:
            path = self._reconstruct_path(previous, start_node, end_node)
            return np.array(path)
        else:
            result = []
            for node in sorted(self._nodes):
                result.append([node, distances.get(node, np.inf), previous.get(node, None)])
            return np.array(result, dtype=object)

    def _dijkstra(self, start_node: int, end_node: Optional[int]=None) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """
        Implementation of Dijkstra's algorithm using a priority queue.
        
        Parameters
        ----------
        start_node : int
            Starting node for the search
        end_node : int, optional
            Target node for early termination
            
        Returns
        -------
        Tuple[Dict[int, float], Dict[int, Optional[int]]]
            Distances to all nodes and previous nodes for path reconstruction
        """
        distances = {node: np.inf for node in self._nodes}
        previous = {node: None for node in self._nodes}
        distances[start_node] = 0
        pq = [(0, start_node)]
        visited = set()
        iteration = 0
        while pq and iteration < self.max_iterations:
            (current_priority, current_node) = heapq.heappop(pq)
            current_distance = -current_priority if self.maximize else current_priority
            if current_node in visited:
                iteration += 1
                continue
            visited.add(current_node)
            if current_node == end_node:
                break
            if current_node in self._graph:
                for (neighbor, weight) in self._graph[current_node]:
                    if neighbor in visited:
                        continue
                    new_distance = current_distance + weight
                    if self.maximize:
                        if new_distance > distances[neighbor]:
                            distances[neighbor] = new_distance
                            previous[neighbor] = current_node
                            heapq.heappush(pq, (-new_distance, neighbor))
                    elif new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (new_distance, neighbor))
            iteration += 1
        return (distances, previous)

    def _reconstruct_path(self, previous: Dict[int, Optional[int]], start_node: int, end_node: int) -> List[int]:
        """
        Reconstruct the path from start to end node using previous node information.
        
        Parameters
        ----------
        previous : Dict[int, Optional[int]]
            Dictionary mapping each node to its predecessor in the shortest path
        start_node : int
            Starting node
        end_node : int
            Target node
            
        Returns
        -------
        List[int]
            Path from start to end node
        """
        if previous.get(end_node) is None and end_node != start_node:
            if end_node not in previous or previous[end_node] is None:
                return []
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            if current not in previous:
                return []
            current = previous[current]
            if self._nodes is not None and len(path) > len(self._nodes):
                return []
        path.reverse()
        if path and path[0] == start_node:
            return path
        else:
            return []

    def inverse_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Reverse the optimization transformation if applicable.
        
        Parameters
        ----------
        data : np.ndarray
            Transformed data to be inverted
        **kwargs : dict
            Additional inversion parameters
            
        Returns
        -------
        np.ndarray
            Original data representation
        """
        return data

def priority_queue_optimization(graph: Union[np.ndarray, List[Tuple]], start_node: int, end_node: Optional[int]=None, priority_metric: str='distance', maximize: bool=False, max_iterations: int=1000, tolerance: float=1e-06) -> Union[List[int], Tuple[List[int], float]]:
    """
    Optimize graph traversal using a priority queue approach.
    
    This function implements Dijkstra's algorithm or A* search depending on the
    priority metric and whether an end node is specified. It efficiently finds
    the shortest path or optimal solution in a graph by always expanding the
    most promising node first.
    
    Parameters
    ----------
    graph : Union[np.ndarray, List[Tuple]]
        Graph representation as adjacency matrix or list of edges (source, target, weight)
    start_node : int
        Index of the starting node for traversal
    end_node : int, optional
        Index of the target node (if None, finds paths to all nodes)
    priority_metric : str, optional
        The metric used to determine node priority ('distance', 'cost', 'heuristic')
    maximize : bool, optional
        Whether to maximize (True) or minimize (False) the priority metric
    max_iterations : int, optional
        Maximum number of iterations for the optimization process
    tolerance : float, optional
        Convergence tolerance for stopping criteria
        
    Returns
    -------
    Union[List[int], Tuple[List[int], float]]
        If end_node is specified: (optimal path as list of node indices, total cost)
        If end_node is None: list of node indices in traversal order
        
    Raises
    ------
    ValueError
        If start_node or end_node are out of bounds
    RuntimeError
        If no path exists between start and end nodes
    """
    optimizer = PriorityQueueOptimizer(priority_metric=priority_metric, maximize=maximize, max_iterations=max_iterations, tolerance=tolerance)
    optimizer.fit(graph)
    if end_node is not None:
        path = optimizer.transform(graph, start_node=start_node, end_node=end_node)
        if len(path) == 0:
            raise RuntimeError(f'No path found from node {start_node} to node {end_node}')
        total_cost = 0.0
        if isinstance(graph, np.ndarray):
            for i in range(len(path) - 1):
                total_cost += graph[path[i], path[i + 1]]
        else:
            edge_map = {}
            for edge in graph:
                (source, target) = (edge[0], edge[1])
                weight = float(edge[2]) if len(edge) > 2 else 1.0
                if source not in edge_map:
                    edge_map[source] = {}
                edge_map[source][target] = weight
            for i in range(len(path) - 1):
                (source, target) = (path[i], path[i + 1])
                if source in edge_map and target in edge_map[source]:
                    total_cost += edge_map[source][target]
                else:
                    raise RuntimeError(f'Edge from {source} to {target} not found in graph')
        return (path.tolist(), float(total_cost))
    else:
        result = optimizer.transform(graph, start_node=start_node)
        reachable_nodes = [(int(row[0]), float(row[1])) for row in result if not np.isinf(row[1])]
        reachable_nodes.sort(key=lambda x: x[1])
        traversal_order = [node_id for (node_id, _) in reachable_nodes]
        return traversal_order