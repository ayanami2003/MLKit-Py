from typing import Any, Dict, List, Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
import heapq

class DijkstraShortestPathComputer(BaseModel):

    def __init__(self, source_node: Any, max_distance: Optional[float]=None, return_predecessors: bool=False, name: Optional[str]=None):
        """
        Initialize the Dijkstra shortest path computer.
        
        Args:
            source_node (Any): The node from which to compute shortest paths.
            max_distance (Optional[float]): Optional maximum distance threshold. Paths longer than
                this will not be computed.
            return_predecessors (bool): If True, the model will also compute and store predecessor
                information to enable path reconstruction.
            name (Optional[str]): Optional name for the model instance.
        """
        super().__init__(name)
        self.source_node = source_node
        self.max_distance = max_distance
        self.return_predecessors = return_predecessors

    def fit(self, X: Union[np.ndarray, DataBatch], y: Optional[Any]=None, **kwargs) -> 'DijkstraShortestPathComputer':
        """
        Fit the model by computing shortest paths from the source node.
        
        For Dijkstra's algorithm, fitting involves running the algorithm on the provided graph
        representation to compute distances (and optionally predecessors) from the source node
        to all reachable nodes.
        
        Args:
            X (Union[np.ndarray, DataBatch]): Graph representation. If numpy array, assumed to be
                an adjacency matrix where X[i,j] represents the weight of edge from node i to node j.
                If DataBatch, should contain graph data in an appropriate format.
            y (Optional[Any]): Not used in this implementation but kept for API consistency.
            **kwargs: Additional parameters (reserved for future extensions).
            
        Returns:
            DijkstraShortestPathComputer: Returns self for method chaining.
        """
        if hasattr(X, 'data') and hasattr(X, '__class__') and (X.__class__.__name__ == 'DataBatch'):
            adj_matrix = X.data
        elif isinstance(X, np.ndarray):
            adj_matrix = X
        else:
            raise TypeError('Input X must be either a numpy array or a DataBatch.')
        n_nodes = adj_matrix.shape[0]
        if not isinstance(self.source_node, int) or not 0 <= self.source_node < n_nodes:
            raise ValueError(f'Source node {self.source_node} is invalid for graph with {n_nodes} nodes.')
        self.distances_ = np.full(n_nodes, np.inf)
        self.distances_[self.source_node] = 0
        if self.return_predecessors:
            self.predecessors_ = np.full(n_nodes, -1)
            self.predecessors_[self.source_node] = -1
        else:
            self.predecessors_ = None
        pq = [(0, self.source_node)]
        visited = set()
        while pq:
            (current_dist, u) = heapq.heappop(pq)
            if u in visited:
                continue
            if self.max_distance is not None and current_dist > self.max_distance:
                self.distances_[u] = np.inf
                visited.add(u)
                continue
            visited.add(u)
            for v in range(n_nodes):
                weight = adj_matrix[u, v]
                if weight <= 0 or np.isinf(weight) or np.isnan(weight):
                    continue
                new_dist = current_dist + weight
                if self.max_distance is not None and new_dist > self.max_distance:
                    continue
                if new_dist < self.distances_[v]:
                    self.distances_[v] = new_dist
                    if self.return_predecessors:
                        self.predecessors_[v] = u
                    heapq.heappush(pq, (new_dist, v))
        return self

    def predict(self, X: Union[np.ndarray, DataBatch], **kwargs) -> np.ndarray:
        """
        Return the computed shortest distances from the source node.
        
        After fitting, this method returns the shortest path distances from the source node
        to all other nodes in the graph.
        
        Args:
            X (Union[np.ndarray, DataBatch]): Graph representation (same format as used in fit).
                This parameter is primarily for API consistency; the actual computation
                was performed during fit().
            **kwargs: Additional parameters (reserved for future extensions).
            
        Returns:
            np.ndarray: Array of shortest distances from source node to all nodes.
                Unreachable nodes will have infinite distance.
        """
        if not hasattr(self, 'distances_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        return self.distances_.copy()

    def score(self, X: Union[np.ndarray, DataBatch], y: Any, **kwargs) -> float:
        """
        Evaluate the model performance (not typically used for this algorithm).
        
        Since Dijkstra's algorithm is deterministic and exact, this method serves mainly
        for API compatibility. It could be extended to verify correctness against known solutions.
        
        Args:
            X (Union[np.ndarray, DataBatch]): Graph representation.
            y (Any): Expected distances or solution for comparison.
            **kwargs: Additional parameters.
            
        Returns:
            float: A score indicating how well the computed paths match expectations.
                For exact algorithms like Dijkstra, this would typically be 1.0 for correct
                solutions and lower for incorrect ones.
        """
        if not hasattr(self, 'distances_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'score'.")
        predicted = self.predict(X)
        if isinstance(y, dict):
            y_array = np.array([y.get(i, np.inf) for i in range(len(predicted))])
        else:
            y_array = np.asarray(y)
        matches = np.isclose(predicted, y_array, equal_nan=True)
        return float(np.mean(matches))

    def get_shortest_path_to(self, target_node: Any) -> List[Any]:
        """
        Reconstruct the shortest path from source to a specific target node.
        
        This method requires that the model was fitted with return_predecessors=True.
        
        Args:
            target_node (Any): The destination node for which to reconstruct the path.
            
        Returns:
            List[Any]: Ordered list of nodes representing the shortest path from
                source_node to target_node. Returns empty list if no path exists.
                
        Raises:
            ValueError: If predecessors were not computed during fitting.
        """
        if not hasattr(self, 'predecessors_') or self.predecessors_ is None:
            raise ValueError('Predecessor information not available. Fit the model with return_predecessors=True.')
        if not isinstance(target_node, int) or target_node < 0 or target_node >= len(self.distances_):
            raise ValueError(f'Target node {target_node} is invalid.')
        if np.isinf(self.distances_[target_node]):
            return []
        path = []
        current = target_node
        while current != -1:
            path.append(current)
            current = self.predecessors_[current]
        if path and path[-1] == self.source_node:
            return path[::-1]
        else:
            return []

    def get_distances(self) -> Dict[Any, float]:
        """
        Get the mapping of all nodes to their shortest distances from source.
        
        Returns:
            Dict[Any, float]: Dictionary mapping each node to its shortest distance
                from the source node. Unreachable nodes are not included.
        """
        if not hasattr(self, 'distances_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'get_distances'.")
        result = {}
        for (i, dist) in enumerate(self.distances_):
            if np.isfinite(dist):
                result[i] = dist
        return result