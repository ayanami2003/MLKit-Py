from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

class HDBSCANClusteringModel(BaseModel):

    def __init__(self, min_cluster_size: int=5, min_samples: Optional[int]=None, cluster_selection_method: str='eom', allow_single_cluster: bool=False, metric: str='euclidean', alpha: float=1.0, match_reference_implementation: bool=False, name: Optional[str]=None):
        """
        Initialize the HDBSCAN clustering model.
        
        Parameters
        ----------
        min_cluster_size : int
            The minimum number of samples in a cluster (default: 5)
        min_samples : Optional[int]
            The number of samples in a neighborhood for a point to be considered a core point.
            If None, defaults to min_cluster_size (default: None)
        cluster_selection_method : str
            The method used to select clusters from the condensed tree ('eom' or 'leaf')
            (default: 'eom')
        allow_single_cluster : bool
            Whether to allow a single cluster to be returned (default: False)
        metric : str
            The metric to use when calculating distance between instances (default: 'euclidean')
        alpha : float
            A distance scaling parameter for stabilizing the cluster hierarchy (default: 1.0)
        match_reference_implementation : bool
            Whether to match the reference implementation's behavior exactly (default: False)
        name : Optional[str]
            Name for the model instance (default: None, uses class name)
        """
        super().__init__(name=name)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.metric = metric
        self.alpha = alpha
        self.match_reference_implementation = match_reference_implementation
        self._labels = None
        self._probabilities = None
        self._outliers = None
        self._X_fit = None

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'HDBSCANClusteringModel':
        """
        Fit the HDBSCAN clustering model to the input data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training data features. If FeatureSet, uses the features attribute.
            If ndarray, expects shape (n_samples, n_features)
        y : Optional[np.ndarray]
            Target values (ignored in unsupervised learning)
        **kwargs : dict
            Additional fitting parameters (reserved for future use)
            
        Returns
        -------
        HDBSCANClusteringModel
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data dimensions are incompatible
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        if X_array.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self._X_fit = X_array.copy()
        core_distances = self._compute_core_distances(X_array)
        mr_distances = self._compute_mutual_reachability(X_array, core_distances)
        mst = self._build_minimum_spanning_tree(mr_distances)
        condensed_tree = self._build_condensed_tree(mst)
        (self._labels, self._probabilities) = self._extract_clusters(condensed_tree, X_array.shape[0])
        self._outliers = np.where(self._labels == -1)[0]
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Note: HDBSCAN does not naturally support predicting on new data.
        This implementation will find the closest cluster member and assign
        that cluster label, or label as noise if no close neighbor exists.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            New data points to assign cluster labels to
        **kwargs : dict
            Additional prediction parameters (reserved for future use)
            
        Returns
        -------
        np.ndarray
            Array of cluster labels for each input point. Noisy samples are
            labeled as -1.
        """
        if self._X_fit is None:
            raise ValueError('Model must be fitted before making predictions')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        from scipy.spatial.distance import cdist
        distances = cdist(X_array, self._X_fit, metric=self.metric)
        closest_indices = np.argmin(distances, axis=1)
        return self._labels[closest_indices]

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and return cluster labels for the training data.
        
        This is the primary method for HDBSCAN clustering, as it computes
        clusters directly from the training data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training data features
        y : Optional[np.ndarray]
            Target values (ignored in unsupervised learning)
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        np.ndarray
            Array of cluster labels for each training point. Noisy samples
            are labeled as -1.
        """
        self.fit(X, y, **kwargs)
        return self._labels

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Calculate the clustering score (higher is better).
        
        Uses the average silhouette score to evaluate clustering quality.
        Noise points (-1 labels) are excluded from the calculation.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data to evaluate clustering on
        y : Optional[np.ndarray]
            True labels (if available, for comparison)
        **kwargs : dict
            Additional scoring parameters
            
        Returns
        -------
        float
            Average silhouette score for the clustering, bounded [-1, 1].
            Higher values indicate better clustering.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        labels = self.fit_predict(X_array, y, **kwargs)
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0
        return silhouette_score(X_array[mask], labels[mask], metric=self.metric)

    def get_probabilities(self) -> np.ndarray:
        """
        Get the probability that each sample is a member of its assigned cluster.
        
        These probabilities are calculated during fitting and represent the
        confidence in cluster assignments.
        
        Returns
        -------
        np.ndarray
            Array of membership probabilities for each sample
        """
        if self._probabilities is None:
            raise ValueError('Model must be fitted before accessing probabilities')
        return self._probabilities

    def get_outliers(self) -> np.ndarray:
        """
        Get indices of outlier/noise points identified during clustering.
        
        Returns
        -------
        np.ndarray
            Indices of samples labeled as noise points (-1)
        """
        if self._outliers is None:
            raise ValueError('Model must be fitted before accessing outliers')
        return self._outliers

    def _compute_core_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute core distances for each point."""
        n_points = X.shape[0]
        core_distances = np.zeros(n_points)
        distances = squareform(pdist(X, metric=self.metric))
        for i in range(n_points):
            sorted_distances = np.sort(distances[i])
            core_distances[i] = sorted_distances[min(self.min_samples, len(sorted_distances) - 1)]
        return core_distances

    def _compute_mutual_reachability(self, X: np.ndarray, core_distances: np.ndarray) -> np.ndarray:
        """Compute mutual reachability distances."""
        distances = squareform(pdist(X, metric=self.metric))
        n_points = X.shape[0]
        mr_distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i + 1, n_points):
                mr_dist = max(core_distances[i], core_distances[j], distances[i, j])
                mr_distances[i, j] = mr_dist
                mr_distances[j, i] = mr_dist
        return mr_distances

    def _build_minimum_spanning_tree(self, distances: np.ndarray) -> np.ndarray:
        """Build minimum spanning tree using Kruskal's algorithm."""
        n_points = distances.shape[0]
        edges = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                edges.append((distances[i, j], i, j))
        edges.sort()
        parent = list(range(n_points))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            (px, py) = (find(x), find(y))
            if px != py:
                parent[px] = py
                return True
            return False
        mst_edges = []
        for (weight, i, j) in edges:
            if union(i, j):
                mst_edges.append([i, j, weight])
                if len(mst_edges) == n_points - 1:
                    break
        return np.array(mst_edges)

    def _build_condensed_tree(self, mst: np.ndarray) -> list:
        """Build condensed tree from minimum spanning tree."""
        mst = mst[mst[:, 2].argsort()]
        n_points = int(max(np.max(mst[:, 0]), np.max(mst[:, 1]))) + 1
        parent = list(range(n_points))
        cluster_sizes = [1] * n_points
        next_cluster_id = n_points
        condensed_tree = []
        for edge in mst:
            (left, right, lambda_val) = (int(edge[0]), int(edge[1]), edge[2])
            root_left = self._uf_find(parent, left)
            root_right = self._uf_find(parent, right)
            if root_left != root_right:
                left_size = cluster_sizes[root_left]
                right_size = cluster_sizes[root_right]
                condensed_tree.append((root_left, next_cluster_id, lambda_val, left_size))
                condensed_tree.append((root_right, next_cluster_id, lambda_val, right_size))
                self._uf_union(parent, cluster_sizes, root_left, root_right, next_cluster_id)
                next_cluster_id += 1
        return condensed_tree

    def _uf_find(self, parent: list, x: int) -> int:
        """Union-find find operation with path compression."""
        if parent[x] != x:
            parent[x] = self._uf_find(parent, parent[x])
        return parent[x]

    def _uf_union(self, parent: list, cluster_sizes: list, x: int, y: int, new_id: int) -> None:
        """Union-find union operation."""
        parent[x] = new_id
        parent[y] = new_id
        cluster_sizes.append(cluster_sizes[x] + cluster_sizes[y])

    def _extract_clusters(self, condensed_tree: list, n_points: int) -> tuple:
        """Extract flat clusters from condensed tree."""
        labels = np.full(n_points, -1, dtype=int)
        probabilities = np.zeros(n_points)
        if not condensed_tree:
            return (labels, probabilities)
        tree_array = np.array(condensed_tree)
        clusters = {}
        for entry in condensed_tree:
            (cluster_id, parent_id, lambda_val, child_size) = entry
            if cluster_id not in clusters:
                clusters[cluster_id] = {'lambda_birth': lambda_val, 'size': child_size, 'death': np.inf}
            else:
                clusters[cluster_id]['death'] = min(clusters[cluster_id]['death'], lambda_val)
        stabilities = {}
        for (cluster_id, info) in clusters.items():
            if info['size'] >= self.min_cluster_size:
                stabilities[cluster_id] = (info['death'] - info['lambda_birth']) * info['size']
        if self.cluster_selection_method == 'eom':
            selected_clusters = self._eom_extract(stabilities, condensed_tree)
        else:
            selected_clusters = [cid for cid in stabilities.keys()]
        cluster_labels = {}
        next_label = 0
        for cluster_id in selected_clusters:
            if cluster_id < n_points:
                labels[cluster_id] = next_label
                cluster_labels[cluster_id] = next_label
                probabilities[cluster_id] = 1.0
                next_label += 1
        for entry in reversed(condensed_tree):
            (child_id, parent_id, lambda_val, child_size) = entry
            if child_id in cluster_labels and parent_id not in cluster_labels:
                labels[parent_id] = cluster_labels[child_id]
                probabilities[parent_id] = max(0, 1 - lambda_val)
        return (labels, probabilities)

    def _eom_extract(self, stabilities: dict, condensed_tree: list) -> list:
        """Extract clusters using Excess of Mass algorithm."""
        sorted_clusters = sorted(stabilities.items(), key=lambda x: x[1], reverse=True)
        selected = []
        for (cluster_id, stability) in sorted_clusters:
            overlap = False
            for selected_id in selected:
                if cluster_id == selected_id:
                    overlap = True
                    break
            if not overlap:
                selected.append(cluster_id)
                if not self.allow_single_cluster:
                    break
        return selected