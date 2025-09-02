from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
DEFAULT_MODEL_NAME = 'OPTICSClusteringModel'

class OPTICSClusteringModel(BaseModel):

    def __init__(self, min_samples: int=5, max_eps: float=np.inf, metric: str='euclidean', cluster_method: str='xi', xi: float=0.05, predecessor_correction: bool=True, min_cluster_size: Optional[Union[int, float]]=None, algorithm: str='auto', leaf_size: int=30, memory: Optional[Union[str, object]]=None, n_jobs: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the OPTICS clustering model.
        
        Parameters
        ----------
        min_samples : int, default=5
            The number of samples in a neighborhood for a point to be considered a core point.
        max_eps : float, default=np.inf
            The maximum neighborhood radius for determining connectivity.
        metric : str, default='euclidean'
            The distance metric to use for computing distances between points.
        cluster_method : str, default='xi'
            The extraction method used to extract clusters from the reachability plot.
        xi : float, default=0.05
            Determines the minimum steepness on the reachability plot that constitutes
            a cluster boundary when using the 'xi' cluster method.
        predecessor_correction : bool, default=True
            Correct clusters based on the calculated predecessors.
        min_cluster_size : int or float, optional
            Minimum number of samples in a cluster.
        algorithm : str, default='auto'
            Algorithm used to compute point-wise distances.
        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree for faster queries.
        memory : Union[str, object], default=None
            Used to cache the output of the computation to speed up consecutive runs.
        n_jobs : int, default=None
            Number of parallel jobs to run for neighbors search.
        name : str, optional
            Name of the model instance.
        """
        BaseModel.__init__(self, name=name)
        if name is None:
            self.name = None
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.cluster_method = cluster_method
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, X: FeatureSet, y: Optional[np.ndarray]=None, **kwargs) -> 'OPTICSClusteringModel':
        """
        Fit the OPTICS clustering model to the input data.
        
        This method computes the core distances, reachability distances, and creates
        an ordered representation of the data based on reachability.
        
        Parameters
        ----------
        X : FeatureSet
            Input features to cluster. Must contain a 2D array of shape (n_samples, n_features).
        y : np.ndarray, optional
            Target values (ignored in unsupervised learning).
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        OPTICSClusteringModel
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is invalid or incompatible.
        """
        if not isinstance(X, FeatureSet):
            raise ValueError('X must be an instance of FeatureSet')
        if X.features is None or len(X.features.shape) != 2:
            raise ValueError('X.features must be a 2D array')
        self._X = X.features
        n_samples = self._X.shape[0]
        self._reachability_ = np.full(n_samples, np.inf)
        self._core_distances_ = np.full(n_samples, np.inf)
        self._ordering_ = np.zeros(n_samples, dtype=int)
        self._predecessor_ = np.full(n_samples, -1, dtype=int)
        nbrs = NearestNeighbors(n_neighbors=self.min_samples, algorithm=self.algorithm, leaf_size=self.leaf_size, metric=self.metric, n_jobs=self.n_jobs)
        nbrs.fit(self._X)
        (distances, indices) = nbrs.kneighbors(self._X)
        core_distances = distances[:, -1]
        self._core_distances_ = np.where(core_distances <= self.max_eps, core_distances, np.inf)
        processed = np.zeros(n_samples, dtype=bool)
        ordered_list = []
        for point_idx in range(n_samples):
            if processed[point_idx]:
                continue
            processed[point_idx] = True
            ordered_list.append(point_idx)
            if self._core_distances_[point_idx] == np.inf:
                continue
            (neighbors_distances, neighbors_indices) = nbrs.radius_neighbors(self._X[point_idx:point_idx + 1], radius=self.max_eps)
            neighbors_distances = neighbors_distances[0]
            neighbors_indices = neighbors_indices[0]
            for (i, neighbor_idx) in enumerate(neighbors_indices):
                if processed[neighbor_idx]:
                    continue
                reach_dist = max(self._core_distances_[point_idx], neighbors_distances[i])
                if reach_dist < self._reachability_[neighbor_idx]:
                    self._reachability_[neighbor_idx] = reach_dist
                    self._predecessor_[neighbor_idx] = point_idx
        unprocessed_indices = np.where(~processed)[0]
        if len(unprocessed_indices) > 0:
            ordered_list.extend(unprocessed_indices)
        self._ordering_ = np.array(ordered_list)
        self._labels_ = np.full(n_samples, -2, dtype=int)
        self.labels_ = self._labels_
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Extract clusters from the fitted OPTICS model.
        
        This method uses the reachability plot generated during fitting to extract
        clusters according to the specified cluster method.
        
        Parameters
        ----------
        X : FeatureSet
            Input features for which to extract clusters. Should be the same data used for fitting.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Cluster labels for each point in the dataset. Noisy samples are given the label -1.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, '_X'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'predict'.")
        if self.cluster_method == 'dbscan':
            labels = self._extract_dbscan_clusters()
        else:
            labels = self._extract_xi_clusters()
        self.labels_ = labels
        return labels

    def _extract_dbscan_clusters(self) -> np.ndarray:
        """Extract clusters using DBSCAN approach."""
        labels = np.full(len(self._ordering_), -1, dtype=int)
        cluster_id = 0
        for i in range(len(self._ordering_)):
            point_idx = self._ordering_[i]
            if self._reachability_[point_idx] <= self.max_eps and labels[point_idx] == -1:
                labels[point_idx] = cluster_id
                j = i + 1
                while j < len(self._ordering_):
                    next_point_idx = self._ordering_[j]
                    if self._reachability_[next_point_idx] > self.max_eps:
                        break
                    if labels[next_point_idx] == -1:
                        labels[next_point_idx] = cluster_id
                    j += 1
                cluster_id += 1
        return labels

    def _extract_xi_clusters(self) -> np.ndarray:
        """Extract clusters using Xi approach."""
        labels = np.full(len(self._ordering_), -1, dtype=int)
        cluster_id = 0
        for i in range(len(self._ordering_)):
            point_idx = self._ordering_[i]
            if self._reachability_[point_idx] < self.max_eps * (1 - self.xi):
                labels[point_idx] = cluster_id
            elif self._reachability_[point_idx] != np.inf:
                cluster_id += 1
                labels[point_idx] = cluster_id
        return labels

    def score(self, X: FeatureSet, y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Calculate silhouette score for the clustering result.
        
        Parameters
        ----------
        X : FeatureSet
            Input features to evaluate.
        y : np.ndarray, optional
            True labels for X (not used in unsupervised evaluation).
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Silhouette score of the clustering result.
        """
        if not hasattr(self, '_X'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before 'score'.")
        labels = self.predict(X)
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0
        filtered_X = self._X[mask]
        filtered_labels = labels[mask]
        unique_labels = np.unique(filtered_labels)
        if len(unique_labels) < 2:
            return 0.0
        return silhouette_score(filtered_X, filtered_labels, metric=self.metric)

    def get_reachability_distances(self) -> np.ndarray:
        """
        Get the reachability distances computed during fitting.
        
        Returns
        -------
        np.ndarray
            Array of reachability distances for each sample in the order they were processed.
        """
        if not hasattr(self, '_reachability_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before accessing reachability distances.")
        return self._reachability_.copy()

    def get_ordering(self) -> np.ndarray:
        """
        Get the cluster ordered indices computed during fitting.
        
        Returns
        -------
        np.ndarray
            Indices of samples in the order they were processed by the OPTICS algorithm.
        """
        if not hasattr(self, '_ordering_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before accessing ordering.")
        return self._ordering_.copy()

    def get_core_distances(self) -> np.ndarray:
        """
        Get the core distances computed during fitting.
        
        Returns
        -------
        np.ndarray
            Array of core distances for each sample.
        """
        if not hasattr(self, '_core_distances_'):
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before accessing core distances.")
        return self._core_distances_.copy()