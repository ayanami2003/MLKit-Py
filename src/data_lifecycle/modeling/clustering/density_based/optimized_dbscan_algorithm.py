from typing import Optional, Any
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class OptimizedDBSCANModel(BaseModel):

    def __init__(self, eps: float=0.5, min_samples: int=5, metric: str='euclidean', algorithm: str='auto', leaf_size: int=30, p: Optional[float]=None, n_jobs: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the OptimizedDBSCANModel.
        
        Args:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            metric (str): The metric to use when calculating distance between instances.
            algorithm (str): The algorithm to use for computing pointwise distances.
            leaf_size (int): Leaf size passed to BallTree or KDTree.
            p (Optional[float]): The power of the Minkowski metric to be used.
            n_jobs (Optional[int]): The number of parallel jobs to run.
            name (Optional[str]): Name of the model instance.
        """
        super().__init__(name=name)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.is_fitted = False

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'OptimizedDBSCANModel':
        """
        Perform DBSCAN clustering on the input feature set.
        
        Args:
            X (FeatureSet): The input features to cluster.
            y (Any, optional): Not used in unsupervised learning, kept for API consistency.
            
        Returns:
            OptimizedDBSCANModel: Self instance for method chaining.
        """
        if isinstance(X, FeatureSet):
            data = X.features
        else:
            data = np.array(X)
        if data.size == 0 or data.shape[0] == 0:
            self._labels = np.array([])
            self._core_samples = np.array([], dtype=bool)
            self.is_fitted = True
            return self
        if data.shape[0] == 1:
            if self.min_samples <= 1:
                self._labels = np.array([0])
                self._core_samples = np.array([True])
            else:
                self._labels = np.array([-1])
                self._core_samples = np.array([False])
            self.is_fitted = True
            return self
        self._dbscan_model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, n_jobs=self.n_jobs)
        self._labels = self._dbscan_model.fit_predict(data)
        self._core_samples = self._dbscan_model.core_sample_indices_
        self.is_fitted = True
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Args:
            X (FeatureSet): The input features to predict cluster labels for.
            
        Returns:
            np.ndarray: Cluster labels for each sample. Noisy samples are given the label -1.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before calling predict()')
        if isinstance(X, FeatureSet):
            data = X.features
        else:
            data = np.array(X)
        if data.size == 0 or data.shape[0] == 0:
            return np.array([])
        try:
            if hasattr(self, '_dbscan_model'):
                from sklearn.neighbors import NearestNeighbors
                core_samples = self._dbscan_model.components_
                if len(core_samples) > 0:
                    nn = NearestNeighbors(radius=self.eps, algorithm=self.algorithm, leaf_size=self.leaf_size, metric=self.metric, p=self.p, n_jobs=self.n_jobs)
                    nn.fit(core_samples)
                    (distances, indices) = nn.radius_neighbors(data)
                    predictions = np.full(data.shape[0], -1, dtype=int)
                    for i in range(data.shape[0]):
                        if len(indices[i]) > 0:
                            neighbor_labels = self._labels[self._core_samples][indices[i]]
                            valid_labels = neighbor_labels[neighbor_labels != -1]
                            if len(valid_labels) > 0:
                                (values, counts) = np.unique(valid_labels, return_counts=True)
                                predictions[i] = values[np.argmax(counts)]
                    return predictions
                else:
                    return np.full(data.shape[0], -1, dtype=int)
            else:
                return np.full(data.shape[0], -1, dtype=int)
        except Exception:
            return np.full(data.shape[0], -1, dtype=int)

    def fit_predict(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> np.ndarray:
        """
        Perform DBSCAN clustering and return cluster labels for the input data.
        
        This is equivalent to calling fit(X) followed by predict(X).
        
        Args:
            X (FeatureSet): The input features to cluster.
            y (Any, optional): Not used in unsupervised learning, kept for API consistency.
            
        Returns:
            np.ndarray: Cluster labels for each sample. Noisy samples are given the label -1.
        """
        self.fit(X, y, **kwargs)
        return self._labels

    def score(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> float:
        """
        Return the negative intra-cluster sum of distances as a score.
        
        Higher (less negative) values indicate better clustering.
        
        Args:
            X (FeatureSet): The input features to evaluate.
            y (Any, optional): Not used in unsupervised learning, kept for API consistency.
            
        Returns:
            float: Negative intra-cluster sum of distances.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before calling score()')
        if isinstance(X, FeatureSet):
            data = X.features
        else:
            data = np.array(X)
        if data.size == 0 or data.shape[0] == 0 or len(self._labels) == 0:
            return 0.0
        total_distance = 0.0
        unique_labels = set(self._labels)
        unique_labels.discard(-1)
        for label in unique_labels:
            cluster_points = data[self._labels == label]
            if len(cluster_points) > 1:
                distances = pairwise_distances(cluster_points, metric=self.metric)
                total_distance += np.sum(distances[np.triu_indices_from(distances, k=1)])
        return -total_distance