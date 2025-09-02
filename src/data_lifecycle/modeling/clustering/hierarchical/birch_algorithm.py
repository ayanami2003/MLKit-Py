from typing import Optional, Any, List, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from sklearn.cluster import AgglomerativeClustering

class BirchClusteringModel(BaseModel):

    def __init__(self, threshold: float=0.5, branching_factor: int=50, n_clusters: Optional[int]=3, compute_labels: bool=True, copy: bool=True):
        """
        Initialize the BIRCH clustering model.
        
        Parameters
        ----------
        threshold : float, default=0.5
            Maximum radius of a cluster in the leaf nodes of the CFT.
            Smaller values result in more clusters.
        branching_factor : int, default=50
            Maximum number of children a node can have in the CFT.
            Larger values result in better clustering quality but higher memory usage.
        n_clusters : int or None, default=3
            Number of clusters to find. If None, automatically determine the number.
            If -1, use the labels from the leaf nodes of the CFT.
        compute_labels : bool, default=True
            Whether to compute labels for the leaf nodes.
        copy : bool, default=True
            Whether to make a copy of the input data.
        """
        super().__init__(name='BirchClustering')
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels
        self.copy = copy
        self._cftree = None
        self._subcluster_centers = None
        self.labels_ = None

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'BirchClusteringModel':
        """
        Build the CFT (Clustering Feature Tree) for the input data.
        
        This method constructs the Clustering Feature Tree by scanning the data once,
        creating subclusters in the leaf nodes according to the specified threshold
        and branching factor.
        
        Parameters
        ----------
        X : FeatureSet
            Training instances with features to cluster. Must be a FeatureSet containing
            a numpy array of shape (n_samples, n_features).
        y : Ignored
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters (ignored in this implementation).
            
        Returns
        -------
        BirchClusteringModel
            Fitted estimator.
            
        Raises
        ------
        ValueError
            If the input data is invalid or incompatible.
        """
        if not isinstance(X, FeatureSet):
            raise ValueError('X must be a FeatureSet instance')
        if X.features.ndim != 2:
            raise ValueError('X.features must be a 2D array')
        data = X.features.copy() if self.copy else X.features
        self._cftree = _CFNode(threshold=self.threshold, branching_factor=self.branching_factor, is_leaf=True)
        for point in data:
            self._cftree.insert_point(point)
            if len(self._cftree.children) > self.branching_factor:
                self._cftree = self._cftree.split_root()
        leaf_nodes = self._cftree.get_leaf_nodes()
        self._subcluster_centers = []
        for node in leaf_nodes:
            for cf_entry in node.cf_entries:
                self._subcluster_centers.append(cf_entry.center)
        if self._subcluster_centers:
            self._subcluster_centers = np.array(self._subcluster_centers)
        else:
            self._subcluster_centers = np.empty((0, data.shape[1]))
        if self.n_clusters is not None and self.n_clusters > 0:
            if len(self._subcluster_centers) >= self.n_clusters and len(self._subcluster_centers) > 1:
                global_clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
                subcluster_labels = global_clusterer.fit_predict(self._subcluster_centers)
                label_idx = 0
                self._subcluster_labels = []
                for node in leaf_nodes:
                    node_labels = []
                    for _ in node.cf_entries:
                        node_labels.append(subcluster_labels[label_idx])
                        label_idx += 1
                    self._subcluster_labels.extend(node_labels)
            elif len(self._subcluster_centers) == 1:
                self._subcluster_labels = np.array([0])
            else:
                self._subcluster_labels = np.array([])
        else:
            self._subcluster_labels = np.arange(len(self._subcluster_centers))
        self.is_fitted = True
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Assigns each data point to the closest subcluster center in the CFT.
        
        Parameters
        ----------
        X : FeatureSet
            New data points to assign to clusters. Must be a FeatureSet containing
            a numpy array of shape (n_samples, n_features).
        **kwargs : dict
            Additional prediction parameters (ignored in this implementation).
            
        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with cluster labels for each point.
            
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("This BirchClusteringModel instance is not fitted yet. Call 'fit' before using this method.")
        if not isinstance(X, FeatureSet):
            raise ValueError('X must be a FeatureSet instance')
        if X.features.ndim != 2:
            raise ValueError('X.features must be a 2D array')
        data = X.features
        if len(self._subcluster_centers) == 0 or len(self._subcluster_labels) == 0:
            return np.full(data.shape[0], -1, dtype=np.int32)
        labels = []
        for point in data:
            distances = np.linalg.norm(self._subcluster_centers - point, axis=1)
            closest_subcluster = np.argmin(distances)
            labels.append(self._subcluster_labels[closest_subcluster])
        return np.array(labels, dtype=np.int32)

    def fit_predict(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> np.ndarray:
        """
        Build the CFT and predict cluster labels for the input data.
        
        Convenience method that combines fitting and prediction in one step.
        
        Parameters
        ----------
        X : FeatureSet
            Training instances with features to cluster. Must be a FeatureSet containing
            a numpy array of shape (n_samples, n_features).
        y : Ignored
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional fitting/prediction parameters (ignored in this implementation).
            
        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) with cluster labels for each point.
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> float:
        """
        Calculate the negative inertia (sum of squared distances to centroids) on the data.
        
        This method computes how well the data points are clustered by calculating
        the sum of squared distances to their respective cluster centers.
        
        Parameters
        ----------
        X : FeatureSet
            Data points to evaluate. Must be a FeatureSet containing
            a numpy array of shape (n_samples, n_features).
        y : Ignored
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional scoring parameters (ignored in this implementation).
            
        Returns
        -------
        float
            Negative sum of squared distances to cluster centers (negative inertia).
            Higher (less negative) values indicate better clustering.
        """
        if not self.is_fitted:
            raise ValueError("This BirchClusteringModel instance is not fitted yet. Call 'fit' before using this method.")
        if not isinstance(X, FeatureSet):
            raise ValueError('X must be a FeatureSet instance')
        data = X.features
        labels = self.predict(X)
        if len(data) == 0 or len(labels) == 0:
            return 0.0
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            return 0.0
        inertia = 0.0
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                cluster_points = data[mask]
                centroid = np.mean(cluster_points, axis=0)
                distances_squared = np.sum((cluster_points - centroid) ** 2, axis=1)
                inertia += np.sum(distances_squared)
        return -inertia