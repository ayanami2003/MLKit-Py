from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform, cdist

class AgglomerativeClusteringModel(BaseModel):

    def __init__(self, n_clusters: int=2, linkage: str='ward', affinity: str='euclidean', compute_full_tree: Union[bool, str]='auto', distance_threshold: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the Agglomerative Clustering Model.
        
        Parameters
        ----------
        n_clusters : int, default=2
            Number of clusters to find
        linkage : str, default='ward'
            Linkage criterion to use ('ward', 'complete', 'average', 'single')
        affinity : str, default='euclidean'
            Metric used to compute linkage ('euclidean', 'l1', 'l2', 'manhattan', 'cosine')
        compute_full_tree : bool or 'auto', default='auto'
            Whether to compute full dendrogram
        distance_threshold : float, default=None
            Distance threshold for forming flat clusters
        name : str, optional
            Name of the model instance
        """
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold
        if self.linkage not in ['ward', 'complete', 'average', 'single']:
            raise ValueError(f"Invalid linkage: {self.linkage}. Must be one of ['ward', 'complete', 'average', 'single']")
        if self.affinity not in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']:
            raise ValueError(f"Invalid affinity: {self.affinity}. Must be one of ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']")

    def _convert_input(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, FeatureSet):
            if hasattr(X.features, 'values'):
                return X.features.values
            else:
                return np.asarray(X.features)
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError('Input must be a FeatureSet or numpy array')

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'AgglomerativeClusteringModel':
        """
        Fit the agglomerative clustering model.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster
        y : array-like, optional
            Not used, present here for API consistency
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        AgglomerativeClusteringModel
            Fitted estimator
        """
        X_array = self._convert_input(X)
        if X_array.ndim != 2:
            raise ValueError('Input data must be 2D')
        (self.n_samples_, self.n_features_) = X_array.shape
        self.n_leaves_ = self.n_samples_
        if self.linkage == 'ward' and self.affinity not in ['euclidean', 'l2']:
            raise ValueError('Ward linkage only supports euclidean or l2 affinity')
        if self.compute_full_tree == 'auto':
            compute_full_tree = self.distance_threshold is not None or self.n_clusters < self.n_samples_ // 2
        else:
            compute_full_tree = self.compute_full_tree
        affinity_mapping = {'l1': 'cityblock', 'l2': 'euclidean', 'manhattan': 'cityblock', 'euclidean': 'euclidean', 'cosine': 'cosine'}
        scipy_affinity = affinity_mapping.get(self.affinity, self.affinity)
        if self.linkage == 'ward':
            distances = pdist(X_array, metric='sqeuclidean')
        else:
            distances = pdist(X_array, metric=scipy_affinity)
        self.linkage_matrix_ = linkage(distances, method=self.linkage)
        self.children_ = self.linkage_matrix_[:, :2].astype(int)
        self.distances_ = self.linkage_matrix_[:, 2]
        self.n_connected_components_ = 1
        if self.distance_threshold is not None:
            self.labels_ = fcluster(self.linkage_matrix_, self.distance_threshold, criterion='distance') - 1
        else:
            self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters, criterion='maxclust') - 1
        self.training_data_ = X_array.copy()
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            New data to predict
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        X_array = self._convert_input(X)
        affinity_mapping = {'l1': 'cityblock', 'l2': 'euclidean', 'manhattan': 'cityblock', 'euclidean': 'euclidean', 'cosine': 'cosine'}
        scipy_affinity = affinity_mapping.get(self.affinity, self.affinity)
        distances = cdist(X_array, self.training_data_, metric=scipy_affinity)
        closest_indices = np.argmin(distances, axis=1)
        return self.labels_[closest_indices]

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster
        y : array-like, optional
            Not used, present here for API consistency
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels
        """
        return self.fit(X, y, **kwargs).labels_

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Calculate silhouette score for the clustering.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Data to cluster
        y : array-like
            Cluster labels
            
        Returns
        -------
        float
            Silhouette score
        """
        from sklearn.metrics import silhouette_score
        X_array = self._convert_input(X)
        sklearn_affinity = 'euclidean' if self.affinity == 'l2' else self.affinity
        if self.affinity == 'l1' or self.affinity == 'manhattan':
            sklearn_affinity = 'manhattan'
        elif self.affinity == 'cosine':
            sklearn_affinity = 'cosine'
        return silhouette_score(X_array, y, metric=sklearn_affinity)

class AverageLinkageClustering(BaseModel):
    """
    Average Linkage Hierarchical Clustering Model.
    
    This model implements hierarchical clustering using average linkage criterion,
    where the distance between two clusters is computed as the average distance
    between all pairs of points from the two clusters. This approach tends to
    produce clusters with similar variance and is less sensitive to outliers
    compared to complete linkage.
    
    Attributes
    ----------
    n_clusters : int, default=2
        Number of clusters to find
    affinity : str, default='euclidean'
        Metric used to compute linkage ('euclidean', 'l1', 'l2', 'manhattan', 'cosine')
    compute_full_tree : bool or 'auto', default='auto'
        Whether to compute full dendrogram
    distance_threshold : float, default=None
        Distance threshold for forming flat clusters
    
    Methods
    -------
    fit(X[, y]) : Fit the model
    predict(X) : Predict cluster labels
    fit_predict(X[, y]) : Fit and predict in one step
    """

    def __init__(self, n_clusters: int=2, affinity: str='euclidean', compute_full_tree: Union[bool, str]='auto', distance_threshold: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the Average Linkage Clustering Model.
        
        Parameters
        ----------
        n_clusters : int, default=2
            Number of clusters to find. Ignored if distance_threshold is not None.
        affinity : str, default='euclidean'
            Metric used to compute linkage. Can be 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'.
        compute_full_tree : bool or 'auto', default='auto'
            Whether to compute full dendrogram. Useful for dendrogram visualization.
        distance_threshold : float, default=None
            Distance threshold for forming flat clusters. If not None, n_clusters is ignored.
        name : str, optional
            Name of the model instance
            
        Raises
        ------
        ValueError
            If n_clusters is less than 1 or if affinity is not supported
        """
        super().__init__(name=name)
        if n_clusters < 1:
            raise ValueError('n_clusters must be at least 1')
        supported_affinities = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
        if affinity not in supported_affinities:
            raise ValueError(f'affinity must be one of {supported_affinities}')
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold

    def _convert_input(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, FeatureSet):
            if hasattr(X.features, 'values'):
                return X.features.values
            else:
                return np.asarray(X.features)
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError('Input must be a FeatureSet or numpy array')

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'AverageLinkageClustering':
        """
        Fit the average linkage clustering model.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster. Shape should be (n_samples, n_features).
        y : array-like, optional
            Not used, present here for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters (reserved for future extensions)
            
        Returns
        -------
        AverageLinkageClustering
            Fitted estimator
            
        Raises
        ------
        ValueError
            If X has incompatible shape or data type
        """
        X_array = self._convert_input(X)
        if X_array.ndim != 2:
            raise ValueError('Input data must be 2D')
        (self.n_samples_, self.n_features_) = X_array.shape
        self.n_leaves_ = self.n_samples_
        if self.compute_full_tree == 'auto':
            compute_full_tree = self.distance_threshold is not None or self.n_clusters < self.n_samples_ // 2
        else:
            compute_full_tree = self.compute_full_tree
        affinity_mapping = {'l1': 'cityblock', 'l2': 'euclidean', 'manhattan': 'cityblock', 'euclidean': 'euclidean', 'cosine': 'cosine'}
        scipy_affinity = affinity_mapping.get(self.affinity, self.affinity)
        distances = pdist(X_array, metric=scipy_affinity)
        self.linkage_matrix_ = linkage(distances, method='average')
        self.children_ = self.linkage_matrix_[:, :2].astype(int)
        self.distances_ = self.linkage_matrix_[:, 2]
        self.n_connected_components_ = 1
        if self.distance_threshold is not None:
            self.labels_ = fcluster(self.linkage_matrix_, self.distance_threshold, criterion='distance') - 1
        else:
            self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters, criterion='maxclust') - 1
        self.training_data_ = X_array.copy()
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data using fitted model.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            New data to predict cluster labels for. Should have same number of features as training data.
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        X_array = self._convert_input(X)
        if X_array.ndim != 2:
            raise ValueError('Input data must be 2D')
        if X_array.shape[1] != self.n_features_:
            raise ValueError(f'X has {X_array.shape[1]} features, but model was trained with {self.n_features_} features.')
        unique_labels = np.unique(self.labels_)
        centroids = []
        for label in unique_labels:
            cluster_points = self.training_data_[self.labels_ == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        affinity_mapping = {'l1': 'cityblock', 'l2': 'euclidean', 'manhattan': 'cityblock', 'euclidean': 'euclidean', 'cosine': 'cosine'}
        scipy_affinity = affinity_mapping.get(self.affinity, self.affinity)
        distances_to_centroids = cdist(X_array, centroids, metric=scipy_affinity)
        closest_centroids = np.argmin(distances_to_centroids, axis=1)
        predicted_labels = unique_labels[closest_centroids]
        return predicted_labels

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster. Shape should be (n_samples, n_features).
        y : array-like, optional
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels of shape (n_samples,)
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Calculate silhouette score for the clustering.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Data that was clustered
        y : array-like
            Cluster labels for X
            
        Returns
        -------
        float
            Silhouette score ranging from -1 to 1, where higher values indicate better clustering
        """
        from sklearn.metrics import silhouette_score
        X_array = self._convert_input(X)
        sklearn_affinity = 'euclidean' if self.affinity == 'l2' else self.affinity
        if self.affinity == 'l1' or self.affinity == 'manhattan':
            sklearn_affinity = 'manhattan'
        elif self.affinity == 'cosine':
            sklearn_affinity = 'cosine'
        return silhouette_score(X_array, y, metric=sklearn_affinity)

class WardLinkageClustering(BaseModel):
    """
    Ward Linkage Hierarchical Clustering Model.
    
    This model implements hierarchical clustering using Ward's linkage criterion,
    which minimizes the variance within clusters at each merge step. Ward's method
    tends to produce compact, spherical clusters of similar sizes and is particularly
    effective when clusters have convex shapes. It requires Euclidean distance
    and is computationally efficient for creating balanced cluster trees.
    
    Attributes
    ----------
    n_clusters : int, default=2
        Number of clusters to find
    compute_full_tree : bool or 'auto', default='auto'
        Whether to compute full dendrogram
    distance_threshold : float, default=None
        Distance threshold for forming flat clusters
    
    Methods
    -------
    fit(X[, y]) : Fit the model
    predict(X) : Predict cluster labels
    fit_predict(X[, y]) : Fit and predict in one step
    """

    def __init__(self, n_clusters: int=2, compute_full_tree: Union[bool, str]='auto', distance_threshold: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the Ward Linkage Clustering Model.
        
        Parameters
        ----------
        n_clusters : int, default=2
            Number of clusters to find. Must be greater than 0.
            Ignored if distance_threshold is not None.
        compute_full_tree : bool or 'auto', default='auto'
            Whether to compute full dendrogram. If 'auto', computes full tree
            when n_clusters is small enough.
        distance_threshold : float, default=None
            Distance threshold for forming flat clusters. If not None,
            n_clusters is ignored. Must be non-negative.
        name : str, optional
            Name of the model instance
            
        Raises
        ------
        ValueError
            If n_clusters < 1 or distance_threshold < 0
        """
        super().__init__(name=name)
        if n_clusters < 1:
            raise ValueError('n_clusters must be greater than 0')
        if distance_threshold is not None and distance_threshold < 0:
            raise ValueError('distance_threshold must be non-negative')
        self.n_clusters = n_clusters
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold

    def _convert_input(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, FeatureSet):
            if hasattr(X.features, 'values'):
                return X.features.values
            else:
                return np.asarray(X.features)
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise TypeError('Input must be a FeatureSet or numpy array')

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'WardLinkageClustering':
        """
        Fit the Ward linkage clustering model.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster. Must be 2D array-like with shape (n_samples, n_features).
            Data will be converted to float64 dtype if not already.
        y : array-like, optional
            Not used, present for API consistency by convention.
        **kwargs : dict
            Additional fitting parameters (reserved for future use)
            
        Returns
        -------
        WardLinkageClustering
            Fitted estimator
            
        Raises
        ------
        ValueError
            If X has incompatible shape or contains non-finite values
        TypeError
            If X is not array-like
        """
        X_array = self._convert_input(X)
        if X_array.ndim != 2:
            raise ValueError('Input data must be 2D')
        if not np.isfinite(X_array).all():
            raise ValueError('Input data contains non-finite values')
        (self.n_samples_, self.n_features_) = X_array.shape
        self.n_leaves_ = self.n_samples_
        if self.n_samples_ == 0:
            raise ValueError('Input data must contain at least one sample')
        elif self.n_samples_ == 1:
            self.labels_ = np.array([0])
            self.children_ = np.empty((0, 2), dtype=int)
            self.distances_ = np.empty(0)
            self.n_connected_components_ = 1
            self.training_data_ = X_array.copy()
            return self
        if self.compute_full_tree == 'auto':
            compute_full_tree = self.distance_threshold is not None or self.n_clusters < self.n_samples_ // 2
        else:
            compute_full_tree = self.compute_full_tree
        distances = pdist(X_array, metric='sqeuclidean')
        self.linkage_matrix_ = linkage(distances, method='ward')
        self.children_ = self.linkage_matrix_[:, :2].astype(int)
        self.distances_ = self.linkage_matrix_[:, 2]
        self.n_connected_components_ = 1
        if self.distance_threshold is not None:
            self.labels_ = fcluster(self.linkage_matrix_, self.distance_threshold, criterion='distance') - 1
        else:
            self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters, criterion='maxclust') - 1
        self.training_data_ = X_array.copy()
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data using fitted model.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            New data to predict cluster labels for. Must have same number of features as training data.
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels of shape (n_samples,) with integer values in [0, n_clusters-1]
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        X_array = self._convert_input(X)
        if X_array.shape[1] != self.n_features_:
            raise ValueError('Number of features in X does not match training data')
        if not hasattr(self, 'training_data_'):
            return np.zeros(X_array.shape[0], dtype=int)
        distances = cdist(X_array, self.training_data_, metric='euclidean')
        closest_indices = np.argmin(distances, axis=1)
        return self.labels_[closest_indices]

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training instances to cluster. Must be 2D array-like with shape (n_samples, n_features).
        y : array-like, optional
            Not used, present for API consistency by convention.
            
        Returns
        -------
        np.ndarray
            Predicted cluster labels of shape (n_samples,) with integer values in [0, n_clusters-1]
        """
        return self.fit(X, y, **kwargs).labels_

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Calculate within-cluster sum of squares as clustering quality metric.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Data that was clustered
        y : array-like
            Cluster labels for X
            
        Returns
        -------
        float
            Negative within-cluster sum of squares (higher values indicate better clustering)
        """
        X_array = self._convert_input(X)
        y_array = np.asarray(y)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError('Number of samples in X and y must match')
        total_wcss = 0.0
        for label in np.unique(y_array):
            cluster_points = X_array[y_array == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss = np.sum((cluster_points - centroid) ** 2)
                total_wcss += wcss
        return -total_wcss