from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix

class ClusterBasedInteractionTransformer(BaseTransformer):

    def __init__(self, n_clusters: int=5, interaction_degree: int=2, clustering_method: str='kmeans', random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the ClusterBasedInteractionTransformer.
        
        Parameters
        ----------
        n_clusters : int, default=5
            Number of clusters to generate for creating interaction features
        interaction_degree : int, default=2
            Degree of interaction terms to create between original features and clusters
        clustering_method : str, default='kmeans'
            Clustering algorithm to use ('kmeans' or 'spectral')
        random_state : int, optional
            Random seed for reproducible results
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.interaction_degree = interaction_degree
        self.clustering_method = clustering_method
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ClusterBasedInteractionTransformer':
        """
        Fit the clustering model to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input features to cluster and use for interaction generation
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        ClusterBasedInteractionTransformer
            Self instance for method chaining
        """
        if self.n_clusters <= 0:
            raise ValueError('n_clusters must be positive')
        if self.interaction_degree <= 0:
            raise ValueError('interaction_degree must be positive')
        if self.clustering_method not in ['kmeans', 'spectral']:
            raise ValueError("clustering_method must be either 'kmeans' or 'spectral'")
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_in_ = data.feature_names
        else:
            X = np.asarray(data)
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if X.shape[0] == 0:
            self._cluster_labels = np.array([])
            self._unique_clusters = np.array([])
        elif X.shape[0] == 1 or self.n_clusters == 1:
            self._cluster_labels = np.zeros(X.shape[0], dtype=int)
            self._unique_clusters = np.array([0])
        elif self.clustering_method == 'kmeans':
            effective_n_clusters = min(self.n_clusters, X.shape[0])
            self._cluster_labels = self._kmeans_clustering(X, effective_n_clusters)
            self._unique_clusters = np.unique(self._cluster_labels)
        else:
            effective_n_clusters = min(self.n_clusters, X.shape[0])
            self._cluster_labels = self._spectral_clustering(X, effective_n_clusters)
            self._unique_clusters = np.unique(self._cluster_labels)
        self.is_fitted_ = True
        return self

    def _kmeans_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Perform K-means clustering on the input data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to cluster
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        np.ndarray
            Cluster labels for each sample
        """
        n_samples = X.shape[0]
        if n_samples < n_clusters:
            n_clusters = n_samples
        if n_clusters == 1:
            return np.zeros(n_samples, dtype=int)
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        indices = rng.choice(n_samples, n_clusters, replace=False)
        centroids = X[indices].copy()
        for _ in range(100):
            distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels

    def _spectral_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Perform spectral clustering on the input data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to cluster
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        np.ndarray
            Cluster labels for each sample
        """
        n_samples = X.shape[0]
        if n_samples < n_clusters:
            n_clusters = n_samples
        if n_clusters == 1:
            return np.zeros(n_samples, dtype=int)
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        gamma = 1.0 / X.shape[1]
        pairwise_sq_dists = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
        similarity_matrix = np.exp(-gamma * pairwise_sq_dists)
        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
        laplacian_matrix = degree_matrix - similarity_matrix
        (eigenvalues, eigenvectors) = np.linalg.eigh(laplacian_matrix)
        embedding = eigenvectors[:, :n_clusters]
        embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-10)
        labels = self._kmeans_clustering(embedding, n_clusters)
        return labels

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Generate cluster-based interaction features.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input features to transform into cluster-based interactions
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed features with cluster-based interactions
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError('Transformer must be fitted before calling transform()')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = np.asarray(data)
        if X.shape[0] == 0:
            feature_names = self.get_feature_names()
            n_output_features = len(feature_names)
            transformed_features = np.empty((0, n_output_features))
            return FeatureSet(features=transformed_features, feature_names=feature_names)
        n_samples = X.shape[0]
        if n_samples <= len(self._cluster_labels):
            cluster_labels = self._cluster_labels[:n_samples]
        else:
            cluster_labels = np.concatenate([self._cluster_labels, np.full(n_samples - len(self._cluster_labels), self._cluster_labels[-1] if len(self._cluster_labels) > 0 else 0)])
        features_list = [X]
        for degree in range(1, self.interaction_degree + 1):
            for cluster_id in self._unique_clusters:
                cluster_mask = (cluster_labels == cluster_id).astype(int).reshape(-1, 1)
                if degree == 1:
                    interaction_features = X * cluster_mask
                else:
                    interaction_features = X ** degree * cluster_mask
                features_list.append(interaction_features)
        transformed_features = np.hstack(features_list)
        feature_names = self.get_feature_names()
        return FeatureSet(features=transformed_features, feature_names=feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation (returns original features without interactions).
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Transformed features with interactions
        **kwargs : dict
            Additional inverse transformation parameters
            
        Returns
        -------
        FeatureSet
            Original features without interaction terms
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError('Transformer must be fitted before calling inverse_transform()')
        if isinstance(data, FeatureSet):
            X_transformed = data.features
        else:
            X_transformed = np.asarray(data)
        if X_transformed.shape[1] >= self.n_features_in_:
            X_original = X_transformed[:, :self.n_features_in_]
        else:
            X_original = X_transformed
        if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
            feature_names = self.feature_names_in_
        else:
            feature_names = [f'feature_{i}' for i in range(self.n_features_in_)]
        return FeatureSet(features=X_original, feature_names=feature_names)

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of the generated cluster-based interaction features.
        
        Parameters
        ----------
        input_features : List[str], optional
            Names of original input features
            
        Returns
        -------
        List[str]
            Names of generated interaction features
        """
        if input_features is not None:
            feature_names_in = input_features
        elif hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
            feature_names_in = self.feature_names_in_
        else:
            feature_names_in = [f'feature_{i}' for i in range(self.n_features_in_)]
        feature_names = feature_names_in.copy()
        if hasattr(self, '_unique_clusters'):
            unique_clusters = self._unique_clusters
        else:
            unique_clusters = [0]
        for degree in range(1, self.interaction_degree + 1):
            for cluster_id in unique_clusters:
                for feature_name in feature_names_in:
                    if degree == 1:
                        feature_names.append(f'{feature_name}_cluster_{cluster_id}')
                    else:
                        feature_names.append(f'{feature_name}^{degree}_cluster_{cluster_id}')
        return feature_names