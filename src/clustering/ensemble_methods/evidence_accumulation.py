from typing import List, Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

class EvidenceAccumulationClusteringModel(BaseModel):

    def __init__(self, base_clusterings: List[BaseModel], n_clusters: int=3, consensus_method: str='hierarchical', name: Optional[str]=None):
        """
        Initialize the Evidence Accumulation Clustering model.

        Args:
            base_clusterings (List[BaseModel]): A list of pre-fitted clustering models that will contribute
                                                to the evidence accumulation process.
            n_clusters (int): Desired number of clusters in the final output. Defaults to 3.
            consensus_method (str): Strategy for generating the final clustering from the co-association
                                    matrix. Supported values: 'hierarchical'. Defaults to 'hierarchical'.
            name (Optional[str]): Optional name for the model instance.
        """
        super().__init__(name=name)
        self.base_clusterings = base_clusterings
        self.n_clusters = n_clusters
        self.consensus_method = consensus_method
        self.coassoc_matrix: Optional[np.ndarray] = None
        self._labels_: Optional[np.ndarray] = None

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'EvidenceAccumulationClusteringModel':
        """
        Fit the EAC model by computing the co-association matrix from base clusterings.

        Args:
            X (Union[FeatureSet, np.ndarray]): Input feature data to which base clusterings were fitted.
            y (Optional[np.ndarray]): Optional target values (not used in unsupervised clustering).
            **kwargs: Additional keyword arguments passed to the fitting process.

        Returns:
            EvidenceAccumulationClusteringModel: The fitted model instance.
        """
        if isinstance(X, FeatureSet):
            data = X.features
        else:
            data = X
        n_samples = data.shape[0]
        if len(self.base_clusterings) == 0:
            raise ValueError('At least one base clustering model must be provided.')
        self.coassoc_matrix = np.zeros((n_samples, n_samples), dtype=int)
        for clustering_model in self.base_clusterings:
            if not hasattr(clustering_model, 'predict'):
                raise ValueError("All base clustering models must be fitted and have a 'predict' method.")
            try:
                labels = clustering_model.predict(data)
            except Exception as e:
                raise ValueError(f'Failed to get predictions from a base clustering model: {e}')
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j]:
                        self.coassoc_matrix[i, j] += 1
                        self.coassoc_matrix[j, i] += 1
        np.fill_diagonal(self.coassoc_matrix, 0)
        if self.consensus_method == 'hierarchical':
            self._labels_ = self._hierarchical_consensus()
        else:
            raise ValueError(f'Unsupported consensus method: {self.consensus_method}')
        self.is_fitted = True
        return self

    def _hierarchical_consensus(self) -> np.ndarray:
        """
        Apply hierarchical clustering on the co-association matrix to obtain final cluster labels.
        
        Returns:
            np.ndarray: Cluster labels for each sample.
        """
        if self.coassoc_matrix is None:
            raise RuntimeError("Co-association matrix not computed. Call 'fit' first.")
        max_assoc = np.max(self.coassoc_matrix) if np.max(self.coassoc_matrix) > 0 else 1
        distance_matrix = max_assoc - self.coassoc_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(distance_matrix, 0)
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='average')
        labels = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')
        return labels - 1

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data based on the fitted EAC model.

        Args:
            X (Union[FeatureSet, np.ndarray]): Input feature data for prediction.
            **kwargs: Additional keyword arguments for prediction.

        Returns:
            np.ndarray: Array of predicted cluster labels for each sample.
        """
        if self._labels_ is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        if isinstance(X, FeatureSet):
            data = X.features
        else:
            data = X
        if data.shape[0] != len(self._labels_):
            raise ValueError('Cannot predict on data with different number of samples than used during fitting.')
        return self._labels_.copy()

    def fit_predict(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Fit the model and immediately predict cluster labels in one step.

        Args:
            X (Union[FeatureSet, np.ndarray]): Input feature data.
            y (Optional[np.ndarray]): Optional target values.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Evaluate the quality of the clustering using a validity index.

        Args:
            X (Union[FeatureSet, np.ndarray]): Input feature data.
            y (Optional[np.ndarray]): True labels for supervised evaluation (if available).
            **kwargs: Additional scoring parameters.

        Returns:
            float: Clustering validity score (higher is better).
        """
        try:
            from sklearn.metrics import silhouette_score
            if isinstance(X, FeatureSet):
                data = X.features
            else:
                data = X
            labels = self.predict(X)
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1 or len(unique_labels) >= len(data):
                return 0.0
            return silhouette_score(data, labels)
        except ImportError:
            if self.coassoc_matrix is not None:
                return np.mean(self.coassoc_matrix)
            else:
                return 0.0

    def get_coassociation_matrix(self) -> Optional[np.ndarray]:
        """
        Retrieve the computed co-association matrix.

        Returns:
            Optional[np.ndarray]: The co-association matrix of pairwise associations, or None if not yet computed.
        """
        return self.coassoc_matrix