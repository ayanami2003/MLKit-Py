from typing import Optional, Union
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class ClusteringMislabeledDataHandler(BaseTransformer):

    def __init__(self, n_clusters: int=8, method: str='kmeans', contamination_ratio: float=0.1, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the ClusteringMislabeledDataHandler.
        
        Args:
            n_clusters (int): Number of clusters to form. Defaults to 8.
            method (str): Clustering method to use. Supported values are 'kmeans', 'dbscan'. Defaults to 'kmeans'.
            contamination_ratio (float): Expected ratio of mislabeled data. Used to determine threshold for flagging.
            random_state (Optional[int]): Random seed for reproducibility.
            name (Optional[str]): Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_clusters = n_clusters
        self.method = method
        self.contamination_ratio = contamination_ratio
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'ClusteringMislabeledDataHandler':
        """
        Fit the clustering model to identify potential mislabels.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data containing features and true labels.
            **kwargs: Additional keyword arguments.
            
        Returns:
            ClusteringMislabeledDataHandler: The fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            y = data.metadata.get('labels') if data.metadata else None
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            y = data.labels
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if y is None:
            raise ValueError('Labels are required for mislabel detection')
        self.X_shape_ = X.shape
        self.original_labels_ = np.array(y)
        if self.method == 'kmeans':
            self.clusterer_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.method == 'dbscan':
            self.clusterer_ = DBSCAN()
        else:
            raise ValueError(f'Unsupported clustering method: {self.method}')
        cluster_labels = self.clusterer_.fit_predict(X)
        self._cluster_labels = cluster_labels
        unique_clusters = np.unique(cluster_labels)
        self._cluster_majority_labels = {}
        self._cluster_label_counts = {}
        self._mislabeled_indices = []
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_labels_subset = self.original_labels_[cluster_mask]
            label_counts = Counter(cluster_labels_subset)
            if len(label_counts) == 0:
                continue
            majority_label = label_counts.most_common(1)[0][0]
            self._cluster_majority_labels[cluster_id] = majority_label
            self._cluster_label_counts[cluster_id] = dict(label_counts)
            mislabeled_in_cluster = cluster_indices[cluster_labels_subset != majority_label]
            self._mislabeled_indices.extend(mislabeled_in_cluster.tolist())
        self._mislabeled_indices = sorted(self._mislabeled_indices)
        n_samples = len(self.original_labels_)
        max_mislabeled = int(n_samples * self.contamination_ratio)
        if len(self._mislabeled_indices) > max_mislabeled:
            self._mislabeled_indices = self._mislabeled_indices[:max_mislabeled]
        self._inconsistent_points = self._mislabeled_indices
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Identify and correct mislabeled data points based on clustering results.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data to process.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Union[FeatureSet, DataBatch]: Processed data with corrected labels.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError('Transformer must be fitted before transform can be called.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            y = np.array(data.metadata.get('labels')).copy() if data.metadata and 'labels' in data.metadata else None
            if y is None:
                raise ValueError('Labels are required in FeatureSet for mislabel correction')
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            y = np.array(data.labels).copy() if data.labels is not None else None
            if y is None:
                raise ValueError('Labels are required in DataBatch for mislabel correction')
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        corrected_labels = y.copy()
        for idx in self._mislabeled_indices:
            if idx < len(corrected_labels):
                cluster_id = self._cluster_labels[idx]
                if cluster_id != -1 and cluster_id in self._cluster_majority_labels:
                    corrected_labels[idx] = self._cluster_majority_labels[cluster_id]
        if isinstance(data, FeatureSet):
            new_metadata = data.metadata.copy() if data.metadata else {}
            new_metadata['labels'] = corrected_labels.tolist()
            return FeatureSet(features=X, metadata=new_metadata, feature_names=data.feature_names)
        else:
            return DataBatch(data=X.tolist() if not isinstance(X, np.ndarray) else X, labels=corrected_labels.tolist() if not isinstance(corrected_labels, np.ndarray) else corrected_labels, metadata=data.metadata.copy() if data.metadata else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, feature_names=data.feature_names.copy() if data.feature_names else None, batch_id=data.batch_id)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Inverse transformation is not supported for this handler.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Transformed data.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Union[FeatureSet, DataBatch]: The input data unchanged.
            
        Raises:
            NotImplementedError: Always raised as inverse transform is not applicable.
        """
        raise NotImplementedError('Inverse transformation is not supported for mislabeled data handling.')