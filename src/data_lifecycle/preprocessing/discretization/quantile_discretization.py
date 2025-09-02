from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from sklearn.exceptions import NotFittedError

class QuantileDiscretizer(BaseTransformer):

    def __init__(self, n_bins: int=5, feature_indices: Optional[List[int]]=None, handle_unknown: str='error', name: Optional[str]=None):
        super().__init__(name=name)
        if n_bins < 2:
            raise ValueError('n_bins must be >= 2')
        self.n_bins = n_bins
        self.feature_indices = feature_indices
        self.handle_unknown = handle_unknown
        self.bin_edges_: List[np.ndarray] = []
        self.n_features_: int = 0

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'QuantileDiscretizer':
        """
        Compute quantile boundaries for discretization.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to fit the discretizer on. Can be a FeatureSet, DataBatch,
            or numpy array of shape (n_samples, n_features).
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        QuantileDiscretizer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If n_bins < 2 or if data contains non-finite values.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data)
        else:
            X = np.asarray(data)
        if not np.isfinite(X).all():
            raise ValueError('Input data contains non-finite values')
        (n_samples, n_features) = X.shape
        self.n_features_ = n_features
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        self.bin_edges_ = []
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        for i in range(n_features):
            if i in feature_indices:
                feature_data = X[:, i]
                bin_edges = np.percentile(feature_data, percentiles)
                bin_edges = np.unique(bin_edges)
                self.bin_edges_.append(bin_edges)
            else:
                self.bin_edges_.append(np.array([]))
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Discretize the input data using computed quantile boundaries.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to transform. Must have the same number of features
            as the data used during fitting.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Discretized data with the same container type as input.
            
        Raises
        ------
        ValueError
            If data has different number of features than fitted data.
        NotFittedError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'n_features_') or self.n_features_ == 0:
            raise NotFittedError("This QuantileDiscretizer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            original_type = 'FeatureSet'
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            original_type = 'DataBatch'
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        else:
            X = np.asarray(data).copy()
            original_type = 'ndarray'
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'X has {X.shape[1]} features, but QuantileDiscretizer is expecting {self.n_features_} features')
        X_discrete = X.copy()
        for i in range(self.n_features_):
            if len(self.bin_edges_[i]) > 0:
                bin_edges = self.bin_edges_[i]
                bin_indices = np.digitize(X[:, i], bin_edges[1:-1], right=True)
                if self.handle_unknown == 'error':
                    if np.any((X[:, i] < bin_edges[0]) | (X[:, i] > bin_edges[-1])):
                        raise ValueError("Values in transform data are outside the range of fitted data. To ignore unknown values during transformation, set 'handle_unknown' to 'ignore'.")
                X_discrete[:, i] = bin_indices
        if original_type == 'FeatureSet':
            return FeatureSet(features=X_discrete, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == 'DataBatch':
            return DataBatch(data=X_discrete, labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            return X_discrete

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Map discretized values back to approximate continuous values.
        
        Uses the midpoint of each bin as the representative continuous value.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Discretized data to convert back to continuous values.
        **kwargs : dict
            Additional inverse transformation parameters.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Approximate continuous values with the same container type as input.
            
        Raises
        ------
        ValueError
            If data has different number of features than fitted data.
        NotFittedError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'n_features_') or self.n_features_ == 0:
            raise NotFittedError("This QuantileDiscretizer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            original_type = 'FeatureSet'
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            original_type = 'DataBatch'
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        else:
            X = np.asarray(data).copy()
            original_type = 'ndarray'
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'X has {X.shape[1]} features, but QuantileDiscretizer is expecting {self.n_features_} features')
        X_continuous = X.copy().astype(float)
        for i in range(self.n_features_):
            if len(self.bin_edges_[i]) > 0:
                bin_edges = self.bin_edges_[i]
                midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
                for j in range(len(midpoints)):
                    X_continuous[X[:, i] == j, i] = midpoints[j]
        if original_type == 'FeatureSet':
            return FeatureSet(features=X_continuous, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == 'DataBatch':
            return DataBatch(data=X_continuous, labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            return X_continuous