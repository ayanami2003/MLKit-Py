from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class EqualWidthDiscretizer(BaseTransformer):
    """
    Discretizes continuous features into equal-width bins.
    
    This transformer divides the range of each selected feature into equal-width intervals (bins)
    and maps each value to its corresponding bin index. It is useful for converting continuous
    variables into categorical ones while preserving uniform bin widths.
    
    Attributes:
        n_bins (int): Number of equal-width bins to create for each feature.
        feature_indices (Optional[List[int]]): Indices of features to discretize. If None, all features are processed.
        bin_edges_ (List[np.ndarray]): Learned bin edges for each feature after fitting.
        
    Example:
        >>> discretizer = EqualWidthDiscretizer(n_bins=5, feature_indices=[0, 2])
        >>> discretizer.fit(feature_set)
        >>> transformed = discretizer.transform(feature_set)
    """

    def __init__(self, n_bins: int=5, feature_indices: Optional[List[int]]=None, name: Optional[str]=None):
        """
        Initialize the EqualWidthDiscretizer.
        
        Args:
            n_bins (int): Number of equal-width bins to create. Must be >= 2.
            feature_indices (Optional[List[int]]): Indices of features to discretize.
                If None, all features in the input will be processed.
            name (Optional[str]): Name identifier for the transformer instance.
        """
        super().__init__(name=name)
        if n_bins < 2:
            raise ValueError('n_bins must be at least 2')
        self.n_bins = n_bins
        self.feature_indices = feature_indices
        self.bin_edges_: List[np.ndarray] = []

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'EqualWidthDiscretizer':
        """
        Learn the equal-width bin edges from the input data.
        
        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Input data containing continuous features.
                Can be a FeatureSet with features attribute, DataBatch with data attribute, or raw numpy array.
            **kwargs: Additional keyword arguments for fitting.
            
        Returns:
            EqualWidthDiscretizer: Returns self for method chaining.
            
        Raises:
            ValueError: If data has inconsistent shapes or invalid values.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
        else:
            X = np.array(data)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (n_samples, n_features) = X.shape
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
            for idx in feature_indices:
                if idx < 0 or idx >= n_features:
                    raise ValueError(f'Feature index {idx} is out of range [0, {n_features - 1}]')
        self.bin_edges_ = []
        for i in feature_indices:
            feature_values = X[:, i]
            valid_values = feature_values[~np.isnan(feature_values)]
            if len(valid_values) == 0:
                raise ValueError(f'All values are NaN in feature {i}')
            (min_val, max_val) = (np.min(valid_values), np.max(valid_values))
            if min_val == max_val:
                epsilon = max(1e-08, abs(min_val) * 1e-08) if min_val != 0 else 1e-08
                bin_edges = np.linspace(min_val - epsilon, max_val + epsilon, self.n_bins + 1)
            else:
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            self.bin_edges_.append(bin_edges)
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply equal-width discretization to the input data.
        
        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Input data to discretize.
                Must have the same number of features as the fitted data.
            **kwargs: Additional keyword arguments for transformation.
            
        Returns:
            Union[FeatureSet, DataBatch]: Transformed data with values mapped to bin indices.
                Preserves the input data type (FeatureSet or DataBatch).
                
        Raises:
            ValueError: If transformer has not been fitted or data dimensions mismatch.
        """
        if not self.bin_edges_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data)
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            original_feature_set = data
        elif isinstance(data, DataBatch):
            X = np.array(data.data).copy()
            original_data_batch = data
        else:
            X = np.array(data).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (n_samples, n_features) = X.shape
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        if len(self.bin_edges_) != len(feature_indices):
            raise ValueError('Mismatch between number of features and fitted bin edges')
        for (idx, bin_edges) in zip(feature_indices, self.bin_edges_):
            if idx >= n_features:
                raise ValueError(f'Feature index {idx} is out of range for input data with {n_features} features')
            bin_indices = np.digitize(X[:, idx], bin_edges, right=False)
            bin_indices = bin_indices - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
            X[:, idx] = bin_indices.astype(int)
        if original_type == FeatureSet:
            return FeatureSet(features=X, feature_names=original_feature_set.feature_names, feature_types=original_feature_set.feature_types, sample_ids=original_feature_set.sample_ids, metadata=original_feature_set.metadata, quality_scores=original_feature_set.quality_scores)
        elif original_type == DataBatch:
            return DataBatch(data=X, labels=original_data_batch.labels, metadata=original_data_batch.metadata, sample_ids=original_data_batch.sample_ids, feature_names=original_data_batch.feature_names, batch_id=original_data_batch.batch_id)
        else:
            return X

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Map discretized bin indices back to approximate continuous values (bin midpoints).
        
        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Discretized data to invert.
            **kwargs: Additional keyword arguments for inverse transformation.
            
        Returns:
            Union[FeatureSet, DataBatch]: Data with bin indices replaced by bin midpoint values.
            
        Raises:
            ValueError: If transformer has not been fitted.
        """
        if not self.bin_edges_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data)
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            original_feature_set = data
        elif isinstance(data, DataBatch):
            X = np.array(data.data).copy()
            original_data_batch = data
        else:
            X = np.array(data).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (n_samples, n_features) = X.shape
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        if len(self.bin_edges_) != len(feature_indices):
            raise ValueError('Mismatch between number of features and fitted bin edges')
        for (idx, bin_edges) in zip(feature_indices, self.bin_edges_):
            if idx >= n_features:
                raise ValueError(f'Feature index {idx} is out of range for input data with {n_features} features')
            bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_indices = X[:, idx].astype(int)
            if len(bin_midpoints) == 1:
                X[:, idx] = bin_midpoints[0]
            else:
                bin_indices = np.clip(bin_indices, 0, len(bin_midpoints) - 1)
                X[:, idx] = bin_midpoints[bin_indices]
        if original_type == FeatureSet:
            return FeatureSet(features=X, feature_names=original_feature_set.feature_names, feature_types=original_feature_set.feature_types, sample_ids=original_feature_set.sample_ids, metadata=original_feature_set.metadata, quality_scores=original_feature_set.quality_scores)
        elif original_type == DataBatch:
            return DataBatch(data=X, labels=original_data_batch.labels, metadata=original_data_batch.metadata, sample_ids=original_data_batch.sample_ids, feature_names=original_data_batch.feature_names, batch_id=original_data_batch.batch_id)
        else:
            return X

    def get_bin_edges(self, feature_index: int) -> np.ndarray:
        """
        Retrieve the bin edges for a specific feature.
        
        Args:
            feature_index (int): Index of the feature to retrieve bin edges for.
            
        Returns:
            np.ndarray: Array of bin edges with length n_bins + 1.
            
        Raises:
            IndexError: If feature_index is out of range.
            ValueError: If transformer has not been fitted.
        """
        if not self.bin_edges_:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.feature_indices is None:
            if feature_index < 0 or feature_index >= len(self.bin_edges_):
                raise IndexError(f'Feature index {feature_index} is out of range [0, {len(self.bin_edges_) - 1}]')
            return self.bin_edges_[feature_index]
        else:
            try:
                position = self.feature_indices.index(feature_index)
                return self.bin_edges_[position]
            except ValueError:
                raise IndexError(f'Feature index {feature_index} was not processed during fitting')