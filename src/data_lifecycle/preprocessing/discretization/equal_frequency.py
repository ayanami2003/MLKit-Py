from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class EqualFrequencyDiscretizer(BaseTransformer):
    """
    Discretizes continuous features into bins with approximately equal frequency counts.
    
    This transformer implements equal-frequency binning (also known as quantile binning),
    where each bin contains roughly the same number of samples. This approach is particularly
    useful for handling skewed data distributions by ensuring uniform sample distribution
    across bins.
    
    The discretization process divides the feature space based on sample quantiles,
    creating bins that adapt to the underlying data distribution rather than using
    fixed-width intervals.
    
    Attributes
    ----------
    n_bins : int
        Number of bins to create for discretization (must be >= 2)
    feature_indices : Optional[List[int]]
        Indices of features to discretize; if None, all features are processed
    bin_edges_ : List[np.ndarray]
        Learned bin edges for each feature after fitting
    name : str
        Name identifier for the transformer instance
        
    Examples
    --------
    >>> import numpy as np
    >>> from general.structures.feature_set import FeatureSet
    >>> 
    >>> # Create sample data with skewed distribution
    >>> data = np.array([[1], [1], [2], [3], [5], [8], [13], [21], [34], [55]])
    >>> feature_set = FeatureSet(features=data, feature_names=['fibonacci'])
    >>> 
    >>> # Initialize and apply equal frequency discretization
    >>> discretizer = EqualFrequencyDiscretizer(n_bins=3)
    >>> discretized = discretizer.fit_transform(feature_set)
    >>> print(discretized.features.flatten())
    [0 0 0 0 1 1 1 2 2 2]
    """

    def __init__(self, n_bins: int=5, feature_indices: Optional[List[int]]=None, name: Optional[str]=None):
        """
        Initialize the EqualFrequencyDiscretizer.
        
        Parameters
        ----------
        n_bins : int, default=5
            Number of bins to create. Must be >= 2.
        feature_indices : Optional[List[int]], optional
            Indices of features to discretize. If None, all features are processed.
        name : Optional[str], optional
            Name identifier for the transformer instance.
            
        Raises
        ------
        ValueError
            If n_bins < 2.
        """
        super().__init__(name=name)
        if n_bins < 2:
            raise ValueError('n_bins must be at least 2')
        self.n_bins = n_bins
        self.feature_indices = feature_indices
        self.bin_edges_: List[np.ndarray] = []

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'EqualFrequencyDiscretizer':
        """
        Learn the bin edges for equal-frequency discretization from the input data.
        
        This method calculates quantile-based bin edges so that each bin will
        contain approximately the same number of samples when transform is called.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to fit the discretizer on. Can be:
            - FeatureSet: Uses the features attribute
            - DataBatch: Uses the data attribute
            - np.ndarray: Direct numpy array (2D expected)
        **kwargs : dict
            Additional fitting parameters (ignored)
            
        Returns
        -------
        EqualFrequencyDiscretizer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data contains non-finite values or insufficient samples for binning
        TypeError
            If data is not in supported format
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data)
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')
        if not np.isfinite(X).all():
            raise ValueError('Input data contains non-finite values')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if n_samples < self.n_bins:
            raise ValueError(f'n_samples ({n_samples}) must be >= n_bins ({self.n_bins})')
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
                if len(bin_edges) == 1:
                    bin_edges = np.array([bin_edges[0], bin_edges[0]])
                self.bin_edges_.append(bin_edges)
            else:
                self.bin_edges_.append(np.array([]))
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Apply equal-frequency discretization to the input data.
        
        Transforms continuous features into discrete bins based on the bin edges
        learned during the fit phase. Each bin is represented by an integer label
        starting from 0.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to transform. Must have same number of features as fitted data.
        **kwargs : dict
            Additional transformation parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Discretized data in the same container type as input:
            - FeatureSet if input was FeatureSet
            - DataBatch if input was DataBatch
            - np.ndarray if input was np.ndarray
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or data dimensions don't match
        """
        if not hasattr(self, 'bin_edges_') or len(self.bin_edges_) == 0:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data).__name__
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        elif isinstance(data, np.ndarray):
            X = data.copy()
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.bin_edges_):
            raise ValueError(f'Data has {X.shape[1]} features, but transformer was fitted on {len(self.bin_edges_)} features')
        X_discrete = X.copy()
        for i in range(X.shape[1]):
            if len(self.bin_edges_[i]) > 0:
                bin_edges = self.bin_edges_[i]
                if len(bin_edges) > 2:
                    bin_indices = np.digitize(X[:, i], bin_edges[1:-1], right=True)
                else:
                    bin_indices = np.zeros(X.shape[0], dtype=int)
                bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
                X_discrete[:, i] = bin_indices
        if original_type == 'FeatureSet':
            return FeatureSet(features=X_discrete, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == 'DataBatch':
            return DataBatch(data=X_discrete, labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            return X_discrete

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Convert discretized data back to approximate continuous values.
        
        Maps discrete bin indices back to the midpoint of their respective bin ranges.
        This is an approximation since the exact original values are lost during discretization.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Discretized data to convert back to continuous space
        **kwargs : dict
            Additional inverse transformation parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Approximate continuous data in same container type as input
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or input data is invalid
        """
        if not hasattr(self, 'bin_edges_') or len(self.bin_edges_) == 0:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data).__name__
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        elif isinstance(data, DataBatch):
            X = np.asarray(data.data).copy()
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            batch_id = data.batch_id
            labels = data.labels
        elif isinstance(data, np.ndarray):
            X = data.copy()
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != len(self.bin_edges_):
            raise ValueError(f'Data has {X.shape[1]} features, but transformer was fitted on {len(self.bin_edges_)} features')
        X_continuous = X.copy().astype(float)
        for i in range(X.shape[1]):
            if len(self.bin_edges_[i]) > 0:
                bin_edges = self.bin_edges_[i]
                if len(bin_edges) > 1:
                    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
                    for j in range(min(len(midpoints), self.n_bins)):
                        X_continuous[X[:, i] == j, i] = midpoints[j]
                else:
                    pass
        if original_type == 'FeatureSet':
            return FeatureSet(features=X_continuous, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == 'DataBatch':
            return DataBatch(data=X_continuous, labels=labels, metadata=metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=batch_id)
        else:
            return X_continuous

    def get_bin_edges(self) -> List[np.ndarray]:
        """
        Retrieve the learned bin edges for all features.
        
        Returns
        -------
        List[np.ndarray]
            List of bin edge arrays, one for each feature. Each array has length n_bins+1.
            
        Raises
        ------
        ValueError
            If transformer has not been fitted yet
        """
        if not hasattr(self, 'bin_edges_') or len(self.bin_edges_) == 0:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return [edges.copy() for edges in self.bin_edges_]