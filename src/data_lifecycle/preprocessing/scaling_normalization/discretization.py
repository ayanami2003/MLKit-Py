import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, List

class Discretizer(BaseTransformer):
    """
    A transformer that converts continuous features into discrete bins or categories.
    
    This class provides a standardized interface for applying various discretization
    techniques to continuous data, transforming numerical features into categorical
    representations. It supports multiple binning strategies and can handle both
    single features and entire feature sets.
    
    Attributes
    ----------
    strategy : str
        The discretization strategy to use (e.g., 'equal_width', 'equal_frequency',
        'kmeans', 'quantile')
    n_bins : int
        Number of bins to create
    bin_edges : dict
        Stores the computed bin edges for each feature after fitting
    """

    def __init__(self, strategy: str='equal_width', n_bins: int=5, name: Optional[str]=None, **strategy_params):
        """
        Initialize the Discretizer.
        
        Parameters
        ----------
        strategy : str, default='equal_width'
            The discretization strategy to use. Supported strategies include:
            - 'equal_width': Equal-width binning
            - 'equal_frequency': Equal-frequency binning
            - 'kmeans': K-means clustering-based discretization
            - 'quantile': Quantile-based discretization
        n_bins : int, default=5
            Number of bins to create for discretization
        name : str, optional
            Name of the transformer instance
        **strategy_params
            Additional parameters specific to the chosen strategy
        """
        super().__init__(name=name)
        self.strategy = strategy
        self.n_bins = n_bins
        self.strategy_params = strategy_params
        self.bin_edges = {}

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'Discretizer':
        """
        Fit the discretizer to the input data by computing bin edges.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to fit the discretizer on. If FeatureSet, uses the features attribute.
        **kwargs
            Additional fitting parameters
            
        Returns
        -------
        Discretizer
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        self.bin_edges = {}
        for i in range(n_features):
            feature_data = X[:, i]
            if self.strategy == 'equal_width':
                (min_val, max_val) = (np.min(feature_data), np.max(feature_data))
                eps = (max_val - min_val) * 1e-07
                bin_edges = np.linspace(min_val, max_val + eps, self.n_bins + 1)
            elif self.strategy == 'equal_frequency':
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                bin_edges = np.percentile(feature_data, quantiles)
                bin_edges = np.unique(bin_edges)
            elif self.strategy == 'kmeans':
                from src.data_lifecycle.modeling.clustering.k_means.kmeans_algorithm import KMeansClusteringModel
                feature_data_reshaped = feature_data.reshape(-1, 1)
                kmeans = KMeansClusteringModel(n_clusters=self.n_bins, random_state=self.strategy_params.get('random_state', None))
                kmeans.fit(FeatureSet(features=feature_data_reshaped))
                centers = np.sort(kmeans.cluster_centers_.flatten())
                bin_edges = np.empty(len(centers) + 1)
                bin_edges[0] = np.min(feature_data)
                bin_edges[-1] = np.max(feature_data)
                for j in range(len(centers) - 1):
                    bin_edges[j + 1] = (centers[j] + centers[j + 1]) / 2
            elif self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                bin_edges = np.percentile(feature_data, quantiles)
                bin_edges = np.unique(bin_edges)
            else:
                raise ValueError(f'Unsupported strategy: {self.strategy}')
            self.bin_edges[i] = bin_edges
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply discretization to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform
        **kwargs
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed data with discretized features
        """
        if not self.bin_edges:
            raise RuntimeError("Discretizer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != len(self.bin_edges):
            raise ValueError(f'Input data has {X.shape[1]} features, but discretizer was fitted on {len(self.bin_edges)} features')
        X_discretized = np.empty_like(X, dtype=int)
        for i in range(X.shape[1]):
            X_discretized[:, i] = np.digitize(X[:, i], self.bin_edges[i]) - 1
            X_discretized[:, i] = np.clip(X_discretized[:, i], 0, self.n_bins - 1)
        metadata['discretization_method'] = self.strategy
        metadata['n_bins'] = self.n_bins
        metadata['bin_edges'] = self.bin_edges
        return FeatureSet(features=X_discretized, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert discretized data back to approximate continuous values.
        
        Uses the midpoints of each bin to represent the original continuous values.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Discretized data to convert back
        **kwargs
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Data with approximate continuous values
        """
        if not self.bin_edges:
            raise RuntimeError("Discretizer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != len(self.bin_edges):
            raise ValueError(f'Input data has {X.shape[1]} features, but discretizer was fitted on {len(self.bin_edges)} features')
        X_continuous = np.empty_like(X, dtype=float)
        for i in range(X.shape[1]):
            bin_edges = self.bin_edges[i]
            for j in range(len(bin_edges) - 1):
                mask = X[:, i] == j
                midpoint = (bin_edges[j] + bin_edges[j + 1]) / 2
                X_continuous[mask, i] = midpoint
        if 'discretization_method' in metadata:
            del metadata['discretization_method']
        if 'n_bins' in metadata:
            del metadata['n_bins']
        if 'bin_edges' in metadata:
            del metadata['bin_edges']
        return FeatureSet(features=X_continuous, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def get_bin_edges(self, feature_index: Union[int, str]) -> np.ndarray:
        """
        Get the bin edges for a specific feature.
        
        Parameters
        ----------
        feature_index : int or str
            Index or name of the feature
            
        Returns
        -------
        np.ndarray
            Array of bin edges for the specified feature
        """
        if not self.bin_edges:
            raise RuntimeError("Discretizer has not been fitted yet. Call 'fit' first.")
        if isinstance(feature_index, str):
            raise ValueError('Feature names not supported in this implementation. Use integer index.')
        if feature_index not in self.bin_edges:
            raise ValueError(f'No bin edges found for feature index {feature_index}')
        return self.bin_edges[feature_index]

    def get_bin_labels(self, feature_index: Union[int, str]) -> List[str]:
        """
        Get descriptive labels for the bins of a specific feature.
        
        Parameters
        ----------
        feature_index : int or str
            Index or name of the feature
            
        Returns
        -------
        List[str]
            List of bin labels
        """
        if not self.bin_edges:
            raise RuntimeError("Discretizer has not been fitted yet. Call 'fit' first.")
        if isinstance(feature_index, str):
            raise ValueError('Feature names not supported in this implementation. Use integer index.')
        if feature_index not in self.bin_edges:
            raise ValueError(f'No bin edges found for feature index {feature_index}')
        n_bins = len(self.bin_edges[feature_index]) - 1
        return [f'Bin {i + 1}' for i in range(n_bins)]