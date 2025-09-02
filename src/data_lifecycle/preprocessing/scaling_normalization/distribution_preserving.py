from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union
from scipy.stats import rankdata
from scipy.stats import rankdata, norm

class DistributionPreservingNormalizer(BaseTransformer):

    def __init__(self, strategy: str='quantile', reference_distribution: Optional[str]=None, preserve_mean: bool=True, preserve_std: bool=True, name: Optional[str]=None):
        """
        Initialize the DistributionPreservingNormalizer.
        
        Parameters
        ----------
        strategy : str, default='quantile'
            Normalization strategy to use ('quantile', 'rank', or 'gaussian')
        reference_distribution : str or None, default=None
            Target distribution type ('normal', 'uniform', or None to preserve original)
        preserve_mean : bool, default=True
            Whether to preserve the original mean after normalization
        preserve_std : bool, default=True
            Whether to preserve the original standard deviation after normalization
        name : str or None, default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        valid_strategies = ['quantile', 'rank', 'gaussian']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
        valid_ref_dist = ['normal', 'uniform', None]
        if reference_distribution not in valid_ref_dist:
            raise ValueError(f"Invalid reference_distribution '{reference_distribution}'. Must be one of {valid_ref_dist}")
        self.strategy = strategy
        self.reference_distribution = reference_distribution
        self.preserve_mean = preserve_mean
        self.preserve_std = preserve_std
        self._fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'DistributionPreservingNormalizer':
        """
        Fit the normalizer to the input data by computing necessary statistics.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to fit the normalizer on
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        DistributionPreservingNormalizer
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
        self.n_features_in_ = X.shape[1]
        self._original_means = np.mean(X, axis=0)
        self._original_stds = np.std(X, axis=0)
        self._original_stds[self._original_stds == 0] = 1.0
        if self.strategy == 'quantile':
            n_samples = X.shape[0]
            self._n_quantiles = min(1000, n_samples)
            self._quantiles = np.linspace(0, 1, self._n_quantiles)
            self._feature_quantiles = np.zeros((self._n_quantiles, self.n_features_in_))
            for i in range(self.n_features_in_):
                self._feature_quantiles[:, i] = np.quantile(X[:, i], self._quantiles)
            self._references = np.zeros((n_samples, self.n_features_in_))
            for i in range(self.n_features_in_):
                if self.reference_distribution == 'normal':
                    self._references[:, i] = norm.ppf(np.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_samples)) * self._original_stds[i] + self._original_means[i]
                elif self.reference_distribution == 'uniform':
                    self._references[:, i] = np.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_samples)
        elif self.strategy == 'rank':
            self._original_data = X.copy()
        elif self.strategy == 'gaussian':
            pass
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply distribution-preserving normalization to the input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to normalize
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Normalized feature set with preserved distribution characteristics
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
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
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        if self.strategy == 'quantile':
            X_transformed = self._quantile_transform(X)
        elif self.strategy == 'rank':
            X_transformed = self._rank_transform(X)
        elif self.strategy == 'gaussian':
            X_transformed = self._gaussian_transform(X)
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')
        if self.preserve_mean:
            X_transformed = X_transformed - np.mean(X_transformed, axis=0) + self._original_means
        if self.preserve_std:
            current_std = np.std(X_transformed, axis=0)
            current_std[current_std == 0] = 1.0
            X_transformed = X_transformed * (self._original_stds / current_std)
        metadata.update({'distribution_preserving_normalization': {'strategy': self.strategy, 'reference_distribution': self.reference_distribution, 'preserve_mean': self.preserve_mean, 'preserve_std': self.preserve_std}})
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse distribution-preserving normalization if possible.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Normalized data to invert
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Data restored to original scale with distribution preserved
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
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
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        X_restored = X.copy()
        if self.strategy == 'quantile':
            X_restored = self._quantile_inverse_transform(X_restored)
        elif self.strategy == 'rank':
            X_restored = self._rank_inverse_transform(X_restored)
        elif self.strategy == 'gaussian':
            X_restored = self._gaussian_inverse_transform(X_restored)
        if self.preserve_std:
            current_std = np.std(X_restored, axis=0)
            current_std[current_std == 0] = 1.0
            X_restored = X_restored * (self._original_stds / current_std)
        if self.preserve_mean:
            X_restored = X_restored - np.mean(X_restored, axis=0) + self._original_means
        if 'distribution_preserving_normalization' in metadata:
            del metadata['distribution_preserving_normalization']
        return FeatureSet(features=X_restored, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def _quantile_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply quantile-based transformation."""
        (n_samples, n_features) = X.shape
        X_transformed = np.zeros_like(X)
        for i in range(n_features):
            quantiles = self._quantiles
            feature_quantiles = self._feature_quantiles[:, i]
            positions = np.searchsorted(feature_quantiles, X[:, i], side='right')
            normalized_positions = positions / len(feature_quantiles)
            X_transformed[:, i] = np.interp(normalized_positions, quantiles, feature_quantiles)
        if self.reference_distribution == 'normal':
            X_uniform = np.zeros_like(X_transformed)
            for i in range(n_features):
                feature_min = X_transformed[:, i].min()
                feature_max = X_transformed[:, i].max()
                if feature_max > feature_min:
                    X_uniform[:, i] = (X_transformed[:, i] - feature_min) / (feature_max - feature_min)
                else:
                    X_uniform[:, i] = 0.5
            X_uniform = np.clip(X_uniform, 1e-10, 1 - 1e-10)
            X_transformed = norm.ppf(X_uniform)
        elif self.reference_distribution == 'uniform':
            for i in range(n_features):
                feature_min = X_transformed[:, i].min()
                feature_max = X_transformed[:, i].max()
                if feature_max > feature_min:
                    X_transformed[:, i] = (X_transformed[:, i] - feature_min) / (feature_max - feature_min)
        return X_transformed

    def _rank_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply rank-based transformation."""
        (n_samples, n_features) = X.shape
        X_transformed = np.zeros_like(X)
        for i in range(n_features):
            ranks = rankdata(X[:, i], method='average')
            X_transformed[:, i] = (ranks - 1) / (n_samples - 1)
        if self.reference_distribution == 'normal':
            X_transformed = np.clip(X_transformed, 1e-10, 1 - 1e-10)
            X_transformed = norm.ppf(X_transformed)
        return X_transformed

    def _gaussian_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply gaussian-based transformation (z-score normalization)."""
        X_transformed = (X - self._original_means) / self._original_stds
        if self.reference_distribution == 'uniform':
            X_transformed = norm.cdf(X_transformed)
        return X_transformed

    def _quantile_inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply inverse quantile-based transformation."""
        X_restored = np.zeros_like(X)
        (n_samples, n_features) = X.shape
        X_input = X.copy()
        if self.reference_distribution == 'normal':
            X_input = norm.cdf(X_input)
        elif self.reference_distribution == 'uniform':
            pass
        for i in range(n_features):
            uniform_values = X_input[:, i]
            quantiles = self._quantiles
            feature_quantiles = self._feature_quantiles[:, i]
            X_restored[:, i] = np.interp(uniform_values, quantiles, feature_quantiles)
        return X_restored

    def _rank_inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply inverse rank-based transformation."""
        X_restored = np.zeros_like(X)
        (n_samples, n_features) = X.shape
        X_input = X.copy()
        if self.reference_distribution == 'normal':
            X_input = norm.cdf(X_input)
        elif self.reference_distribution == 'uniform':
            pass
        for i in range(n_features):
            if hasattr(self, '_original_data'):
                sorted_original = np.sort(self._original_data[:, i])
                ranks = rankdata(X_input[:, i], method='average')
                indices = ((ranks - 1) / (len(sorted_original) - 1) * (len(sorted_original) - 1)).astype(int)
                indices = np.clip(indices, 0, len(sorted_original) - 1)
                X_restored[:, i] = sorted_original[indices]
            else:
                quantiles = np.linspace(0, 1, n_samples)
                original_data_sorted = np.sort(np.linspace(self._original_means[i] - 3 * self._original_stds[i], self._original_means[i] + 3 * self._original_stds[i], n_samples))
                X_restored[:, i] = np.interp(X_input[:, i], quantiles, original_data_sorted)
        return X_restored

    def _gaussian_inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply inverse gaussian-based transformation."""
        X_input = X.copy()
        if self.reference_distribution == 'uniform':
            X_input = norm.ppf(np.clip(X_input, 1e-10, 1 - 1e-10))
        elif self.reference_distribution == 'normal':
            pass
        X_restored = X_input * self._original_stds + self._original_means
        return X_restored