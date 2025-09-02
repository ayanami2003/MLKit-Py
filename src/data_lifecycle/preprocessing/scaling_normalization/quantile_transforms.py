from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet


# ...(code omitted)...


class QuantileNormalizer(BaseTransformer):

    def __init__(self, n_quantiles: int=1000, output_distribution: str='uniform', name: Optional[str]=None):
        """
        Initialize the QuantileNormalizer.
        
        Parameters
        ----------
        n_quantiles : int, optional
            Number of quantiles to use for the transformation (default: 1000).
        output_distribution : str, optional
            Desired distribution shape ('uniform' or 'normal') (default: 'uniform').
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        if output_distribution not in ['uniform', 'normal']:
            raise ValueError("output_distribution must be either 'uniform' or 'normal'")
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.reference_distribution = None
        self._quantiles = None
        self._references = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'QuantileNormalizer':
        """
        Compute the reference quantile distribution from the training data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Training data to compute reference distribution from.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        QuantileNormalizer
            Fitted transformer instance.
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
        n_quantiles = min(self.n_quantiles, n_samples)
        quantiles = np.linspace(0, 1, n_quantiles)
        self._quantiles = quantiles
        feature_quantiles = np.zeros((n_quantiles, n_features))
        for i in range(n_features):
            feature_quantiles[:, i] = np.quantile(X[:, i], quantiles)
        self.reference_distribution = np.mean(feature_quantiles, axis=1)
        if self.output_distribution == 'uniform':
            self._references = quantiles
        else:
            self._references = np.random.normal(size=n_quantiles)
            self._references.sort()
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply quantile normalization to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed data with quantile normalization applied.
        """
        if self._quantiles is None or self._references is None:
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
        (n_samples, n_features) = X.shape
        X_transformed = np.zeros_like(X)
        for i in range(n_features):
            feature_quantiles = np.quantile(X[:, i], self._quantiles)
            X_transformed[:, i] = np.interp(X[:, i], feature_quantiles, self._references)
        metadata['transformation'] = 'quantile_normalization'
        metadata['output_distribution'] = self.output_distribution
        metadata['n_quantiles'] = self.n_quantiles
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reverse the quantile normalization transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to reverse.
        **kwargs : dict
            Additional inverse transformation parameters.
            
        Returns
        -------
        FeatureSet
            Data restored to original scale.
        """
        if self._quantiles is None or self._references is None or self._original_quantiles is None:
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
        (n_samples, n_features) = X.shape
        if n_features != self._original_quantiles.shape[1]:
            raise ValueError(f'Input data has {n_features} features, but transformer was fitted on {self._original_quantiles.shape[1]} features')
        X_inverse = np.zeros_like(X)
        for i in range(n_features):
            X_inverse[:, i] = np.interp(X[:, i], self._references, self._original_quantiles[:, i])
        if 'transformation' in metadata:
            del metadata['transformation']
        if 'output_distribution' in metadata:
            del metadata['output_distribution']
        if 'n_quantiles' in metadata:
            del metadata['n_quantiles']
        return FeatureSet(features=X_inverse, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)