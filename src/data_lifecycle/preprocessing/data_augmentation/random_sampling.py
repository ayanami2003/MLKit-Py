import numpy as np
from general.structures.feature_set import FeatureSet
from general.base_classes.transformer_base import BaseTransformer
from typing import Optional

class RandomSamplingTransformer(BaseTransformer):

    def __init__(self, sampling_ratio: float=1.0, replace: bool=True, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None):
        """
        Initialize the RandomSamplingTransformer.
        
        Parameters
        ----------
        sampling_ratio : float, default=1.0
            Ratio of samples to generate relative to the original dataset size.
            Values > 1.0 indicate upsampling, < 1.0 indicate downsampling.
        replace : bool, default=True
            Whether to sample with replacement (True) or without (False).
        random_state : Optional[int], default=None
            Random seed for reproducibility.
        preserve_original : bool, default=True
            If True, includes original samples in the output when upsampling.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.sampling_ratio = sampling_ratio
        self.replace = replace
        self.random_state = random_state
        self.preserve_original = preserve_original

    def fit(self, data: FeatureSet, **kwargs) -> 'RandomSamplingTransformer':
        """
        Fit the transformer to the input data.
        
        For random sampling, fitting simply stores the data shape and feature information.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to fit the transformer on.
        **kwargs : dict
            Additional parameters (unused).
            
        Returns
        -------
        RandomSamplingTransformer
            Self instance for method chaining.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        self._n_input_samples = data.features.shape[0]
        self._n_features = data.features.shape[1]
        self._feature_names = data.feature_names
        self._feature_types = data.feature_types
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply random sampling to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs : dict
            Additional parameters (unused).
            
        Returns
        -------
        FeatureSet
            Transformed feature set with sampled data.
        """
        if not hasattr(self, '_n_input_samples'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        n_input_samples = data.features.shape[0]
        if n_input_samples != self._n_input_samples:
            raise ValueError(f'Input data has {n_input_samples} samples, but transformer was fitted on {self._n_input_samples} samples.')
        n_samples_to_generate = int(round(self.sampling_ratio * self._n_input_samples))
        if n_samples_to_generate <= 0:
            raise ValueError('sampling_ratio must be positive.')
        if not self.replace and n_samples_to_generate > self._n_input_samples:
            raise ValueError(f'Cannot sample {n_samples_to_generate} samples without replacement from a population of size {self._n_input_samples}.')
        rng = np.random.default_rng(self.random_state)
        sampled_indices = rng.choice(self._n_input_samples, size=n_samples_to_generate, replace=self.replace)
        sampled_features = data.features[sampled_indices]
        sampled_sample_ids = None
        if data.sample_ids is not None:
            sampled_sample_ids = [data.sample_ids[i] for i in sampled_indices]
        if self.preserve_original and self.sampling_ratio > 1.0:
            sampled_features = np.vstack([data.features, sampled_features])
            if sampled_sample_ids is not None and data.sample_ids is not None:
                sampled_sample_ids = data.sample_ids + sampled_sample_ids
        return FeatureSet(features=sampled_features, feature_names=self._feature_names, feature_types=self._feature_types, sample_ids=sampled_sample_ids, metadata=data.metadata.copy() if data.metadata else {}, quality_scores=data.quality_scores.copy() if data.quality_scores else {})

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for random sampling.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            The same data passed as input since inverse transformation
            is not meaningful for random sampling.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('inverse_transform is not implemented for RandomSamplingTransformer')