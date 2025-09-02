from typing import Optional
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np

class DataMixingTransformer(BaseTransformer):

    def __init__(self, mixing_strategy: str='convex', alpha: float=0.2, apply_to_labels: bool=True, preserve_original: bool=True, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.mixing_strategy = mixing_strategy
        self.alpha = alpha
        self.apply_to_labels = apply_to_labels
        self.preserve_original = preserve_original
        self.random_state = random_state

    def fit(self, data: FeatureSet, **kwargs) -> 'DataMixingTransformer':
        """
        Fit the transformer to the input data by recording its structure.

        Parameters
        ----------
        data : FeatureSet
            Input data to fit the transformer on
        **kwargs : dict
            Additional parameters (ignored)

        Returns
        -------
        DataMixingTransformer
            Self instance for method chaining
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        self.n_features_ = data.features.shape[1]
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply data mixing transformation to generate augmented samples.

        Parameters
        ----------
        data : FeatureSet
            Input data to transform
        **kwargs : dict
            Additional parameters (ignored)

        Returns
        -------
        FeatureSet
            Augmented data with mixed samples
        """
        if not hasattr(self, 'n_features_'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        if data.features.shape[1] != self.n_features_:
            raise ValueError(f'Input data has {data.features.shape[1]} features, but transformer was fitted on {self.n_features_} features.')
        rng = np.random.default_rng(self.random_state)
        n_samples = data.features.shape[0]
        if n_samples <= 1:
            if self.preserve_original:
                return data
            else:
                empty_features = np.empty((0, data.features.shape[1]), dtype=data.features.dtype)
                empty_sample_ids = [] if data.sample_ids is not None else None
                empty_metadata = data.metadata.copy() if data.metadata else {}
                empty_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
                return FeatureSet(features=empty_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=empty_sample_ids, metadata=empty_metadata, quality_scores=empty_quality_scores)
        mixed_features = None
        mixed_labels = None
        if self.mixing_strategy == 'convex':
            indices1 = rng.integers(0, n_samples, n_samples)
            indices2 = rng.integers(0, n_samples, n_samples)
            mixed_features = self.alpha * data.features[indices1] + (1 - self.alpha) * data.features[indices2]
        elif self.mixing_strategy == 'cutmix':
            indices1 = rng.integers(0, n_samples, n_samples)
            indices2 = rng.integers(0, n_samples, n_samples)
            mixed_features = data.features[indices1].copy()
            for i in range(n_samples):
                start_idx = rng.integers(0, self.n_features_)
                end_idx = rng.integers(start_idx, self.n_features_)
                if end_idx > start_idx:
                    mixed_features[i, start_idx:end_idx] = data.features[indices2[i], start_idx:end_idx]
        elif self.mixing_strategy == 'mixup':
            indices1 = rng.integers(0, n_samples, n_samples)
            indices2 = rng.integers(0, n_samples, n_samples)
            lambda_vals = rng.beta(self.alpha, self.alpha, n_samples)
            mixed_features = lambda_vals[:, np.newaxis] * data.features[indices1] + (1 - lambda_vals[:, np.newaxis]) * data.features[indices2]
        else:
            raise ValueError(f'Unsupported mixing strategy: {self.mixing_strategy}')
        original_labels = data.metadata.get('labels')
        if original_labels is not None and self.apply_to_labels:
            if self.mixing_strategy == 'mixup':
                mixed_labels = lambda_vals[:, np.newaxis] * original_labels[indices1] + (1 - lambda_vals[:, np.newaxis]) * original_labels[indices2]
            else:
                mixed_labels = original_labels[indices1]
        else:
            mixed_labels = None
        if self.preserve_original:
            final_features = np.vstack([data.features, mixed_features])
            if mixed_labels is not None and original_labels is not None and self.apply_to_labels:
                final_labels = np.concatenate([original_labels, mixed_labels], axis=0)
            else:
                final_labels = original_labels
            if data.sample_ids is not None:
                mixed_sample_ids = [f'mixed_{i}' for i in range(n_samples)]
                final_sample_ids = data.sample_ids + mixed_sample_ids
            else:
                final_sample_ids = None
        else:
            final_features = mixed_features
            final_labels = mixed_labels
            if data.sample_ids is not None:
                final_sample_ids = [f'mixed_{i}' for i in range(n_samples)]
            else:
                final_sample_ids = None
        final_metadata = data.metadata.copy() if data.metadata else {}
        if self.apply_to_labels:
            if final_labels is not None:
                final_metadata['labels'] = final_labels
            elif 'labels' in final_metadata:
                del final_metadata['labels']
        elif original_labels is not None:
            final_metadata['labels'] = original_labels
        elif 'labels' in final_metadata:
            del final_metadata['labels']
        final_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        return FeatureSet(features=final_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=final_sample_ids, metadata=final_metadata, quality_scores=final_quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the data mixing transformation (not supported for this transformer).

        Parameters
        ----------
        data : FeatureSet
            Transformed data to inverse
        **kwargs : dict
            Additional parameters (ignored)

        Returns
        -------
        FeatureSet
            Original data without mixing (if preserve_original=True) or raises NotImplementedError

        Raises
        ------
        NotImplementedError
            If attempting to perform inverse transformation
        """
        raise NotImplementedError('inverse_transform is not implemented for DataMixingTransformer')