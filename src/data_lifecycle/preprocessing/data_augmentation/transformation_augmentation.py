from typing import Optional, Union, Callable, Any
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np

class TransformationAugmentationTransformer(BaseTransformer):

    def __init__(self, transformation_func: Callable[[Union[np.ndarray, FeatureSet]], Union[np.ndarray, FeatureSet]], augmentation_factor: float=0.5, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None):
        """
        Initialize the TransformationAugmentationTransformer.
        
        Args:
            transformation_func (Callable): Function that defines how to transform data samples.
            augmentation_factor (float): Factor determining how many new samples to generate.
            random_state (Optional[int]): Random seed for reproducibility.
            preserve_original (bool): If True, includes original data in the output.
            name (Optional[str]): Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.transformation_func = transformation_func
        self.augmentation_factor = augmentation_factor
        self.random_state = random_state
        self.preserve_original = preserve_original

    def fit(self, data: FeatureSet, **kwargs) -> 'TransformationAugmentationTransformer':
        """
        Fit the transformer to the input data (no-op for this transformer).
        
        Args:
            data (FeatureSet): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            TransformationAugmentationTransformer: Returns self.
        """
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply transformations to augment the dataset.
        
        Args:
            data (FeatureSet): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Augmented dataset with transformed samples.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        rng = np.random.default_rng(self.random_state)
        n_original_samples = data.features.shape[0]
        n_new_samples = int(np.round(n_original_samples * self.augmentation_factor))
        if n_new_samples <= 0:
            if self.preserve_original:
                return data
            else:
                empty_features = np.empty((0, data.features.shape[1]), dtype=data.features.dtype)
                final_sample_ids = [] if data.sample_ids is not None else None
                final_feature_names = data.feature_names
                final_feature_types = data.feature_types
                final_metadata = data.metadata.copy() if data.metadata else {}
                final_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
                return FeatureSet(features=empty_features, feature_names=final_feature_names, feature_types=final_feature_types, sample_ids=final_sample_ids, metadata=final_metadata, quality_scores=final_quality_scores)
        sample_indices = rng.choice(n_original_samples, size=n_new_samples, replace=True)
        augmented_features_list = []
        for idx in sample_indices:
            original_sample = data.features[idx:idx + 1]
            transformed_sample = self.transformation_func(original_sample)
            if isinstance(transformed_sample, np.ndarray):
                if transformed_sample.ndim == 1:
                    augmented_features_list.append(transformed_sample.reshape(1, -1))
                else:
                    augmented_features_list.append(transformed_sample)
            elif isinstance(transformed_sample, FeatureSet):
                if transformed_sample.features.ndim == 1:
                    augmented_features_list.append(transformed_sample.features.reshape(1, -1))
                else:
                    augmented_features_list.append(transformed_sample.features)
            else:
                raise TypeError('transformation_func must return either a numpy array or FeatureSet')
        if augmented_features_list:
            augmented_features = np.vstack(augmented_features_list)
        else:
            augmented_features = np.empty((0, data.features.shape[1]))
        if self.preserve_original:
            final_features = np.vstack([data.features, augmented_features])
        else:
            final_features = augmented_features
        final_sample_ids = None
        if data.sample_ids is not None:
            if self.preserve_original:
                augmented_ids = [f'aug_{i}' for i in range(n_new_samples)]
                final_sample_ids = data.sample_ids + augmented_ids
            else:
                augmented_ids = [f'aug_{i}' for i in range(n_new_samples)]
                final_sample_ids = augmented_ids
        final_feature_names = data.feature_names
        final_feature_types = data.feature_types
        final_metadata = data.metadata.copy() if data.metadata else {}
        final_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        return FeatureSet(features=final_features, feature_names=final_feature_names, feature_types=final_feature_types, sample_ids=final_sample_ids, metadata=final_metadata, quality_scores=final_quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for augmentation.
        
        Args:
            data (FeatureSet): Transformed data.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Original data (not supported).
            
        Raises:
            NotImplementedError: Always raised since inverse is not meaningful for augmentation.
        """
        raise NotImplementedError('Inverse transformation is not supported for data augmentation.')