from typing import Any, Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class DataAugmentationTransformer(BaseTransformer):

    def __init__(self, augmentation_factor: float=0.5, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None):
        """
        Initialize the DataAugmentationTransformer.
        
        Parameters
        ----------
        augmentation_factor : float, default=0.5
            Factor by which to increase the dataset size. Must be non-negative.
            For example, 0.5 adds 50% more samples, 1.0 doubles the dataset size.
        random_state : int, optional
            Random seed for reproducible augmentations
        preserve_original : bool, default=True
            If True, includes original samples in the output along with augmented ones
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.augmentation_factor = augmentation_factor
        self.random_state = random_state
        self.preserve_original = preserve_original

    def fit(self, data: FeatureSet, **kwargs) -> 'DataAugmentationTransformer':
        """
        Fit the transformer to the input data.
        
        This method analyzes the input data to understand its structure and
        characteristics, which might be needed for certain augmentation strategies.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to fit the transformer on
        **kwargs : dict
            Additional parameters for fitting (not used in this base implementation)
            
        Returns
        -------
        DataAugmentationTransformer
            Self instance for method chaining
        """
        if self.augmentation_factor < 0:
            raise ValueError('augmentation_factor must be non-negative')
        self.n_samples_ = data.get_n_samples()
        self.n_features_ = data.get_n_features()
        self.feature_names_ = data.feature_names
        self.feature_types_ = data.feature_types
        self.sample_ids_ = data.sample_ids
        self.rng_ = np.random.default_rng(self.random_state)
        self.fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply data augmentation transformations to the input data.
        
        Generates new synthetic samples based on the fitted data characteristics
        and configured augmentation strategy. The number of generated samples
        is determined by the augmentation_factor parameter.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to augment
        **kwargs : dict
            Additional parameters for transformation (not used in this base implementation)
            
        Returns
        -------
        FeatureSet
            Augmented feature set containing original and/or synthetic samples
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError('Transformer must be fitted before transform')
        n_samples = data.get_n_samples()
        n_augmented = int(round(n_samples * self.augmentation_factor))
        if n_augmented == 0:
            if self.preserve_original:
                return data
            else:
                empty_features = np.empty((0, data.features.shape[1]), dtype=data.features.dtype)
                return FeatureSet(features=empty_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=[] if data.sample_ids is not None else None, metadata=data.metadata.copy() if data.metadata is not None else None, quality_scores=data.quality_scores.copy() if data.quality_scores is not None else None)
        augmented_indices = self.rng_.choice(n_samples, size=n_augmented, replace=True)
        augmented_features = data.features[augmented_indices]
        if self.preserve_original:
            combined_features = np.vstack([data.features, augmented_features])
            combined_sample_ids = None
            if data.sample_ids is not None:
                augmented_sample_ids = [f'aug_{i}' for i in range(n_augmented)]
                combined_sample_ids = data.sample_ids + augmented_sample_ids
            return FeatureSet(features=combined_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=combined_sample_ids, metadata=data.metadata.copy() if data.metadata is not None else None, quality_scores=data.quality_scores.copy() if data.quality_scores is not None else None)
        else:
            augmented_sample_ids = None
            if data.sample_ids is not None:
                augmented_sample_ids = [f'aug_{i}' for i in range(n_augmented)]
            return FeatureSet(features=augmented_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=augmented_sample_ids, metadata=data.metadata.copy() if data.metadata is not None else None, quality_scores=data.quality_scores.copy() if data.quality_scores is not None else None)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for data augmentation.
        
        Since augmentation creates new synthetic samples rather than modifying
        existing ones in a reversible way, there's no meaningful inverse operation.
        
        Parameters
        ----------
        data : FeatureSet
            Augmented data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Never returns successfully - always raises NotImplementedError
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported
        """
        raise NotImplementedError('inverse_transform is not implemented for DataAugmentationTransformer')