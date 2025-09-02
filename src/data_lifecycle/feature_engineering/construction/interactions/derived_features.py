from general.structures.feature_set import FeatureSet
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.base_classes.validator_base import BaseValidator
from typing import List, Optional, Union, Callable

class DerivedFeatureCreator(BaseTransformer):

    def __init__(self, transformations: List[Callable[[np.ndarray], np.ndarray]], feature_names: List[str], name: Optional[str]=None):
        """
        Initialize the DerivedFeatureCreator.
        
        Args:
            transformations: List of functions that take a feature matrix and return a new feature vector.
            feature_names: Names for the derived features.
            name: Optional name for the transformer.
        """
        super().__init__(name)
        self.transformations = transformations
        self.feature_names = feature_names

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'DerivedFeatureCreator':
        """
        Fit the transformer to the input data.
        
        For derived feature creation, fitting typically just validates the input structure.
        
        Args:
            data: Input features as FeatureSet or numpy array.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            Self instance for method chaining.
        """
        if not isinstance(data, (FeatureSet, np.ndarray)):
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if not self.transformations:
            raise ValueError('At least one transformation function must be provided')
        if len(self.feature_names) != len(self.transformations):
            raise ValueError('Number of feature names must match number of transformations')
        for (i, func) in enumerate(self.transformations):
            if not callable(func):
                raise TypeError(f'Transformation at index {i} is not callable')
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply derived feature transformations to the input data.
        
        Args:
            data: Input features as FeatureSet or numpy array.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet with original features plus derived features.
        """
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
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        derived_features_list = []
        for func in self.transformations:
            derived_feature = func(X)
            if derived_feature.ndim == 1:
                derived_feature = derived_feature.reshape(-1, 1)
            elif derived_feature.ndim != 2 or derived_feature.shape[1] != 1:
                raise ValueError('Each transformation must return a 1D array or a 2D array with one column')
            derived_features_list.append(derived_feature)
        if derived_features_list:
            derived_features = np.hstack(derived_features_list)
            all_features = np.hstack([X, derived_features])
        else:
            all_features = X
        if feature_names is not None:
            all_feature_names = feature_names + self.feature_names
        else:
            all_feature_names = None
        if feature_types is not None:
            derived_feature_types = ['numeric'] * len(self.feature_names)
            all_feature_types = feature_types + derived_feature_types
        else:
            all_feature_types = None
        return FeatureSet(features=all_features, feature_names=all_feature_names, feature_types=all_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for derived features.
        
        Args:
            data: Transformed data (ignored).
            **kwargs: Additional parameters (ignored).
            
        Returns:
            Original data without derived features.
            
        Raises:
            NotImplementedError: Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for derived features.')