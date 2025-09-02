from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from collections import Counter
from src.data_lifecycle.preprocessing.imbalanced_data.oversampling.smote_variants import SMOTEOversampler
from src.data_lifecycle.preprocessing.imbalanced_data.oversampling.adasyn import ADASYNOversampler
from src.data_lifecycle.preprocessing.imbalanced_data.oversampling.basic_oversampling import SimpleOversampler

class GenericOversampler(BaseTransformer):

    def __init__(self, sampling_strategy: Union[str, float]='auto', random_state: Optional[int]=None, method: str='smote', k_neighbors: int=5, name: Optional[str]=None):
        """
        Initialize the GenericOversampler.
        
        Args:
            sampling_strategy: Strategy to use for sampling ('auto', 'minority', or float ratio)
            random_state: Random seed for reproducible results
            method: Oversampling technique to apply ('smote', 'adasyn', 'random')
            k_neighbors: Number of neighbors for SMOTE-based methods
            name: Optional name for the transformer
        """
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.method = method
        self.k_neighbors = k_neighbors
        if method not in ['smote', 'adasyn', 'random']:
            raise ValueError("method must be one of 'smote', 'adasyn', or 'random'")
        if not isinstance(k_neighbors, int) or k_neighbors <= 0:
            raise ValueError('k_neighbors must be a positive integer')
        if not (isinstance(sampling_strategy, (str, float)) and (sampling_strategy in ['auto', 'minority'] or (isinstance(sampling_strategy, float) and 0 < sampling_strategy <= 1))):
            raise ValueError("sampling_strategy must be 'auto', 'minority', or a float between 0 and 1")

    def fit(self, data: FeatureSet, **kwargs) -> 'GenericOversampler':
        """
        Fit the oversampler to the input data.
        
        Analyzes the class distribution in the provided data and prepares
        the oversampling mechanism according to the specified method.
        
        Args:
            data: FeatureSet containing features and labels to fit on
            **kwargs: Additional parameters for fitting
            
        Returns:
            GenericOversampler: Self instance for method chaining
        """
        if data.labels is None:
            raise ValueError('Input data must contain labels for oversampling.')
        self._labels = np.array(data.labels) if not isinstance(data.labels, np.ndarray) else data.labels
        self._class_counts = Counter(self._labels)
        self._classes = list(self._class_counts.keys())
        if len(self._classes) < 2:
            self._needs_oversampling = False
            self._fitted = True
            return self
        majority_count = max(self._class_counts.values())
        minority_counts = [count for (cls, count) in self._class_counts.items() if count < majority_count]
        self._needs_oversampling = len(minority_counts) > 0
        if not self._needs_oversampling:
            self._fitted = True
            return self
        sampling_strategy = self.sampling_strategy
        if sampling_strategy == 'minority':
            minority_class = min(self._class_counts, key=self._class_counts.get)
            if self.method in ['smote', 'adasyn']:
                sampling_strategy = 'minority'
            else:
                sampling_strategy = majority_count / self._class_counts[minority_class]
        elif isinstance(sampling_strategy, float) and 0 < sampling_strategy <= 1 and (self.method == 'random'):
            if sampling_strategy <= 0:
                raise ValueError('sampling_strategy as float must be positive')
        if self.method == 'smote':
            self._oversampler = SMOTEOversampler(sampling_strategy=sampling_strategy, k_neighbors=self.k_neighbors, random_state=self.random_state, smote_variant='regular')
        elif self.method == 'adasyn':
            self._oversampler = ADASYNOversampler(sampling_strategy=sampling_strategy, k_neighbors=self.k_neighbors, random_state=self.random_state)
        elif self.method == 'random':
            self._oversampler = SimpleOversampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
        self._oversampler.fit(data, **kwargs)
        self._fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply oversampling to the input data.
        
        Generates synthetic or duplicated samples for minority classes based
        on the fitted oversampling strategy and method.
        
        Args:
            data: FeatureSet to transform with oversampling
            **kwargs: Additional parameters for transformation
            
        Returns:
            FeatureSet: Transformed data with balanced class distribution
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise RuntimeError('GenericOversampler must be fitted before transform()')
        if not hasattr(self, '_needs_oversampling') or not self._needs_oversampling:
            return data
        return self._oversampler.transform(data, **kwargs)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the oversampling transformation (not supported).
        
        Since oversampling adds synthetic data points, exact inversion
        is not possible. This method raises NotImplementedError.
        
        Args:
            data: Transformed FeatureSet to invert
            **kwargs: Additional parameters
            
        Returns:
            FeatureSet: Original data without synthetic samples
            
        Raises:
            NotImplementedError: Always raised as inversion is not supported
        """
        raise NotImplementedError('Inverse transform is not implemented for oversampling techniques.')