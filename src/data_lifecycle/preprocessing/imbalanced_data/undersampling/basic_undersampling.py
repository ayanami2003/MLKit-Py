from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from collections import Counter
import numpy as np

class Undersampler(BaseTransformer):

    def __init__(self, sampling_strategy: Union[str, float, dict]='majority', random_state: Optional[int]=None, replacement: bool=False, name: Optional[str]=None):
        """
        Initialize the Undersampler.
        
        Parameters
        ----------
        sampling_strategy : Union[str, float, dict], optional
            Strategy to use for undersampling (default is 'majority')
        random_state : int, optional
            Random seed for reproducibility (default is None)
        replacement : bool, default False
            Whether to sample with replacement
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement
        self._sampling_targets = None

    def fit(self, data: FeatureSet, **kwargs) -> 'Undersampler':
        """
        Fit the undersampler to the input data.
        
        This method analyzes the class distribution in the input data to determine
        how to perform undersampling. It stores necessary information for the
        transform method.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing features and labels
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Undersampler
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data does not contain labels
        """
        if data.metadata is None or 'target' not in data.metadata:
            raise ValueError("FeatureSet must contain target labels in metadata['target']")
        y = data.metadata['target']
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('Target labels must be a numpy array or list')
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        class_counts = Counter(y)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        if len(classes) < 2:
            self._sampling_targets = {}
            return self
        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy == 'majority':
                max_count = max(counts)
                min_count = min(counts)
                self._sampling_targets = {}
                for (cls, count) in class_counts.items():
                    if count == max_count:
                        self._sampling_targets[int(cls)] = min_count
            elif self.sampling_strategy == 'not minority':
                min_count = min(counts)
                self._sampling_targets = {}
                for (cls, count) in class_counts.items():
                    if count != min_count:
                        self._sampling_targets[int(cls)] = min_count
            elif self.sampling_strategy == 'not majority':
                max_count = max(counts)
                self._sampling_targets = {}
                non_majority_counts = [c for c in counts if c != max_count]
                target_count = max(non_majority_counts) if non_majority_counts else max_count
                for (cls, count) in class_counts.items():
                    if count == max_count:
                        self._sampling_targets[int(cls)] = target_count
            elif self.sampling_strategy == 'all':
                min_count = min(counts)
                self._sampling_targets = {}
                for cls in classes:
                    self._sampling_targets[int(cls)] = min_count
            else:
                raise ValueError(f'Unknown sampling strategy: {self.sampling_strategy}')
        elif isinstance(self.sampling_strategy, float):
            if not 0 < self.sampling_strategy <= 1:
                raise ValueError('sampling_strategy as float must be in range (0, 1]')
            majority_count = max(counts)
            target_count = int(majority_count * self.sampling_strategy)
            self._sampling_targets = {}
            for (cls, count) in class_counts.items():
                if count == majority_count:
                    self._sampling_targets[int(cls)] = target_count
        elif isinstance(self.sampling_strategy, dict):
            self._sampling_targets = {}
            for (cls, count) in class_counts.items():
                if cls in self.sampling_strategy:
                    self._sampling_targets[int(cls)] = min(self.sampling_strategy[cls], count)
                else:
                    self._sampling_targets[int(cls)] = count
        else:
            raise ValueError('sampling_strategy must be a string, float, or dictionary')
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply undersampling to the input data.
        
        This method reduces the number of samples in the majority class(es)
        according to the specified sampling strategy.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to undersample
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Undersampled feature set with balanced class distribution
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if self._sampling_targets is None:
            raise RuntimeError("Undersampler has not been fitted yet. Call 'fit' first.")
        if data.metadata is None or 'target' not in data.metadata:
            raise ValueError("FeatureSet must contain target labels in metadata['target']")
        X = data.features
        y = data.metadata['target']
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices_to_keep = []
        for (cls, target_count) in self._sampling_targets.items():
            class_indices = np.where(y == cls)[0]
            current_count = len(class_indices)
            if target_count >= current_count:
                indices_to_keep.extend(class_indices)
            else:
                if self.replacement:
                    selected_indices = np.random.choice(class_indices, size=target_count, replace=True)
                else:
                    selected_indices = np.random.choice(class_indices, size=target_count, replace=False)
                indices_to_keep.extend(selected_indices)
        sampled_classes = set(self._sampling_targets.keys())
        all_classes = set(np.unique(y))
        unchanged_classes = all_classes - sampled_classes
        for cls in unchanged_classes:
            class_indices = np.where(y == cls)[0]
            indices_to_keep.extend(class_indices)
        indices_to_keep = sorted(indices_to_keep)
        X_resampled = X[indices_to_keep]
        y_resampled = y[indices_to_keep]
        new_metadata = data.metadata.copy() if data.metadata else {}
        new_metadata['target'] = y_resampled
        new_metadata['resampling_method'] = 'undersampling'
        sample_ids = None
        if data.sample_ids is not None:
            sample_ids = [data.sample_ids[i] for i in indices_to_keep]
        return FeatureSet(features=X_resampled, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=sample_ids, metadata=new_metadata, quality_scores=data.quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the undersampling transformation.
        
        Since undersampling is a lossy transformation (samples are removed),
        this method cannot perfectly reconstruct the original dataset. It
        returns the input data unchanged.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed feature set
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Input feature set unchanged
        """
        return data