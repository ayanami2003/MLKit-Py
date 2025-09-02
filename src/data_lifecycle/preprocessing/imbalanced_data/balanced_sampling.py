from typing import Optional, Union, Dict, Any
import numpy as np
from collections import Counter
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class BalancedSampler(BaseTransformer):

    def __init__(self, sampling_strategy: Union[str, float]='auto', random_state: Optional[int]=None, method: str='mixed', replacement: bool=True, name: Optional[str]=None):
        """
        Initialize the BalancedSampler.
        
        Parameters
        ----------
        sampling_strategy : str or float, default='auto'
            Strategy to balance the dataset. Can be 'auto', 'majority', 'not_majority',
            'all', 'minority', or a float representing the desired ratio.
        random_state : int, optional
            Random seed for reproducibility of sampling operations.
        method : str, default='mixed'
            Sampling method to use. Options include 'oversample', 'undersample', or 'mixed'.
        replacement : bool, default=True
            Whether to sample with replacement when oversampling.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.method = method
        self.replacement = replacement

    def fit(self, data: FeatureSet, **kwargs) -> 'BalancedSampler':
        """
        Fit the sampler to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing features and labels.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        BalancedSampler
            Self instance for method chaining.
        """
        if 'labels' not in data.metadata or data.metadata['labels'] is None:
            raise ValueError('Input data must contain labels in metadata to perform balanced sampling')
        y = np.asarray(data.metadata['labels'])
        if len(y) != data.features.shape[0]:
            raise ValueError('Number of samples in features and labels must match')
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.fitted_ = True
            self.classes_ = unique_classes
            self.class_counts_ = {int(unique_classes[0]): len(y)}
            self.majority_class_ = int(unique_classes[0])
            self.minority_class_ = int(unique_classes[0])
            self.target_ratios_ = {int(unique_classes[0]): len(y)}
            self.original_n_samples_ = data.features.shape[0]
            return self
        if self.random_state is not None:
            np.random.seed(self.random_state)
        (unique_classes, counts) = np.unique(y, return_counts=True)
        self.classes_ = unique_classes
        self.class_counts_ = dict(zip(unique_classes, counts))
        self.majority_class_ = unique_classes[np.argmax(counts)]
        self.minority_class_ = unique_classes[np.argmin(counts)]
        valid_methods = ['oversample', 'undersample', 'mixed']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{self.method}'")
        valid_strategies = ['auto', 'majority', 'not_majority', 'all', 'minority']
        if isinstance(self.sampling_strategy, str) and self.sampling_strategy not in valid_strategies:
            raise ValueError(f"When sampling_strategy is a string, it must be one of {valid_strategies}, got '{self.sampling_strategy}'")
        elif isinstance(self.sampling_strategy, (int, float)):
            if not (0 < self.sampling_strategy <= 1 or self.sampling_strategy > 1):
                raise ValueError(f'When sampling_strategy is a float, it must be in range (0, 1] or greater than 1, got {self.sampling_strategy}')
        self.target_ratios_ = {}
        if self.sampling_strategy == 'auto':
            target_count = self.class_counts_[self.minority_class_]
            for cls in self.classes_:
                self.target_ratios_[int(cls)] = min(target_count, self.class_counts_[cls])
        elif self.sampling_strategy == 'majority':
            target_count = self.class_counts_[self.minority_class_]
            for cls in self.classes_:
                if cls == self.majority_class_:
                    self.target_ratios_[int(cls)] = target_count
                else:
                    self.target_ratios_[int(cls)] = self.class_counts_[cls]
        elif self.sampling_strategy == 'not_majority':
            target_count = self.class_counts_[self.majority_class_]
            for cls in self.classes_:
                if cls != self.majority_class_:
                    self.target_ratios_[int(cls)] = target_count
                else:
                    self.target_ratios_[int(cls)] = self.class_counts_[cls]
        elif self.sampling_strategy == 'all':
            avg_count = int(np.mean(list(self.class_counts_.values())))
            for cls in self.classes_:
                self.target_ratios_[int(cls)] = avg_count
        elif self.sampling_strategy == 'minority':
            target_count = self.class_counts_[self.majority_class_]
            for cls in self.classes_:
                if cls == self.minority_class_:
                    self.target_ratios_[int(cls)] = target_count
                else:
                    self.target_ratios_[int(cls)] = self.class_counts_[cls]
        elif isinstance(self.sampling_strategy, (int, float)):
            if isinstance(self.sampling_strategy, float) and 0 < self.sampling_strategy <= 1:
                majority_count = self.class_counts_[self.majority_class_]
                target_minority_count = int(majority_count * self.sampling_strategy)
                original_minority_count = self.class_counts_[self.minority_class_]
                if self.method in ['oversample', 'mixed']:
                    target_minority_count = max(target_minority_count, original_minority_count)
                elif self.method == 'undersample':
                    target_minority_count = min(target_minority_count, original_minority_count)
                for cls in self.classes_:
                    if cls == self.minority_class_:
                        self.target_ratios_[int(cls)] = target_minority_count
                    elif cls == self.majority_class_:
                        if self.method in ['undersample', 'mixed']:
                            self.target_ratios_[int(cls)] = int(target_minority_count / self.sampling_strategy)
                        else:
                            self.target_ratios_[int(cls)] = self.class_counts_[cls]
                    elif self.method == 'undersample':
                        self.target_ratios_[int(cls)] = int(target_minority_count / self.sampling_strategy)
                    else:
                        self.target_ratios_[int(cls)] = self.class_counts_[cls]
            elif isinstance(self.sampling_strategy, (int, float)) and self.sampling_strategy > 1:
                target_count = int(self.sampling_strategy)
                for cls in self.classes_:
                    self.target_ratios_[int(cls)] = target_count
            else:
                raise ValueError(f'Invalid sampling_strategy value: {self.sampling_strategy}')
        self.original_n_samples_ = data.features.shape[0]
        self.fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply balanced sampling to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Balanced feature set after applying sampling techniques.
        """
        if not self.fitted_:
            raise ValueError('Transformer has not been fitted yet.')
        if 'labels' not in data.metadata or data.metadata['labels'] is None:
            raise ValueError('Input data must contain labels in metadata to perform balanced sampling')
        y = np.asarray(data.metadata['labels'])
        if len(y) != data.features.shape[0]:
            raise ValueError('Number of samples in features and labels must match')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices_to_keep = []
        for cls in self.classes_:
            class_indices = np.where(y == cls)[0]
            current_count = len(class_indices)
            target_count = self.target_ratios_[int(cls)]
            if self.method == 'oversample' or (self.method == 'mixed' and target_count > current_count):
                if self.replacement:
                    selected_indices = np.random.choice(class_indices, target_count, replace=True)
                elif target_count <= current_count:
                    selected_indices = np.random.choice(class_indices, target_count, replace=False)
                else:
                    n_full_sets = target_count // current_count
                    remainder = target_count % current_count
                    selected_indices = np.concatenate([np.tile(class_indices, n_full_sets), np.random.choice(class_indices, remainder, replace=False) if remainder > 0 else np.array([])])
                    selected_indices = selected_indices.astype(int)
            elif self.method == 'undersample' or (self.method == 'mixed' and target_count < current_count):
                selected_indices = np.random.choice(class_indices, target_count, replace=False)
            else:
                selected_indices = class_indices
            indices_to_keep.extend(selected_indices)
        indices_to_keep = sorted(indices_to_keep)
        new_features = data.features[indices_to_keep]
        new_labels = y[indices_to_keep]
        new_sample_ids = None
        if data.sample_ids is not None:
            new_sample_ids = [data.sample_ids[i] for i in indices_to_keep]
        new_metadata = data.metadata.copy()
        new_metadata['labels'] = new_labels.tolist() if isinstance(new_labels, np.ndarray) else new_labels
        new_fs = FeatureSet(features=new_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=new_sample_ids, metadata=new_metadata, quality_scores=data.quality_scores)
        return new_fs

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the sampling transformation (if possible).
        
        Parameters
        ----------
        data : FeatureSet
            Transformed feature set to inverse transform.
        **kwargs : dict
            Additional parameters for inverse transformation.
            
        Returns
        -------
        FeatureSet
            Original feature set before sampling (if possible).
        """
        return data