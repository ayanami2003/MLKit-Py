from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

class ADASYNOversampler(BaseTransformer):
    """
    ADASYN (Adaptive Synthetic Sampling) oversampler for imbalanced datasets.
    
    This transformer implements the ADASYN algorithm which generates synthetic samples
    for minority classes by focusing on those that are harder to learn. It uses nearest
    neighbors to identify regions where synthetic samples should be generated.
    
    The algorithm works by:
    1. Calculating the desired number of synthetic samples for each minority class instance
    2. Finding k nearest neighbors for each minority sample
    3. Generating synthetic samples along the line segments connecting minority samples to their neighbors
    
    Attributes
    ----------
    sampling_strategy : Union[str, float], default='auto'
        Strategy to use for sampling. Can be 'auto', 'minority', 'not majority',
        or a float representing the desired ratio of minority to majority samples.
    random_state : Optional[int], default=None
        Random seed for reproducibility of synthetic sample generation.
    k_neighbors : int, default=5
        Number of nearest neighbors to consider when generating synthetic samples.
    name : Optional[str], default=None
        Name identifier for the transformer instance.
        
    Methods
    -------
    fit : Fits the transformer to the input data
    transform : Applies ADASYN oversampling to generate synthetic samples
    inverse_transform : Not supported for this transformer (raises NotImplementedError)
    """

    def __init__(self, sampling_strategy: Union[str, float]='auto', random_state: Optional[int]=None, k_neighbors: int=5, name: Optional[str]=None):
        """
        Initialize the ADASYNOversampler.
        
        Parameters
        ----------
        sampling_strategy : Union[str, float], default='auto'
            Strategy to use for sampling. Can be 'auto', 'minority', 'not majority',
            or a float representing the desired ratio of minority to majority samples.
        random_state : Optional[int], default=None
            Random seed for reproducibility of synthetic sample generation.
        k_neighbors : int, default=5
            Number of nearest neighbors to consider when generating synthetic samples.
        name : Optional[str], default=None
            Name identifier for the transformer instance.
        """
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self._fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'ADASYNOversampler':
        """
        Fit the ADASYN oversampler to the input data.
        
        This method analyzes the class distribution in the input data to determine
        where synthetic samples need to be generated.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with features and labels. Labels are required for
            identifying minority and majority classes.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        ADASYNOversampler
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data does not contain labels.
        """
        if data.labels is None:
            raise ValueError('Input data must contain labels for ADASYN oversampling.')
        if hasattr(data.features, 'to_numpy'):
            self._features = data.features.to_numpy()
        else:
            self._features = np.array(data.features)
        self._labels = np.array(data.labels) if not isinstance(data.labels, np.ndarray) else data.labels
        self._class_counts = Counter(self._labels)
        self._classes = list(self._class_counts.keys())
        self._majority_class = max(self._class_counts, key=self._class_counts.get)
        self._minority_classes = [cls for cls in self._classes if cls != self._majority_class]
        self._sampling_targets = self._calculate_sampling_targets()
        self._fitted = True
        return self

    def _calculate_sampling_targets(self):
        """Calculate how many samples to generate for each class based on sampling strategy."""
        targets = {}
        majority_count = self._class_counts[self._majority_class]
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'not majority':
            for cls in self._classes:
                targets[cls] = majority_count
        elif self.sampling_strategy == 'minority':
            for cls in self._classes:
                if cls in self._minority_classes:
                    targets[cls] = majority_count
                else:
                    targets[cls] = self._class_counts[cls]
        elif isinstance(self.sampling_strategy, float):
            if not 0 < self.sampling_strategy <= 1:
                raise ValueError('sampling_strategy as float must be in (0, 1]')
            target_count = int(majority_count * self.sampling_strategy)
            for cls in self._classes:
                if cls in self._minority_classes:
                    targets[cls] = max(target_count, self._class_counts[cls])
                else:
                    targets[cls] = self._class_counts[cls]
        else:
            raise ValueError(f'Invalid sampling_strategy: {self.sampling_strategy}')
        return targets

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply ADASYN oversampling to generate synthetic samples.
        
        This method generates synthetic samples for minority classes based on
        the ADASYN algorithm, using nearest neighbors to identify difficult regions.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to be oversampled.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Oversampled feature set with synthetic samples added for minority classes.
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError('ADASYNOversampler must be fitted before transform()')
        rng = np.random.default_rng(self.random_state)
        if hasattr(data.features, 'to_numpy'):
            features = data.features.to_numpy()
        else:
            features = np.array(data.features)
        labels = np.array(data.labels) if not isinstance(data.labels, np.ndarray) else data.labels
        needs_oversampling = any((self._sampling_targets[cls] > self._class_counts[cls] for cls in self._minority_classes))
        if not needs_oversampling or len(self._minority_classes) == 0:
            return data
        synthetic_features = []
        synthetic_labels = []
        for minority_class in self._minority_classes:
            minority_indices = np.where(labels == minority_class)[0]
            minority_samples = features[minority_indices]
            if len(minority_samples) == 0:
                continue
            samples_needed = self._sampling_targets[minority_class] - self._class_counts[minority_class]
            if samples_needed <= 0:
                continue
            effective_k = min(self.k_neighbors, len(features) - 1)
            if effective_k <= 0:
                continue
            nn = NearestNeighbors(n_neighbors=effective_k + 1, algorithm='auto')
            nn.fit(features)
            (distances, indices) = nn.kneighbors(minority_samples)
            neighbor_indices = indices[:, 1:]
            density_ratios = np.zeros(len(minority_samples))
            for i in range(len(minority_samples)):
                neighbor_labels = labels[neighbor_indices[i]]
                diff_class_count = np.sum(neighbor_labels != minority_class)
                density_ratios[i] = diff_class_count / effective_k
            total_ratio = np.sum(density_ratios)
            if total_ratio == 0:
                generation_distribution = np.ones(len(minority_samples)) / len(minority_samples)
            else:
                generation_distribution = density_ratios / total_ratio
            for i in range(samples_needed):
                idx = rng.choice(len(minority_samples), p=generation_distribution)
                selected_sample = minority_samples[idx]
                neighbor_idx = rng.choice(neighbor_indices[idx])
                neighbor_sample = features[neighbor_idx]
                gap = rng.random()
                synthetic_sample = selected_sample + gap * (neighbor_sample - selected_sample)
                synthetic_features.append(synthetic_sample)
                synthetic_labels.append(minority_class)
        if synthetic_features:
            new_features = np.vstack([features, np.array(synthetic_features)])
            new_labels = np.hstack([labels, np.array(synthetic_labels)])
        else:
            new_features = features
            new_labels = labels
        new_data = FeatureSet(features=new_features, labels=new_labels.tolist() if isinstance(data.labels, list) else new_labels, feature_names=data.feature_names, metadata=data.metadata)
        return new_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for ADASYN oversampling.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            This method always raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for oversampling.
        """
        raise NotImplementedError('ADASYN oversampling does not support inverse transformation')