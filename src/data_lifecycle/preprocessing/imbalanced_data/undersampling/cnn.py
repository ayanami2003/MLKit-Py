from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, Dict, List
import numpy as np
from collections import Counter
import warnings

class CondensedNearestNeighborSampler(BaseTransformer):
    """
    A transformer that implements the Condensed Nearest Neighbor (CNN) undersampling technique.
    
    This method selects a subset of samples from the majority class that are sufficient
    to correctly classify the minority class samples using a 1-NN rule. It aims to
    reduce the number of majority class samples while preserving the decision boundary.
    
    The algorithm works by:
    1. Starting with all minority class samples and a random subset of majority samples
    2. Iteratively adding misclassified majority samples to the stored set
    3. Stopping when no more samples need to be added
    
    This implementation supports both binary and multiclass imbalanced datasets.
    
    Parameters
    ----------
    sampling_strategy : Union[str, float, dict], default='auto'
        Strategy to sample the dataset.
        - 'auto': equivalent to 'not minority'
        - 'majority': resample only the majority class
        - 'not minority': resample all classes but the minority class
        - 'not majority': resample all classes but the majority class
        - dict: key-value pairs specifying the number of samples to keep for each class
        - float: desired ratio of minority to majority samples after resampling
    random_state : Optional[int], default=None
        Random seed for reproducibility
    n_neighbors : int, default=1
        Number of neighbors to use for KNN classification during CNN process
    allow_minority : bool, default=False
        Whether to allow the minority class to be undersampled
    n_seeds_S : int, default=1
        Number of samples to initialize the stored set S with
    n_jobs : Optional[int], default=None
        Number of CPU cores to use for computation (-1 for all cores)
    name : Optional[str], default=None
        Name of the transformer instance
        
    Attributes
    ----------
    sampling_strategy_ : dict
        Actual sampling strategy used during fitting
    sample_indices_ : list
        Indices of samples selected for the condensed set
        
    Examples
    --------
    >>> from general.structures.feature_set import FeatureSet
    >>> import numpy as np
    >>> 
    >>> # Create example imbalanced dataset
    >>> X = np.random.rand(1000, 4)
    >>> y = np.hstack([np.zeros(900), np.ones(100)])  # 90% majority class
    >>> feature_set = FeatureSet(features=X, labels=y.tolist())
    >>> 
    >>> # Apply CNN undersampling
    >>> cnn_sampler = CondensedNearestNeighborSampler(
    ...     sampling_strategy='auto',
    ...     random_state=42
    ... )
    >>> balanced_features = cnn_sampler.fit_transform(feature_set)
    >>> print(f"Original shape: {feature_set.features.shape}")
    >>> print(f"Balanced shape: {balanced_features.features.shape}")
    """

    def __init__(self, sampling_strategy: Union[str, float, dict]='auto', random_state: Optional[int]=None, n_neighbors: int=1, allow_minority: bool=False, n_seeds_S: int=1, n_jobs: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.allow_minority = allow_minority
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, data: FeatureSet, **kwargs) -> 'CondensedNearestNeighborSampler':
        """
        Fit the CNN sampler to the input data.
        
        This method identifies which samples to retain based on the
        condensed nearest neighbor algorithm.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with features and labels
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        CondensedNearestNeighborSampler
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data does not contain labels
        """
        if data.labels is None:
            raise ValueError('Input data must contain labels for CNN undersampling')
        X = np.array(data.features)
        y = np.array(data.labels)
        if len(X) != len(y):
            raise ValueError('Features and labels must have the same length')
        (unique_classes, class_counts) = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        self.sampling_strategy_ = self._validate_sampling_strategy(self.sampling_strategy, class_distribution, unique_classes)
        minority_class = unique_classes[np.argmin(class_counts)]
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_indices = np.where(y == minority_class)[0]
        selected_indices = minority_indices.tolist()
        for (target_class, target_count) in self.sampling_strategy_.items():
            if target_class == minority_class and (not self.allow_minority):
                continue
            class_indices = np.where(y == target_class)[0]
            if len(class_indices) <= target_count:
                selected_indices.extend(class_indices.tolist())
                continue
            class_selected = self._apply_cnn_to_class(X, y, class_indices, minority_indices, target_count)
            selected_indices.extend(class_selected)
        self.sample_indices_ = sorted(selected_indices)
        self._fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the CNN undersampling transformation to the input data.
        
        This method returns a new FeatureSet containing only the samples
        selected by the CNN algorithm.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed feature set with undersampled samples
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        selected_features = [data.features[i] for i in self.sample_indices_]
        selected_labels = [data.labels[i] for i in self.sample_indices_] if data.labels else None
        selected_sample_ids = None
        if data.sample_ids:
            selected_sample_ids = [data.sample_ids[i] for i in self.sample_indices_]
        transformed_data = FeatureSet(features=selected_features, labels=selected_labels, feature_names=data.feature_names, sample_ids=selected_sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        return transformed_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the CNN transformation (not supported).
        
        CNN undersampling is a lossy transformation that discards samples,
        so exact inversion is not possible.
        
        Parameters
        ----------
        data : FeatureSet
            Feature set to "invert" (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            The same data passed in (identity operation)
            
        Warning
        -------
        This method simply returns the input data unchanged, as CNN
        undersampling cannot be inverted due to loss of information.
        """
        warnings.warn('CNN undersampling is a lossy transformation. inverse_transform cannot recover the original samples that were removed.', UserWarning)
        return data

    def _validate_sampling_strategy(self, sampling_strategy, class_distribution, unique_classes):
        """Validate and convert sampling strategy to a dictionary format."""
        if isinstance(sampling_strategy, str):
            if sampling_strategy == 'auto':
                sampling_strategy = 'not minority'
            result = {}
            if sampling_strategy == 'majority':
                majority_class = max(class_distribution, key=class_distribution.get)
                result[majority_class] = class_distribution[majority_class]
            elif sampling_strategy == 'not minority':
                minority_class = min(class_distribution, key=class_distribution.get)
                for cls in unique_classes:
                    if cls != minority_class:
                        result[cls] = class_distribution[cls]
            elif sampling_strategy == 'not majority':
                majority_class = max(class_distribution, key=class_distribution.get)
                for cls in unique_classes:
                    if cls != majority_class:
                        result[cls] = class_distribution[cls]
            else:
                raise ValueError(f'Unknown sampling strategy: {sampling_strategy}')
            return result
        elif isinstance(sampling_strategy, dict):
            return sampling_strategy
        elif isinstance(sampling_strategy, (int, float)):
            minority_class = min(class_distribution, key=class_distribution.get)
            result = {}
            for cls in unique_classes:
                if cls != minority_class:
                    target_count = int(sampling_strategy * class_distribution[cls])
                    result[cls] = max(target_count, 1)
            return result
        else:
            raise ValueError(f'Unsupported sampling_strategy type: {type(sampling_strategy)}')

    def _apply_cnn_to_class(self, X, y, class_indices, minority_indices, target_count):
        """Apply the CNN algorithm to a specific class."""
        if len(class_indices) <= self.n_seeds_S:
            selected_from_class = class_indices.tolist()
        else:
            selected_from_class = np.random.choice(class_indices, size=self.n_seeds_S, replace=False).tolist()
        stored_indices = selected_from_class + minority_indices.tolist()
        changed = True
        while changed and len(selected_from_class) < target_count:
            changed = False
            for idx in class_indices:
                if idx in selected_from_class:
                    continue
                nearest_neighbor_idx = self._find_nearest_neighbor(X[idx], X, stored_indices, y, y[idx])
                if y[nearest_neighbor_idx] != y[idx]:
                    selected_from_class.append(idx)
                    stored_indices.append(idx)
                    changed = True
                    if len(selected_from_class) >= target_count:
                        break
        return selected_from_class

    def _find_nearest_neighbor(self, sample, X, candidate_indices, y, sample_label):
        """Find the nearest neighbor of a sample among candidates using k-NN."""
        if len(candidate_indices) == 1:
            return candidate_indices[0]
        distances = []
        for idx in candidate_indices:
            dist = np.linalg.norm(sample - X[idx])
            distances.append((dist, idx))
        distances.sort(key=lambda x: x[0])
        if self.n_neighbors == 1:
            return distances[0][1]
        k = min(self.n_neighbors, len(distances))
        neighbor_labels = [y[distances[i][1]] for i in range(k)]
        label_counts = Counter(neighbor_labels)
        most_common_label = label_counts.most_common(1)[0][0]
        for (dist, idx) in distances:
            if y[idx] == most_common_label:
                return idx
        return distances[0][1]