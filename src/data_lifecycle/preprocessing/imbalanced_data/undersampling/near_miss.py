import numpy as np
from typing import Union, Dict, List
from collections import Counter
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.mathematical_foundations.specialized_functions.distance_metrics import euclidean_distance


class NearMissUndersampler(BaseTransformer):
    """
    NearMiss undersampling transformer for balancing imbalanced datasets.
    
    This transformer implements the NearMiss undersampling technique, which selects examples
    from the majority class based on their distance to the minority class examples. Three
    variants are supported:
    - NearMiss-1: Selects samples with the smallest average distance to the N closest
                  minority class samples
    - NearMiss-2: Selects samples with the smallest average distance to the N farthest
                  minority class samples
    - NearMiss-3: Selects samples that maximize the minimum distance to minority samples
    
    This approach helps preserve important samples from the majority class while removing
    potentially noisy or borderline samples.
    
    Parameters
    ----------
    sampling_strategy : Union[str, float, dict], default='auto'
        Strategy to resample the dataset:
        - 'majority': resample only the majority class
        - 'not minority': resample all classes except minority
        - 'not majority': resample all classes except majority
        - 'all': resample all classes
        - float: desired ratio of minority to majority samples
        - dict: specify ratio for each class
    version : int, default=1
        Variant of NearMiss to use (1, 2, or 3)
    n_neighbors : int, default=3
        Number of neighbors to consider when calculating distances
    random_state : Optional[int], default=None
        Random seed for reproducible results
    name : Optional[str], default=None
        Name of the transformer instance
    
    Attributes
    ----------
    sampling_strategy_ : dict
        Actual sampling strategy used
    target_class_count_ : dict
        Count of samples per class after resampling
    
    Examples
    --------
    >>> from general.structures.feature_set import FeatureSet
    >>> import numpy as np
    >>> 
    >>> # Create imbalanced dataset
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])  # Imbalanced classes
    >>> feature_set = FeatureSet(features=X, labels=y.tolist())
    >>> 
    >>> # Apply NearMiss undersampling
    >>> undersampler = NearMissUndersampler(version=1, n_neighbors=2)
    >>> balanced_set = undersampler.fit_transform(feature_set)
    """

    def __init__(self, sampling_strategy: Union[str, float, dict]='majority', version: int=1, n_neighbors: int=3, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.version = version
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        if self.version not in [1, 2, 3]:
            raise ValueError('version must be 1, 2, or 3')
        if self.n_neighbors <= 0:
            raise ValueError('n_neighbors must be positive')
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, data: FeatureSet, **kwargs) -> 'NearMissUndersampler':
        """
        Fit the undersampler to the input data.
        
        Computes the sampling strategy and identifies which samples to keep
        based on the NearMiss algorithm variant.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set with features and labels
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        NearMissUndersampler
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the data does not contain labels or if version is not 1, 2, or 3
        """
        if data.metadata is None or 'labels' not in data.metadata or len(data.metadata['labels']) == 0:
            raise ValueError('Data must contain labels for undersampling')
        y = np.array(data.metadata['labels'])
        X = np.array(data.features)
        if len(X) != len(y):
            raise ValueError('Features and labels must have the same length')
        (unique_classes, class_counts) = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        minority_class = unique_classes[np.argmin(class_counts)]
        majority_class = unique_classes[np.argmax(class_counts)]
        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy == 'majority':
                self.sampling_strategy_ = {majority_class: class_distribution[minority_class]}
            elif self.sampling_strategy == 'not minority':
                self.sampling_strategy_ = {cls: class_distribution[minority_class] for cls in unique_classes if cls != minority_class}
            elif self.sampling_strategy == 'not majority':
                self.sampling_strategy_ = {cls: max(class_distribution[cls], class_distribution[majority_class]) for cls in unique_classes if cls != majority_class}
            elif self.sampling_strategy == 'all':
                min_count = class_distribution[minority_class]
                self.sampling_strategy_ = {cls: min_count for cls in unique_classes}
            else:
                raise ValueError(f'Unknown sampling_strategy: {self.sampling_strategy}')
        elif isinstance(self.sampling_strategy, float):
            if not 0 < self.sampling_strategy <= 1:
                raise ValueError('sampling_strategy as float must be in (0, 1]')
            target_majority_count = int(class_distribution[minority_class] / self.sampling_strategy)
            self.sampling_strategy_ = {majority_class: target_majority_count}
        elif isinstance(self.sampling_strategy, dict):
            self.sampling_strategy_ = self.sampling_strategy
        else:
            raise ValueError('sampling_strategy must be str, float, or dict')
        for (cls, target_count) in self.sampling_strategy_.items():
            if cls in class_distribution and target_count > class_distribution[cls]:
                self.sampling_strategy_[cls] = class_distribution[cls]
        self.target_class_count_ = {}
        for cls in unique_classes:
            if cls in self.sampling_strategy_:
                self.target_class_count_[cls] = self.sampling_strategy_[cls]
            else:
                self.target_class_count_[cls] = class_distribution[cls]
        self._identify_samples_to_keep(X, y)
        self._is_fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply NearMiss undersampling to the input data.
        
        Selects samples according to the fitted sampling strategy using
        the specified NearMiss variant.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed feature set with balanced classes
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if data.metadata is None or 'labels' not in data.metadata:
            raise ValueError('Data must contain labels for undersampling')
        X = np.array(data.features)
        y = np.array(data.metadata['labels'])
        filtered_features = X[self.indices_to_keep_]
        filtered_labels = y[self.indices_to_keep_].tolist()
        new_metadata = data.metadata.copy() if data.metadata else {}
        new_metadata['labels'] = filtered_labels
        balanced_data = FeatureSet(features=filtered_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=[data.sample_ids[i] for i in self.indices_to_keep_] if data.sample_ids else None, metadata=new_metadata, quality_scores=data.quality_scores)
        return balanced_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for undersampling methods.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            The same data passed as input
            
        Warning
        -------
        This method simply returns the input data unchanged as undersampling
        is a destructive operation that cannot be inverted.
        """
        return data

    def _identify_samples_to_keep(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Identify which samples to keep based on the NearMiss variant.
        
        Parameters
        ----------
        X : np.ndarray
            Feature array
        y : np.ndarray
            Label array
        """
        unique_classes = np.unique(y)
        minority_class = unique_classes[np.argmin([np.sum(y == cls) for cls in unique_classes])]
        indices_to_keep = list(np.where(y == minority_class)[0])
        if not hasattr(self, 'sampling_strategy_'):
            self.sampling_strategy_ = {}
        for (target_class, target_count) in self.sampling_strategy_.items():
            if target_class == minority_class:
                continue
            class_indices = np.where(y == target_class)[0]
            current_count = len(class_indices)
            if target_count >= current_count:
                indices_to_keep.extend(class_indices)
                continue
            if self.version == 1:
                selected_indices = self._nearmiss_1(X, y, class_indices, minority_class, target_count)
            elif self.version == 2:
                selected_indices = self._nearmiss_2(X, y, class_indices, minority_class, target_count)
            else:
                selected_indices = self._nearmiss_3(X, y, class_indices, minority_class, target_count)
            indices_to_keep.extend(selected_indices)
        self.indices_to_keep_ = np.array(indices_to_keep)

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all samples.
        
        Parameters
        ----------
        X : np.ndarray
            Array of features.
            
        Returns
        -------
        np.ndarray
            Pairwise distance matrix.
        """
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        return distances

    def _nearmiss_1(self, X: np.ndarray, y: np.ndarray, class_indices: np.ndarray, minority_class: int, target_count: int) -> List[int]:
        """
        NearMiss-1: Select samples with smallest average distance to N closest minority samples.
        """
        minority_indices = np.where(y == minority_class)[0]
        distances = self._compute_distances(X)
        avg_distances = []
        for idx in class_indices:
            minority_distances = [distances[idx, minor_idx] for minor_idx in minority_indices]
            minority_distances.sort()
            closest_distances = minority_distances[:self.n_neighbors]
            avg_dist = np.mean(closest_distances) if closest_distances else np.inf
            avg_distances.append((idx, avg_dist))
        avg_distances.sort(key=lambda x: x[1])
        selected_indices = [idx for (idx, _) in avg_distances[:target_count]]
        return selected_indices

    def _nearmiss_2(self, X: np.ndarray, y: np.ndarray, class_indices: np.ndarray, minority_class: int, target_count: int) -> List[int]:
        """
        NearMiss-2: Select samples with smallest average distance to N farthest minority samples.
        """
        minority_indices = np.where(y == minority_class)[0]
        distances = self._compute_distances(X)
        avg_distances = []
        for idx in class_indices:
            minority_distances = [distances[idx, minor_idx] for minor_idx in minority_indices]
            minority_distances.sort(reverse=True)
            farthest_distances = minority_distances[:self.n_neighbors]
            avg_dist = np.mean(farthest_distances) if farthest_distances else np.inf
            avg_distances.append((idx, avg_dist))
        avg_distances.sort(key=lambda x: x[1])
        selected_indices = [idx for (idx, _) in avg_distances[:target_count]]
        return selected_indices

    def _nearmiss_3(self, X: np.ndarray, y: np.ndarray, class_indices: np.ndarray, minority_class: int, target_count: int) -> List[int]:
        """
        NearMiss-3: Select samples that maximize the minimum distance to minority samples.
        """
        minority_indices = np.where(y == minority_class)[0]
        distances = self._compute_distances(X)
        min_distances = []
        for idx in class_indices:
            minority_distances = [distances[idx, minor_idx] for minor_idx in minority_indices]
            min_dist = min(minority_distances) if minority_distances else np.inf
            min_distances.append((idx, min_dist))
        min_distances.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for (idx, _) in min_distances[:target_count]]
        return selected_indices