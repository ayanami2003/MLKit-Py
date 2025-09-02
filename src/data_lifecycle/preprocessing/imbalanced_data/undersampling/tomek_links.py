from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, Dict, Any
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.neighbors import NearestNeighbors

class TomekLinksUndersampler(BaseTransformer):
    """
    Transformer that performs undersampling using Tomek Links technique.
    
    Tomek Links undersampling removes samples that form Tomek links - pairs of samples from 
    different classes that are each other's nearest neighbors. Removing one sample from each 
    link helps to clean up the separation between classes.
    
    This technique is particularly useful for binary classification problems where class 
    boundaries need to be clarified by removing borderline samples.
    
    Parameters
    ----------
    sampling_strategy : Union[str, float, dict], default='majority'
        Strategy to use for sampling. Can be 'majority', 'not minority', 'not majority',
        'all', float (ratio of minority to majority), or dict (class-specific ratios).
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    n_jobs : Optional[int], default=None
        Number of CPU cores to use for computation. None means 1, -1 means all cores.
    name : Optional[str], default=None
        Name of the transformer instance.
        
    Attributes
    ----------
    sampling_strategy_ : dict
        Actual sampling strategy used during fitting.
    """

    def __init__(self, sampling_strategy: Union[str, float, dict]='majority', random_state: Optional[int]=None, n_jobs: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, data: FeatureSet, **kwargs) -> 'TomekLinksUndersampler':
        """
        Fit the Tomek Links undersampler to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing features and labels for fitting.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        TomekLinksUndersampler
            Self instance for method chaining.
        """
        if data.metadata is None or 'labels' not in data.metadata:
            raise ValueError('FeatureSet must contain labels in metadata')
        labels = np.array(data.metadata['labels'])
        (unique_classes, class_counts) = np.unique(labels, return_counts=True)
        if len(unique_classes) < 2:
            raise ValueError('Tomek Links undersampling requires at least 2 classes')
        self.sampling_strategy_ = self._validate_sampling_strategy(self.sampling_strategy, unique_classes, class_counts)
        self._indices_to_remove = self._find_tomek_links(data.features, labels)
        return self

    def _validate_sampling_strategy(self, sampling_strategy, unique_classes, class_counts):
        """Validate and convert sampling strategy to a dictionary."""
        class_count_dict = dict(zip(unique_classes, class_counts))
        if isinstance(sampling_strategy, str):
            if sampling_strategy == 'majority':
                majority_class = unique_classes[np.argmax(class_counts)]
                return {int(majority_class): int(class_count_dict[majority_class])}
            elif sampling_strategy == 'not minority':
                minority_count = int(np.min(list(class_count_dict.values())))
                return {int(cls): int(count) for (cls, count) in class_count_dict.items() if int(count) > minority_count}
            elif sampling_strategy == 'not majority':
                majority_count = int(np.max(list(class_count_dict.values())))
                return {int(cls): int(count) for (cls, count) in class_count_dict.items() if int(count) < majority_count}
            elif sampling_strategy == 'all':
                return {int(cls): int(count) for (cls, count) in class_count_dict.items()}
            else:
                raise ValueError(f'Unknown sampling_strategy: {sampling_strategy}')
        elif isinstance(sampling_strategy, float):
            if not 0 < sampling_strategy <= 1:
                raise ValueError('Float sampling_strategy must be in (0, 1]')
            minority_count = int(np.min(list(class_count_dict.values())))
            majority_class = unique_classes[np.argmax(class_counts)]
            target_count = min(int(minority_count / sampling_strategy), class_count_dict[majority_class])
            return {int(majority_class): target_count}
        elif isinstance(sampling_strategy, dict):
            return {int(cls): int(count) for (cls, count) in sampling_strategy.items()}
        else:
            raise TypeError('sampling_strategy must be str, float, or dict')

    def _find_tomek_links(self, X, y):
        """Find indices of samples that are part of Tomek links."""
        if not hasattr(self, 'sampling_strategy_'):
            return []
        if len(X) < 2:
            return []
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        if n_jobs == -1:
            n_jobs = None
        n_neighbors = min(max(2, len(X)), len(np.unique(y)) * 2)
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
        nn.fit(X)
        (distances, indices) = nn.kneighbors(X)
        tomek_links = []
        for i in range(len(X)):
            for neighbor_idx in indices[i][1:]:
                if y[i] != y[neighbor_idx]:
                    (_, reverse_indices) = nn.kneighbors([X[neighbor_idx]])
                    if i in reverse_indices[0]:
                        tomek_links.append((i, neighbor_idx))
        unique_links = []
        seen = set()
        for (i, j) in tomek_links:
            pair = tuple(sorted([i, j]))
            if pair not in seen:
                unique_links.append(pair)
                seen.add(pair)
        indices_to_remove = []
        rng = np.random.RandomState(self.random_state)
        for (i, j) in unique_links:
            class_i = int(y[i])
            class_j = int(y[j])
            remove_i = class_i in self.sampling_strategy_
            remove_j = class_j in self.sampling_strategy_
            if remove_i and remove_j:
                to_remove = rng.choice([i, j])
                indices_to_remove.append(to_remove)
            elif remove_i:
                indices_to_remove.append(i)
            elif remove_j:
                indices_to_remove.append(j)
            else:
                to_remove = rng.choice([i, j])
                indices_to_remove.append(to_remove)
        return list(set(indices_to_remove))

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply Tomek Links undersampling to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed feature set with Tomek Links samples removed.
        """
        if not hasattr(self, '_indices_to_remove'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if data.metadata is None or 'labels' not in data.metadata:
            raise ValueError('FeatureSet must contain labels in metadata')
        n_samples = data.features.shape[0]
        mask = np.ones(n_samples, dtype=bool)
        mask[self._indices_to_remove] = False
        new_features = data.features[mask]
        new_feature_names = data.feature_names
        new_feature_types = data.feature_types
        new_sample_ids = None
        if data.sample_ids is not None:
            new_sample_ids = [data.sample_ids[i] for i in range(n_samples) if mask[i]]
        new_labels = [data.metadata['labels'][i] for i in range(n_samples) if mask[i]]
        new_metadata = data.metadata.copy()
        new_metadata['labels'] = new_labels
        new_quality_scores = data.quality_scores
        return FeatureSet(features=new_features, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=new_sample_ids, metadata=new_metadata, quality_scores=new_quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform operation (not applicable for undersampling).
        
        Parameters
        ----------
        data : FeatureSet
            Feature set to inverse transform.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            The same feature set as input (identity operation).
        """
        return data