from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import Optional, Union
import numpy as np
from collections import Counter

class SimpleOversampler(BaseTransformer):
    """
    A transformer that performs simple oversampling on imbalanced datasets.
    
    This class implements basic oversampling techniques to balance class distributions
    by increasing the number of instances in minority classes. It is primarily intended
    for classification tasks where class imbalance affects model performance.
    
    The transformer can be configured with different oversampling strategies and
    maintains compatibility with the standard fit-transform interface.
    
    Attributes
    ----------
    sampling_strategy : Union[str, float], optional
        Strategy to use for oversampling. Can be 'auto', 'minority', or a float
        representing the desired ratio of samples (default is 'auto').
    random_state : Optional[int], optional
        Random seed for reproducibility (default is None).
    name : Optional[str], optional
        Name of the transformer instance (default is None).
        
    Methods
    -------
    fit(data: FeatureSet, **kwargs) -> 'SimpleOversampler'
        Fit the oversampler to the input data.
    transform(data: FeatureSet, **kwargs) -> FeatureSet
        Apply oversampling to the input data.
    inverse_transform(data: FeatureSet, **kwargs) -> FeatureSet
        Reverse the oversampling transformation (not implemented for oversampling).
    """

    def __init__(self, sampling_strategy: Union[str, float]='auto', random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self._sampling_targets = None

    def fit(self, data: FeatureSet, **kwargs) -> 'SimpleOversampler':
        """
        Fit the oversampler to the input data.
        
        This method analyzes the class distribution in the input data and prepares
        the oversampling strategy accordingly.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing features and labels.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        SimpleOversampler
            Self instance for method chaining.
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
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'minority':
            max_count = max(counts)
            self._sampling_targets = {}
            for (cls, count) in class_counts.items():
                if count < max_count:
                    self._sampling_targets[cls] = max_count
        elif isinstance(self.sampling_strategy, float):
            if self.sampling_strategy <= 0:
                raise ValueError('sampling_strategy as float must be positive')
            max_count = max(counts)
            target_count = int(max_count / self.sampling_strategy)
            self._sampling_targets = {}
            for (cls, count) in class_counts.items():
                if count < target_count:
                    self._sampling_targets[cls] = target_count
        else:
            raise ValueError("sampling_strategy must be 'auto', 'minority', or a positive float")
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply oversampling to the input data.
        
        This method increases the number of samples in minority classes according
        to the configured sampling strategy.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to be oversampled.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Oversampled feature set with balanced class distribution.
        """
        if self._sampling_targets is None:
            raise RuntimeError("Oversampler has not been fitted yet. Call 'fit' first.")
        if data.metadata is None or 'target' not in data.metadata:
            raise ValueError("FeatureSet must contain target labels in metadata['target']")
        X = data.features
        y = data.metadata['target']
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        original_indices = np.arange(len(y))
        additional_indices = []
        for (cls, target_count) in self._sampling_targets.items():
            class_indices = np.where(y == cls)[0]
            current_count = len(class_indices)
            samples_to_add = target_count - current_count
            if samples_to_add > 0:
                new_indices = np.random.choice(class_indices, size=samples_to_add, replace=True)
                additional_indices.extend(new_indices)
        if len(additional_indices) == 0:
            return data
        all_indices = np.concatenate([original_indices, additional_indices])
        X_resampled = X[all_indices]
        y_resampled = y[all_indices]
        new_metadata = data.metadata.copy() if data.metadata else {}
        new_metadata['target'] = y_resampled
        new_metadata['resampling_method'] = 'simple_oversampling'
        feature_names = data.feature_names
        feature_types = data.feature_types
        quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        sample_ids = None
        if data.sample_ids is not None:
            original_sample_ids = list(data.sample_ids) if not isinstance(data.sample_ids, list) else data.sample_ids
            additional_sample_ids = [f'{original_sample_ids[i]}_oversampled_{j}' for (j, i) in enumerate(additional_indices)]
            sample_ids = original_sample_ids + additional_sample_ids
        return FeatureSet(features=X_resampled, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=new_metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the oversampling transformation.
        
        Since oversampling adds synthetic or duplicated samples, reversing
        the transformation would require identifying and removing those samples.
        This method raises NotImplementedError as it's not generally meaningful
        for oversampling operations.
        
        Parameters
        ----------
        data : FeatureSet
            Oversampled feature set.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set before oversampling.
            
        Raises
        ------
        NotImplementedError
            If the method is called, as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for oversampling operations.')

class DataOversampler(BaseTransformer):

    def __init__(self, ratio: float=0.5, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the DataOversampler.
        
        Parameters
        ----------
        ratio : float, optional
            Ratio to increase the dataset size (e.g., 0.5 means increase by 50%)
            (default is 0.5).
        random_state : Optional[int], optional
            Random seed for reproducibility (default is None).
        name : Optional[str], optional
            Name of the transformer instance (default is None).
        """
        super().__init__(name=name)
        if ratio < 0:
            raise ValueError('ratio must be non-negative')
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, data: FeatureSet, **kwargs) -> 'DataOversampler':
        """
        Fit the oversampler to the input data.
        
        This method prepares the oversampling process by analyzing the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to analyze.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        DataOversampler
            Self instance for method chaining.
        """
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
        Apply data oversampling to the input data.
        
        This method increases the dataset size by randomly duplicating existing
        samples according to the specified ratio.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to be augmented.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Augmented feature set with increased sample count.
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError('Transformer must be fitted before transform')
        n_samples = data.get_n_samples()
        n_additional = int(n_samples * self.ratio)
        if n_additional == 0:
            return data
        additional_indices = self.rng_.choice(n_samples, size=n_additional, replace=True)
        additional_features = data.features[additional_indices]
        combined_features = np.vstack([data.features, additional_features])
        combined_sample_ids = None
        if data.sample_ids is not None:
            additional_sample_ids = [f'{data.sample_ids[i]}_dup_{j}' for (j, i) in enumerate(additional_indices)]
            combined_sample_ids = data.sample_ids + additional_sample_ids
        new_metadata = data.metadata.copy() if data.metadata is not None else None
        new_quality_scores = data.quality_scores.copy() if data.quality_scores is not None else None
        return FeatureSet(features=combined_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=combined_sample_ids, metadata=new_metadata, quality_scores=new_quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the oversampling transformation.
        
        This method removes duplicated samples to restore the original dataset.
        It identifies and removes samples added during the oversampling process.
        
        Parameters
        ----------
        data : FeatureSet
            Oversampled feature set.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set before oversampling.
            
        Raises
        ------
        NotImplementedError
            If attempting to inverse transform without proper tracking of sample origins.
        """
        raise NotImplementedError('Inverse transformation is not supported for DataOversampler as it would require tracking of duplicated samples.')