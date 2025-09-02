from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from collections import Counter
from typing import Optional, Union, Dict
from sklearn.neighbors import NearestNeighbors

class BasicSMOTEOversampler(BaseTransformer):
    """
    A transformer that applies basic SMOTE (Synthetic Minority Oversampling Technique) to handle imbalanced datasets.
    
    This class generates synthetic samples for minority classes by interpolating between existing samples
    and their nearest neighbors. It is particularly useful for classification problems with imbalanced class distributions.
    
    Parameters
    ----------
    sampling_strategy : Union[str, float], default='auto'
        Strategy to sample the dataset. If 'auto', balances all classes.
        If float, specifies the desired ratio of minority to majority samples.
    k_neighbors : int, default=5
        Number of nearest neighbors to use for generating synthetic samples.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    name : Optional[str], default=None
        Name of the transformer instance.
        
    Attributes
    ----------
    sampling_strategy_ : dict
        Actual sampling strategy used after fitting.
    n_features_in_ : int
        Number of features seen during fit.
        
    Methods
    -------
    fit(data: FeatureSet, **kwargs) -> 'BasicSMOTEOversampler'
        Fit the SMOTE sampler to the input data.
    transform(data: FeatureSet, **kwargs) -> FeatureSet
        Apply SMOTE oversampling to the input data.
    inverse_transform(data: FeatureSet, **kwargs) -> FeatureSet
        Not implemented for oversampling techniques.
    """

    def __init__(self, sampling_strategy: Union[str, float]='auto', k_neighbors: int=5, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        if not (isinstance(sampling_strategy, (str, float)) and (sampling_strategy == 'auto' or (isinstance(sampling_strategy, float) and 0 < sampling_strategy <= 1))):
            raise ValueError("sampling_strategy must be 'auto' or a float between 0 and 1")
        if not isinstance(k_neighbors, int) or k_neighbors <= 0:
            raise ValueError('k_neighbors must be a positive integer')

    def fit(self, data: FeatureSet, **kwargs) -> 'BasicSMOTEOversampler':
        """
        Fit the SMOTE sampler to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing features and labels.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        BasicSMOTEOversampler
            Fitted transformer instance.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        if data.y is None:
            raise ValueError('FeatureSet must contain labels for SMOTE oversampling.')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.n_features_in_ = data.X.shape[1]
        y_values = data.y.values.flatten()
        class_counts = Counter(y_values)
        unique_classes = list(class_counts.keys())
        n_samples_per_class = list(class_counts.values())
        max_samples = max(n_samples_per_class)
        self.sampling_strategy_: Dict = {}
        if self.sampling_strategy == 'auto':
            for cls in unique_classes:
                current_count = class_counts[cls]
                if current_count < max_samples:
                    self.sampling_strategy_[cls] = max_samples - current_count
        else:
            for cls in unique_classes:
                current_count = class_counts[cls]
                desired_count = int(np.ceil(current_count / self.sampling_strategy))
                if desired_count > current_count:
                    self.sampling_strategy_[cls] = desired_count - current_count
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply SMOTE oversampling to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed feature set with synthetic samples added.
        """
        if not hasattr(self, 'sampling_strategy_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        if data.y is None:
            raise ValueError('FeatureSet must contain labels for SMOTE oversampling.')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_original = data.X.copy()
        y_original = data.y.copy().values.flatten()
        synthetic_X = []
        synthetic_y = []
        for (cls, n_samples_to_generate) in self.sampling_strategy_.items():
            class_indices = np.where(y_original == cls)[0]
            if len(class_indices) == 0:
                continue
            class_samples = X_original.iloc[class_indices].values
            effective_k = min(self.k_neighbors, len(class_samples) - 1)
            if effective_k <= 0:
                continue
            nn = NearestNeighbors(n_neighbors=effective_k + 1)
            nn.fit(class_samples)
            (distances, indices) = nn.kneighbors(class_samples)
            for _ in range(n_samples_to_generate):
                idx = np.random.randint(0, len(class_samples))
                selected_sample = class_samples[idx]
                neighbor_indices = indices[idx][1:]
                neighbor_idx = neighbor_indices[np.random.randint(0, len(neighbor_indices))]
                neighbor_sample = class_samples[neighbor_idx]
                gap = np.random.rand()
                synthetic_sample = selected_sample + gap * (neighbor_sample - selected_sample)
                synthetic_X.append(synthetic_sample)
                synthetic_y.append(cls)
        if synthetic_X:
            synthetic_X = np.array(synthetic_X)
            synthetic_y = np.array(synthetic_y).reshape(-1, 1)
            new_X = np.vstack([X_original.values, synthetic_X])
            new_y = np.vstack([y_original.reshape(-1, 1), synthetic_y])
        else:
            new_X = X_original.values
            new_y = y_original.reshape(-1, 1)
        transformed_data = FeatureSet(X=new_X, y=new_y, feature_names=data.feature_names, target_names=data.target_names)
        return transformed_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Not implemented for oversampling techniques.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not applicable.
        """
        raise NotImplementedError('Inverse transform is not implemented for oversampling techniques.')

class RegularSMOTEOversampler(BaseTransformer):
    """
    A transformer that applies regular SMOTE (Synthetic Minority Oversampling Technique) to handle imbalanced datasets.
    
    This implementation follows the standard SMOTE algorithm which generates synthetic samples for minority classes 
    by interpolating between existing samples and their k-nearest neighbors. It extends basic SMOTE with additional
    parameters for more fine-grained control.
    
    Parameters
    ----------
    sampling_strategy : Union[str, float], default='auto'
        Strategy to sample the dataset. If 'auto', balances all classes.
        If float, specifies the desired ratio of minority to majority samples.
    k_neighbors : int, default=5
        Number of nearest neighbors to use for generating synthetic samples.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    name : Optional[str], default=None
        Name of the transformer instance.
        
    Attributes
    ----------
    sampling_strategy_ : dict
        Actual sampling strategy used after fitting.
    n_features_in_ : int
        Number of features seen during fit.
    is_fitted : bool
        Whether the transformer has been fitted.
        
    Methods
    -------
    fit(data: FeatureSet, **kwargs) -> 'RegularSMOTEOversampler'
        Fit the SMOTE sampler to the input data.
    transform(data: FeatureSet, **kwargs) -> FeatureSet
        Apply SMOTE oversampling to the input data.
    inverse_transform(data: FeatureSet, **kwargs) -> FeatureSet
        Not implemented for oversampling techniques.
    """

    def __init__(self, sampling_strategy: Union[str, float]='auto', k_neighbors: int=5, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy_ = {}
        self.n_features_in_ = 0
        self.is_fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'RegularSMOTEOversampler':
        """
        Fit the SMOTE sampler to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing features and labels.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        RegularSMOTEOversampler
            Fitted transformer instance.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        if data.labels is None:
            raise ValueError('FeatureSet must contain labels for SMOTE oversampling.')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        y = data.labels.values
        class_counts = Counter(y)
        unique_classes = list(class_counts.keys())
        n_majority = max(class_counts.values())
        self.sampling_strategy_ = {}
        if self.sampling_strategy == 'auto':
            for cls in unique_classes:
                self.sampling_strategy_[cls] = n_majority
        elif isinstance(self.sampling_strategy, float):
            for cls in unique_classes:
                target_count = int(class_counts[cls] + (n_majority - class_counts[cls]) * self.sampling_strategy)
                self.sampling_strategy_[cls] = max(target_count, class_counts[cls])
        else:
            raise ValueError("sampling_strategy must be 'auto' or a float.")
        self.n_features_in_ = data.features.shape[1]
        self.is_fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply SMOTE oversampling to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed feature set with synthetic samples added.
        """
        if not self.is_fitted:
            raise RuntimeError('Transformer must be fitted before calling transform.')
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        if data.labels is None:
            raise ValueError('FeatureSet must contain labels for SMOTE oversampling.')
        if data.features.shape[1] != self.n_features_in_:
            raise ValueError(f'Feature dimension mismatch. Expected {self.n_features_in_} features, got {data.features.shape[1]}.')
        X = data.features.values
        y = data.labels.values
        class_counts = Counter(y)
        synthetic_X = []
        synthetic_y = []
        for (cls, target_count) in self.sampling_strategy_.items():
            current_count = class_counts[cls]
            n_samples_needed = target_count - current_count
            if n_samples_needed <= 0:
                continue
            class_indices = np.where(y == cls)[0]
            class_samples = X[class_indices]
            k = min(self.k_neighbors, len(class_samples) - 1)
            if k < 1:
                continue
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
            nn.fit(class_samples)
            (_, indices) = nn.kneighbors(class_samples)
            for _ in range(n_samples_needed):
                idx = np.random.randint(0, len(class_samples))
                sample = class_samples[idx]
                neighbor_indices = indices[idx][1:]
                neighbor_idx = neighbor_indices[np.random.randint(0, len(neighbor_indices))]
                neighbor = class_samples[neighbor_idx]
                gap = np.random.rand()
                synthetic_sample = sample + gap * (neighbor - sample)
                synthetic_X.append(synthetic_sample)
                synthetic_y.append(cls)
        if synthetic_X:
            X_resampled = np.vstack([X, np.array(synthetic_X)])
            y_resampled = np.hstack([y, np.array(synthetic_y)])
        else:
            X_resampled = X
            y_resampled = y
        resampled_features = data.features.copy()
        resampled_features = resampled_features.iloc[:len(X_resampled)]
        resampled_features[:] = X_resampled
        resampled_labels = data.labels.copy() if data.labels is not None else None
        if resampled_labels is not None:
            resampled_labels = resampled_labels.iloc[:len(y_resampled)]
            resampled_labels[:] = y_resampled
        result = FeatureSet(features=resampled_features, labels=resampled_labels, metadata=data.metadata.copy() if data.metadata else None)
        return result

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Not implemented for oversampling techniques.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not applicable.
        """
        raise NotImplementedError('Inverse transform is not implemented for SMOTE oversampling.')

class SMOTEOversampler(BaseTransformer):

    def __init__(self, sampling_strategy: Union[str, float]='auto', k_neighbors: int=5, random_state: Optional[int]=None, smote_variant: str='regular', name: Optional[str]=None):
        super().__init__(name=name)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.smote_variant = smote_variant
        if not isinstance(k_neighbors, int) or k_neighbors <= 0:
            raise ValueError('k_neighbors must be a positive integer')
        if smote_variant not in ['basic', 'regular', 'advanced']:
            raise ValueError("smote_variant must be one of 'basic', 'regular', or 'advanced'")
        if not (isinstance(sampling_strategy, (str, float)) and (sampling_strategy == 'auto' or (isinstance(sampling_strategy, float) and 0 < sampling_strategy <= 1))):
            raise ValueError("sampling_strategy must be 'auto' or a float between 0 and 1")
        if smote_variant == 'basic':
            self._smote_instance = BasicSMOTEOversampler(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
        elif smote_variant == 'regular':
            self._smote_instance = RegularSMOTEOversampler(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
        else:
            self._smote_instance = AdvancedSMOTEOversampler(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
        self.sampling_strategy_ = {}
        self.n_features_in_ = 0
        if hasattr(self._smote_instance, 'is_fitted'):
            self.is_fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'SMOTEOversampler':
        """
        Fit the SMOTE sampler to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing features and labels.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        SMOTEOversampler
            Fitted transformer instance.
        """
        self._smote_instance.fit(data, **kwargs)
        if hasattr(self._smote_instance, 'sampling_strategy_'):
            self.sampling_strategy_ = self._smote_instance.sampling_strategy_
        if hasattr(self._smote_instance, 'n_features_in_'):
            self.n_features_in_ = self._smote_instance.n_features_in_
        if hasattr(self._smote_instance, 'is_fitted'):
            self.is_fitted = self._smote_instance.is_fitted
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply SMOTE oversampling to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed feature set with synthetic samples added.
        """
        return self._smote_instance.transform(data, **kwargs)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Not implemented for oversampling techniques.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not applicable.
        """
        raise NotImplementedError('Inverse transform is not implemented for SMOTE oversampling.')