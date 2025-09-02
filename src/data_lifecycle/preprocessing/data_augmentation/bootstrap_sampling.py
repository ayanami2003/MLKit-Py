from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer

class BootstrapSampler(BaseTransformer):

    def __init__(self, n_samples: Optional[int]=None, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.random_state = random_state
        self.preserve_original = preserve_original

    def fit(self, data: FeatureSet, **kwargs) -> 'BootstrapSampler':
        """
        Fit the transformer to the input data.
        
        For bootstrap sampling, fitting simply validates the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to fit on.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        BootstrapSampler
            The fitted transformer instance.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        self._n_input_samples = data.features.shape[0]
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply bootstrap sampling to the input data.
        
        Draws samples with replacement from the input data to create a new dataset.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        FeatureSet
            Bootstrapped feature set with potentially duplicated samples.
        """
        if not hasattr(self, '_n_input_samples'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        n_input_samples = data.features.shape[0]
        if n_input_samples != self._n_input_samples:
            raise ValueError(f'Input data has {n_input_samples} samples, but transformer was fitted on {self._n_input_samples} samples.')
        n_samples = self.n_samples if self.n_samples is not None else n_input_samples
        rng = np.random.default_rng(self.random_state)
        bootstrap_indices = rng.choice(n_input_samples, size=n_samples, replace=True)
        sampled_features = data.features[bootstrap_indices]
        sampled_feature_names = data.feature_names
        sampled_feature_types = data.feature_types
        sampled_metadata = data.metadata.copy() if data.metadata else {}
        sampled_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        if data.sample_ids is not None:
            sampled_sample_ids = [data.sample_ids[i] for i in bootstrap_indices]
        else:
            sampled_sample_ids = None
        if self.preserve_original:
            sampled_features = np.vstack([data.features, sampled_features])
            if sampled_sample_ids is not None and data.sample_ids is not None:
                sampled_sample_ids = data.sample_ids + sampled_sample_ids
        return FeatureSet(features=sampled_features, feature_names=sampled_feature_names, feature_types=sampled_feature_types, sample_ids=sampled_sample_ids, metadata=sampled_metadata, quality_scores=sampled_quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for bootstrap sampling.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set.
        **kwargs
            Additional keyword arguments.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not meaningful for sampling operations.
        """
        raise NotImplementedError('inverse_transform is not implemented for BootstrapSampler')

class BalancedBootstrapSampler(BaseTransformer):

    def __init__(self, n_samples_per_class: Optional[int]=None, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None, target_column: Optional[str]=None):
        super().__init__(name=name)
        self.n_samples_per_class = n_samples_per_class
        self.random_state = random_state
        self.preserve_original = preserve_original
        self.target_column = target_column
        self._class_indices = None
        self._fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'BalancedBootstrapSampler':
        """
        Fit the transformer to the input data, identifying class distributions.
        
        Validates that target labels exist and computes class distributions to
        determine sampling parameters for balanced bootstrap.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to fit on. Must contain labels in metadata or as a named feature.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        BalancedBootstrapSampler
            The fitted transformer instance.
            
        Raises
        ------
        ValueError
            If target labels are not found or target_column is not specified when needed.
        """
        if self.target_column is None:
            if data.metadata is None or 'target' not in data.metadata:
                raise ValueError('Target column must be specified or target must be present in metadata')
            y = data.metadata['target']
        else:
            if data.feature_names is None or self.target_column not in data.feature_names:
                raise ValueError(f"Target column '{self.target_column}' not found in feature names")
            target_index = data.feature_names.index(self.target_column)
            y = data.features[:, target_index]
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        (unique_classes, inverse_indices) = np.unique(y, return_inverse=True)
        self._class_indices = {}
        for (i, cls) in enumerate(unique_classes):
            self._class_indices[cls] = np.where(inverse_indices == i)[0]
        self._fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply balanced bootstrap sampling to the input data.
        
        For each class, draws samples with replacement to create a balanced dataset.
        If preserve_original is True, the original samples are included in the output.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        FeatureSet
            Balanced bootstrapped feature set with equal representation of classes.
            
        Raises
        ------
        ValueError
            If the transformer was not properly fitted or if class labels are inconsistent.
        """
        if not self._fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.target_column is None:
            if data.metadata is None or 'target' not in data.metadata:
                raise ValueError('Target column must be specified or target must be present in metadata')
            y = data.metadata['target']
        else:
            if data.feature_names is None or self.target_column not in data.feature_names:
                raise ValueError(f"Target column '{self.target_column}' not found in feature names")
            target_index = data.feature_names.index(self.target_column)
            y = data.features[:, target_index]
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        transform_class_indices = {}
        for class_label in self._class_indices.keys():
            class_mask = y == class_label
            transform_class_indices[class_label] = np.where(class_mask)[0]
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.n_samples_per_class is None:
            available_sample_counts = [len(indices) for indices in transform_class_indices.values() if len(indices) > 0]
            if not available_sample_counts:
                raise ValueError('No classes from the fitted data are present in the transform data')
            n_samples = max(available_sample_counts)
        else:
            n_samples = self.n_samples_per_class
        sampled_indices = []
        if self.preserve_original:
            sampled_indices.extend(range(data.features.shape[0]))
        for class_label in self._class_indices.keys():
            indices = transform_class_indices.get(class_label, np.array([]))
            if len(indices) == 0:
                continue
            sampled_class_indices = np.random.choice(indices, size=n_samples, replace=True)
            if not self.preserve_original:
                sampled_indices.extend(sampled_class_indices)
            else:
                sampled_indices.extend(sampled_class_indices)
        if len(sampled_indices) == 0:
            raise ValueError('No valid samples found for sampling')
        sampled_features = data.features[sampled_indices]
        feature_names = data.feature_names
        feature_types = data.feature_types
        if self.target_column is not None and feature_names is not None and (self.target_column in feature_names):
            target_index = feature_names.index(self.target_column)
            feature_mask = np.ones(sampled_features.shape[1], dtype=bool)
            feature_mask[target_index] = False
            sampled_features = sampled_features[:, feature_mask]
            if feature_names is not None:
                feature_names = [name for (i, name) in enumerate(feature_names) if i != target_index]
            if feature_types is not None:
                feature_types = [ftype for (i, ftype) in enumerate(feature_types) if i != target_index]
        sample_ids = None
        if data.sample_ids is not None:
            sample_ids = [data.sample_ids[i] for i in sampled_indices]
        metadata = {}
        if data.metadata is not None:
            metadata = data.metadata.copy()
            if 'target' in metadata:
                original_target = metadata['target']
                if not isinstance(original_target, np.ndarray):
                    original_target = np.array(original_target)
                sampled_target = original_target[sampled_indices]
                metadata['target'] = sampled_target
        quality_scores = None
        if data.quality_scores is not None:
            if isinstance(data.quality_scores, dict):
                quality_scores = {str(i): data.quality_scores.get(str(orig_idx), 0.0) for (i, orig_idx) in enumerate(sampled_indices) if str(orig_idx) in data.quality_scores}
            elif isinstance(data.quality_scores, np.ndarray):
                quality_scores = data.quality_scores[sampled_indices]
            else:
                quality_scores = []
                for idx in sampled_indices:
                    if idx < len(data.quality_scores):
                        quality_scores.append(data.quality_scores[idx])
                    else:
                        quality_scores.append(0.0)
        return FeatureSet(features=sampled_features, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for balanced bootstrap sampling.
        
        Balanced bootstrap sampling is a stochastic process that creates new samples
        by sampling with replacement from each class. Since information is lost during this process
        (some samples are duplicated while others are omitted), it's not possible
        to reconstruct the original dataset from the bootstrapped version.
            
        Parameters
        ----------
        data : FeatureSet
            Input feature set.
        **kwargs
            Additional keyword arguments.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not defined for sampling operations.
        """
        raise NotImplementedError('Inverse transform is not supported for balanced bootstrap sampling.')

class WeightedBootstrapSampler(BaseTransformer):

    def __init__(self, n_samples: Optional[int]=None, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None, weights: Optional[Union[List[float], np.ndarray]]=None):
        super().__init__(name=name)
        self.n_samples = n_samples
        self.random_state = random_state
        self.preserve_original = preserve_original
        self.weights = weights

    def fit(self, data: FeatureSet, **kwargs) -> 'WeightedBootstrapSampler':
        """
        Fit the transformer to the input data.
        
        Stores sample weights if provided in kwargs. Validates weight dimensions
        if weights were provided during initialization.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to fit on.
        **kwargs
            Additional keyword arguments. Can include 'sample_weights' for per-sample weights.
            
        Returns
        -------
        WeightedBootstrapSampler
            The fitted transformer instance.
            
        Raises
        ------
        ValueError
            If provided weights have incorrect dimensions or contain invalid values.
        """
        self._n_input_samples = data.features.shape[0]
        weights = kwargs.get('sample_weights', self.weights)
        if weights is not None:
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)
            if weights.shape[0] != self._n_input_samples:
                raise ValueError(f'Weight array size ({weights.shape[0]}) does not match number of samples ({self._n_input_samples})')
            if not np.all(np.isfinite(weights)):
                raise ValueError('Weights contain non-finite values (NaN or infinity)')
            if np.any(weights < 0):
                raise ValueError('Weights contain negative values')
            weight_sum = np.sum(weights)
            if weight_sum == 0:
                raise ValueError('Sum of weights is zero')
            weights = weights / weight_sum
            self._fitted_weights = weights
        else:
            self._fitted_weights = None
        self._fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply weighted bootstrap sampling to the input data.
        
        Draws samples with replacement using specified weights. Higher weighted samples
        have a greater probability of being selected. If preserve_original is True,
        the original samples are included in the output.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform.
        **kwargs
            Additional keyword arguments. Can include 'sample_weights' for per-sample weights.
            
        Returns
        -------
        FeatureSet
            Weighted bootstrapped feature set with samples selected according to weights.
            
        Raises
        ------
        ValueError
            If weights are not provided or have incorrect dimensions.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        n_input_samples = data.features.shape[0]
        weights = kwargs.get('sample_weights', getattr(self, '_fitted_weights', self.weights))
        if weights is None:
            raise ValueError('Sample weights must be provided either during initialization, fitting, or transformation')
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        if weights.shape[0] != n_input_samples:
            raise ValueError(f'Weight array size ({weights.shape[0]}) does not match number of samples ({n_input_samples})')
        if not np.all(np.isfinite(weights)):
            raise ValueError('Weights contain non-finite values (NaN or infinity)')
        if np.any(weights < 0):
            raise ValueError('Weights contain negative values')
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            raise ValueError('Sum of weights is zero')
        weights = weights / weight_sum
        if not hasattr(self, '_n_input_samples'):
            self._n_input_samples = n_input_samples
        elif n_input_samples != self._n_input_samples:
            raise ValueError(f'Input data has {n_input_samples} samples, but transformer was fitted on {self._n_input_samples} samples.')
        n_samples = self.n_samples if self.n_samples is not None else n_input_samples
        rng = np.random.default_rng(self.random_state)
        bootstrap_indices = rng.choice(n_input_samples, size=n_samples, replace=True, p=weights)
        sampled_features = data.features[bootstrap_indices]
        sampled_feature_names = data.feature_names
        sampled_feature_types = data.feature_types
        sampled_metadata = data.metadata.copy() if data.metadata else {}
        sampled_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        if data.sample_ids is not None:
            sampled_sample_ids = [data.sample_ids[i] for i in bootstrap_indices]
        else:
            sampled_sample_ids = None
        if self.preserve_original:
            sampled_features = np.vstack([data.features, sampled_features])
            if sampled_sample_ids is not None and data.sample_ids is not None:
                sampled_sample_ids = data.sample_ids + sampled_sample_ids
        return FeatureSet(features=sampled_features, feature_names=sampled_feature_names, feature_types=sampled_feature_types, sample_ids=sampled_sample_ids, metadata=sampled_metadata, quality_scores=sampled_quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for weighted bootstrap sampling.
        
        Weighted bootstrap sampling is a stochastic process that creates new samples
        by sampling with replacement according to specified weights. Since information 
        is lost during this process (some samples are duplicated while others are omitted), 
        it's not possible to reconstruct the original dataset from the bootstrapped version.
            
        Parameters
        ----------
        data : FeatureSet
            Input feature set.
        **kwargs
            Additional keyword arguments.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not defined for sampling operations.
        """
        raise NotImplementedError('Inverse transform is not supported for weighted bootstrap sampling.')