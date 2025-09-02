from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from src.data_lifecycle.computational_utilities.random_operations.sampling_methods import generate_random_samples
from src.data_lifecycle.feature_engineering.selection.methods.generic_selection import GenericFeatureSelector
import copy

class StabilitySelectionFeatureSelector(BaseTransformer):

    def __init__(self, subsample_ratio: float=0.5, n_iterations: int=100, threshold: float=0.6, base_selector: Optional[BaseTransformer]=None, name: Optional[str]=None):
        """
        Initialize the stability selection feature selector.
        
        Parameters
        ----------
        subsample_ratio : float, default=0.5
            Proportion of samples to use in each subsample.
            Must be between 0 and 1.
        n_iterations : int, default=100
            Number of subsampling iterations to perform.
            Higher values provide more robust stability estimates.
        threshold : float, default=0.6
            Minimum selection frequency for a feature to be considered stable.
            Features selected in at least this proportion of iterations are kept.
        base_selector : BaseTransformer, optional
            Underlying feature selection method to apply on each subsample.
            If None, uses variance threshold selection by default.
        name : str, optional
            Name of the transformer instance.
            
        Raises
        ------
        ValueError
            If subsample_ratio is not between 0 and 1, or if
            n_iterations or threshold are not positive.
        """
        super().__init__(name=name)
        self.subsample_ratio = subsample_ratio
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.base_selector = base_selector
        self.stable_features_ = []
        self.selection_freq_ = np.array([])

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'StabilitySelectionFeatureSelector':
        """
        Fit the stability selection model by evaluating feature stability across subsamples.
        
        This method performs multiple iterations of subsampling, applies the base
        selector to each subsample, and tracks how frequently each feature is selected.
        Features meeting the stability threshold are marked as stable.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing features to select from.
        **kwargs : dict
            Additional parameters passed to the base selector.
            
        Returns
        -------
        StabilitySelectionFeatureSelector
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the data is malformed or incompatible with the selector.
        """
        if not 0 < self.subsample_ratio <= 1:
            raise ValueError('subsample_ratio must be between 0 and 1')
        if self.n_iterations <= 0:
            raise ValueError('n_iterations must be positive')
        if not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            y = kwargs.get('y', None)
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            y = data.labels
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        (n_samples, n_features) = X.shape
        if self.base_selector is None:
            self.base_selector = GenericFeatureSelector(method='variance_threshold')
        self.selection_freq_ = np.zeros(n_features)
        if isinstance(data, FeatureSet):
            data_for_sampling = np.array(data.features)
        else:
            data_for_sampling = data
        for _ in range(self.n_iterations):
            n_subsamples = max(1, int(self.subsample_ratio * n_samples))
            indices = np.random.choice(n_samples, size=n_subsamples, replace=False)
            if isinstance(data, FeatureSet):
                X_sub = X[indices]
                y_sub = np.array(y)[indices] if y is not None else None
                subsample = FeatureSet(features=X_sub, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=[data.sample_ids[i] if data.sample_ids else None for i in indices] if data.sample_ids else None, metadata=data.metadata)
            elif isinstance(data, DataBatch):
                X_sub = X[indices]
                y_sub = np.array(y)[indices] if y is not None else None
                subsample = DataBatch(data=X_sub.tolist() if not isinstance(data.data, np.ndarray) else X_sub, labels=y_sub.tolist() if y_sub is not None and (not isinstance(data.labels, np.ndarray)) else y_sub, metadata=data.metadata, sample_ids=[data.sample_ids[i] if data.sample_ids else None for i in indices] if data.sample_ids else None, feature_names=data.feature_names, batch_id=data.batch_id)
            if hasattr(self.base_selector, '__dict__'):
                selector = self.base_selector.__class__(**self.base_selector.get_params())
            else:
                selector = self.base_selector.__class__()
            if y_sub is not None and hasattr(selector, 'fit'):
                selector.fit(subsample, y=y_sub)
            elif hasattr(selector, 'fit'):
                selector.fit(subsample)
            if hasattr(selector, 'get_support'):
                support = selector.get_support()
                if isinstance(support, list):
                    selected_indices = support
                else:
                    selected_indices = np.where(support)[0].tolist()
            elif hasattr(selector, 'selected_features_'):
                selected_indices = selector.selected_features_
            else:
                selected_indices = list(range(n_features))
            for idx in selected_indices:
                if 0 <= idx < n_features:
                    self.selection_freq_[idx] += 1
        self.selection_freq_ = self.selection_freq_ / self.n_iterations
        self.stable_features_ = np.where(self.selection_freq_ >= self.threshold)[0].tolist()
        self.n_features_ = n_features
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Transform the input data by selecting only the stable features.
        
        Applies the feature selection based on previously computed stability scores,
        returning a new FeatureSet containing only features identified as stable.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed data containing only stable features.
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'stable_features_') or len(self.stable_features_) == 0:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            feature_types = None
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        X_selected = X[:, self.stable_features_]
        selected_feature_names = None
        if feature_names is not None:
            selected_feature_names = [feature_names[i] for i in self.stable_features_]
        selected_feature_types = None
        if feature_types is not None:
            selected_feature_types = [feature_types[i] for i in self.stable_features_]
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for stability selection.
        
        Since stability selection reduces dimensionality by removing features,
        it's generally not possible to reconstruct the original feature space.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Data to inverse transform (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Always raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        pass

    def get_support(self, indices: bool=False) -> Union[np.ndarray, list]:
        """
        Get a mask or indices of the stable features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return indices of stable features.
            If False, return a boolean mask.
            
        Returns
        -------
        Union[np.ndarray, list]
            Boolean mask or list of indices indicating stable features.
        """
        pass