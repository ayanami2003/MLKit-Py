from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class VarianceThresholdSelector(BaseTransformer):

    def __init__(self, threshold: float=0.0, name: Optional[str]=None):
        """
        Initialize the VarianceThresholdSelector.
        
        Parameters
        ----------
        threshold : float, default=0.0
            Features with a training-set variance lower than this threshold will
            be removed. The default of 0.0 removes only features that have the
            same value in all samples.
            
        name : str, optional
            Name of the transformer instance. If None, uses class name.
        """
        super().__init__(name=name)
        if threshold < 0:
            raise ValueError('Threshold must be non-negative')
        self.threshold = threshold
        self.selected_features_ = None
        self.feature_names_ = None
        self._variances = None
        self._selected_features_mask = None
        self._n_features_in = None
        self._is_fitted = False
        self._original_feature_names = None

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'VarianceThresholdSelector':
        """
        Learn empirical variances from the input data.
        
        Computes the variance for each feature and determines which features
        meet the variance threshold requirement.
        
        Parameters
        ----------
        data : FeatureSet, DataBatch, or ndarray of shape (n_samples, n_features)
            Training data where n_samples is the number of samples and
            n_features is the number of features.
            
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        VarianceThresholdSelector
            This instance.
            
        Raises
        ------
        ValueError
            If any feature has negative variance (should not happen with real data).
        """
        if isinstance(data, np.ndarray):
            X = data
            feature_names = None
        elif isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
        else:
            raise TypeError('Input data must be numpy array, FeatureSet, or DataBatch')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        self._n_features_in = X.shape[1]
        self._original_feature_names = feature_names
        if X.shape[0] > 1:
            self._variances = np.var(X, axis=0, ddof=0)
        else:
            self._variances = np.zeros(X.shape[1])
        self._variances = np.maximum(self._variances, 0.0)
        self._selected_features_mask = self._variances >= self.threshold
        if not np.any(self._selected_features_mask) and len(self._variances) > 0:
            max_var_idx = np.argmax(self._variances)
            self._selected_features_mask[:] = False
            self._selected_features_mask[max_var_idx] = True
        self.selected_features_ = self._selected_features_mask
        if feature_names is not None:
            self.feature_names_ = [name for (i, name) in enumerate(feature_names) if self._selected_features_mask[i]]
        else:
            self.feature_names_ = None
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Reduce data to features with sufficiently high variance.
        
        Applies the variance threshold to select only features that meet
        the variance requirement determined during fitting.
        
        Parameters
        ----------
        data : FeatureSet, DataBatch, or ndarray of shape (n_samples, n_features)
            Data to transform with variance thresholding.
            
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        FeatureSet, DataBatch, or ndarray
            Transformed data with low-variance features removed.
            Same type as input data.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
            If input data has a different number of features than during fitting.
        """
        if not self._is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, np.ndarray):
            X = data
            original_type = 'ndarray'
            feature_names = None
        elif isinstance(data, FeatureSet):
            X = data.features
            original_type = 'FeatureSet'
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            original_type = 'DataBatch'
            feature_names = data.feature_names
        else:
            raise TypeError('Input data must be numpy array, FeatureSet, or DataBatch')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        n_features = X.shape[1]
        if n_features != self._n_features_in:
            raise ValueError(f'Input data has {n_features} features, but transformer was fitted on {self._n_features_in} features')
        X_reduced = X[:, self._selected_features_mask]
        if original_type == 'ndarray':
            return X_reduced
        elif original_type == 'FeatureSet':
            selected_feature_names = None
            if feature_names is not None:
                selected_feature_names = [name for (i, name) in enumerate(feature_names) if self._selected_features_mask[i]]
            return FeatureSet(features=X_reduced, feature_names=selected_feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata)
        elif original_type == 'DataBatch':
            selected_feature_names = None
            if feature_names is not None:
                selected_feature_names = [name for (i, name) in enumerate(feature_names) if self._selected_features_mask[i]]
            return DataBatch(data=X_reduced, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=selected_feature_names, batch_id=data.batch_id)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Reverse the variance thresholding transformation.
        
        Inserts zero-valued columns for the removed features at their original positions.
        Note that this cannot recover the original feature values, only restore the structure.
        
        Parameters
        ----------
        data : FeatureSet, DataBatch, or ndarray
            Data with reduced features to inverse transform.
            
        **kwargs : dict
            Additional parameters (ignored in this implementation).
            
        Returns
        -------
        FeatureSet, DataBatch, or ndarray
            Data with features restored to original dimensionality.
            Removed features are filled with zeros.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
            If input data dimensions don't match the expected shape.
        """
        if not self._is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, np.ndarray):
            X = data
            original_type = 'ndarray'
            feature_names = None
        elif isinstance(data, FeatureSet):
            X = data.features
            original_type = 'FeatureSet'
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            original_type = 'DataBatch'
            feature_names = data.feature_names
        else:
            raise TypeError('Input data must be numpy array, FeatureSet, or DataBatch')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        n_features_out = X.shape[1]
        expected_n_features_out = np.sum(self._selected_features_mask)
        if n_features_out != expected_n_features_out:
            raise ValueError(f'Input data has {n_features_out} features, but expected {expected_n_features_out} features based on fitting')
        X_full = np.zeros((X.shape[0], self._n_features_in), dtype=X.dtype)
        X_full[:, self._selected_features_mask] = X
        if original_type == 'ndarray':
            return X_full
        elif original_type == 'FeatureSet':
            restored_feature_names = None
            if self._original_feature_names is not None:
                restored_feature_names = []
                selected_idx = 0
                for i in range(self._n_features_in):
                    if self._selected_features_mask[i]:
                        if feature_names is not None and selected_idx < len(feature_names):
                            restored_feature_names.append(feature_names[selected_idx])
                        else:
                            restored_feature_names.append(self._original_feature_names[i] if self._original_feature_names and i < len(self._original_feature_names) else f'feature_{i}')
                        selected_idx += 1
                    else:
                        restored_feature_names.append(self._original_feature_names[i] if self._original_feature_names and i < len(self._original_feature_names) else f'feature_{i}')
            return FeatureSet(features=X_full, feature_names=restored_feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata)
        elif original_type == 'DataBatch':
            restored_feature_names = None
            if self._original_feature_names is not None:
                restored_feature_names = []
                selected_idx = 0
                for i in range(self._n_features_in):
                    if self._selected_features_mask[i]:
                        if feature_names is not None and selected_idx < len(feature_names):
                            restored_feature_names.append(feature_names[selected_idx])
                        else:
                            restored_feature_names.append(self._original_feature_names[i] if self._original_feature_names and i < len(self._original_feature_names) else f'feature_{i}')
                        selected_idx += 1
                    else:
                        restored_feature_names.append(self._original_feature_names[i] if self._original_feature_names and i < len(self._original_feature_names) else f'feature_{i}')
            return DataBatch(data=X_full, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=restored_feature_names, batch_id=data.batch_id)