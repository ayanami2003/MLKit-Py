from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

class TreeBasedFeatureSelector(BaseTransformer):

    def __init__(self, method: str='random_forest', n_features: Optional[int]=None, importance_threshold: Optional[float]=None, random_state: Optional[int]=None, name: Optional[str]=None, **kwargs):
        """
        Initialize the TreeBasedFeatureSelector.
        
        Parameters
        ----------
        method : str, default='random_forest'
            The tree-based method to use ('random_forest', 'gradient_boosting', etc.).
        n_features : int, optional
            Number of top features to select. If None, uses importance_threshold.
        importance_threshold : float, optional
            Minimum importance threshold for feature selection. Ignored if n_features is set.
        random_state : int, optional
            Random state for reproducibility.
        name : str, optional
            Name of the transformer instance.
        **kwargs : dict
            Additional parameters passed to the underlying tree-based estimator.
        """
        super().__init__(name=name)
        self.method = method
        self.n_features = n_features
        self.importance_threshold = importance_threshold
        self.random_state = random_state
        self.kwargs = kwargs
        self._selected_features_mask = None
        self._feature_importances = None

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'TreeBasedFeatureSelector':
        """
        Fit the selector to the input data by training a tree-based model and computing feature importances.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing features and optionally labels.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        TreeBasedFeatureSelector
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            y = kwargs.get('y')
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            y = data.labels
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if y is None:
            raise ValueError('Labels (y) must be provided for supervised feature selection')
        if self.n_features is None and self.importance_threshold is None:
            raise ValueError('Either n_features or importance_threshold must be specified')
        if self.n_features is not None and self.n_features <= 0:
            raise ValueError('n_features must be positive')
        n_features = X.shape[1]
        if self.n_features is not None and self.n_features > n_features:
            self.n_features = n_features
        is_classification = len(np.unique(y)) <= 20
        if self.method == 'random_forest':
            if is_classification:
                estimator = RandomForestClassifier(random_state=self.random_state, **self.kwargs)
            else:
                estimator = RandomForestRegressor(random_state=self.random_state, **self.kwargs)
        elif self.method == 'gradient_boosting':
            if is_classification:
                estimator = GradientBoostingClassifier(random_state=self.random_state, **self.kwargs)
            else:
                estimator = GradientBoostingRegressor(random_state=self.random_state, **self.kwargs)
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        estimator.fit(X, y)
        self._feature_importances = estimator.feature_importances_
        if self.n_features is not None:
            indices = np.argsort(self._feature_importances)[::-1][:self.n_features]
            self._selected_features_mask = np.zeros(n_features, dtype=bool)
            self._selected_features_mask[indices] = True
        else:
            self._selected_features_mask = self._feature_importances >= self.importance_threshold
            if not np.any(self._selected_features_mask):
                best_idx = np.argmax(self._feature_importances)
                self._selected_features_mask = np.zeros(n_features, dtype=bool)
                self._selected_features_mask[best_idx] = True
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Apply feature selection to the input data using computed importances.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed data with only selected features.
        """
        if not getattr(self, '_is_fitted', False):
            raise RuntimeError('Transformer has not been fitted yet.')
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
        X_selected = X[:, self._selected_features_mask]
        selected_feature_names = [feature_names[i] for (i, selected) in enumerate(self._selected_features_mask) if selected]
        selected_feature_types = None
        if feature_types is not None:
            selected_feature_types = [feature_types[i] for (i, selected) in enumerate(self._selected_features_mask) if selected]
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the transformation (not supported for feature selection).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original data structure (raises NotImplementedError as inversion is not meaningful for selection).
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection.')

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get the feature importances computed during fitting.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of feature importances or None if not yet fitted.
        """
        return self._feature_importances

    def get_selected_features_mask(self) -> Optional[np.ndarray]:
        """
        Get boolean mask indicating which features were selected.
        
        Returns
        -------
        Optional[np.ndarray]
            Boolean array where True indicates selected features, or None if not yet fitted.
        """
        return self._selected_features_mask