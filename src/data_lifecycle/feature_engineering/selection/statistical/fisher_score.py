from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class FisherScoreSelector(BaseTransformer):

    def __init__(self, k: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the FisherScoreSelector.
        
        Parameters
        ----------
        k : Optional[int], default=None
            Number of top features to select. If None, all features are selected.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.k = k

    def fit(self, data: FeatureSet, y: Union[np.ndarray, List], **kwargs) -> 'FisherScoreSelector':
        """
        Compute Fisher scores for each feature.
        
        Parameters
        ----------
        data : FeatureSet
            Input features with shape (n_samples, n_features).
        y : array-like of shape (n_samples,)
            Target values (class labels for classification).
        **kwargs : dict
            Additional fitting parameters (not used).
            
        Returns
        -------
        self : FisherScoreSelector
            Fitted selector.
            
        Raises
        ------
        ValueError
            If data is not labeled or if k is larger than the number of features.
        """
        X = data.features
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in features and target must match')
        if X.ndim != 2:
            raise ValueError('Features must be a 2D array')
        (n_samples, n_features) = X.shape
        unique_labels = np.unique(y)
        if not np.issubdtype(y.dtype, np.number):
            label_map = {label: idx for (idx, label) in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
        if self.k is not None and self.k < 0:
            raise ValueError('k must be non-negative.')
        X = X.astype(np.float64)
        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError('At least 2 classes are required for Fisher Score computation')
        class_means = np.zeros((n_classes, n_features))
        class_vars = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)
        for (i, cls) in enumerate(classes):
            X_cls = X[y == cls]
            class_counts[i] = X_cls.shape[0]
            class_means[i] = np.mean(X_cls, axis=0)
            class_vars[i] = np.var(X_cls, axis=0)
        overall_mean = np.mean(X, axis=0)
        between_class_var = np.zeros(n_features)
        for i in range(n_classes):
            diff = class_means[i] - overall_mean
            between_class_var += class_counts[i] * diff * diff
        within_class_var = np.zeros(n_features)
        for i in range(n_classes):
            within_class_var += class_counts[i] * class_vars[i]
        epsilon = 1e-12
        scores = between_class_var / (within_class_var + epsilon)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.scores_ = scores
        if self.k is None:
            k_to_select = n_features
        elif self.k <= 0:
            k_to_select = 0
        else:
            k_to_select = min(self.k, n_features)
        if k_to_select == 0:
            self.selected_indices_ = np.array([], dtype=int)
        elif k_to_select >= n_features:
            self.selected_indices_ = np.arange(n_features)
        else:
            indices = np.argsort(scores)[::-1][:k_to_select]
            self.selected_indices_ = np.sort(indices)
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reduce data to the selected features.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform.
        **kwargs : dict
            Additional transformation parameters (not used).
            
        Returns
        -------
        FeatureSet
            Transformed feature set with only the selected features.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'selected_indices_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        selected_features = data.features[:, self.selected_indices_]
        selected_feature_names = None
        if data.feature_names:
            selected_feature_names = [data.feature_names[i] for i in self.selected_indices_]
        selected_feature_types = None
        if data.feature_types:
            selected_feature_types = [data.feature_types[i] for i in self.selected_indices_]
        transformed_data = FeatureSet(features=selected_features, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})
        return transformed_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the transformation (not supported for feature selection).
        
        Parameters
        ----------
        data : FeatureSet
            Transformed data to reverse.
        **kwargs : dict
            Additional parameters (not used).
            
        Returns
        -------
        FeatureSet
            Original feature set with placeholder values for unselected features.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection.
        """
        raise NotImplementedError('Inverse transform is not supported for feature selection.')

    def get_support(self) -> np.ndarray:
        """
        Get a mask of the features selected.
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            Boolean mask indicating which features are selected.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'scores_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'get_support'.")
        mask = np.zeros(len(self.scores_), dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_scores(self) -> np.ndarray:
        """
        Get the Fisher scores for all features.
        
        Returns
        -------
        np.ndarray of shape (n_features,)
            Fisher scores for each feature.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'scores_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'get_scores'.")
        return self.scores_