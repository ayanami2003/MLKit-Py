from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class MutualInformationSelector(BaseTransformer):

    def __init__(self, method: str='classification', k: Optional[int]=None, threshold: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the MutualInformationSelector.
        
        Args:
            method (str): Method for computing mutual information ('classification' or 'regression').
            k (Optional[int]): Number of top features to select.
            threshold (Optional[float]): Minimum score for feature selection.
            name (Optional[str]): Name of the transformer instance.
        """
        super().__init__(name=name)
        self.method = method
        self.k = k
        self.threshold = threshold
        self._scores = None
        self._selected_indices = None

    def fit(self, data: FeatureSet, y: Union[np.ndarray, List], **kwargs) -> 'MutualInformationSelector':
        """
        Compute mutual information scores between features and target variable.
        
        Args:
            data (FeatureSet): Input features to evaluate.
            y (Union[np.ndarray, List]): Target variable values.
            **kwargs: Additional fitting parameters.
            
        Returns:
            MutualInformationSelector: Fitted transformer instance.
        """
        if self.method not in ['classification', 'regression']:
            raise ValueError("Method must be either 'classification' or 'regression'")
        if self.k is None and self.threshold is None:
            self.k = data.features.shape[1]
        if self.k is not None and (not isinstance(self.k, int) or self.k <= 0):
            raise ValueError('k must be a positive integer or None')
        if self.threshold is not None and (not isinstance(self.threshold, (int, float)) or self.threshold < 0):
            raise ValueError('threshold must be a non-negative number or None')
        X = data.features
        n_features = X.shape[1]
        self._feature_names = data.feature_names if data.feature_names else [f'feature_{i}' for i in range(n_features)]
        if self.method == 'classification':
            self._scores = mutual_info_classif(X, y, random_state=kwargs.get('random_state', 0))
        else:
            self._scores = mutual_info_regression(X, y, random_state=kwargs.get('random_state', 0))
        self._scores = np.nan_to_num(self._scores, nan=0.0, posinf=0.0, neginf=0.0)
        if self.threshold is not None:
            self._selected_indices = np.where(self._scores >= self.threshold)[0]
        else:
            k = min(self.k, n_features) if self.k is not None else n_features
            k = max(1, k)
            if k >= n_features:
                self._selected_indices = np.arange(n_features)
            else:
                sorted_indices = np.argsort(self._scores)[::-1]
                self._selected_indices = sorted_indices[:k]
        self._selected_indices = np.sort(self._selected_indices)
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply feature selection based on mutual information scores.
        
        Args:
            data (FeatureSet): Input features to transform.
            **kwargs: Additional transformation parameters.
            
        Returns:
            FeatureSet: Transformed feature set containing only selected features.
        """
        if self._selected_indices is None:
            raise ValueError('Transformer has not been fitted yet.')
        if len(self._selected_indices) == 0:
            X_selected = np.empty((data.features.shape[0], 0))
            selected_feature_names = [] if data.feature_names else None
            selected_feature_types = [] if data.feature_types else None
            sample_ids = data.sample_ids
        else:
            X_selected = data.features[:, self._selected_indices]
            selected_feature_names = [data.feature_names[i] for i in self._selected_indices] if data.feature_names else None
            selected_feature_types = [data.feature_types[i] for i in self._selected_indices] if data.feature_types else None
            sample_ids = data.sample_ids
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, metadata=data.metadata, sample_ids=sample_ids, quality_scores=data.quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Not implemented for feature selection transformers.
        
        Args:
            data (FeatureSet): Transformed data.
            **kwargs: Additional parameters.
            
        Returns:
            FeatureSet: Original feature set.
            
        Raises:
            NotImplementedError: Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection.')

    def get_support(self) -> np.ndarray:
        """
        Get indices of selected features.
        
        Returns:
            np.ndarray: Boolean array indicating which features are selected.
        """
        if self._selected_indices is None:
            return None
        mask = np.zeros(len(self._scores), dtype=bool)
        mask[self._selected_indices] = True
        return mask

    def get_scores(self) -> np.ndarray:
        """
        Get mutual information scores for all features.
        
        Returns:
            np.ndarray: Array of mutual information scores.
        """
        if self._scores is None:
            return None
        return self._scores.copy()