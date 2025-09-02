from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from sklearn.feature_selection import chi2

class ChiSquareFeatureSelector(BaseTransformer):

    def __init__(self, k: Optional[int]=None, threshold: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the Chi-Square feature selector.
        
        Parameters
        ----------
        k : Optional[int], default=None
            Number of top features to select based on Chi-Square scores.
            If None, all features with scores above threshold are selected.
        threshold : Optional[float], default=None
            Minimum Chi-Square score required for feature selection.
            If specified, overrides k parameter.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        if k is not None and threshold is not None:
            self.k = None
        else:
            self.k = k
        self.threshold = threshold
        self._scores = None
        self._selected_indices = None

    def fit(self, data: FeatureSet, y: Optional[Union[np.ndarray, List]]=None, **kwargs) -> 'ChiSquareFeatureSelector':
        """
        Fit the selector to training data by computing Chi-Square scores.
        
        Parameters
        ----------
        data : FeatureSet
            Input features with shape (n_samples, n_features).
        y : Union[np.ndarray, List]
            Target values with shape (n_samples,).
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        ChiSquareFeatureSelector
            Self instance for method chaining.
        """
        X = data.features
        if y is None:
            raise ValueError('y parameter is required for fitting')
        if np.any(X < 0):
            raise ValueError('Chi-square test requires non-negative feature values.')
        if len(np.unique(y)) < 2:
            raise ValueError('Chi-square test requires at least two classes in target variable.')
        (scores, _) = chi2(X, y)
        self._scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        n_features = X.shape[1]
        if self.k is not None and self.k < 0:
            raise ValueError('k must be non-negative.')
        if self.threshold is not None and self.threshold < 0:
            raise ValueError('threshold must be non-negative.')
        if self.k is not None and self.k > n_features:
            raise ValueError(f'k ({self.k}) cannot be larger than number of features ({n_features}).')
        if self.threshold is not None:
            self._selected_indices = np.where(self._scores >= self.threshold)[0]
        elif self.k is not None:
            if self.k >= n_features:
                self._selected_indices = np.arange(n_features)
            else:
                sorted_indices = np.argsort(self._scores)[::-1]
                self._selected_indices = sorted_indices[:self.k]
        else:
            self._selected_indices = np.arange(n_features)
        self._selected_indices = np.sort(self._selected_indices)
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply feature selection to input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed feature set containing only selected features.
        """
        if self._selected_indices is None:
            raise ValueError('Transformer has not been fitted yet.')
        X_selected = data.features[:, self._selected_indices]
        selected_feature_names = None
        selected_feature_types = None
        if data.feature_names is not None:
            selected_feature_names = [data.feature_names[i] for i in self._selected_indices]
        if data.feature_types is not None:
            selected_feature_types = [data.feature_types[i] for i in self._selected_indices]
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=getattr(data, 'metadata', None), quality_scores=getattr(data, 'quality_scores', None))

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reverse the transformation (not implemented for feature selection).
        
        Parameters
        ----------
        data : FeatureSet
            Transformed feature set.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set with unselected features filled with zeros.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection.
        """
        raise NotImplementedError('Inverse transformation is not implemented for feature selection.')

    def get_support(self) -> np.ndarray:
        """
        Get the indices of selected features.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating which features are selected.
        """
        if self._selected_indices is None:
            raise ValueError('Transformer has not been fitted yet.')
        support_mask = np.zeros(len(self._scores), dtype=bool)
        support_mask[self._selected_indices] = True
        return support_mask

    def get_scores(self) -> np.ndarray:
        """
        Get the Chi-Square scores for all features.
        
        Returns
        -------
        np.ndarray
            Array of Chi-Square scores for each feature.
        """
        if self._scores is None:
            raise ValueError('Transformer has not been fitted yet.')
        return self._scores.copy()