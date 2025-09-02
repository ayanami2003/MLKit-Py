from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.feature_selection import chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_X_y, check_array
from scipy.stats import pearsonr
from typing import Optional, List, Union

class StatisticalFilterSelector(BaseTransformer):

    def __init__(self, method: str='f_classif', k: Optional[int]=None, threshold: Optional[float]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.method = method
        self.k = k
        self.threshold = threshold
        self._selected_indices = None
        self._feature_scores = None
        self._original_feature_names = None
        self._original_feature_types = None
        supported_methods = {'chi2', 'f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression', 'correlation'}
        if self.method not in supported_methods:
            raise ValueError(f"Method '{self.method}' is not supported. Supported methods: {supported_methods}")
        if self.k is not None and self.k <= 0:
            raise ValueError('k must be a positive integer or None')
        if self.threshold is not None and (not isinstance(self.threshold, (int, float))):
            raise ValueError('threshold must be a number or None')
        if self.k is None and self.threshold is None:
            raise ValueError('Either k or threshold must be specified')

    def fit(self, data: FeatureSet, y: Union[np.ndarray, List], **kwargs) -> 'StatisticalFilterSelector':
        """
        Fit the statistical filter selector to the data.
        
        Computes the statistical scores for each feature based on the specified method
        and determines which features to select based on k or threshold parameters.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to evaluate for selection
        y : array-like
            Target values (required for supervised methods)
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        StatisticalFilterSelector
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If method is not supported or if both k and threshold are None
        """
        X = data.features
        (X, y) = check_X_y(X, y)
        n_features = X.shape[1]
        self._original_feature_names = data.feature_names if data.feature_names else [f'feature_{i}' for i in range(n_features)]
        self._original_feature_types = data.feature_types if data.feature_types else None
        if self.method == 'chi2':
            if np.any(X < 0):
                raise ValueError('Chi2 method requires non-negative features')
            (scores, _) = chi2(X, y)
        elif self.method == 'f_classif':
            (scores, _) = f_classif(X, y)
        elif self.method == 'f_regression':
            (scores, _) = f_regression(X, y)
        elif self.method == 'mutual_info_classif':
            scores = mutual_info_classif(X, y, random_state=kwargs.get('random_state', 0))
        elif self.method == 'mutual_info_regression':
            scores = mutual_info_regression(X, y, random_state=kwargs.get('random_state', 0))
        elif self.method == 'correlation':
            scores = np.zeros(n_features)
            for i in range(n_features):
                if np.std(X[:, i]) == 0 or np.std(y) == 0:
                    scores[i] = 0.0
                else:
                    try:
                        (corr, _) = pearsonr(X[:, i], y)
                        scores[i] = abs(corr)
                    except:
                        scores[i] = 0.0
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self._feature_scores = scores
        if self.threshold is not None:
            self._selected_indices = np.where(scores >= self.threshold)[0]
        else:
            k = min(self.k, n_features) if self.k is not None else n_features
            k = max(1, k)
            if k >= n_features:
                self._selected_indices = np.arange(n_features)
            else:
                sorted_indices = np.argsort(scores)[::-1]
                self._selected_indices = sorted_indices[:k]
        self._selected_indices = np.sort(self._selected_indices)
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform the data by selecting features according to the fitted selector.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed feature set containing only selected features
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if self._selected_indices is None:
            raise ValueError('Transformer has not been fitted yet.')
        X_selected = data.features[:, self._selected_indices]
        selected_feature_names = [self._original_feature_names[i] for i in self._selected_indices] if self._original_feature_names else None
        selected_feature_types = [self._original_feature_types[i] for i in self._selected_indices] if self._original_feature_types else None
        selected_feature_set = FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})
        return selected_feature_set

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform (identity operation - returns original feature space representation).
        
        Since filter methods don't modify feature values, this operation reconstructs
        the full feature space by placing selected features back in their original positions.
        
        Parameters
        ----------
        data : FeatureSet
            Selected features to inverse transform
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Feature set with all original features (selected ones preserved, others filled with zeros)
        """
        if self._selected_indices is None:
            raise ValueError('Transformer has not been fitted yet.')
        if self._feature_scores is None:
            raise ValueError('Transformer has not been fitted yet.')
        n_features_original = len(self._feature_scores)
        n_samples = data.features.shape[0]
        X_full = np.zeros((n_samples, n_features_original))
        X_full[:, self._selected_indices] = data.features
        inverse_feature_set = FeatureSet(features=X_full, feature_names=self._original_feature_names, feature_types=self._original_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})
        return inverse_feature_set

    def get_support(self) -> np.ndarray:
        """
        Get the indices of the selected features.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating which features were selected
        """
        if self._selected_indices is None:
            return None
        mask = np.zeros(len(self._feature_scores), dtype=bool)
        mask[self._selected_indices] = True
        return mask

    def get_scores(self) -> np.ndarray:
        """
        Get the statistical scores for all features.
        
        Returns
        -------
        np.ndarray
            Array of scores for each feature
        """
        return self._feature_scores