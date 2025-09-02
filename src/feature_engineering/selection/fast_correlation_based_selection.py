from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Union, List
from scipy.stats import pearsonr, spearmanr, kendalltau

class FastCorrelationBasedSelector(BaseTransformer):
    """
    A transformer that performs fast correlation-based feature selection.
    
    This selector identifies and removes features that are highly correlated with each other,
    retaining only those that contribute unique information. It uses an efficient algorithm
    to compute pairwise correlations and select a subset of features with low inter-correlations.
    
    The method is particularly useful for reducing multicollinearity in datasets while preserving
    predictive power. It works by:
    1. Computing a correlation matrix for all features
    2. Applying a correlation threshold to identify highly correlated pairs
    3. Selecting features to maximize the number of retained features while maintaining
       low inter-feature correlations
    
    Attributes
    ----------
    threshold : float
        Maximum allowed absolute correlation between features (default: 0.95)
    method : str
        Correlation computation method ('pearson', 'spearman', or 'kendall')
    selected_features_ : List[int]
        Indices of selected features after fitting
    correlation_matrix_ : np.ndarray
        Computed correlation matrix of the input features
    """

    def __init__(self, threshold: float=0.95, method: str='pearson', name: Optional[str]=None):
        """
        Initialize the FastCorrelationBasedSelector.
        
        Parameters
        ----------
        threshold : float, optional
            Maximum allowed absolute correlation between features (default: 0.95)
        method : str, optional
            Correlation computation method ('pearson', 'spearman', or 'kendall') (default: 'pearson')
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
        self.threshold = threshold
        self.method = method
        self.selected_features_: List[int] = []
        self.correlation_matrix_: Optional[np.ndarray] = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'FastCorrelationBasedSelector':
        """
        Fit the selector to the input data by computing correlations and selecting features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to fit the selector on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FastCorrelationBasedSelector
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.correlation_matrix_ = self._compute_correlation_matrix(X)
        self.selected_features_ = self._select_features()
        return self

    def _compute_correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the correlation matrix using the specified method.
        
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix
            
        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        n_features = X.shape[1]
        if n_features == 1:
            return np.array([[1.0]])
        corr_matrix = np.eye(n_features)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if self.method == 'pearson':
                    (corr, _) = pearsonr(X[:, i], X[:, j])
                elif self.method == 'spearman':
                    (corr, _) = spearmanr(X[:, i], X[:, j])
                elif self.method == 'kendall':
                    (corr, _) = kendalltau(X[:, i], X[:, j])
                if np.isnan(corr):
                    corr = 0.0
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        return corr_matrix

    def _select_features(self) -> List[int]:
        """
        Select features using a greedy approach to minimize correlations.
        
        Returns
        -------
        List[int]
            Indices of selected features
        """
        if self.correlation_matrix_ is None:
            raise RuntimeError('Correlation matrix not computed. Call fit() first.')
        n_features = self.correlation_matrix_.shape[0]
        if n_features == 0:
            return []
        if n_features == 1:
            return [0]
        selected = []
        candidates = list(range(n_features))
        while candidates:
            best_candidate = None
            best_max_corr = float('inf')
            for candidate in candidates:
                if not selected:
                    best_candidate = candidate
                    best_max_corr = 0
                    break
                max_corr = 0
                for selected_feature in selected:
                    corr = abs(self.correlation_matrix_[candidate, selected_feature])
                    max_corr = max(max_corr, corr)
                if max_corr < best_max_corr:
                    best_max_corr = max_corr
                    best_candidate = candidate
            if best_max_corr >= self.threshold and selected:
                break
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        return selected

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply feature selection to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to transform
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with only selected features
        """
        if not self.selected_features_:
            raise RuntimeError('Transformer has not been fitted yet. Call fit() first.')
        if isinstance(data, FeatureSet):
            selected_features = data.features[:, self.selected_features_]
            selected_feature_names = None
            if data.feature_names:
                selected_feature_names = [data.feature_names[i] for i in self.selected_features_]
            selected_feature_types = None
            if data.feature_types:
                selected_feature_types = [data.feature_types[i] for i in self.selected_features_]
            return FeatureSet(features=selected_features, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        else:
            return data[:, self.selected_features_]

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for this selector as information is lost during selection.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Same as input since inverse transform is not meaningful
        """
        raise NotImplementedError('Inverse transform is not supported for feature selection as information is lost during selection.')

    def get_selected_features(self) -> List[int]:
        """
        Get the indices of selected features.
        
        Returns
        -------
        List[int]
            Indices of selected features
        """
        if not self.selected_features_:
            raise RuntimeError('Transformer has not been fitted yet. Call fit() first.')
        return self.selected_features_.copy()

    def get_support(self, indices: bool=True) -> Union[List[bool], List[int]]:
        """
        Get a mask or indices of the selected features.
        
        Parameters
        ----------
        indices : bool, optional
            If True, return indices of selected features. If False, return boolean mask (default: True)
            
        Returns
        -------
        Union[List[bool], List[int]]
            Either boolean mask or indices of selected features
        """
        if not self.selected_features_:
            raise RuntimeError('Transformer has not been fitted yet. Call fit() first.')
        if indices:
            return self.selected_features_.copy()
        else:
            n_features = self.correlation_matrix_.shape[0] if self.correlation_matrix_ is not None else len(self.selected_features_)
            mask = [False] * n_features
            for idx in self.selected_features_:
                if 0 <= idx < n_features:
                    mask[idx] = True
            return mask