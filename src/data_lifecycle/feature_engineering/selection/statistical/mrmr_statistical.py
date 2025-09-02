from typing import Optional, Union, List
import numpy as np
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class MRMRFeatureSelector(BaseTransformer):

    def __init__(self, k: Optional[int]=None, method: str='MIQ', threshold: Optional[float]=None, discrete_features: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the mRMR feature selector.
        
        Parameters
        ----------
        k : Optional[int], default=None
            Number of features to select. Exactly one of `k` or `threshold` must be specified.
        method : str, default='MIQ'
            Method for computing relevance and redundancy. Supported methods:
            - 'MIQ': Mutual Information Quotient
            - 'MID': Mutual Information Difference
            - 'F': F-statistic
            - 'CORR': Correlation-based
        threshold : Optional[float], default=None
            Threshold for feature selection. Exactly one of `k` or `threshold` must be specified.
        discrete_features : bool, default=False
            Whether features should be treated as discrete variables for mutual information calculation.
        random_state : Optional[int], default=None
            Random state for reproducibility.
        name : Optional[str], default=None
            Name of the transformer instance.
            
        Raises
        ------
        ValueError
            If neither k nor threshold is specified, or if both are specified.
        """
        super().__init__(name=name)
        if (k is None) == (threshold is None):
            raise ValueError("Exactly one of 'k' or 'threshold' must be specified.")
        self.k = k
        self.method = method
        self.threshold = threshold
        self.discrete_features = discrete_features
        self.random_state = random_state
        self._selected_features_mask = None
        self._feature_scores = None
        self.scores_ = None
        self.support_ = None

    def _calculate_relevance(self, X: np.ndarray, y: Union[np.ndarray, List]) -> np.ndarray:
        """
        Calculate relevance scores for features based on the specified method.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : Union[np.ndarray, List]
            Target values.
            
        Returns
        -------
        np.ndarray
            Relevance scores for each feature.
        """
        n_features = X.shape[1]
        relevance_scores = np.zeros(n_features)
        if self.method in ['MIQ', 'MID']:
            unique_y = np.unique(y)
            is_classification = len(unique_y) < max(10, 0.05 * len(y))
            try:
                if is_classification:
                    relevance_scores = mutual_info_classif(X, y, discrete_features=self.discrete_features, random_state=self.random_state)
                else:
                    relevance_scores = mutual_info_regression(X, y, discrete_features=self.discrete_features, random_state=self.random_state)
            except Exception:
                for i in range(n_features):
                    try:
                        corr = np.corrcoef(X[:, i], y)[0, 1]
                        relevance_scores[i] = abs(corr) if not np.isnan(corr) else 0
                    except:
                        relevance_scores[i] = 0
        elif self.method == 'F':
            for i in range(n_features):
                try:
                    unique_labels = np.unique(y)
                    if len(unique_labels) > 1:
                        groups = [X[y == label, i] for label in unique_labels]
                        groups = [group for group in groups if len(group) > 0]
                        if len(groups) > 1 and all((len(group) > 1 for group in groups)):
                            (f_stat, _) = f_oneway(*groups)
                            relevance_scores[i] = f_stat if not np.isnan(f_stat) else 0
                        else:
                            relevance_scores[i] = 0
                    else:
                        relevance_scores[i] = 0
                except:
                    relevance_scores[i] = 0
        elif self.method == 'CORR':
            for i in range(n_features):
                try:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    relevance_scores[i] = abs(corr) if not np.isnan(corr) else 0
                except:
                    relevance_scores[i] = 0
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        return relevance_scores

    def _calculate_redundancy(self, X: np.ndarray, selected_features: List[int], candidate_feature: int) -> float:
        """
        Calculate redundancy of candidate feature with already selected features.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        selected_features : List[int]
            Indices of already selected features.
        candidate_feature : int
            Index of candidate feature.
            
        Returns
        -------
        float
            Redundancy score.
        """
        if not selected_features:
            return 0.0
        redundancies = []
        for selected_idx in selected_features:
            if self.method in ['MIQ', 'MID']:
                try:
                    x_selected = X[:, selected_idx].reshape(-1, 1)
                    x_candidate = X[:, candidate_feature]
                    if len(x_selected) > 1 and len(x_candidate) > 1:
                        mi = mutual_info_regression(x_selected, x_candidate, discrete_features=self.discrete_features, random_state=self.random_state)
                        redundancies.append(mi[0] if len(mi) > 0 else 0)
                    else:
                        corr = np.corrcoef(X[:, selected_idx], X[:, candidate_feature])[0, 1]
                        redundancies.append(abs(corr) if not np.isnan(corr) else 0)
                except Exception:
                    corr = np.corrcoef(X[:, selected_idx], X[:, candidate_feature])[0, 1]
                    redundancies.append(abs(corr) if not np.isnan(corr) else 0)
            elif self.method == 'F':
                try:
                    corr = np.corrcoef(X[:, selected_idx], X[:, candidate_feature])[0, 1]
                    redundancies.append(abs(corr) if not np.isnan(corr) else 0)
                except:
                    redundancies.append(0.0)
            elif self.method == 'CORR':
                try:
                    corr = np.corrcoef(X[:, selected_idx], X[:, candidate_feature])[0, 1]
                    redundancies.append(abs(corr) if not np.isnan(corr) else 0)
                except:
                    redundancies.append(0.0)
        return np.mean(redundancies) if redundancies else 0.0

    def fit(self, data: FeatureSet, y: Union[np.ndarray, List], **kwargs) -> 'MRMRFeatureSelector':
        """
        Fit the mRMR feature selector to training data.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to fit the selector on.
        y : Union[np.ndarray, List]
            Target values for supervised feature selection.
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        MRMRFeatureSelector
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If data is not a FeatureSet or if shapes don't match.
        """
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet instance')
        X = data.features
        if X.shape[0] != len(y):
            raise ValueError('Number of samples in X and y must match')
        n_features = X.shape[1]
        feature_indices = list(range(n_features))
        if self.random_state is not None:
            np.random.seed(self.random_state)
        relevance_scores = self._calculate_relevance(X, y)
        selected_features = []
        remaining_features = feature_indices.copy()
        feature_scores = np.zeros(n_features)
        if np.all(relevance_scores == 0):
            if n_features > 0:
                selected_features.append(0)
                remaining_features.remove(0)
                feature_scores[0] = 0
        else:
            first_feature_idx = np.argmax(relevance_scores)
            selected_features.append(first_feature_idx)
            remaining_features.remove(first_feature_idx)
            feature_scores[first_feature_idx] = relevance_scores[first_feature_idx]
        n_select = self.k if self.k is not None else n_features
        for _ in range(1, min(n_select, n_features)):
            best_score = -np.inf
            best_feature = None
            for feature_idx in remaining_features:
                redundancy = self._calculate_redundancy(X, selected_features, feature_idx)
                if self.method == 'MIQ':
                    if redundancy == 0:
                        mrmr_score = np.inf if relevance_scores[feature_idx] > 0 else 0
                    else:
                        mrmr_score = relevance_scores[feature_idx] / redundancy
                elif self.method == 'MID':
                    mrmr_score = relevance_scores[feature_idx] - redundancy
                elif self.method == 'F':
                    mrmr_score = relevance_scores[feature_idx] / (1 + redundancy)
                elif self.method == 'CORR':
                    if redundancy == 0:
                        mrmr_score = np.inf if relevance_scores[feature_idx] > 0 else 0
                    else:
                        mrmr_score = relevance_scores[feature_idx] / redundancy
                else:
                    raise ValueError(f'Unsupported method: {self.method}')
                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_feature = feature_idx
            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                feature_scores[best_feature] = best_score
            else:
                break
        self._selected_features_mask = np.zeros(n_features, dtype=bool)
        self._selected_features_mask[selected_features] = True
        self._feature_scores = feature_scores
        self.scores_ = feature_scores.copy()
        if self.threshold is not None:
            threshold_mask = feature_scores >= self.threshold
            self.support_ = threshold_mask
        else:
            self.support_ = self._selected_features_mask.copy()
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply feature selection to input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed feature set containing only selected features.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if self.support_ is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet instance')
        if data.features.shape[1] != len(self.support_):
            raise ValueError(f'Feature dimension mismatch: expected {len(self.support_)} features, got {data.features.shape[1]}')
        selected_features = data.features[:, self.support_]
        selected_feature_names = None
        selected_feature_types = None
        selected_sample_ids = data.sample_ids
        if data.feature_names:
            selected_feature_names = [name for (i, name) in enumerate(data.feature_names) if self.support_[i]]
        if data.feature_types:
            selected_feature_types = [ftype for (i, ftype) in enumerate(data.feature_types) if self.support_[i]]
        return FeatureSet(features=selected_features, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=selected_sample_ids, metadata=data.metadata.copy() if data.metadata else {})

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform (not supported).
        
        Parameters
        ----------
        data : FeatureSet
            Transformed features to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Original feature set with zeros for removed features.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transform is not supported for MRMRFeatureSelector.')

    def get_support(self) -> np.ndarray:
        """
        Get boolean mask of selected features.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating which features were selected.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if self.support_ is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'get_support'.")
        return self.support_.copy()

    def get_scores(self) -> np.ndarray:
        """
        Get scores for all features.
        
        Returns
        -------
        np.ndarray
            Array of scores for each feature.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if self._feature_scores is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'get_scores'.")
        return self._feature_scores.copy()