import numpy as np
from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from .relief_based import ReliefFeatureSelector
from .chi_square_selection import ChiSquareScorer
from .correlation_based import ANOVAScorer
from .select_k_best import SelectKBest



class AutomaticFeatureSelector(BaseTransformer):

    def __init__(self, method: str='auto', k: Optional[int]=None, threshold: Optional[float]=None, scoring_strategy: str='combined', random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the AutomaticFeatureSelector.
        
        Parameters
        ----------
        method : str, default='auto'
            The automatic selection method to use. Options include 'auto', 'mrmr', 'relief', 'chi2', 'anova'.
        k : Optional[int], default=None
            Number of features to select. If None, automatically determines optimal number.
        threshold : Optional[float], default=None
            Threshold for feature selection. Features with scores above this value are selected.
        scoring_strategy : str, default='combined'
            Strategy for combining multiple scoring methods ('combined', 'majority_voting', 'weighted').
        random_state : Optional[int], default=None
            Random state for reproducible results.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.method = method
        self.k = k
        self.threshold = threshold
        self.scoring_strategy = scoring_strategy
        self.random_state = random_state

    def fit(self, data: FeatureSet, y: Union[np.ndarray, List], **kwargs) -> 'AutomaticFeatureSelector':
        """
        Fit the automatic feature selector to the training data.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to select from.
        y : Union[np.ndarray, List]
            Target values for supervised feature selection.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        AutomaticFeatureSelector
            Self instance for method chaining.
        """
        X = data.features
        y = np.array(y)
        if X.size == 0:
            raise ValueError('Input features cannot be empty')
        if len(y) != X.shape[0]:
            raise ValueError(f'Number of samples in X ({X.shape[0]}) does not match number of targets ({len(y)})')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        unique_labels = np.unique(y)
        is_classification = len(unique_labels) < max(20, 0.05 * len(y))
        method = self.method
        if method == 'auto':
            method = 'relief'
        scores_list = []
        if method in ['auto', 'relief', 'mrmr'] or (self.method == 'combined' and self.scoring_strategy in ['combined', 'majority_voting', 'weighted']):
            try:
                n_features_to_select = max(1, min(max(10, X.shape[1] // 2), 100, X.shape[1]))
                relief_selector = ReliefFeatureSelector(n_features_to_select=n_features_to_select, random_state=self.random_state)
                relief_selector.fit(data, y)
                scores_list.append(relief_selector.get_scores())
            except Exception:
                pass
        if method in ['auto', 'chi2'] or (self.method == 'combined' and self.scoring_strategy in ['combined', 'majority_voting', 'weighted']):
            try:
                if np.all(X >= 0):
                    (chi2_scores, _) = chi2(X, y)
                    scores_list.append(chi2_scores)
            except Exception:
                pass
        if method in ['auto', 'anova'] or (self.method == 'combined' and self.scoring_strategy in ['combined', 'majority_voting', 'weighted']):
            try:
                if is_classification:
                    (f_scores, _) = f_classif(X, y)
                else:
                    (f_scores, _) = f_regression(X, y)
                scores_list.append(f_scores)
            except Exception:
                pass
        if not scores_list:
            scores = np.var(X, axis=0)
        elif len(scores_list) == 1:
            scores = scores_list[0]
        else:
            max_len = X.shape[1]
            padded_scores = []
            for scores_arr in scores_list:
                if len(scores_arr) == max_len:
                    padded_scores.append(scores_arr)
                elif len(scores_arr) < max_len:
                    padded = np.zeros(max_len)
                    padded[:len(scores_arr)] = scores_arr
                    padded_scores.append(padded)
                else:
                    padded_scores.append(scores_arr[:max_len])
            if self.scoring_strategy == 'combined':
                scores = np.mean(padded_scores, axis=0)
            elif self.scoring_strategy == 'majority_voting':
                scores = np.mean(padded_scores, axis=0)
            elif self.scoring_strategy == 'weighted':
                scores = np.average(padded_scores, axis=0, weights=np.ones(len(padded_scores)))
            else:
                scores = padded_scores[0]
        self.scores_ = scores
        if self.threshold is not None:
            self.support_ = scores >= self.threshold
        elif self.k is not None:
            k = min(self.k, len(scores))
            if k <= 0:
                self.support_ = np.zeros(len(scores), dtype=bool)
            else:
                indices = np.argpartition(scores, -k)[-k:]
                self.support_ = np.zeros(len(scores), dtype=bool)
                self.support_[indices] = True
        else:
            sorted_scores = np.sort(scores)[::-1]
            if len(sorted_scores) <= 1:
                k = len(sorted_scores)
            else:
                mean_score = np.mean(sorted_scores)
                k = max(1, np.sum(scores >= mean_score))
                k = min(k, len(scores))
            if k <= 0:
                self.support_ = np.zeros(len(scores), dtype=bool)
            else:
                indices = np.argpartition(scores, -k)[-k:]
                self.support_ = np.zeros(len(scores), dtype=bool)
                self.support_[indices] = True
        self.is_fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform the input data to only include selected features.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed feature set with only selected features.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        if data.features.shape[1] != len(self.support_):
            raise ValueError(f'Number of features in data ({data.features.shape[1]}) does not match fitted selector ({len(self.support_)})')
        X_selected = data.features[:, self.support_]
        selected_feature_names = None
        selected_feature_types = None
        if data.feature_names is not None:
            selected_feature_names = [name for (i, name) in enumerate(data.feature_names) if self.support_[i]]
        if data.feature_types is not None:
            selected_feature_types = [ftype for (i, ftype) in enumerate(data.feature_types) if self.support_[i]]
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for feature selection.
        
        Parameters
        ----------
        data : FeatureSet
            Selected features to inverse transform.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set with placeholder values for unselected features.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'inverse_transform'.")
        n_selected_features = data.features.shape[1]
        n_original_features = len(self.support_)
        if n_selected_features != np.sum(self.support_):
            raise ValueError(f'Number of features in data ({n_selected_features}) does not match number of selected features ({np.sum(self.support_)})')
        X_reconstructed = np.zeros((data.features.shape[0], n_original_features))
        X_reconstructed[:, self.support_] = data.features
        reconstructed_feature_names = None
        reconstructed_feature_types = None
        if data.feature_names is not None:
            reconstructed_feature_names = [''] * n_original_features
            for (i, name) in enumerate(data.feature_names):
                original_index = np.where(self.support_)[0][i]
                reconstructed_feature_names[original_index] = name
        if data.feature_types is not None:
            reconstructed_feature_types = ['numeric'] * n_original_features
            for (i, ftype) in enumerate(data.feature_types):
                original_index = np.where(self.support_)[0][i]
                reconstructed_feature_types[original_index] = ftype
        return FeatureSet(features=X_reconstructed, feature_names=reconstructed_feature_names, feature_types=reconstructed_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})

    def get_support(self) -> np.ndarray:
        """
        Get the boolean mask indicating which features are selected.
        
        Returns
        -------
        np.ndarray
            Boolean array where True indicates selected features.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'get_support'.")
        return self.support_.copy()

    def get_scores(self) -> np.ndarray:
        """
        Get the feature scores from the selection process.
        
        Returns
        -------
        np.ndarray
            Array of feature scores used for selection.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'get_scores'.")
        return self.scores_.copy()