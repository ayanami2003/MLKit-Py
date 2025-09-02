from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ICAPFeatureSelector(BaseTransformer):
    """
    Implements the ICAP (Interaction Capping) feature selection method.
    
    ICAP is a multivariate feature selection technique that evaluates feature relevance
    while considering feature redundancy and interaction effects. It aims to select
    a subset of features that are highly relevant to the target variable while minimizing
    redundancy among selected features and accounting for feature interactions.
    
    This selector extends BaseTransformer and can be used in ML pipelines for feature selection.
    
    Parameters
    ----------
    k : int, optional (default=10)
        Number of features to select. If None, a stopping criterion based on scores is used.
    alpha : float, optional (default=0.01)
        Significance level for feature inclusion.
    max_iterations : int, optional (default=100)
        Maximum number of iterations for the selection process.
    name : str, optional
        Name of the transformer instance.
        
    Attributes
    ----------
    selected_features_ : List[int]
        Indices of selected features.
    feature_scores_ : np.ndarray
        Scores of all features based on ICAP criterion.
    """

    def __init__(self, k: Optional[int]=10, alpha: float=0.01, max_iterations: int=100, name: Optional[str]=None):
        super().__init__(name=name)
        self.k = k
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.selected_features_: List[int] = []
        self.feature_scores_: np.ndarray = np.array([])

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'ICAPFeatureSelector':
        """
        Fit the ICAP feature selector on the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to select from. If FeatureSet, uses the features attribute.
        y : np.ndarray
            Target values for supervised feature selection.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        ICAPFeatureSelector
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in X and y must match')
        (n_samples, n_features) = X.shape
        relevance_scores = self._compute_relevance(X, y)
        self.selected_features_ = []
        self.feature_scores_ = np.zeros(n_features)
        remaining_features = list(range(n_features))
        if self.k is None:
            k = np.sum(relevance_scores > self.alpha)
            k = max(1, min(k, n_features))
        else:
            k = min(self.k, n_features)
        for iteration in range(min(k, self.max_iterations)):
            if not remaining_features:
                break
            best_score = -np.inf
            best_feature = None
            for feature_idx in remaining_features:
                icap_score = self._calculate_icap_score(feature_idx, X, y, relevance_scores, self.selected_features_)
                if icap_score > best_score:
                    best_score = icap_score
                    best_feature = feature_idx
            if best_feature is None or best_score <= 0:
                break
            self.selected_features_.append(best_feature)
            self.feature_scores_[best_feature] = best_score
            remaining_features.remove(best_feature)
            if self.k is None and best_score < self.alpha:
                break
        return self

    def _compute_relevance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute relevance scores between features and target using mutual information approximation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
            
        Returns
        -------
        np.ndarray
            Relevance scores for each feature
        """
        n_features = X.shape[1]
        relevance_scores = np.zeros(n_features)
        if len(np.unique(y)) > 20:
            for i in range(n_features):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                relevance_scores[i] = abs(corr) if not np.isnan(corr) else 0
        else:
            for i in range(n_features):
                relevance_scores[i] = self._f_score(X[:, i], y)
        return relevance_scores

    def _f_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute F-score for feature-target relationship.
        
        Parameters
        ----------
        x : np.ndarray
            Single feature values
        y : np.ndarray
            Target values
            
        Returns
        -------
        float
            F-score for the feature
        """
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0
        overall_mean = np.mean(x)
        between_class_sum = 0.0
        within_class_sum = 0.0
        for cls in classes:
            x_cls = x[y == cls]
            class_mean = np.mean(x_cls)
            between_class_sum += len(x_cls) * (class_mean - overall_mean) ** 2
            within_class_sum += np.sum((x_cls - class_mean) ** 2)
        if within_class_sum == 0:
            return 0.0
        f_score = between_class_sum / (len(classes) - 1) / (within_class_sum / (len(x) - len(classes)))
        return f_score if not np.isnan(f_score) else 0.0

    def _calculate_icap_score(self, feature_idx: int, X: np.ndarray, y: np.ndarray, relevance_scores: np.ndarray, selected_features: List[int]) -> float:
        """
        Calculate ICAP score for a feature given already selected features.
        
        Parameters
        ----------
        feature_idx : int
            Index of the feature to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        relevance_scores : np.ndarray
            Precomputed relevance scores for all features
        selected_features : List[int]
            Indices of already selected features
            
        Returns
        -------
        float
            ICAP score for the feature
        """
        rel = relevance_scores[feature_idx]
        if not selected_features:
            return rel
        redundancy_sum = 0.0
        interaction_sum = 0.0
        for selected_idx in selected_features:
            if X.dtype in [np.float32, np.float64]:
                corr = np.corrcoef(X[:, feature_idx], X[:, selected_idx])[0, 1]
                redundancy = abs(corr) if not np.isnan(corr) else 0
            else:
                redundancy = self._conditional_redundancy(X[:, feature_idx], X[:, selected_idx], y)
            redundancy_sum += redundancy
            interaction = self._interaction_gain(feature_idx, selected_idx, X, y)
            interaction_sum += max(0, interaction)
        num_selected = len(selected_features)
        if num_selected > 0:
            redundancy_term = redundancy_sum / num_selected
            interaction_term = interaction_sum / num_selected
        else:
            redundancy_term = 0
            interaction_term = 0
        icap_score = rel - redundancy_term + interaction_term
        return max(0, icap_score)

    def _conditional_redundancy(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
        """
        Compute conditional redundancy between two features given the target.
        
        Parameters
        ----------
        x1 : np.ndarray
            First feature
        x2 : np.ndarray
            Second feature
        y : np.ndarray
            Target values
            
        Returns
        -------
        float
            Conditional redundancy measure
        """
        unique_y = np.unique(y)
        redundancy = 0.0
        for cls in unique_y:
            mask = y == cls
            if np.sum(mask) > 1:
                x1_cls = x1[mask]
                x2_cls = x2[mask]
                if len(np.unique(x1_cls)) > 1 and len(np.unique(x2_cls)) > 1:
                    corr = np.corrcoef(x1_cls, x2_cls)[0, 1]
                    redundancy += abs(corr) * np.sum(mask) / len(y) if not np.isnan(corr) else 0
        return redundancy

    def _interaction_gain(self, f1_idx: int, f2_idx: int, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute interaction gain between two features for predicting the target.
        
        Parameters
        ----------
        f1_idx : int
            Index of first feature
        f2_idx : int
            Index of second feature
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
            
        Returns
        -------
        float
            Interaction gain measure
        """
        interaction_feature = X[:, f1_idx] * X[:, f2_idx]
        if len(np.unique(y)) > 20:
            corr = np.corrcoef(interaction_feature, y)[0, 1]
            interaction_relevance = abs(corr) if not np.isnan(corr) else 0
        else:
            interaction_relevance = self._f_score(interaction_feature, y)
        if len(np.unique(y)) > 20:
            corr1 = np.corrcoef(X[:, f1_idx], y)[0, 1]
            corr2 = np.corrcoef(X[:, f2_idx], y)[0, 1]
            rel1 = abs(corr1) if not np.isnan(corr1) else 0
            rel2 = abs(corr2) if not np.isnan(corr2) else 0
        else:
            rel1 = self._f_score(X[:, f1_idx], y)
            rel2 = self._f_score(X[:, f2_idx], y)
        individual_predictiveness = (rel1 + rel2) / 2
        interaction_gain = interaction_relevance - individual_predictiveness
        return interaction_gain

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply feature selection to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to transform by selecting features.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with only selected features.
        """
        if not hasattr(self, 'selected_features_') or len(self.selected_features_) == 0:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X_selected = data.features[:, self.selected_features_]
            selected_feature_names = None
            if data.feature_names is not None:
                selected_feature_names = [data.feature_names[i] for i in self.selected_features_]
            selected_feature_types = None
            if data.feature_types is not None:
                selected_feature_types = [data.feature_types[i] for i in self.selected_features_]
            return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})
        else:
            return data[:, self.selected_features_]

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for feature selection methods.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to inverse transform (not used).
        **kwargs : dict
            Additional parameters (not used).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Same as input since inverse transformation is not applicable.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transform is not meaningful for feature selection.
        """
        raise NotImplementedError('ICAPFeatureSelector does not support inverse transformation.')

    def get_support(self, indices: bool=True) -> Union[List[int], np.ndarray]:
        """
        Get the indices of selected features.
        
        Parameters
        ----------
        indices : bool, optional (default=True)
            If True, return indices of selected features. If False, return a mask.
            
        Returns
        -------
        Union[List[int], np.ndarray]
            Either list of selected feature indices or boolean mask.
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if indices:
            return self.selected_features_.copy()
        else:
            mask = np.zeros(len(self.feature_scores_), dtype=bool)
            mask[self.selected_features_] = True
            return mask