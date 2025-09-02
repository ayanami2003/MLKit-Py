from typing import Optional, Union, List, Dict, Any
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.component_config import ComponentConfig
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
import warnings

class RecursiveFeatureEliminationCV(BaseTransformer):

    def __init__(self, estimator: object, step: Union[int, float]=1, cv: Optional[Union[int, object]]=None, scoring: Optional[Union[str, callable]]=None, min_features_to_select: int=1, n_jobs: Optional[int]=None, verbose: int=0, name: Optional[str]=None):
        """
        Initialize the RecursiveFeatureEliminationCV transformer.
        
        Parameters
        ----------
        estimator : object
            A supervised learning estimator with a fit method that provides
            information about feature importance.
        step : int or float, default=1
            Number or percentage of features to remove at each iteration.
        cv : int, cross-validation generator or iterable, default=None
            Cross-validation strategy for evaluating feature subsets.
        scoring : str, callable or None, default=None
            Scoring function to use for evaluation.
        min_features_to_select : int, default=1
            Minimum number of features to retain.
        n_jobs : int, default=None
            Number of parallel jobs to run.
        verbose : int, default=0
            Verbosity level.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._support = None
        self._ranking = None
        self._scores = None
        self._n_features_selected = None
        self._feature_names = None
        self._cv_scores = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> 'RecursiveFeatureEliminationCV':
        """
        Fit the RFE transformer with cross-validation to find optimal features.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Training data with shape (n_samples, n_features).
        y : np.ndarray
            Target values with shape (n_samples,).
        **kwargs : dict
            Additional parameters passed to the estimator's fit method.
            
        Returns
        -------
        RecursiveFeatureEliminationCV
            Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        else:
            X = data
            self._feature_names = None
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y.ndim != 1 and (y.ndim == 2 and y.shape[1] != 1):
            raise ValueError('y must be a 1D array or 2D array with single column')
        (n_samples, n_features) = X.shape
        if self.min_features_to_select < 1 or self.min_features_to_select > n_features:
            raise ValueError('min_features_to_select must be between 1 and number of features')
        if isinstance(self.step, int):
            if self.step < 1:
                raise ValueError('step must be >= 1 when integer')
        elif isinstance(self.step, float):
            if self.step <= 0 or self.step >= 1:
                raise ValueError('step must be in (0, 1) when float')
        else:
            raise ValueError('step must be int or float')
        if self.cv is None:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=5)
        else:
            cv = self.cv
        if self.scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) > 2 else 'r2'
        else:
            scoring = self.scoring
        feature_indices = np.arange(n_features)
        scores_dict = {}
        rankings = np.ones(n_features, dtype=int)
        current_rank = 1
        feature_sets_and_scores = []
        while len(feature_indices) >= self.min_features_to_select:
            current_X = X[:, feature_indices]
            try:
                cv_results = cross_validate(self.estimator, current_X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs, error_score='raise')
                mean_score = np.mean(cv_results['test_score'])
            except Exception as e:
                if self.verbose > 0:
                    print(f'Warning: Cross-validation failed with error: {e}')
                mean_score = -np.inf
            scores_dict[len(feature_indices)] = mean_score
            feature_sets_and_scores.append((len(feature_indices), feature_indices.copy(), mean_score))
            if self.verbose > 0:
                print(f'Features: {len(feature_indices)}, Score: {mean_score:.4f}')
            if len(feature_indices) <= self.min_features_to_select:
                break
            estimator_clone = self._clone_estimator()
            try:
                estimator_clone.fit(current_X, y, **kwargs)
            except Exception as e:
                raise RuntimeError(f'Estimator failed to fit: {e}')
            if hasattr(estimator_clone, 'coef_'):
                if len(estimator_clone.coef_.shape) > 1 and estimator_clone.coef_.shape[0] > 1:
                    importances = np.mean(np.abs(estimator_clone.coef_), axis=0)
                else:
                    importances = np.abs(estimator_clone.coef_).ravel()
            elif hasattr(estimator_clone, 'feature_importances_'):
                importances = estimator_clone.feature_importances_
            else:
                raise ValueError('Estimator must have either coef_ or feature_importances_ attribute')
            if isinstance(self.step, int):
                n_features_to_remove = min(self.step, len(feature_indices) - self.min_features_to_select)
            else:
                n_features_to_remove = max(1, int(len(feature_indices) * self.step))
                n_features_to_remove = min(n_features_to_remove, len(feature_indices) - self.min_features_to_select)
            least_important_indices = np.argsort(importances)[:n_features_to_remove]
            for idx in least_important_indices:
                global_idx = feature_indices[idx]
                rankings[global_idx] = current_rank + 1
            feature_indices = np.delete(feature_indices, least_important_indices)
            current_rank += 1
        if scores_dict:
            best_n_features = max(scores_dict.keys(), key=lambda k: scores_dict[k])
            best_score = scores_dict[best_n_features]
        else:
            best_n_features = self.min_features_to_select
            best_score = -np.inf
        best_feature_indices = None
        best_cv_scores = None
        for (n_feat, feat_indices, score) in feature_sets_and_scores:
            if n_feat == best_n_features:
                best_feature_indices = feat_indices
                best_X = X[:, best_feature_indices]
                try:
                    cv_results = cross_validate(self.estimator, best_X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs, error_score='raise')
                    best_cv_scores = cv_results['test_score']
                except Exception:
                    if hasattr(cv, 'n_splits'):
                        n_folds = cv.n_splits
                    elif isinstance(cv, int):
                        n_folds = cv
                    else:
                        n_folds = 5
                    best_cv_scores = np.array([best_score] * n_folds)
                break
        if best_feature_indices is None and feature_sets_and_scores:
            best_item = max(feature_sets_and_scores, key=lambda x: x[2])
            best_feature_indices = best_item[1]
            best_score = best_item[2]
            best_X = X[:, best_feature_indices]
            try:
                cv_results = cross_validate(self.estimator, best_X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs, error_score='raise')
                best_cv_scores = cv_results['test_score']
            except Exception:
                if hasattr(cv, 'n_splits'):
                    n_folds = cv.n_splits
                elif isinstance(cv, int):
                    n_folds = cv
                else:
                    n_folds = 5
                best_cv_scores = np.array([best_score] * n_folds)
        if best_feature_indices is None:
            best_feature_indices = np.arange(n_features)
            try:
                cv_results = cross_validate(self.estimator, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs, error_score='raise')
                best_cv_scores = cv_results['test_score']
            except Exception:
                if hasattr(cv, 'n_splits'):
                    n_folds = cv.n_splits
                elif isinstance(cv, int):
                    n_folds = cv
                else:
                    n_folds = 5
                best_cv_scores = np.array([best_score] * n_folds)
        self._support = np.zeros(n_features, dtype=bool)
        self._support[best_feature_indices] = True
        self._n_features_selected = len(best_feature_indices)
        self._scores = scores_dict
        self.n_features_ = n_features
        self._ranking = rankings
        for idx in best_feature_indices:
            self._ranking[idx] = 1
        self._cv_scores = best_cv_scores
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the input data by selecting only the relevant features.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform with shape (n_samples, n_features).
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Transformed data with selected features.
        """
        if self._support is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        else:
            X = data
            feature_names = self._feature_names
            feature_types = None
            sample_ids = None
            metadata = {}
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'X has {X.shape[1]} features, but transformer was fitted with {self.n_features_} features.')
        X_selected = X[:, self._support]
        selected_feature_names = None
        if feature_names is not None:
            selected_feature_names = [feature_names[i] for (i, support) in enumerate(self._support) if support]
        elif self._feature_names is not None:
            selected_feature_names = [self._feature_names[i] for (i, support) in enumerate(self._support) if support]
        selected_feature_types = None
        if feature_types is not None:
            selected_feature_types = [feature_types[i] for (i, support) in enumerate(self._support) if support]
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reverse the transformation by returning features to original space.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Transformed data with selected features.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Data with features in original space (zero-padded for eliminated features).
        """
        if self._support is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X_transformed = data.features
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        else:
            X_transformed = data
            sample_ids = None
            metadata = {}
        if not isinstance(X_transformed, np.ndarray):
            X_transformed = np.array(X_transformed)
        if X_transformed.shape[1] != self._n_features_selected:
            raise ValueError(f'X has {X_transformed.shape[1]} features, but {self._n_features_selected} were selected.')
        X_original = np.zeros((X_transformed.shape[0], self.n_features_), dtype=X_transformed.dtype)
        X_original[:, self._support] = X_transformed
        original_feature_names = self._feature_names
        original_feature_types = None
        return FeatureSet(features=X_original, feature_names=original_feature_names, feature_types=original_feature_types, sample_ids=sample_ids, metadata=metadata)

    def get_support(self, indices: bool=False) -> Union[np.ndarray, List[bool]]:
        """
        Get a mask or indices of the features selected.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return indices instead of a boolean mask.
            
        Returns
        -------
        np.ndarray or List[bool]
            Boolean mask or indices of selected features.
        """
        if self._support is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if indices:
            return np.where(self._support)[0]
        else:
            return self._support.copy()

    def get_ranking(self) -> np.ndarray:
        """
        Get the ranking of features based on their elimination order.
        
        Features with rank 1 are selected, higher ranks indicate earlier elimination.
        
        Returns
        -------
        np.ndarray
            Array of feature rankings with shape (n_features,).
        """
        if self._ranking is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self._ranking.copy()

    def get_scores(self) -> List[float]:
        """
        Get cross-validation scores for different numbers of features.
        
        Returns
        -------
        List[float]
            List of CV scores ordered by number of features (ascending).
        """
        if self._cv_scores is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self._cv_scores.tolist()

    def get_n_features_selected(self) -> int:
        """
        Get the number of features selected.
        
        Returns
        -------
        int
            Number of selected features.
        """
        if self._n_features_selected is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self._n_features_selected

    def get_feature_names_out(self) -> Optional[List[str]]:
        """
        Get names of selected features.
        
        Returns
        -------
        Optional[List[str]]
            Names of selected features if available, None otherwise.
        """
        if self._support is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if self._feature_names is not None:
            return [name for (i, name) in enumerate(self._feature_names) if self._support[i]]
        else:
            return None

    def _clone_estimator(self):
        """Create a fresh copy of the estimator."""
        from sklearn.base import clone
        return clone(self.estimator)