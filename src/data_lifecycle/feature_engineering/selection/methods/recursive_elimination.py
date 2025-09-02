import numpy as np
from typing import Optional, Union, Callable
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate
from copy import deepcopy
import warnings

class RecursiveFeatureElimination(BaseTransformer):

    def __init__(self, estimator, n_features_to_select: Optional[int]=None, step: Union[int, float]=1, verbose: int=0, name: Optional[str]=None):
        super().__init__(name=name)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'RecursiveFeatureElimination':
        """
        Fit the RFE model and determine the optimal features to keep.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            The input data containing features and labels. Must have both features and labels.
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        RecursiveFeatureElimination
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
            y = kwargs.get('y')
            if y is None and data.metadata and ('labels' in data.metadata):
                y = data.metadata['labels']
            feature_names = data.feature_names
            feature_types = data.feature_types
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            y = data.labels
            feature_names = data.feature_names
            feature_types = None
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if y is None:
            raise ValueError('Input data must contain labels for supervised feature selection')
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.ndim != 2:
            raise ValueError('Features must be a 2D array')
        (n_samples, n_features) = X.shape
        if self.n_features_to_select is None:
            self.n_features_to_select = max(1, n_features // 2)
        if self.n_features_to_select > n_features:
            raise ValueError(f'n_features_to_select ({self.n_features_to_select}) cannot be larger than the number of features ({n_features})')
        self.original_feature_names_ = feature_names
        self.original_feature_types_ = feature_types
        self.original_n_features_ = n_features
        feature_indices = np.arange(n_features)
        while len(feature_indices) > self.n_features_to_select:
            if isinstance(self.step, int) and self.step >= 1:
                n_features_to_remove = min(self.step, len(feature_indices) - self.n_features_to_select)
            elif isinstance(self.step, float) and 0.0 < self.step < 1.0:
                n_features_to_remove = max(1, int(len(feature_indices) * self.step))
                n_features_to_remove = min(n_features_to_remove, len(feature_indices) - self.n_features_to_select)
            else:
                raise ValueError('step must be either an integer >= 1 or a float in (0.0, 1.0)')
            current_X = X[:, feature_indices]
            if self.verbose > 0:
                print(f'Fitting estimator with {len(feature_indices)} features')
            estimator_clone = self._clone_estimator()
            estimator_clone.fit(current_X, y)
            if hasattr(estimator_clone, 'coef_'):
                if len(estimator_clone.coef_.shape) > 1 and estimator_clone.coef_.shape[0] > 1:
                    importances = np.mean(np.abs(estimator_clone.coef_), axis=0)
                else:
                    importances = np.abs(estimator_clone.coef_).ravel()
            elif hasattr(estimator_clone, 'feature_importances_'):
                importances = estimator_clone.feature_importances_
            else:
                raise ValueError('Estimator must have either coef_ or feature_importances_ attribute')
            least_important_indices = np.argsort(importances)[:n_features_to_remove]
            feature_indices = np.delete(feature_indices, least_important_indices)
            if self.verbose > 0:
                print(f'Eliminated {n_features_to_remove} features. Remaining: {len(feature_indices)}')
        self.selected_features_ = feature_indices.tolist()
        if feature_names is not None:
            self.selected_feature_names_ = [feature_names[i] for i in self.selected_features_]
        else:
            self.selected_feature_names_ = None
        if feature_types is not None:
            self.selected_feature_types_ = [feature_types[i] for i in self.selected_features_]
        else:
            self.selected_feature_types_ = None
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reduce data to the selected features.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            The input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with only the selected features
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
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
        X_selected = X[:, self.selected_features_]
        selected_feature_names = None
        if feature_names is not None:
            selected_feature_names = [feature_names[i] for i in self.selected_features_]
        elif hasattr(self, 'selected_feature_names_') and self.selected_feature_names_ is not None:
            selected_feature_names = self.selected_feature_names_
        selected_feature_types = None
        if feature_types is not None:
            selected_feature_types = [feature_types[i] for i in self.selected_features_]
        elif hasattr(self, 'selected_feature_types_') and self.selected_feature_types_ is not None:
            selected_feature_types = self.selected_feature_types_
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the transformation, adding back the eliminated features as zeros.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            The transformed data to inverse transform
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Data with eliminated features added back as zeros
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X_transformed = data.features
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        elif isinstance(data, DataBatch):
            X_transformed = np.array(data.data)
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        n_selected_features = len(self.selected_features_)
        if X_transformed.shape[1] != n_selected_features:
            raise ValueError(f'Input data has {X_transformed.shape[1]} features, but {n_selected_features} were selected during fitting.')
        n_original_features = self.original_n_features_
        X_original = np.zeros((X_transformed.shape[0], n_original_features), dtype=X_transformed.dtype)
        X_original[:, self.selected_features_] = X_transformed
        original_feature_names = self.original_feature_names_
        original_feature_types = self.original_feature_types_
        return FeatureSet(features=X_original, feature_names=original_feature_names, feature_types=original_feature_types, sample_ids=sample_ids, metadata=metadata)

    def _clone_estimator(self):
        """Create a fresh copy of the estimator."""
        from sklearn.base import clone
        return clone(self.estimator)

class RecursiveFeatureEliminationCV(BaseTransformer):

    def __init__(self, estimator, step: Union[int, float]=1, cv: int=5, scoring: Optional[Union[str, callable]]=None, verbose: int=0, n_jobs: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_: Optional[np.ndarray] = None
        self.ranking_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.scores_: Optional[np.ndarray] = None

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'RecursiveFeatureEliminationCV':
        """
        Fit the RFECV model and determine the optimal features to keep using cross-validation.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            The input data containing features and labels. Must have both features and labels.
            For DataBatch, labels must not be None.
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        RecursiveFeatureEliminationCV
            Fitted transformer instance
            
        Raises
        ------
        ValueError
            If data does not contain labels or if parameters are invalid
        """
        if isinstance(data, FeatureSet):
            if data.metadata is None or 'labels' not in data.metadata:
                raise ValueError('FeatureSet must contain labels in metadata')
            X = data.features
            y = data.metadata['labels']
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            if not data.is_labeled():
                raise ValueError('DataBatch must contain labels')
            X = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data
            y = np.array(data.labels) if not isinstance(data.labels, np.array) else data.labels
            feature_names = data.feature_names
        else:
            raise TypeError('Data must be either FeatureSet or DataBatch')
        if X.ndim != 2:
            raise ValueError('Features must be a 2D array')
        (n_samples, n_features) = X.shape
        if isinstance(self.step, int):
            if self.step < 1:
                raise ValueError('Step must be >= 1 when integer')
        elif isinstance(self.step, float):
            if not 0.0 < self.step < 1.0:
                raise ValueError('Step must be in (0.0, 1.0) when float')
        else:
            raise TypeError('Step must be int or float')
        supports = []
        scores = []
        rankings = np.ones(n_features, dtype=int)
        current_features = np.arange(n_features)
        current_support = np.ones(n_features, dtype=bool)
        iteration = 0
        while np.sum(current_support) > 1:
            X_subset = X[:, current_features]
            try:
                estimator_clone = deepcopy(self.estimator)
                if self.scoring is not None:
                    scorer = get_scorer(self.scoring)
                else:
                    scorer = None
                cv_results = cross_validate(estimator_clone, X_subset, y, cv=self.cv, scoring=scorer, n_jobs=self.n_jobs, return_train_score=False)
                mean_score = np.mean(cv_results['test_score'])
                scores.append(mean_score)
                supports.append(current_support.copy())
                if self.verbose > 0:
                    print(f'Iteration {iteration}: {np.sum(current_support)} features, score={mean_score:.4f}')
                estimator_clone.fit(X_subset, y)
                if hasattr(estimator_clone, 'coef_'):
                    importances = np.abs(estimator_clone.coef_)
                elif hasattr(estimator_clone, 'feature_importances_'):
                    importances = estimator_clone.feature_importances_
                else:
                    raise ValueError('Estimator must have either coef_ or feature_importances_ attribute')
                if importances.ndim > 1:
                    importances = np.mean(importances, axis=0)
                if isinstance(self.step, int):
                    n_eliminate = min(self.step, len(current_features))
                else:
                    n_eliminate = max(1, int(self.step * len(current_features)))
                eliminate_indices = np.argsort(importances)[:n_eliminate]
                eliminated_features = current_features[eliminate_indices]
                rankings[eliminated_features] = iteration + 2
                keep_mask = np.ones(len(current_features), dtype=bool)
                keep_mask[eliminate_indices] = False
                current_features = current_features[keep_mask]
                current_support[eliminated_features] = False
                iteration += 1
            except Exception as e:
                warnings.warn(f'Cross-validation failed at iteration {iteration}: {str(e)}')
                break
        if np.sum(current_support) == 1:
            X_subset = X[:, current_features]
            try:
                estimator_clone = deepcopy(self.estimator)
                cv_results = cross_validate(estimator_clone, X_subset, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs, return_train_score=False)
                mean_score = np.mean(cv_results['test_score'])
                scores.append(mean_score)
                supports.append(current_support.copy())
                if self.verbose > 0:
                    print(f'Iteration {iteration}: 1 feature, score={mean_score:.4f}')
            except Exception as e:
                warnings.warn(f'Cross-validation failed for final feature: {str(e)}')
        self.scores_ = np.array(scores)
        if len(self.scores_) > 0:
            best_idx = np.argmax(self.scores_)
            self.support_ = supports[best_idx]
        else:
            self.support_ = np.ones(n_features, dtype=bool)
        self.ranking_ = rankings
        self.n_features_ = np.sum(self.support_)
        self.ranking_[self.support_] = 1
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reduce data to the selected features determined by cross-validation.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            The input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed data with only the selected features
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        pass

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the transformation, adding back the eliminated features as zeros.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            The transformed data to inverse transform
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Data with eliminated features added back as zeros
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        pass