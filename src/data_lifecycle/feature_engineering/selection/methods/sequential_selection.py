from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, List
import numpy as np
from sklearn.model_selection import cross_val_score

class SequentialFeatureSelector(BaseTransformer):

    def __init__(self, estimator, n_features_to_select: Optional[int]=None, direction: str='forward', scoring: Optional[str]=None, cv: int=5, name: Optional[str]=None):
        """
        Initialize the SequentialFeatureSelector.
        
        Args:
            estimator: A supervised learning estimator with a fit method.
            n_features_to_select: Number of features to select. If None, half the features are selected.
            direction: Direction of selection ('forward' or 'backward').
            scoring: Scoring metric to optimize (e.g., 'accuracy', 'f1', 'roc_auc').
            cv: Number of cross-validation folds for evaluating feature subsets.
            name: Optional name for the transformer.
        """
        super().__init__(name=name)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.selected_features_: List[int] = []
        self.scores_: List[float] = []

    def fit(self, data: Union[FeatureSet, DataBatch], y: Optional[Union[List, DataBatch]]=None, **kwargs) -> 'SequentialFeatureSelector':
        """
        Fit the sequential feature selector on the input data.
        
        Args:
            data: Input features as FeatureSet or DataBatch.
            y: Target values for supervised learning.
            **kwargs: Additional fitting parameters.
            
        Returns:
            SequentialFeatureSelector: The fitted transformer.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            if y is None and data.metadata and ('labels' in data.metadata):
                y = data.metadata['labels']
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            y = data.labels if y is None else y
            feature_names = data.feature_names
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if y is None:
            raise ValueError('Target values (y) must be provided for supervised feature selection')
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
        self.selected_features_ = []
        self.scores_ = []
        if self.direction == 'forward':
            remaining_features = list(range(n_features))
            while len(self.selected_features_) < self.n_features_to_select:
                best_score = -np.inf
                best_feature = None
                for feature_idx in remaining_features:
                    candidate_features = self.selected_features_ + [feature_idx]
                    X_candidate = X[:, candidate_features]
                    if self.cv < 2:
                        from sklearn.model_selection import KFold
                        kf = KFold(n_splits=2, shuffle=True, random_state=42)
                        scores = cross_val_score(self.estimator, X_candidate, y, cv=kf, scoring=self.scoring)
                    else:
                        scores = cross_val_score(self.estimator, X_candidate, y, cv=self.cv, scoring=self.scoring)
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_feature = feature_idx
                if best_feature is not None:
                    self.selected_features_.append(best_feature)
                    self.scores_.append(best_score)
                    remaining_features.remove(best_feature)
                else:
                    break
        elif self.direction == 'backward':
            self.selected_features_ = list(range(n_features))
            while len(self.selected_features_) > self.n_features_to_select:
                worst_score = np.inf
                worst_feature = None
                for feature_idx in self.selected_features_:
                    candidate_features = [f for f in self.selected_features_ if f != feature_idx]
                    X_candidate = X[:, candidate_features]
                    if self.cv < 2:
                        from sklearn.model_selection import KFold
                        kf = KFold(n_splits=2, shuffle=True, random_state=42)
                        scores = cross_val_score(self.estimator, X_candidate, y, cv=kf, scoring=self.scoring)
                    else:
                        scores = cross_val_score(self.estimator, X_candidate, y, cv=self.cv, scoring=self.scoring)
                    avg_score = np.mean(scores)
                    if avg_score < worst_score:
                        worst_score = avg_score
                        worst_feature = feature_idx
                if worst_feature is not None:
                    self.selected_features_.remove(worst_feature)
                    self.scores_.append(worst_score)
                else:
                    break
        else:
            raise ValueError("direction must be either 'forward' or 'backward'")
        if feature_names is not None:
            self.selected_feature_names_ = [feature_names[i] for i in self.selected_features_]
        else:
            self.selected_feature_names_ = None
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Transform the input data to only include selected features.
        
        Args:
            data: Input features as FeatureSet or DataBatch.
            **kwargs: Additional transformation parameters.
            
        Returns:
            FeatureSet: Transformed data with only selected features.
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            metadata = data.metadata
            sample_ids = data.sample_ids
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            feature_types = None
            metadata = getattr(data, 'metadata', None)
            sample_ids = getattr(data, 'sample_ids', None)
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X_transformed = X[:, self.selected_features_]
        if feature_names is not None:
            new_feature_names = [feature_names[i] for i in self.selected_features_]
        else:
            new_feature_names = None
        if feature_types is not None:
            new_feature_types = [feature_types[i] for i in self.selected_features_]
        else:
            new_feature_types = None
        return FeatureSet(features=X_transformed, feature_names=new_feature_names, feature_types=new_feature_types, metadata=metadata, sample_ids=sample_ids)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for feature selection.
        
        Args:
            data: Transformed data.
            **kwargs: Additional parameters.
            
        Returns:
            FeatureSet: Original data structure (placeholder implementation).
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection.')