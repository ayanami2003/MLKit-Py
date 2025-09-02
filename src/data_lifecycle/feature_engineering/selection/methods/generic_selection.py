from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, List
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

class GenericFeatureSelector(BaseTransformer):

    def __init__(self, method: str='variance_threshold', n_features: Optional[int]=None, name: Optional[str]=None, **kwargs):
        """
        Initialize the GenericFeatureSelector.

        Parameters
        ----------
        method : str, default='variance_threshold'
            The feature selection method to apply. Supported methods include:
            - 'variance_threshold': Remove features with low variance
            - 'recursive_elimination': Recursively remove least important features
            - 'tree_based': Use tree-based feature importance
            - 'mutual_information': Select based on mutual information with target
        n_features : int, optional
            Number of features to select. If None, method-specific defaults are used
        name : str, optional
            Name for the transformer instance
        **kwargs : dict
            Additional method-specific parameters
        """
        super().__init__(name=name)
        self.method = method
        self.n_features = n_features
        self.kwargs = kwargs
        self.selected_features_: Optional[List[int]] = None
        self.feature_names_: Optional[List[str]] = None
        supported_methods = {'variance_threshold', 'recursive_elimination', 'tree_based', 'mutual_information'}
        if self.method not in supported_methods:
            raise ValueError(f"Method '{self.method}' is not supported. Supported methods: {supported_methods}")

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'GenericFeatureSelector':
        """
        Fit the feature selector to the input data.

        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing features and optionally labels for supervised methods
        **kwargs : dict
            Additional fitting parameters

        Returns
        -------
        GenericFeatureSelector
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            y = kwargs.get('y', None)
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
            y = data.labels
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        n_features = X.shape[1]
        if feature_names is not None:
            self.feature_names_ = feature_names
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(n_features)]
        if self.method == 'variance_threshold':
            threshold = self.kwargs.get('threshold', 0.0)
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X)
            self.selected_features_ = np.where(selector.get_support())[0].tolist()
            if self.n_features is not None:
                if len(self.selected_features_) >= self.n_features:
                    variances = selector.variances_
                    selected_vars = variances[self.selected_features_]
                    sorted_indices = np.argsort(selected_vars)[::-1][:self.n_features]
                    self.selected_features_ = [self.selected_features_[i] for i in sorted_indices]
                else:
                    rejected_features = [i for i in range(n_features) if i not in self.selected_features_]
                    if rejected_features:
                        variances = selector.variances_
                        rejected_vars = variances[rejected_features]
                        n_needed = min(self.n_features - len(self.selected_features_), len(rejected_features))
                        top_rejected = np.argsort(rejected_vars)[::-1][:n_needed]
                        additional_features = [rejected_features[i] for i in top_rejected]
                        self.selected_features_.extend(additional_features)
                        self.selected_features_.sort()
        elif self.method == 'recursive_elimination':
            if y is not None:
                estimator = self.kwargs.get('estimator', RandomForestClassifier(random_state=0))
                n_features_to_select = self.n_features if self.n_features is not None else max(1, n_features // 2)
                selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
                selector.fit(X, y)
                self.selected_features_ = np.where(selector.support_)[0].tolist()
            else:
                self.selected_features_ = list(range(n_features))
        elif self.method == 'tree_based':
            if y is not None:
                if len(np.unique(y)) > 20:
                    estimator = self.kwargs.get('estimator', RandomForestRegressor(random_state=0))
                else:
                    estimator = self.kwargs.get('estimator', RandomForestClassifier(random_state=0))
                if self.n_features is not None:
                    selector = SelectFromModel(estimator=estimator, max_features=None, **{k: v for (k, v) in self.kwargs.items() if k != 'estimator'})
                    selector.fit(X, y)
                    importances = selector.estimator_.feature_importances_
                    top_indices = np.argsort(importances)[::-1][:self.n_features]
                    self.selected_features_ = sorted(top_indices.tolist())
                else:
                    max_features = max(1, n_features // 2)
                    selector = SelectFromModel(estimator=estimator, max_features=max_features, **{k: v for (k, v) in self.kwargs.items() if k != 'estimator'})
                    selector.fit(X, y)
                    support = selector.get_support()
                    self.selected_features_ = np.where(support)[0].tolist()
                    if len(self.selected_features_) == 0 and n_features > 0:
                        importances = selector.estimator_.feature_importances_
                        best_feature_idx = np.argmax(importances)
                        self.selected_features_ = [best_feature_idx]
            else:
                self.selected_features_ = list(range(n_features))
        elif self.method == 'mutual_information':
            if y is not None:
                random_state = self.kwargs.get('random_state', 0)
                n_features_to_select = self.n_features if self.n_features is not None else max(1, n_features // 2)
                if len(np.unique(y)) > 20:
                    mi_scores = mutual_info_regression(X, y, random_state=random_state)
                else:
                    mi_scores = mutual_info_classif(X, y, random_state=random_state)
                if n_features_to_select >= n_features:
                    self.selected_features_ = list(range(n_features))
                else:
                    top_indices = np.argsort(mi_scores)[-n_features_to_select:]
                    self.selected_features_ = sorted(top_indices.tolist())
            else:
                self.selected_features_ = list(range(n_features))
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Apply feature selection to reduce the dimensionality of the input data.

        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform by selecting features
        **kwargs : dict
            Additional transformation parameters

        Returns
        -------
        FeatureSet
            Transformed data with reduced feature dimensionality
        """
        if self.selected_features_ is None:
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
        selected_feature_types = None
        if feature_names is not None:
            selected_feature_names = [feature_names[i] for i in self.selected_features_]
        elif self.feature_names_ is not None:
            selected_feature_names = [self.feature_names_[i] for i in self.selected_features_]
        if feature_types is not None:
            selected_feature_types = [feature_types[i] for i in self.selected_features_]
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Attempt to restore the original feature set (not generally supported).

        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data to restore
        **kwargs : dict
            Additional parameters

        Returns
        -------
        FeatureSet
            Original feature set (if supported by method)

        Raises
        ------
        NotImplementedError
            If the method does not support inverse transformation
        """
        raise NotImplementedError('GenericFeatureSelector does not support inverse transformation for most selection methods.')