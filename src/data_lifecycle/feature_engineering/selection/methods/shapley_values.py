from typing import Optional, Union, Dict, Any, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations
from sklearn.model_selection import train_test_split
import warnings

class ShapleyValueFeatureSelector(BaseTransformer):

    def __init__(self, n_features: Optional[int]=None, utility_function: str='accuracy', normalize: bool=True, random_state: Optional[int]=None, name: Optional[str]=None, **kwargs):
        """
        Initialize the Shapley value feature selector.
        
        Parameters
        ----------
        n_features : Optional[int], default=None
            Number of top features to select. If None, selects features with positive Shapley values.
        utility_function : str, default='accuracy'
            Utility function to evaluate subset performance. Supported values include:
            'accuracy', 'f1', 'precision', 'recall', 'mse', 'mae', 'r2'.
        normalize : bool, default=True
            Whether to normalize Shapley values so they sum to 1.
        random_state : Optional[int], default=None
            Random seed for reproducibility of sampling procedures.
        name : Optional[str], default=None
            Name of the transformer instance.
        **kwargs : dict
            Additional parameters passed to the underlying model (if applicable).
        """
        super().__init__(name=name)
        self.n_features = n_features
        self.utility_function = utility_function
        self.normalize = normalize
        self.random_state = random_state
        self._extra_params = kwargs
        self.shapley_values_ = None
        self.selected_features_ = None

    def _compute_utility(self, X_subset: np.ndarray, y: np.ndarray, estimator: Any, is_classification: bool) -> float:
        """
        Compute utility score for a feature subset.
        
        Parameters
        ----------
        X_subset : np.ndarray
            Feature subset to evaluate
        y : np.ndarray
            Target values
        estimator : sklearn-compatible estimator
            Estimator to use for evaluation
        is_classification : bool
            Whether this is a classification task
            
        Returns
        -------
        float
            Utility score based on the configured utility function
        """
        from sklearn.model_selection import train_test_split
        if X_subset.shape[1] == 0:
            if self.utility_function in ['mse', 'mae']:
                return np.inf if self.utility_function == 'mse' else np.inf
            else:
                return 0.0
        if len(y) <= 1:
            return 0.0
        test_size = 0.3 if len(y) > 10 else 0.5
        try:
            (X_train, X_test, y_train, y_test) = train_test_split(X_subset, y, test_size=test_size, random_state=self.random_state or 0)
        except ValueError:
            (X_train, X_test, y_train, y_test) = (X_subset, X_subset, y, y)
        if len(X_train) == 0 or len(X_test) == 0:
            return 0.0
        try:
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
        except Exception:
            return 0.0
        if self.utility_function == 'accuracy':
            if not is_classification:
                return 0.0
            return accuracy_score(y_test, y_pred)
        elif self.utility_function == 'f1':
            if not is_classification:
                return 0.0
            if len(np.unique(y_test)) <= 2:
                try:
                    return f1_score(y_test, y_pred, average='binary', zero_division=0)
                except ValueError:
                    return f1_score(y_test, y_pred, average='macro', zero_division=0)
            else:
                return f1_score(y_test, y_pred, average='weighted', zero_division=0)
        elif self.utility_function == 'precision':
            if not is_classification:
                return 0.0
            if len(np.unique(y_test)) <= 2:
                try:
                    return precision_score(y_test, y_pred, average='binary', zero_division=0)
                except ValueError:
                    return precision_score(y_test, y_pred, average='macro', zero_division=0)
            else:
                return precision_score(y_test, y_pred, average='weighted', zero_division=0)
        elif self.utility_function == 'recall':
            if not is_classification:
                return 0.0
            if len(np.unique(y_test)) <= 2:
                try:
                    return recall_score(y_test, y_pred, average='binary', zero_division=0)
                except ValueError:
                    return recall_score(y_test, y_pred, average='macro', zero_division=0)
            else:
                return recall_score(y_test, y_pred, average='weighted', zero_division=0)
        elif self.utility_function == 'mse':
            if is_classification:
                return 0.0
            return -mean_squared_error(y_test, y_pred)
        elif self.utility_function == 'mae':
            if is_classification:
                return 0.0
            return -mean_absolute_error(y_test, y_pred)
        elif self.utility_function == 'r2':
            if is_classification:
                return 0.0
            return r2_score(y_test, y_pred)
        else:
            if not is_classification:
                return 0.0
            return accuracy_score(y_test, y_pred)

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'ShapleyValueFeatureSelector':
        """
        Compute Shapley values for features based on their contribution to model performance.
        
        This method calculates the Shapley value for each feature by evaluating the
        marginal contribution of each feature across all possible feature subsets.
        Due to the computational complexity, approximation methods may be used.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing features and labels. Must have 'labels' attribute.
        **kwargs : dict
            Additional fitting parameters. May include:
            - estimator : sklearn-compatible estimator to use for utility computation
            - n_iterations : int, number of iterations for approximation methods
            - y : array-like, target values (for FeatureSet)
            
        Returns
        -------
        ShapleyValueFeatureSelector
            Fitted transformer instance.
            
        Raises
        ------
        ValueError
            If data does not contain labels or if utility_function is not supported.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(data, FeatureSet):
            X = data.features
            y = kwargs.get('y', None)
            if y is None:
                if data.metadata:
                    y = data.metadata.get('labels', None)
                    if y is None:
                        y = data.metadata.get('target', None)
            if y is None:
                raise ValueError('FeatureSet must contain labels in metadata or as y parameter in kwargs')
            self.feature_names_ = data.feature_names
        elif isinstance(data, DataBatch):
            X = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data
            y = data.labels
            if y is None:
                raise ValueError('DataBatch must contain labels')
            self.feature_names_ = data.feature_names
        else:
            raise TypeError('data must be either FeatureSet or DataBatch')
        supported_utilities = ['accuracy', 'f1', 'precision', 'recall', 'mse', 'mae', 'r2']
        if self.utility_function not in supported_utilities:
            raise ValueError(f'utility_function must be one of {supported_utilities}')
        unique_labels = np.unique(y)
        is_classification = len(unique_labels) < len(y) / 2 and (np.issubdtype(y.dtype, np.integer) or len(unique_labels) <= 20)
        estimator = kwargs.get('estimator')
        if estimator is None:
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                estimator = RandomForestClassifier(random_state=self.random_state or 0)
            else:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(random_state=self.random_state or 0)
        n_iterations = kwargs.get('n_iterations', min(1000, 2 ** X.shape[1]))
        n_features = X.shape[1]
        shapley_values = np.zeros(n_features)
        if n_features <= 10:
            for feature_idx in range(n_features):
                for subset_size in range(n_features):
                    for subset in combinations([i for i in range(n_features) if i != feature_idx], subset_size):
                        subset = list(subset)
                        utility_without = self._compute_utility(X[:, subset] if subset else np.empty((X.shape[0], 0)), y, estimator, is_classification)
                        subset_with_feature = subset + [feature_idx]
                        utility_with = self._compute_utility(X[:, subset_with_feature], y, estimator, is_classification)
                        marginal_contribution = utility_with - utility_without
                        weight = np.math.factorial(len(subset)) * np.math.factorial(n_features - len(subset) - 1) / np.math.factorial(n_features)
                        shapley_values[feature_idx] += weight * marginal_contribution
        else:
            for _ in range(n_iterations):
                perm = np.random.permutation(n_features)
                current_utility = 0
                for (i, feature_idx) in enumerate(perm):
                    subset = perm[:i + 1]
                    new_utility = self._compute_utility(X[:, subset], y, estimator, is_classification)
                    marginal_contribution = new_utility - current_utility
                    shapley_values[feature_idx] += marginal_contribution
                    current_utility = new_utility
            shapley_values /= n_iterations
        if self.normalize and np.sum(np.abs(shapley_values)) > 0:
            shapley_values /= np.sum(np.abs(shapley_values))
        self.shapley_values_ = shapley_values
        if self.n_features is not None:
            self.selected_features_ = np.argsort(np.abs(shapley_values))[::-1][:self.n_features].tolist()
        else:
            self.selected_features_ = np.where(shapley_values > 0)[0].tolist()
        self.is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Select features based on computed Shapley values.
        
        Retains only the top `n_features` features with highest Shapley values,
        or all features with positive Shapley values if `n_features` is None.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform. Must be compatible with fitted selector.
        **kwargs : dict
            Additional transformation parameters (unused).
            
        Returns
        -------
        FeatureSet
            Transformed data with selected features only.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            metadata = data.metadata.copy() if data.metadata else {}
        elif isinstance(data, DataBatch):
            X = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data
            feature_names = data.feature_names
            metadata = {}
        else:
            raise TypeError('data must be either FeatureSet or DataBatch')
        X_selected = X[:, self.selected_features_]
        selected_feature_names = [feature_names[i] for i in self.selected_features_] if feature_names else None
        return FeatureSet(features=X_selected, feature_names=selected_feature_names, metadata=metadata, name=f'{data.name}_shapley_selected' if hasattr(data, 'name') and data.name else 'shapley_selected_features')

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse transformation (not supported for feature selection).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters (unused).
            
        Returns
        -------
        FeatureSet
            Original data with unselected features filled with zeros or NaNs.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection.
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection.')

    def get_feature_importance(self) -> Dict[str, Union[np.ndarray, list]]:
        """
        Get feature importance scores based on computed Shapley values.
        
        Returns
        -------
        Dict[str, Union[np.ndarray, list]]
            Dictionary containing:
            - 'shapley_values': Array of Shapley values for each feature
            - 'feature_indices': Indices of selected features
            - 'feature_names': Names of selected features (if available)
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError('Transformer has not been fitted yet.')
        feature_names = getattr(self, 'feature_names_', None)
        selected_feature_names = [feature_names[i] for i in self.selected_features_] if feature_names else None
        return {'shapley_values': self.shapley_values_, 'feature_indices': self.selected_features_, 'feature_names': selected_feature_names}