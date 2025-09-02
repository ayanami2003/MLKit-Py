from typing import Optional, Union, List, Dict, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class DecisionTreeClassifier(BaseModel):

    def __init__(self, criterion: str='gini', max_depth: Optional[int]=None, min_samples_split: int=2, min_samples_leaf: int=1, max_features: Optional[Union[int, float, str]]=None, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the DecisionTreeClassifier.
        
        Parameters
        ----------
        criterion : str, default='gini'
            The function to measure the quality of a split. Supported criteria are
            'gini' for the Gini impurity and 'entropy' for the information gain.
        max_depth : int, optional
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node.
        max_features : int, float, str, or None, default=None
            The number of features to consider when looking for the best split:
            - If int, then consider max_features features at each split.
            - If float, then max_features is a fraction and
              int(max_features * n_features) features are considered at each split.
            - If 'auto', then max_features=sqrt(n_features).
            - If 'sqrt', then max_features=sqrt(n_features).
            - If 'log2', then max_features=log2(n_features).
            - If None, then max_features=n_features.
        random_state : int, optional
            Controls the randomness of the estimator.
        name : str, optional
            Name of the model instance.
            
        Raises
        ------
        ValueError
            If criterion is not 'gini' or 'entropy'
            If max_depth is not positive
            If min_samples_split is less than 2
            If min_samples_leaf is less than 1
        """
        super().__init__(name=name)
        if criterion not in ('gini', 'entropy'):
            raise ValueError("Criterion must be 'gini' or 'entropy'")
        if max_depth is not None and max_depth <= 0:
            raise ValueError('max_depth must be positive')
        if min_samples_split < 2:
            raise ValueError('min_samples_split must be at least 2')
        if min_samples_leaf < 1:
            raise ValueError('min_samples_leaf must be at least 1')
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.tree_: Optional[Dict[str, Any]] = None

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs) -> 'DecisionTreeClassifier':
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            The training input samples. If FeatureSet, uses the features attribute.
            If ndarray, expects shape (n_samples, n_features).
        y : np.ndarray or List
            The target values (class labels) as integers or strings.
        **kwargs : dict
            Additional fitting parameters (not used but present for compatibility).
            
        Returns
        -------
        DecisionTreeClassifier
            Fitted estimator.
            
        Raises
        ------
        ValueError
            If X and y have incompatible shapes
            If y contains unsupported data types
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y_array.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        (self.n_samples_, self.n_features_) = X_array.shape
        self.classes_ = np.unique(y_array)
        self.n_classes_ = len(self.classes_)
        self.feature_importances_ = np.zeros(self.n_features_)
        self.tree_ = self._build_tree(X_array, y_array)
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)
        else:
            self.feature_importances_ = np.zeros(self.n_features_)
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class for X.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            The input samples. If FeatureSet, uses the features attribute.
            If ndarray, expects shape (n_samples, n_features).
        **kwargs : dict
            Additional prediction parameters (not used but present for compatibility).
            
        Returns
        -------
        np.ndarray
            Predicted classes with shape (n_samples,).
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        ValueError
            If X has incompatible shape with training data
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.n_features_:
            raise ValueError(f'X must have {self.n_features_} features, but got {X_array.shape[1]}')
        predictions = np.array([self._predict_sample(x, self.tree_) for x in X_array])
        return predictions

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            The input samples. If FeatureSet, uses the features attribute.
            If ndarray, expects shape (n_samples, n_features).
        **kwargs : dict
            Additional prediction parameters (not used but present for compatibility).
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities with shape (n_samples, n_classes).
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        ValueError
            If X has incompatible shape with training data
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.n_features_:
            raise ValueError(f'X must have {self.n_features_} features, but got {X_array.shape[1]}')
        probabilities = np.array([self._predict_proba_sample(x, self.tree_) for x in X_array])
        return probabilities

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Test samples. If FeatureSet, uses the features attribute.
            If ndarray, expects shape (n_samples, n_features).
        y : np.ndarray or List
            True labels for X.
        **kwargs : dict
            Additional scoring parameters (not used but present for compatibility).
            
        Returns
        -------
        float
            Mean accuracy of self.predict(X) wrt. y.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        ValueError
            If X and y have incompatible shapes
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before using this method.")
        predictions = self.predict(X)
        y_array = np.asarray(y)
        if predictions.shape[0] != y_array.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        return np.mean(predictions == y_array)

    def get_feature_importances(self) -> np.ndarray:
        """
        Return the feature importances.
        
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature. It is also known as
        the Gini importance.
        
        Returns
        -------
        np.ndarray
            Normalized total reduction of criteria by feature (Gini importance).
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before using this method.")
        return self.feature_importances_.copy()

    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity of a node."""
        if len(y) == 0:
            return 0.0
        (_, counts) = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a node."""
        if len(y) == 0:
            return 0.0
        (_, counts) = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        nonzero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
        return entropy

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity of a node based on the criterion."""
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        else:
            return self._calculate_entropy(y)

    def _calculate_split_gain(self, parent_y: np.ndarray, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """Calculate the information gain from a split."""
        n_parent = len(parent_y)
        n_left = len(left_y)
        n_right = len(right_y)
        if n_parent == 0 or n_left == 0 or n_right == 0:
            return 0.0
        parent_impurity = self._calculate_impurity(parent_y)
        left_impurity = self._calculate_impurity(left_y)
        right_impurity = self._calculate_impurity(right_y)
        weighted_child_impurity = n_left / n_parent * left_impurity + n_right / n_parent * right_impurity
        gain = parent_impurity - weighted_child_impurity
        return gain

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> tuple:
        """Find the best split for the given data among the selected features."""
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        parent_impurity = self._calculate_impurity(y)
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            if len(unique_values) < 2:
                continue
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                left_y = y[left_mask]
                right_y = y[right_mask]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                gain = self._calculate_split_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        if best_gain == -np.inf:
            best_gain = 0.0
        return (best_feature_idx, best_threshold, best_gain)

    def _get_max_features(self, n_features: int) -> int:
        """Determine the number of features to consider for splitting."""
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt' or self.max_features == 'auto':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            return n_features

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0) -> Dict[str, Any]:
        """Recursively build the decision tree."""
        (n_samples, n_features) = X.shape
        n_classes = len(np.unique(y))
        if self.max_depth is not None and depth >= self.max_depth or n_samples < self.min_samples_split or n_classes <= 1:
            (unique_classes, counts) = np.unique(y, return_counts=True)
            most_common_class = unique_classes[np.argmax(counts)]
            class_probs = np.zeros(self.n_classes_)
            for (i, cls) in enumerate(self.classes_):
                class_probs[i] = np.sum(y == cls) / len(y)
            impurity = self._calculate_impurity(y)
            return {'type': 'leaf', 'prediction': most_common_class, 'class_probabilities': class_probs, 'samples': n_samples, 'impurity': impurity}
        n_selected_features = self._get_max_features(n_features)
        if n_selected_features == n_features:
            feature_indices = np.arange(n_features)
        else:
            feature_indices = np.random.choice(n_features, n_selected_features, replace=False)
        (best_feature_idx, best_threshold, best_gain) = self._find_best_split(X, y, feature_indices)
        if best_feature_idx is None or best_gain <= 0:
            (unique_classes, counts) = np.unique(y, return_counts=True)
            most_common_class = unique_classes[np.argmax(counts)]
            class_probs = np.zeros(self.n_classes_)
            for (i, cls) in enumerate(self.classes_):
                class_probs[i] = np.sum(y == cls) / len(y)
            impurity = self._calculate_impurity(y)
            return {'type': 'leaf', 'prediction': most_common_class, 'class_probabilities': class_probs, 'samples': n_samples, 'impurity': impurity}
        feature_values = X[:, best_feature_idx]
        left_mask = feature_values <= best_threshold
        right_mask = feature_values > best_threshold
        (left_X, left_y) = (X[left_mask], y[left_mask])
        (right_X, right_y) = (X[right_mask], y[right_mask])
        left_samples = len(left_y)
        right_samples = len(right_y)
        total_samples = len(y)
        parent_impurity = self._calculate_impurity(y)
        left_impurity = self._calculate_impurity(left_y)
        right_impurity = self._calculate_impurity(right_y)
        impurity_reduction = parent_impurity - (left_samples / total_samples * left_impurity + right_samples / total_samples * right_impurity)
        self.feature_importances_[best_feature_idx] += impurity_reduction * total_samples
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        return {'type': 'internal', 'feature_idx': best_feature_idx, 'threshold': best_threshold, 'left': left_child, 'right': right_child, 'samples': n_samples, 'impurity': parent_impurity}

    def _predict_sample(self, x: np.ndarray, node: Dict[str, Any]) -> Any:
        """Predict a single sample by traversing the tree."""
        if node['type'] == 'leaf':
            return node['prediction']
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def _predict_proba_sample(self, x: np.ndarray, node: Dict[str, Any]) -> np.ndarray:
        """Predict class probabilities for a single sample."""
        if node['type'] == 'leaf':
            return node['class_probabilities']
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_proba_sample(x, node['left'])
        else:
            return self._predict_proba_sample(x, node['right'])