from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, Tuple, List
from collections import Counter
from typing import Optional, Union, Tuple

class DecisionTreeRegressor(BaseModel):

    def __init__(self, criterion: str='mse', max_depth: Optional[int]=None, min_samples_split: Union[int, float]=2, min_samples_leaf: Union[int, float]=1, max_features: Optional[Union[int, float, str]]=None, random_state: Optional[int]=None, ccp_alpha: float=0.0):
        """
        Initialize the DecisionTreeRegressor.

        Args:
            criterion (str): The function to measure the quality of a split.
                            Supported criteria are 'mse' (mean squared error),
                            'friedman_mse' (improved mse for gradient boosting),
                            and 'mae' (mean absolute error). Defaults to 'mse'.
            max_depth (Optional[int]): The maximum depth of the tree. If None,
                                       nodes are expanded until all leaves are pure
                                       or until min_samples_split is reached.
            min_samples_split (Union[int, float]): The minimum number of samples
                                                   required to split an internal node.
                                                   If float, interpreted as fraction
                                                   of total samples. Defaults to 2.
            min_samples_leaf (Union[int, float]): The minimum number of samples
                                                  required to be at a leaf node.
                                                  If float, interpreted as fraction
                                                  of total samples. Defaults to 1.
            max_features (Optional[Union[int, float, str]]): The number of features
                                                             to consider when looking
                                                             for the best split.
                                                             If int, consider max_features features;
                                                             if float, consider max_features*total_features;
                                                             if 'auto' or 'sqrt', sqrt(total_features);
                                                             if 'log2', log2(total_features);
                                                             if None, consider all features.
            random_state (Optional[int]): Random state for reproducibility.
            ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning.
                               Defaults to 0.0 (no pruning).
        """
        super().__init__(name='DecisionTreeRegressor')
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.tree_ = None
        self.n_features_ = None
        self.n_outputs_ = None

    def _validate_inputs(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and convert inputs to numpy arrays."""
        if isinstance(X, FeatureSet):
            X_array = X.features.values if hasattr(X.features, 'values') else np.array(X.features)
        else:
            X_array = np.asarray(X)
        if y is not None:
            y_array = np.asarray(y)
            if X_array.shape[0] != y_array.shape[0]:
                raise ValueError(f'X and y have inconsistent numbers of samples: {X_array.shape[0]} != {y_array.shape[0]}')
            return (X_array, y_array)
        else:
            return (X_array, None)

    def _calculate_criterion(self, y: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        """Calculate the impurity improvement for a split."""
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        if self.criterion == 'mse':
            return self._mse_criterion(y, left_indices, right_indices)
        elif self.criterion == 'friedman_mse':
            return self._friedman_mse_criterion(y, left_indices, right_indices)
        elif self.criterion == 'mae':
            return self._mae_criterion(y, left_indices, right_indices)
        else:
            raise ValueError(f'Unsupported criterion: {self.criterion}')

    def _mse_criterion(self, y: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        """Calculate MSE improvement for a split."""
        (left_y, right_y) = (y[left_indices], y[right_indices])
        parent_mse = np.var(y) * len(y)
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        improvement = parent_mse - left_mse - right_mse
        return improvement / len(y)

    def _friedman_mse_criterion(self, y: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        """Calculate Friedman MSE improvement for a split."""
        (left_y, right_y) = (y[left_indices], y[right_indices])
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        (left_mean, right_mean) = (np.mean(left_y), np.mean(right_y))
        diff = left_mean - right_mean
        friedman_mse = len(left_y) * len(right_y) * diff ** 2 / len(y)
        return friedman_mse

    def _mae_criterion(self, y: np.ndarray, left_indices: np.ndarray, right_indices: np.ndarray) -> float:
        """Calculate MAE improvement for a split."""
        (left_y, right_y) = (y[left_indices], y[right_indices])
        parent_mae = np.mean(np.abs(y - np.mean(y))) * len(y)
        left_mae = np.mean(np.abs(left_y - np.mean(left_y))) * len(left_y)
        right_mae = np.mean(np.abs(right_y - np.mean(right_y))) * len(right_y)
        improvement = parent_mae - left_mae - right_mae
        return improvement / len(y)

    def _get_best_split(self, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray) -> dict:
        """Find the best split for a node."""
        best_score = -np.inf
        best_split = {}
        n_features = X.shape[1]
        if self.max_features is None:
            features_to_consider = np.arange(n_features)
        elif isinstance(self.max_features, str):
            if self.max_features == 'auto' or self.max_features == 'sqrt':
                n_features_to_consider = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                n_features_to_consider = int(np.log2(n_features))
            else:
                raise ValueError(f'Invalid max_features string value: {self.max_features}')
            features_to_consider = np.random.choice(n_features, size=n_features_to_consider, replace=False)
        elif isinstance(self.max_features, float):
            n_features_to_consider = int(self.max_features * n_features)
            features_to_consider = np.random.choice(n_features, size=n_features_to_consider, replace=False)
        else:
            n_features_to_consider = min(self.max_features, n_features)
            features_to_consider = np.random.choice(n_features, size=n_features_to_consider, replace=False)
        for feature_idx in features_to_consider:
            feature_values = X[sample_indices, feature_idx]
            unique_values = np.unique(feature_values)
            if len(unique_values) > 1:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            else:
                continue
            for threshold in thresholds:
                left_indices = sample_indices[feature_values <= threshold]
                right_indices = sample_indices[feature_values > threshold]
                if len(left_indices) < self._effective_min_samples_leaf or len(right_indices) < self._effective_min_samples_leaf:
                    continue
                score = self._calculate_criterion(y, left_indices, right_indices)
                if score > best_score:
                    best_score = score
                    best_split = {'feature_idx': feature_idx, 'threshold': threshold, 'left_indices': left_indices, 'right_indices': right_indices, 'improvement': score}
        return best_split

    def _build_tree(self, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray, depth: int=0) -> dict:
        """Recursively build the decision tree."""
        n_samples = len(sample_indices)
        if self.max_depth is not None and depth >= self.max_depth:
            return {'value': np.mean(y[sample_indices]), 'n_samples': n_samples}
        if n_samples < self._effective_min_samples_split:
            return {'value': np.mean(y[sample_indices]), 'n_samples': n_samples}
        if len(np.unique(y[sample_indices])) == 1:
            return {'value': np.mean(y[sample_indices]), 'n_samples': n_samples}
        best_split = self._get_best_split(X, y, sample_indices)
        if not best_split:
            return {'value': np.mean(y[sample_indices]), 'n_samples': n_samples}
        node = {'feature_idx': best_split['feature_idx'], 'threshold': best_split['threshold'], 'n_samples': n_samples, 'improvement': best_split['improvement']}
        node['left'] = self._build_tree(X, y, best_split['left_indices'], depth + 1)
        node['right'] = self._build_tree(X, y, best_split['right_indices'], depth + 1)
        return node

    def _predict_sample(self, x: np.ndarray, tree: dict) -> float:
        """Predict a single sample."""
        if 'value' in tree:
            return tree['value']
        if 'feature_idx' not in tree or 'threshold' not in tree or 'left' not in tree or ('right' not in tree):
            return 0.0
        try:
            if x[tree['feature_idx']] <= tree['threshold']:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        except (IndexError, KeyError):
            return 0.0

    def _compute_cost_complexity(self, tree: dict, alpha: float) -> Tuple[float, int]:
        """Compute cost-complexity of a subtree."""
        if 'value' in tree:
            return (0.0, 1)
        (left_cost, left_leaves) = self._compute_cost_complexity(tree['left'], alpha)
        (right_cost, right_leaves) = self._compute_cost_complexity(tree['right'], alpha)
        total_leaves = left_leaves + right_leaves
        cost_complexity = left_cost + right_cost + alpha * total_leaves
        return (cost_complexity, total_leaves)

    def _prune_tree(self, tree: dict, X_val: np.ndarray, y_val: np.ndarray, alpha: float) -> dict:
        """Prune the tree using cost-complexity pruning."""
        if 'value' in tree:
            return tree
        tree['left'] = self._prune_tree(tree['left'], X_val, y_val, alpha)
        tree['right'] = self._prune_tree(tree['right'], X_val, y_val, alpha)
        if 'value' in tree['left'] and 'value' in tree['right']:
            predictions_with_subtree = []
            for i in range(len(X_val)):
                pred = self._predict_sample(X_val[i], tree)
                predictions_with_subtree.append(pred)
            error_with_subtree = np.mean((y_val - np.array(predictions_with_subtree)) ** 2)
            leaf_value = (tree['left']['value'] * tree['left']['n_samples'] + tree['right']['value'] * tree['right']['n_samples']) / tree['n_samples']
            predictions_merged = np.array([leaf_value] * len(y_val))
            error_merged = np.mean((y_val - predictions_merged) ** 2)
            if error_merged <= error_with_subtree + alpha:
                return {'value': leaf_value, 'n_samples': tree['n_samples']}
        return tree

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'DecisionTreeRegressor':
        """
        Build a decision tree regressor from the training set (X, y).

        Args:
            X (Union[FeatureSet, np.ndarray]): The training input samples.
                                              If FeatureSet, uses features attribute.
            y (Optional[Union[np.ndarray, list]]): The target values (real numbers).
                                                  Must be aligned with X.

        Returns:
            DecisionTreeRegressor: Fitted estimator.

        Raises:
            ValueError: If dimensions of X and y do not match.
        """
        (X_array, y_array) = self._validate_inputs(X, y)
        self.n_features_ = X_array.shape[1]
        self.n_outputs_ = 1
        n_samples = X_array.shape[0]
        self._effective_min_samples_split = max(int(self.min_samples_split * n_samples) if isinstance(self.min_samples_split, float) else self.min_samples_split, 2)
        self._effective_min_samples_leaf = max(int(self.min_samples_leaf * n_samples) if isinstance(self.min_samples_leaf, float) else self.min_samples_leaf, 1)
        sample_indices = np.arange(n_samples)
        self.tree_ = self._build_tree(X_array, y_array, sample_indices)
        self._tree = self.tree_
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict continuous target values for X.

        Args:
            X (Union[FeatureSet, np.ndarray]): The input samples.
                                              If FeatureSet, uses features attribute.

        Returns:
            np.ndarray: Predicted target values for X.
        """
        if self.tree_ is None:
            raise ValueError("This DecisionTreeRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        (X_array, _) = self._validate_inputs(X)
        predictions = []
        for i in range(X_array.shape[0]):
            pred = self._predict_sample(X_array[i], self.tree_)
            predictions.append(pred)
        return np.array(predictions)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative.

        Args:
            X (Union[FeatureSet, np.ndarray]): Test samples.
            y (Union[np.ndarray, list]): True values for X.

        Returns:
            float: R^2 of self.predict(X) wrt. y.
        """
        (X_array, y_array) = self._validate_inputs(X, y)
        y_pred = self.predict(X_array)
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def prune(self, X_val: Union[FeatureSet, np.ndarray], y_val: Union[np.ndarray, list], **kwargs) -> 'DecisionTreeRegressor':
        """
        Prune the decision tree using validation data to improve generalization.

        This method implements cost-complexity pruning to reduce overfitting
        by removing sections of the tree that provide little predictive power.

        Args:
            X_val (Union[FeatureSet, np.ndarray]): Validation input samples.
            y_val (Union[np.ndarray, list]): Validation target values.

        Returns:
            DecisionTreeRegressor: Pruned estimator.
        """
        if self.tree_ is None:
            raise ValueError("This DecisionTreeRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        (X_val_array, y_val_array) = self._validate_inputs(X_val, y_val)
        self.tree_ = self._prune_tree(self.tree_, X_val_array, y_val_array, self.ccp_alpha)
        self._tree = self.tree_
        return self