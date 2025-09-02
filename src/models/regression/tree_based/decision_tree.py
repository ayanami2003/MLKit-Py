from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from typing import Optional, Union, Tuple
from collections import namedtuple
TreeNode = namedtuple('TreeNode', ['feature_idx', 'threshold', 'left', 'right', 'value', 'samples', 'impurity'])

class DecisionTreeRegressor(BaseModel):

    def __init__(self, criterion: str='mse', max_depth: Optional[int]=None, min_samples_split: Union[int, float]=2, min_samples_leaf: Union[int, float]=1, max_features: Optional[Union[int, float, str]]=None, random_state: Optional[int]=None, ccp_alpha: float=0.0):
        """
        Initialize the DecisionTreeRegressor.
        
        Parameters
        ----------
        criterion : str, default='mse'
            The function to measure the quality of a split. Supported criteria
            are 'mse' for the mean squared error and 'mae' for the mean absolute error.
        max_depth : int, optional
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
        max_features : int, float, str, or None, default=None
            The number of features to consider when looking for the best split.
        random_state : int, RandomState instance, or None, default=None
            Controls the randomness of the estimator.
        ccp_alpha : float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning.
        """
        super().__init__(name='DecisionTreeRegressor')
        if criterion not in ('mse', 'mae'):
            raise ValueError("Criterion must be 'mse' or 'mae'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        if random_state is not None:
            np.random.seed(random_state)
        self.tree_ = None
        self.n_features_ = None
        self.n_outputs_ = None

    def _validate_inputs(self, X, y):
        """Validate input data."""
        if isinstance(X, FeatureSet):
            X = X.to_numpy()
        if isinstance(y, list):
            y = np.array(y)
        if not isinstance(X, np.ndarray):
            raise TypeError('X must be a numpy array or FeatureSet')
        if not isinstance(y, np.ndarray):
            raise TypeError('y must be a numpy array or list')
        if len(X.shape) != 2:
            raise ValueError('X must be a 2D array')
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) != 2:
            raise ValueError('y must be a 1D or 2D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        return (X, y)

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity of a node based on the criterion."""
        if len(y) == 0:
            return 0.0
        if self.criterion == 'mse':
            return np.var(y) if len(y) > 1 else 0.0
        else:
            median = np.median(y)
            return np.mean(np.abs(y - median))

    def _calculate_split_score(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """Calculate the improvement in impurity from a split."""
        (n_left, n_right) = (len(left_y), len(right_y))
        n_total = n_left + n_right
        if n_total == 0:
            return 0.0
        left_impurity = self._calculate_impurity(left_y)
        right_impurity = self._calculate_impurity(right_y)
        weighted_impurity = n_left / n_total * left_impurity + n_right / n_total * right_impurity
        return weighted_impurity

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split for the given data."""
        best_feature_idx = None
        best_threshold = None
        best_score = float('inf')
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                score = self._calculate_split_score(left_y, right_y)
                if score < best_score:
                    best_score = score
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        return (best_feature_idx, best_threshold, best_score)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0) -> TreeNode:
        """Recursively build the decision tree."""
        (n_samples, n_features) = X.shape
        if self.max_depth is not None and depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            value = np.mean(y, axis=0)
            impurity = self._calculate_impurity(y)
            return TreeNode(feature_idx=None, threshold=None, left=None, right=None, value=value, samples=n_samples, impurity=impurity)
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
            feature_indices = np.random.choice(n_features, n_selected, replace=False)
        elif self.max_features == 'log2':
            n_selected = int(np.log2(n_features))
            feature_indices = np.random.choice(n_features, n_selected, replace=False)
        elif isinstance(self.max_features, int):
            feature_indices = np.random.choice(n_features, min(self.max_features, n_features), replace=False)
        elif isinstance(self.max_features, float):
            n_selected = int(self.max_features * n_features)
            feature_indices = np.random.choice(n_features, n_selected, replace=False)
        else:
            feature_indices = np.arange(n_features)
        (best_feature_idx, best_threshold, best_score) = self._find_best_split(X, y, feature_indices)
        if best_feature_idx is None:
            value = np.mean(y, axis=0)
            impurity = self._calculate_impurity(y)
            return TreeNode(feature_idx=None, threshold=None, left=None, right=None, value=value, samples=n_samples, impurity=impurity)
        feature_values = X[:, best_feature_idx]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask
        (left_X, left_y) = (X[left_mask], y[left_mask])
        (right_X, right_y) = (X[right_mask], y[right_mask])
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)
        impurity = self._calculate_impurity(y)
        return TreeNode(feature_idx=best_feature_idx, threshold=best_threshold, left=left_child, right=right_child, value=np.mean(y, axis=0), samples=n_samples, impurity=impurity)

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'DecisionTreeRegressor':
        """
        Build a decision tree regressor from the training set (X, y).
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            Target values (real numbers in regression).
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        DecisionTreeRegressor
            Fitted estimator.
        """
        (X, y) = self._validate_inputs(X, y)
        self.n_features_ = X.shape[1]
        self.n_outputs_ = y.shape[1] if len(y.shape) > 1 else 1
        self.tree_ = self._build_tree(X, y)
        return self

    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> np.ndarray:
        """Predict a single sample by traversing the tree."""
        if node.feature_idx is None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict regression value for X.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            The input samples.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for X.
        """
        if self.tree_ is None:
            raise ValueError('Model has not been fitted yet.')
        if isinstance(X, FeatureSet):
            X = X.to_numpy()
        if len(X.shape) != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f'X must have {self.n_features_} features')
        predictions = np.array([self._predict_sample(x, self.tree_) for x in X])
        if self.n_outputs_ == 1 and len(predictions.shape) > 1 and (predictions.shape[1] == 1):
            predictions = predictions.ravel()
        return predictions

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        if isinstance(y, list):
            y = np.array(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def _compute_subtree_cost_complexity(self, node: TreeNode) -> Tuple[float, int]:
        """Compute the cost complexity of a subtree."""
        if node.feature_idx is None:
            return (node.impurity * node.samples, 1)
        (left_cost, left_leaves) = self._compute_subtree_cost_complexity(node.left)
        (right_cost, right_leaves) = self._compute_subtree_cost_complexity(node.right)
        total_cost = left_cost + right_cost
        total_leaves = left_leaves + right_leaves
        cost_complexity = total_cost + self.ccp_alpha * total_leaves
        return (cost_complexity, total_leaves)

    def _prune_tree(self, node: TreeNode, X_val: np.ndarray, y_val: np.ndarray) -> TreeNode:
        """Prune the tree using cost complexity pruning."""
        if node.feature_idx is None:
            return node

        def get_samples_mask(X, node_path):
            mask = np.ones(len(X), dtype=bool)
            for (feature_idx, threshold, is_left) in node_path:
                if is_left:
                    mask &= X[:, feature_idx] <= threshold
                else:
                    mask &= X[:, feature_idx] > threshold
            return mask

        def prune_recursive(node, path):
            if node.feature_idx is None:
                return node
            left_path = path + [(node.feature_idx, node.threshold, True)]
            right_path = path + [(node.feature_idx, node.threshold, False)]
            pruned_left = prune_recursive(node.left, left_path)
            pruned_right = prune_recursive(node.right, right_path)
            pruned_node = TreeNode(feature_idx=node.feature_idx, threshold=node.threshold, left=pruned_left, right=pruned_right, value=node.value, samples=node.samples, impurity=node.impurity)
            node_mask = get_samples_mask(X_val, path)
            if np.sum(node_mask) == 0:
                return pruned_node
            subtree_predictions = np.zeros(np.sum(node_mask))
            for (i, x) in enumerate(X_val[node_mask]):
                subtree_predictions[i] = self._predict_sample(x, pruned_node)
            subtree_error = np.mean((y_val[node_mask] - subtree_predictions) ** 2)
            leaf_prediction = node.value
            leaf_error = np.mean((y_val[node_mask] - leaf_prediction) ** 2)
            if leaf_error <= subtree_error + self.ccp_alpha:
                return TreeNode(feature_idx=None, threshold=None, left=None, right=None, value=node.value, samples=node.samples, impurity=node.impurity)
            return pruned_node
        return prune_recursive(node, [])

    def prune(self, X_val: Union[FeatureSet, np.ndarray], y_val: Union[np.ndarray, list], **kwargs) -> 'DecisionTreeRegressor':
        """
        Prune the decision tree using a validation set to improve generalization.
        
        This method applies cost-complexity pruning to reduce the size of the tree
        and prevent overfitting. It uses the validation set to determine the optimal
        subtree that minimizes the prediction error.
        
        Parameters
        ----------
        X_val : FeatureSet or array-like of shape (n_samples, n_features)
            Validation input samples.
        y_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for validation samples.
        **kwargs : dict
            Additional pruning parameters.
            
        Returns
        -------
        DecisionTreeRegressor
            Pruned estimator.
        """
        if self.tree_ is None:
            raise ValueError('Model has not been fitted yet.')
        (X_val, y_val) = self._validate_inputs(X_val, y_val)
        self.tree_ = self._prune_tree(self.tree_, X_val, y_val)
        return self