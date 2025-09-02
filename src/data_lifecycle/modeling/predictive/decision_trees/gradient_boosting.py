from typing import Optional, Dict, Any, List
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class GradientBoostingTreeModel(BaseModel):

    def __init__(self, n_estimators: int=100, learning_rate: float=0.1, max_depth: int=3, min_samples_split: int=2, min_samples_leaf: int=1, subsample: float=1.0, random_state: Optional[int]=None, loss_function: str='ls'):
        super().__init__(name='GradientBoostingTreeModel')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.loss_function = loss_function
        self._trees: List[Dict[str, Any]] = []
        self._initial_prediction: float = 0.0

    def fit(self, X: FeatureSet, y: DataBatch, **kwargs) -> 'GradientBoostingTreeModel':
        """
        Train the gradient boosting model on the provided dataset.
        
        This method builds an ensemble of decision trees by sequentially fitting new trees
        to the residuals of the previous stage. It initializes with a constant prediction
        and then iteratively adds trees that approximate the negative gradient of the loss function.
        
        Args:
            X (FeatureSet): Training features as a FeatureSet containing feature matrix and metadata.
            y (DataBatch): Target values as a DataBatch containing labels.
            **kwargs: Additional fitting parameters (not currently used).
            
        Returns:
            GradientBoostingTreeModel: The fitted model instance.
            
        Raises:
            ValueError: If the dimensions of X and y do not match.
            RuntimeError: If training fails due to numerical issues.
        """
        if X.features is None or y.labels is None:
            raise ValueError('X.features and y.labels must not be None')
        if X.features.shape[0] != len(y.labels):
            raise ValueError('Number of samples in X and y must match')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_data = X.features
        y_data = np.array(y.labels)
        if self.loss_function == 'ls':
            self._initial_prediction = np.mean(y_data)
        elif self.loss_function == 'lad':
            self._initial_prediction = np.median(y_data)
        else:
            raise ValueError(f'Unsupported loss function: {self.loss_function}')
        predictions = np.full_like(y_data, self._initial_prediction, dtype=float)
        self._trees = []
        for _ in range(self.n_estimators):
            if self.loss_function == 'ls':
                residuals = y_data - predictions
            elif self.loss_function == 'lad':
                residuals = np.sign(y_data - predictions)
            tree = self._create_simple_tree(X_data, residuals)
            tree_predictions = self._predict_with_tree(tree, X_data)
            predictions += self.learning_rate * tree_predictions
            self._trees.append(tree)
        self.is_fitted = True
        return self

    def predict(self, X: FeatureSet, **kwargs) -> List[float]:
        """
        Make predictions on new data using the trained gradient boosting model.
        
        For each sample, this method computes the sum of predictions from all trees in the ensemble,
        starting from the initial prediction value. The final prediction represents the accumulated
        corrections learned during training.
        
        Args:
            X (FeatureSet): Input features for prediction.
            **kwargs: Additional prediction parameters (not currently used).
            
        Returns:
            List[float]: Predicted values for each sample in X.
            
        Raises:
            RuntimeError: If called before the model is fitted.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before making predictions.')
        X_data = X.features
        predictions = np.full(X_data.shape[0], self._initial_prediction, dtype=float)
        for tree in self._trees:
            tree_predictions = self._predict_with_tree(tree, X_data)
            predictions += self.learning_rate * tree_predictions
        return predictions.tolist()

    def score(self, X: FeatureSet, y: DataBatch, **kwargs) -> float:
        """
        Evaluate model performance on test data using mean squared error.
        
        Computes the mean squared error between true target values and model predictions.
        Lower values indicate better performance.
        
        Args:
            X (FeatureSet): Test features.
            y (DataBatch): True target values.
            **kwargs: Additional scoring parameters (not currently used).
            
        Returns:
            float: Mean squared error score.
            
        Raises:
            RuntimeError: If called before the model is fitted.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before scoring.')
        predictions = np.array(self.predict(X))
        true_values = np.array(y.labels)
        if len(predictions) != len(true_values):
            raise ValueError('Number of predictions must match number of true values.')
        mse = np.mean((true_values - predictions) ** 2)
        return float(mse)

    def _create_simple_tree(self, X: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Create a simple decision tree (stump) for the given data and residuals.
        
        Args:
            X: Feature matrix
            residuals: Residuals to fit
            
        Returns:
            Dictionary representing a simple decision tree
        """
        best_split = None
        best_gain = -np.inf
        (n_samples, n_features) = X.shape
        if n_samples < self.min_samples_split:
            return {'value': np.mean(residuals) if len(residuals) > 0 else 0.0}
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            if len(unique_values) < 2:
                continue
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                if left_count < self.min_samples_split or right_count < self.min_samples_split:
                    continue
                left_residuals = residuals[left_mask]
                right_residuals = residuals[right_mask]
                if len(left_residuals) == 0 or len(right_residuals) == 0:
                    continue
                total_var = np.var(residuals)
                left_var = np.var(left_residuals)
                right_var = np.var(right_residuals)
                weighted_var = (len(left_residuals) * left_var + len(right_residuals) * right_var) / len(residuals)
                gain = total_var - weighted_var if not np.isnan(total_var) else 0
                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature_index': feature_idx, 'threshold': threshold, 'left_child': {'value': np.mean(left_residuals)}, 'right_child': {'value': np.mean(right_residuals)}}
        if best_split is None or best_gain <= 0:
            return {'value': np.mean(residuals) if len(residuals) > 0 else 0.0}
        return best_split

    def _predict_with_tree(self, tree: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a single tree.
        
        Args:
            tree: Tree dictionary
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if 'value' in tree:
            return np.full(X.shape[0], tree['value'])
        if 'feature_idx' not in tree or 'threshold' not in tree:
            return np.zeros(X.shape[0])
        feature_values = X[:, tree['feature_idx']]
        predictions = np.where(feature_values <= tree['threshold'], tree['left_value'], tree['right_value'])
        return predictions