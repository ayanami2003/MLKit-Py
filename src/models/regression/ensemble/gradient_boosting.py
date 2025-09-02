from typing import Optional, Union, Dict, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from src.models.regression.tree_based.decision_tree import DecisionTreeRegressor

class GradientBoostingRegressor(BaseModel):

    def __init__(self, n_estimators: int=100, learning_rate: float=0.1, max_depth: int=3, min_samples_split: int=2, min_samples_leaf: int=1, subsample: float=1.0, loss: str='ls', random_state: Optional[int]=None, **kwargs: Any):
        """
        Initialize the GradientBoostingRegressor.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting stages to perform.
        learning_rate : float, default=0.1
            Learning rate shrinks the contribution of each tree.
        max_depth : int, default=3
            Maximum depth of the individual regression estimators.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        min_samples_leaf : int, default=1
            Minimum number of samples required to be at a leaf node.
        subsample : float, default=1.0
            Fraction of samples to be used for fitting the individual base learners.
        loss : str, default='ls'
            Loss function to be optimized ('ls', 'lad', 'huber', 'quantile').
        random_state : Optional[int], default=None
            Random seed for reproducibility.
        **kwargs : dict
            Additional parameters passed to the base class.
        """
        super().__init__(name=kwargs.pop('name', None))
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state
        self._estimators = []
        self._initial_prediction = 0.0

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs: Any) -> 'GradientBoostingRegressor':
        """
        Build a gradient boosting regressor from the training set (X, y).
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training data features. Can be a FeatureSet or numpy array of shape (n_samples, n_features).
        y : Optional[Union[np.ndarray, list]]
            Target values. Can be a numpy array or list of shape (n_samples,).
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        GradientBoostingRegressor
            Fitted gradient boosting regressor.
            
        Raises
        ------
        ValueError
            If X and y have incompatible shapes.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        if isinstance(y, list):
            y_array = np.array(y)
        else:
            y_array = y
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        self.n_features_ = X_array.shape[1]
        self._estimators = []
        if self.loss == 'ls':
            self._initial_prediction = np.mean(y_array)
        else:
            raise ValueError(f"Loss function '{self.loss}' is not supported")
        predictions = np.full(y_array.shape[0], self._initial_prediction)
        for i in range(self.n_estimators):
            if self.loss == 'ls':
                residuals = y_array - predictions
            else:
                raise ValueError(f"Loss function '{self.loss}' is not supported")
            if self.subsample < 1.0:
                sample_mask = np.random.rand(len(y_array)) < self.subsample
                X_sample = X_array[sample_mask]
                residuals_sample = residuals[sample_mask]
            else:
                X_sample = X_array
                residuals_sample = residuals
            tree = self._create_decision_tree()
            tree.fit(X_sample, residuals_sample)
            tree_predictions = tree.predict(X_array)
            predictions += self.learning_rate * tree_predictions
            self._estimators.append(tree)
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Predict regression target for X.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input data features. Can be a FeatureSet or numpy array of shape (n_samples, n_features).
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Predicted values for X of shape (n_samples,).
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        predictions = np.full(X_array.shape[0], self._initial_prediction)
        for tree in self._estimators:
            predictions += self.learning_rate * tree.predict(X_array)
        return predictions

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs: Any) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Test samples. Can be a FeatureSet or numpy array of shape (n_samples, n_features).
        y : Union[np.ndarray, list]
            True values for X of shape (n_samples,).
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            R^2 score of self.predict(X) wrt. y.
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

    def get_feature_importance(self) -> np.ndarray:
        """
        Return the feature importances (the higher, the more important the feature).
        
        Returns
        -------
        np.ndarray
            Array of feature importances of shape (n_features,).
        """
        if not self._estimators:
            return np.zeros(self.n_features_)
        importances = np.zeros(self.n_features_)
        for tree in self._estimators:
            importances += tree.feature_importances_
        importances /= len(self._estimators)
        return importances

    def _create_decision_tree(self):
        """
        Create a decision tree regressor with specified parameters.
        For now, we'll use a simple implementation. In a full implementation,
        this would be replaced with a proper decision tree regressor.
        """
        return DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)