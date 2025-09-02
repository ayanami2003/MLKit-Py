from typing import Optional, Union, List, Dict, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class OrderedBoostingClassifier(BaseModel):

    def __init__(self, n_estimators: int=100, learning_rate: float=0.1, feature_order: Optional[List[str]]=None, max_depth: Optional[int]=3, random_state: Optional[int]=None, **kwargs: Any):
        """
        Initialize the OrderedBoostingClassifier.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting stages to perform.
        learning_rate : float, default=0.1
            Shrinks the contribution of each classifier.
        feature_order : List[str], optional
            Explicit ordering of features to respect during training.
        max_depth : int, default=3
            Maximum depth of individual estimators.
        random_state : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional parameters passed to the base class.
        """
        super().__init__(name=kwargs.pop('name', None))
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.feature_order = feature_order
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self._estimator_weights = []
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _get_feature_order(self, X: Union[FeatureSet, np.ndarray]) -> List[str]:
        """Extract feature order from input data."""
        if self.feature_order is not None:
            return self.feature_order
        elif isinstance(X, FeatureSet):
            if hasattr(X, 'feature_names') and X.feature_names is not None:
                return list(X.feature_names)
            else:
                if hasattr(X.features, 'shape') and len(X.features.shape) >= 2:
                    n_features = X.features.shape[1]
                else:
                    n_features = 0
                return [f'feature_{i}' for i in range(n_features)]
        else:
            if hasattr(X, 'shape') and len(X.shape) >= 2:
                n_features = X.shape[1]
            else:
                n_features = 0
            return [f'feature_{i}' for i in range(n_features)]

    def _validate_inputs(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, List]]) -> tuple:
        """Validate and preprocess inputs."""
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        if y is not None:
            y_array = np.asarray(y)
        else:
            y_array = None
        if y_array is not None:
            if isinstance(X, FeatureSet):
                if hasattr(X.features, 'shape'):
                    n_samples = X.features.shape[0]
                else:
                    n_samples = len(X.features)
            else:
                n_samples = X_array.shape[0]
            if n_samples != len(y_array):
                raise ValueError('X and y must have the same number of samples')
        return (X_array, y_array)

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, List]]=None, **kwargs: Any) -> 'OrderedBoostingClassifier':
        """
        Train the ordered boosting classifier on the input data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training features. If FeatureSet, feature ordering will be extracted if not explicitly provided.
        y : np.ndarray or List, optional
            Target values.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        OrderedBoostingClassifier
            Fitted classifier.
        """
        (X_array, y_array) = self._validate_inputs(X, y)
        feature_order = self._get_feature_order(X)
        X_array = np.asarray(X_array)
        if len(feature_order) != X_array.shape[1]:
            raise ValueError('Length of feature_order must match number of features in X')
        self._estimators = []
        self._estimator_weights = []
        n_samples = X_array.shape[0]
        sample_weights = np.full(n_samples, 1 / n_samples)
        self.classes_ = np.unique(y_array)
        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            estimator.fit(X_array, y_array, sample_weight=sample_weights)
            predictions = estimator.predict(X_array)
            incorrect = (predictions != y_array).astype(int)
            estimator_error = np.average(incorrect, weights=sample_weights)
            if estimator_error >= 1.0 - 1.0 / len(self.classes_):
                continue
            if estimator_error <= 0:
                estimator_weight = 1.0
            else:
                estimator_weight = self.learning_rate * np.log((1 - estimator_error) / estimator_error) + np.log(len(self.classes_) - 1)
            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= np.sum(sample_weights)
            self._estimators.append(estimator)
            self._estimator_weights.append(estimator_weight)
        self._estimator_weights = np.array(self._estimator_weights)
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input features for prediction.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input features for which to predict probabilities.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        np.ndarray
            Predicted class probabilities.
        """
        if not self._estimators:
            raise ValueError("This OrderedBoostingClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        X_array = np.asarray(X_array)
        feature_order = self._get_feature_order(X)
        if len(feature_order) != X_array.shape[1]:
            raise ValueError('Number of features in X does not match the fitted model')
        n_samples = X_array.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        for (estimator, weight) in zip(self._estimators, self._estimator_weights):
            estimator_probas = estimator.predict_proba(X_array)
            estimator_classes = estimator.classes_
            for (i, class_label) in enumerate(estimator_classes):
                class_idx = np.where(self.classes_ == class_label)[0][0]
                proba[:, class_idx] += weight * estimator_probas[:, i]
        proba_max = np.max(proba, axis=1, keepdims=True)
        proba_exp = np.exp(proba - proba_max)
        proba = proba_exp / np.sum(proba_exp, axis=1, keepdims=True)
        return proba

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Test samples.
        y : np.ndarray or List
            True labels for X.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Mean accuracy of self.predict(X) wrt. y.
        """
        predictions = self.predict(X)
        y_array = np.asarray(y)
        return accuracy_score(y_array, predictions)