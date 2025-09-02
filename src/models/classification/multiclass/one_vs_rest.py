from typing import Any, Optional, List, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
import copy

class OneVsRestClassifier(BaseModel):

    def __init__(self, estimator: BaseModel, name: Optional[str]=None):
        """
        Initialize the OneVsRestClassifier.

        Parameters
        ----------
        estimator : BaseModel
            A binary classifier that implements the BaseModel interface.
            This will be used as the base classifier for each OvR problem.
        name : Optional[str], optional
            Name of the classifier. If None, uses the class name.
        """
        super().__init__(name)
        self.estimator = estimator
        self.estimators_: List[BaseModel] = []
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs) -> 'OneVsRestClassifier':
        """
        Fit the One-vs-Rest classifier.

        This trains one binary classifier per class, where each classifier
        learns to distinguish one class from all others.

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training features. Can be a FeatureSet or numpy array.
        y : Union[np.ndarray, List]
            Target values as class labels (integers or strings).
        **kwargs : dict
            Additional fitting parameters passed to the base estimator.

        Returns
        -------
        OneVsRestClassifier
            Fitted classifier instance.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        self.classes_ = np.unique(y_array)
        if len(self.classes_) < 2:
            raise ValueError('OneVsRestClassifier requires at least 2 classes')
        self.estimators_ = []
        for class_label in self.classes_:
            estimator_copy = copy.deepcopy(self.estimator)
            binary_y = (y_array == class_label).astype(int)
            estimator_copy.fit(X_array, binary_y, **kwargs)
            self.estimators_.append(estimator_copy)
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class labels for samples.

        The predicted class is the one that receives the highest confidence
        score from the set of binary classifiers.

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input features for prediction.
        **kwargs : dict
            Additional prediction parameters passed to the base estimators.

        Returns
        -------
        np.ndarray
            Predicted class labels for each sample.
        """
        if not self.is_fitted:
            raise ValueError("This OneVsRestClassifier instance is not fitted yet. Call 'fit' before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        n_samples = X_array.shape[0]
        scores = self.decision_function(X_array, **kwargs)
        if len(self.classes_) == 2:
            predictions = np.where(scores.ravel() > 0, self.classes_[1], self.classes_[0])
        else:
            indices = np.argmax(scores, axis=1)
            predictions = self.classes_[indices]
        return predictions

    def decision_function(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Compute the decision function for X.

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input features.

        Returns
        -------
        np.ndarray
            Decision function values for each sample and class.
        """
        if not self.is_fitted:
            raise ValueError("This OneVsRestClassifier instance is not fitted yet. Call 'fit' before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        n_samples = X_array.shape[0]
        if len(self.classes_) == 2:
            if hasattr(self.estimators_[0], 'decision_function'):
                scores = self.estimators_[0].decision_function(X_array, **kwargs)
            else:
                scores = self.estimators_[0].predict(X_array, **kwargs)
            return scores.ravel() if scores.ndim > 1 else scores
        else:
            scores = np.zeros((n_samples, len(self.estimators_)))
            for (i, estimator) in enumerate(self.estimators_):
                if hasattr(estimator, 'decision_function'):
                    scores[:, i] = estimator.decision_function(X_array, **kwargs)
                else:
                    scores[:, i] = estimator.predict(X_array, **kwargs)
        return scores

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class probabilities for samples.

        The probability of a sample belonging to a class is determined
        by the confidence of the corresponding binary classifier.

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input features for prediction.
        **kwargs : dict
            Additional prediction parameters passed to the base estimators.

        Returns
        -------
        np.ndarray
            Probability estimates for each class.
        """
        if not self.is_fitted:
            raise ValueError("This OneVsRestClassifier instance is not fitted yet. Call 'fit' before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        n_samples = X_array.shape[0]
        if len(self.classes_) == 2:
            if hasattr(self.estimators_[0], 'predict_proba'):
                proba = self.estimators_[0].predict_proba(X_array, **kwargs)
                return proba[:, 1:] if proba.shape[1] > 1 else proba
            else:
                scores = self.decision_function(X_array, **kwargs)
                proba_positive = 1 / (1 + np.exp(-scores.ravel()))
                proba = np.column_stack([1 - proba_positive, proba_positive])
                return proba[:, 1:]
        else:
            proba = np.zeros((n_samples, len(self.estimators_)))
            for (i, estimator) in enumerate(self.estimators_):
                if hasattr(estimator, 'predict_proba'):
                    binary_proba = estimator.predict_proba(X_array, **kwargs)
                    if binary_proba.shape[1] > 1:
                        proba[:, i] = binary_proba[:, 1]
                    else:
                        proba[:, i] = binary_proba[:, 0]
                else:
                    scores = self.decision_function(X_array, **kwargs)[:, i]
                    proba[:, i] = 1 / (1 + np.exp(-scores))
            normalizer = proba.sum(axis=1, keepdims=True)
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer
            return proba

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Test samples.
        y : Union[np.ndarray, List]
            True labels for X.
        **kwargs : dict
            Additional scoring parameters.

        Returns
        -------
        float
            Mean accuracy of self.predict(X) wrt. y.
        """
        if not self.is_fitted:
            raise ValueError("This OneVsRestClassifier instance is not fitted yet. Call 'fit' before using this estimator.")
        y_pred = self.predict(X, **kwargs)
        if isinstance(y, list):
            y_true = np.array(y)
        else:
            y_true = np.asarray(y)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        accuracy = np.mean(y_pred == y_true)
        return float(accuracy)