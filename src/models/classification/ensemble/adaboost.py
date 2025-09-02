import numpy as np
from typing import Optional, Union, Any, Iterator
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier(BaseModel):
    """
    AdaBoost Classifier implementation for binary and multiclass classification.
    
    This classifier implements the Adaptive Boosting algorithm, which combines multiple weak learners
    to create a strong classifier. It works by iteratively training weak classifiers on reweighted 
    versions of the training data, focusing more on misclassified samples in each iteration.
    
    The classifier supports both binary and multiclass classification problems and can work with
    any sklearn-compatible weak learner (default is DecisionTreeClassifier with max_depth=1).
    
    Attributes
    ----------
    n_estimators : int
        Number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting
        so a large number usually results in better performance.
    learning_rate : float
        Learning rate shrinks the contribution of each classifier by learning_rate.
        There is a trade-off between learning_rate and n_estimators.
    algorithm : str
        Algorithm type - either 'SAMME' or 'SAMME.R'. 'SAMME.R' uses the probability estimates
        to update the additive model, while 'SAMME' uses the classifications only.
    random_state : Optional[int]
        Random state for reproducibility.
    estimator : Any
        Base estimator from which the boosted ensemble is built.
    
    Methods
    -------
    fit(X, y) -> 'AdaBoostClassifier'
        Build a boosted classifier from the training set (X, y).
    predict(X) -> np.ndarray
        Predict classes for X.
    predict_proba(X) -> np.ndarray
        Predict class probabilities for X.
    score(X, y) -> float
        Return the mean accuracy on the given test data and labels.
    staged_predict(X)
        Return predictions for X at each boosting stage.
    staged_predict_proba(X)
        Return class probability estimates for X at each boosting stage.
    """

    def __init__(self, n_estimators: int=50, learning_rate: float=1.0, algorithm: str='SAMME.R', random_state: Optional[int]=None, estimator: Optional[Any]=None, **kwargs: Any):
        """
        Initialize the AdaBoostClassifier.
        
        Parameters
        ----------
        n_estimators : int, default=50
            The maximum number of estimators at which boosting is terminated.
        learning_rate : float, default=1.0
            Weight applied to each classifier at each boosting iteration.
        algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
            The algorithm to use. 'SAMME.R' uses the probability estimates to update the additive model,
            while 'SAMME' uses the classifications only.
        random_state : int, RandomState instance or None, default=None
            Controls the random seed given at each estimator at each boosting iteration.
        estimator : object, default=None
            The base estimator from which the boosted ensemble is built.
            If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
        **kwargs : dict
            Additional parameters to pass to the base estimator.
        """
        super().__init__(name='AdaBoostClassifier')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.estimator = estimator
        self.kwargs = kwargs
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm must be 'SAMME' or 'SAMME.R'")
        if self.estimator is None:
            self.estimator = DecisionTreeClassifier(max_depth=1)

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs: Any) -> 'AdaBoostClassifier':
        """
        Build a boosted classifier from the training set (X, y).
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), default=None
            The target values (class labels).
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        AdaBoostClassifier
            Fitted estimator.
            
        Raises
        ------
        ValueError
            If input data dimensions don't match or if algorithm parameters are invalid.
        """
        if isinstance(X, FeatureSet):
            X_array = X.X
            y_array = X.y if y is None else y
        else:
            X_array = X
            y_array = y
        if X_array is None or y_array is None:
            raise ValueError('X and y must be provided')
        X_array = np.asarray(X_array)
        y_array = np.asarray(y_array)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y_array.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        self.classes_ = np.unique(y_array)
        self.n_classes_ = len(self.classes_)
        self._label_map = {cls: i for (i, cls) in enumerate(self.classes_)}
        y_encoded = np.array([self._label_map[label] for label in y_array])
        n_samples = X_array.shape[0]
        sample_weight = np.full(n_samples, 1 / n_samples)
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        rng = np.random.RandomState(self.random_state)
        for iboost in range(self.n_estimators):
            estimator = type(self.estimator)(**self.estimator.get_params())
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=sample_weight)
            X_resampled = X_array[indices]
            y_resampled = y_encoded[indices]
            estimator.fit(X_resampled, y_resampled)
            if self.algorithm == 'SAMME.R':
                y_predict_proba = estimator.predict_proba(X_array)
                y_predict = np.argmax(y_predict_proba, axis=1)
            else:
                y_predict = estimator.predict(X_array)
            incorrect = y_predict != y_encoded
            estimator_error = np.average(incorrect, weights=sample_weight)
            if estimator_error >= 1.0 - 1.0 / self.n_classes_:
                if iboost == 0:
                    self.estimators_.append(estimator)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(estimator_error)
                break
            if self.algorithm == 'SAMME.R':
                estimator_weight = self.learning_rate * (np.log((1 - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1))
            else:
                estimator_weight = self.learning_rate * (np.log((1 - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1))
            if self.algorithm == 'SAMME.R':
                proba = y_predict_proba
                proba = np.clip(proba, 1e-10, 1 - 1e-10)
                sample_weight *= np.exp(-estimator_weight * ((y_encoded == y_predict).astype(int) - 1.0 / self.n_classes_))
            else:
                sample_weight *= np.exp(estimator_weight * incorrect * (sample_weight > 0))
            sample_weight /= np.sum(sample_weight)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(estimator_error)
        self.estimator_weights_ = np.array(self.estimator_weights_)
        self.estimator_errors_ = np.array(self.estimator_errors_)
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Predict classes for X.
        
        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            The input samples.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        predictions = self.classes_[np.argmax(proba, axis=1)]
        return predictions

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            The input samples.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        if isinstance(X, FeatureSet):
            X_array = X.X
        else:
            X_array = X
        X_array = np.asarray(X_array)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.estimators_[0].n_features_in_:
            raise ValueError('X has {} features, but AdaBoostClassifier was trained with {} features'.format(X_array.shape[1], self.estimators_[0].n_features_in_))
        n_samples = X_array.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        for (estimator, weight) in zip(self.estimators_, self.estimator_weights_):
            if self.algorithm == 'SAMME.R':
                estimator_proba = estimator.predict_proba(X_array)
            else:
                predictions = estimator.predict(X_array)
                estimator_proba = np.zeros((n_samples, self.n_classes_))
                estimator_proba[np.arange(n_samples), predictions] = 1
            proba += estimator_proba * weight
        proba = np.exp(proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer
        return proba

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Mean accuracy of self.predict(X) wrt. y.
        """
        if isinstance(X, FeatureSet):
            X_array = X.X
            y_array = X.y if hasattr(X, 'y') and y is None else y
        else:
            X_array = X
            y_array = y
        predictions = self.predict(X_array)
        return np.mean(predictions == y_array)

    def staged_predict(self, X: Union[FeatureSet, np.ndarray]) -> Iterator[np.ndarray]:
        """
        Return predictions for X at each boosting stage.
        
        This method allows monitoring (i.e. determine the optimal number of iterations)
        by computing the prediction at each stage.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            The input samples.
            
        Yields
        ------
        ndarray of shape (n_samples,)
            The predicted classes at each boosting stage.
        """
        if isinstance(X, FeatureSet):
            X_array = X.X
        else:
            X_array = X
        X_array = np.asarray(X_array)
        n_samples = X_array.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        for (estimator, weight) in zip(self.estimators_, self.estimator_weights_):
            if self.algorithm == 'SAMME.R':
                estimator_proba = estimator.predict_proba(X_array)
            else:
                predictions = estimator.predict(X_array)
                estimator_proba = np.zeros((n_samples, self.n_classes_))
                estimator_proba[np.arange(n_samples), predictions] = 1
            proba += estimator_proba * weight
            stage_proba = np.exp(proba.copy())
            normalizer = stage_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            stage_proba /= normalizer
            predictions = self.classes_[np.argmax(stage_proba, axis=1)]
            yield predictions

    def staged_predict_proba(self, X: Union[FeatureSet, np.ndarray]) -> Iterator[np.ndarray]:
        """
        Return class probability estimates for X at each boosting stage.
        
        This method allows monitoring (i.e. determine the optimal number of iterations)
        by computing the prediction at each stage.
        
        Parameters
        ----------
        X : FeatureSet or array-like of shape (n_samples, n_features)
            The input samples.
            
        Yields
        ------
        ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples at each boosting stage.
        """
        if isinstance(X, FeatureSet):
            X_array = X.X
        else:
            X_array = X
        X_array = np.asarray(X_array)
        n_samples = X_array.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        for (estimator, weight) in zip(self.estimators_, self.estimator_weights_):
            if self.algorithm == 'SAMME.R':
                estimator_proba = estimator.predict_proba(X_array)
            else:
                predictions = estimator.predict(X_array)
                estimator_proba = np.zeros((n_samples, self.n_classes_))
                estimator_proba[np.arange(n_samples), predictions] = 1
            proba += estimator_proba * weight
            stage_proba = np.exp(proba.copy())
            normalizer = stage_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            stage_proba /= normalizer
            yield stage_proba