from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class NaiveBayesClassifier(BaseModel):

    def __init__(self, distribution: str='gaussian', priors: Optional[np.ndarray]=None, var_smoothing: float=1e-09, name: Optional[str]=None):
        """
        Initialize the Naive Bayes classifier.

        Args:
            distribution (str): The type of distribution to assume for feature likelihoods.
                               Options: 'gaussian', 'multinomial', 'bernoulli'. Default: 'gaussian'.
            priors (Optional[np.ndarray]): Prior probabilities of the classes. If specified,
                                           the priors are not adjusted according to the data.
            var_smoothing (float): Portion of the largest variance added to variances for
                                  calculation stability. Default: 1e-9.
            name (Optional[str]): Name of the model instance.
        """
        super().__init__(name=name)
        valid_distributions = ['gaussian', 'multinomial', 'bernoulli']
        if distribution not in valid_distributions:
            raise ValueError(f'distribution must be one of {valid_distributions}')
        self.distribution = distribution
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> 'NaiveBayesClassifier':
        """
        Train the Naive Bayes classifier on the provided data.

        Args:
            X (Union[FeatureSet, np.ndarray]): Training data features.
            y (Union[np.ndarray, list]): Target values (class labels).
            **kwargs: Additional fitting parameters.

        Returns:
            NaiveBayesClassifier: The fitted classifier.
        """
        if isinstance(X, FeatureSet):
            X = X.features
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if self.priors is None:
            (_, counts) = np.unique(y, return_counts=True)
            self.class_log_prior_ = np.log(counts / y.shape[0])
        else:
            if len(self.priors) != n_classes:
                raise ValueError('Number of priors must match number of classes')
            if not np.isclose(np.sum(self.priors), 1.0):
                raise ValueError('The sum of the priors must be 1')
            self.class_log_prior_ = np.log(self.priors)
        if self.distribution == 'gaussian':
            self._fit_gaussian(X, y)
        elif self.distribution == 'multinomial':
            self._fit_multinomial(X, y)
        elif self.distribution == 'bernoulli':
            self._fit_bernoulli(X, y)
        self.is_fitted = True
        return self

    def _fit_gaussian(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Naive Bayes model."""
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)
        for (i, cls) in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.class_count_[i] = X_cls.shape[0]
            self.theta_[i, :] = np.mean(X_cls, axis=0)
            self.var_[i, :] = np.var(X_cls, axis=0)
        epsilon = self.var_smoothing * np.max(self.var_) if np.max(self.var_) > 0 else self.var_smoothing
        self.var_ += epsilon

    def _fit_multinomial(self, X: np.ndarray, y: np.ndarray):
        """Fit Multinomial Naive Bayes model."""
        if np.any(X < 0):
            raise ValueError('Multinomial distribution requires non-negative features')
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)
        self.feature_count_ = np.zeros((n_classes, n_features))
        for (i, cls) in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.class_count_[i] = X_cls.shape[0]
            feature_count = np.sum(X_cls, axis=0)
            self.feature_count_[i, :] = feature_count
            total_count = np.sum(feature_count)
            self.feature_log_prob_[i, :] = np.log((feature_count + 1) / (total_count + n_features))

    def _fit_bernoulli(self, X: np.ndarray, y: np.ndarray):
        """Fit Bernoulli Naive Bayes model."""
        if np.any((X < 0) | (X > 1)):
            raise ValueError('Bernoulli distribution requires binary features in range [0, 1]')
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)
        self.feature_count_ = np.zeros((n_classes, n_features))
        for (i, cls) in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.class_count_[i] = X_cls.shape[0]
            feature_count = np.sum(X_cls, axis=0)
            self.feature_count_[i, :] = feature_count
            alpha = 1.0
            self.feature_log_prob_[i, :] = np.log((feature_count + alpha) / (X_cls.shape[0] + 2 * alpha))

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X (Union[FeatureSet, np.ndarray]): Samples to predict.
            **kwargs: Additional prediction parameters.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("This NaiveBayesClassifier instance is not fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X = X.features
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict probability estimates for samples.

        Args:
            X (Union[FeatureSet, np.ndarray]): Samples to predict.
            **kwargs: Additional prediction parameters.

        Returns:
            np.ndarray: Probability estimates for each class.
        """
        if not self.is_fitted:
            raise ValueError("This NaiveBayesClassifier instance is not fitted yet. Call 'fit' before using this method.")
        if isinstance(X, FeatureSet):
            X = X.features
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        if self.distribution == 'gaussian':
            jll = self._joint_log_likelihood_gaussian(X)
        elif self.distribution == 'multinomial':
            jll = self._joint_log_likelihood_multinomial(X)
        elif self.distribution == 'bernoulli':
            jll = self._joint_log_likelihood_bernoulli(X)
        log_proba = jll - np.max(jll, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def _joint_log_likelihood_gaussian(self, X: np.ndarray) -> np.ndarray:
        """Calculate joint log likelihood for Gaussian distribution."""
        (n_samples, n_features) = X.shape
        n_classes = len(self.classes_)
        jll = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            jll[:, i] += self.class_log_prior_[i]
            mean = self.theta_[i, :]
            var = self.var_[i, :]
            log_prob_per_feature = -0.5 * np.log(2 * np.pi * var) - 0.5 * (X - mean) ** 2 / var
            jll[:, i] += np.sum(log_prob_per_feature, axis=1)
        return jll

    def _joint_log_likelihood_multinomial(self, X: np.ndarray) -> np.ndarray:
        """Calculate joint log likelihood for Multinomial distribution."""
        (n_samples, n_features) = X.shape
        n_classes = len(self.classes_)
        jll = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            jll[:, i] += self.class_log_prior_[i]
            jll[:, i] += np.dot(X, self.feature_log_prob_[i, :])
        return jll

    def _joint_log_likelihood_bernoulli(self, X: np.ndarray) -> np.ndarray:
        """Calculate joint log likelihood for Bernoulli distribution."""
        (n_samples, n_features) = X.shape
        n_classes = len(self.classes_)
        jll = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            jll[:, i] += self.class_log_prior_[i]
            feature_log_prob = self.feature_log_prob_[i, :]
            jll[:, i] += np.dot(X, feature_log_prob)
            jll[:, i] += np.dot(1 - X, np.log(1 - np.exp(feature_log_prob)))
        return jll

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (Union[FeatureSet, np.ndarray]): Test samples.
            y (Union[np.ndarray, list]): True labels for X.
            **kwargs: Additional scoring parameters.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        if not self.is_fitted:
            raise ValueError("This NaiveBayesClassifier instance is not fitted yet. Call 'fit' before using this method.")
        predictions = self.predict(X)
        y = np.array(y)
        return np.mean(predictions == y)