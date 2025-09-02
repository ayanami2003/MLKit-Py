from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet


# ...(code omitted)...


class LinearDiscriminantAnalysisClassifier(BaseModel):

    def __init__(self, solver: str='svd', shrinkage: Optional[Union[str, float]]=None, priors: Optional[np.ndarray]=None, name: Optional[str]=None):
        """
        Initialize Linear Discriminant Analysis classifier.
        
        Parameters
        ----------
        solver : str, default='svd'
            Solver to use: 'svd', 'lsqr', or 'eigen'
        shrinkage : str or float, optional
            Shrinkage parameter: 'auto', float in [0,1], or None
        priors : array-like, optional
            Prior probabilities of the classes
        name : str, optional
            Name for the classifier instance
        """
        super().__init__(name=name)
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, list]]=None, **kwargs) -> 'LinearDiscriminantAnalysisClassifier':
        """
        Fit the LDA classifier according to the given training data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray or list, optional
            Target values of shape (n_samples,)
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        LinearDiscriminantAnalysisClassifier
            Fitted classifier instance
            
        Raises
        ------
        ValueError
            If data and target dimensions don't match or invalid parameters
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
            if y is None:
                raise ValueError('Target values y must be provided')
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y) if y is not None else None
        if len(X_array.shape) != 2:
            raise ValueError('X must be a 2D array')
        if y_array is None:
            raise ValueError('Target values y must be provided')
        if len(X_array) != len(y_array):
            raise ValueError('X and y must have the same number of samples')
        valid_solvers = ['svd', 'lsqr', 'eigen']
        if self.solver not in valid_solvers:
            raise ValueError(f'solver must be one of {valid_solvers}')
        if self.shrinkage is not None:
            if isinstance(self.shrinkage, str):
                if self.shrinkage != 'auto':
                    raise ValueError("shrinkage can only be 'auto' or a float in [0,1]")
            elif not isinstance(self.shrinkage, (int, float)) or not 0 <= self.shrinkage <= 1:
                raise ValueError("shrinkage must be 'auto' or a float in [0,1]")
        (self.n_samples_, self.n_features_) = X_array.shape
        self.classes_ = np.unique(y_array)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError('LDA requires at least 2 classes')
        if self.priors is None:
            (_, counts) = np.unique(y_array, return_counts=True)
            self.priors_ = counts / self.n_samples_
        else:
            self.priors_ = np.asarray(self.priors)
            if len(self.priors_) != self.n_classes_:
                raise ValueError('Number of priors must match number of classes')
            if not np.isclose(np.sum(self.priors_), 1.0):
                raise ValueError('Priors must sum to 1')
        self.means_ = np.zeros((self.n_classes_, self.n_features_))
        for (i, cls) in enumerate(self.classes_):
            self.means_[i] = np.mean(X_array[y_array == cls], axis=0)
        self.xbar_ = np.mean(X_array, axis=0)
        if self.shrinkage is None:
            self.covariance_ = self._estimate_covariance(X_array, y_array)
        else:
            cov = self._estimate_covariance(X_array, y_array)
            if self.shrinkage == 'auto':
                shrinkage_val = 0.1
            else:
                shrinkage_val = self.shrinkage
            target = np.diag(np.diag(cov))
            self.covariance_ = (1 - shrinkage_val) * cov + shrinkage_val * target
        self._solve_eigensystem(X_array)
        self.is_fitted = True
        return self

    def _estimate_covariance(self, X, y):
        """Estimate the covariance matrix."""
        cov = np.zeros((self.n_features_, self.n_features_))
        for (i, cls) in enumerate(self.classes_):
            X_cls = X[y == cls]
            diff = X_cls - self.means_[i]
            cov += (X_cls.shape[0] - 1) * np.dot(diff.T, diff)
        return cov / (X.shape[0] - self.n_classes_)

    def _solve_eigensystem(self, X_train):
        """Solve the eigenvalue problem for LDA."""
        self._X_train = X_train
        if self.solver == 'svd':
            self._solve_svd()
        elif self.solver == 'eigen':
            self._solve_eigen()
        else:
            self._solve_lsqr()

    def _solve_svd(self):
        """Solve using SVD approach."""
        if self.n_classes_ == 2:
            diff_means = self.means_[1] - self.means_[0]
            try:
                coef = np.linalg.solve(self.covariance_, diff_means)
            except np.linalg.LinAlgError:
                coef = np.dot(np.linalg.pinv(self.covariance_), diff_means)
            self.coef_ = coef.reshape(1, -1)
            intercept = -0.5 * np.dot(np.dot(self.means_[0] + self.means_[1], np.linalg.inv(self.covariance_)), self.means_[1] - self.means_[0]) + np.log(self.priors_[1] / self.priors_[0])
            self.intercept_ = np.array([intercept])
        else:
            self.coef_ = np.zeros((self.n_classes_, self.n_features_))
            self.intercept_ = np.zeros(self.n_classes_)
            try:
                cov_inv = np.linalg.inv(self.covariance_)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(self.covariance_)
            for i in range(self.n_classes_):
                self.coef_[i] = np.dot(cov_inv, self.means_[i])
                self.intercept_[i] = -0.5 * np.dot(np.dot(self.means_[i], cov_inv), self.means_[i]) + np.log(self.priors_[i])

    def _solve_eigen(self):
        """Solve using eigenvalue decomposition."""
        self._solve_svd()

    def _solve_lsqr(self):
        """Solve using least squares approach."""
        self._solve_svd()

    def _validate_inputs(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Validate and convert input data."""
        if not self.is_fitted:
            raise ValueError("This LinearDiscriminantAnalysisClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if len(X_array.shape) != 2:
            raise ValueError('X must be a 2D array')
        if X_array.shape[1] != self.n_features_:
            raise ValueError(f'X has {X_array.shape[1]} features, but LDA is expecting {self.n_features_} features')
        return X_array

    def decision_function(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Apply decision function to input data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Decision function values of shape (n_samples,) for binary classification
            or (n_samples, n_classes) for multi-class classification
        """
        X_array = self._validate_inputs(X)
        if self.n_classes_ == 2:
            scores = np.dot(X_array, self.coef_.T) + self.intercept_
            return scores.ravel()
        else:
            scores = np.zeros((X_array.shape[0], self.n_classes_))
            for i in range(self.n_classes_):
                scores[:, i] = np.dot(X_array, self.coef_[i]) + self.intercept_[i]
            return scores

    def predict(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Make predictions for input data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,)
        """
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            predictions = (scores > 0).astype(int)
            return self.classes_[predictions]
        else:
            indices = np.argmax(scores, axis=1)
            return self.classes_[indices]

    def predict_proba(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Class probabilities of shape (n_samples, n_classes)
        """
        scores = self.decision_function(X)
        if self.n_classes_ == 2:
            proba = 1 / (1 + np.exp(-scores))
            proba = np.column_stack([1 - proba, proba])
        else:
            scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            proba = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
        return proba

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list]) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : FeatureSet or np.ndarray
            Test samples of shape (n_samples, n_features)
        y : np.ndarray or list
            True labels for X of shape (n_samples,)
            
        Returns
        -------
        float
            Mean accuracy of self.predict(X) wrt. y
        """
        predictions = self.predict(X)
        y_array = np.asarray(y)
        return np.mean(predictions == y_array)

    def get_class_means(self) -> np.ndarray:
        """
        Get the class means.
        
        Returns
        -------
        np.ndarray
            Class means of shape (n_classes, n_features)
        """
        if not self.is_fitted:
            raise ValueError('Classifier is not fitted yet.')
        return self.means_

    def get_covariance(self) -> np.ndarray:
        """
        Get the shared covariance matrix.
        
        Returns
        -------
        np.ndarray
            Shared covariance matrix of shape (n_features, n_features)
        """
        if not self.is_fitted:
            raise ValueError('Classifier is not fitted yet.')
        return self.covariance_