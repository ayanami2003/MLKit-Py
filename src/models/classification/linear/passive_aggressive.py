from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class PassiveAggressiveClassifier(BaseModel):

    def __init__(self, C: float=1.0, fit_intercept: bool=True, max_iter: int=1000, tol: float=0.001, early_stopping: bool=False, validation_fraction: float=0.1, n_iter_no_change: int=5, shuffle: bool=True, verbose: int=0, random_state: Optional[int]=None, loss: str='hinge', warm_start: bool=False):
        """
        Initialize the PassiveAggressiveClassifier.
        
        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter. The strength of the regularization is
            inversely proportional to C.
        fit_intercept : bool, default=True
            Whether to calculate the intercept for this model.
        max_iter : int, default=1000
            Maximum number of iterations for training.
        tol : float, default=1e-3
            Tolerance for stopping criterion.
        early_stopping : bool, default=False
            Whether to use early stopping.
        validation_fraction : float, default=0.1
            Fraction of training data to use for validation when early stopping is enabled.
        n_iter_no_change : int, default=5
            Number of iterations with no improvement to wait before stopping.
        shuffle : bool, default=True
            Whether to shuffle the training data after each epoch.
        verbose : int, default=0
            Verbosity level.
        random_state : int, default=None
            Random seed for reproducibility.
        loss : str, default="hinge"
            Loss function to use ('hinge' for PA-I, 'squared_hinge' for PA-II).
        warm_start : bool, default=False
            Whether to reuse the solution from previous fits.
        """
        super().__init__(name='PassiveAggressiveClassifier')
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.loss = loss
        self.warm_start = warm_start
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.classes_ = None

    def fit(self, X: Union[np.ndarray, FeatureSet], y: Union[np.ndarray, list], **kwargs) -> 'PassiveAggressiveClassifier':
        """
        Fit the model according to the given training data.
        
        Parameters
        ----------
        X : numpy.ndarray or FeatureSet
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : numpy.ndarray or list
            Target values (class labels).
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        PassiveAggressiveClassifier
            Fitted estimator.
        """
        if isinstance(X, FeatureSet):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        y = np.asarray(y)
        if self.loss not in ['hinge', 'squared_hinge']:
            raise ValueError("loss must be 'hinge' or 'squared_hinge'")
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if self.coef_ is None or not self.warm_start:
            self.coef_ = np.zeros((1, self.n_features_in_)) if n_classes == 2 else np.zeros((n_classes, self.n_features_in_))
            self.intercept_ = np.zeros(1) if n_classes == 2 else np.zeros(n_classes)
        else:
            expected_coef_shape = (1, self.n_features_in_) if n_classes == 2 else (n_classes, self.n_features_in_)
            expected_intercept_shape = (1,) if n_classes == 2 else (n_classes,)
            if self.coef_.shape != expected_coef_shape or self.intercept_.shape != expected_intercept_shape:
                self.coef_ = np.zeros(expected_coef_shape)
                self.intercept_ = np.zeros(expected_intercept_shape)
        if self.early_stopping and self.validation_fraction > 0:
            n_validation = max(int(X.shape[0] * self.validation_fraction), 1)
            n_training = X.shape[0] - n_validation
            (X_train, X_val) = (X[:n_training], X[n_training:])
            (y_train, y_val) = (y[:n_training], y[n_training:])
        else:
            (X_train, y_train) = (X, y)
            (X_val, y_val) = (None, None)
        if n_classes == 2:
            y_train_binary = np.where(y_train == self.classes_[1], 1, -1)
            if y_val is not None:
                y_val_binary = np.where(y_val == self.classes_[1], 1, -1)
        else:
            y_train_binary = np.zeros((len(y_train), n_classes))
            for (i, cls) in enumerate(self.classes_):
                y_train_binary[:, i] = np.where(y_train == cls, 1, -1)
            if y_val is not None:
                y_val_binary = np.zeros((len(y_val), n_classes))
                for (i, cls) in enumerate(self.classes_):
                    y_val_binary[:, i] = np.where(y_val == cls, 1, -1)
        best_score = -np.inf
        no_improvement_count = 0
        best_coef = self.coef_.copy()
        best_intercept = self.intercept_.copy()
        for iteration in range(self.max_iter):
            if self.shuffle:
                indices = np.random.permutation(len(X_train))
                X_train_shuffled = X_train[indices]
                if n_classes == 2:
                    y_train_binary_shuffled = y_train_binary[indices]
                else:
                    y_train_binary_shuffled = y_train_binary[indices]
            else:
                X_train_shuffled = X_train
                y_train_binary_shuffled = y_train_binary
            if n_classes == 2:
                for i in range(len(X_train_shuffled)):
                    self._update_weights_single(X_train_shuffled[i], y_train_binary_shuffled[i], 0)
            else:
                for i in range(len(X_train_shuffled)):
                    for cls_idx in range(n_classes):
                        self._update_weights_single(X_train_shuffled[i], y_train_binary_shuffled[i, cls_idx], cls_idx)
            if self.early_stopping and X_val is not None:
                val_score = self._compute_score(X_val, y_val)
                if val_score > best_score + self.tol:
                    best_score = val_score
                    no_improvement_count = 0
                    best_coef = self.coef_.copy()
                    best_intercept = self.intercept_.copy()
                else:
                    no_improvement_count += 1
                if no_improvement_count >= self.n_iter_no_change:
                    if self.verbose > 0:
                        print(f'Early stopping at iteration {iteration}')
                    break
        if self.early_stopping and X_val is not None and (best_score > -np.inf):
            self.coef_ = best_coef
            self.intercept_ = best_intercept
        self.is_fitted = True
        return self

    def partial_fit(self, X: Union[np.ndarray, FeatureSet], y: Union[np.ndarray, list], classes: Optional[np.ndarray]=None, **kwargs) -> 'PassiveAggressiveClassifier':
        """
        Update the model with a single batch of features and labels.
        
        This method is particularly useful for online learning where data
        arrives in batches.
        
        Parameters
        ----------
        X : numpy.ndarray or FeatureSet
            Training vectors.
        y : numpy.ndarray or list
            Target values (class labels).
        classes : numpy.ndarray, optional
            List of all possible classes. Required for the first call to partial_fit.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        PassiveAggressiveClassifier
            Updated estimator.
        """
        import numpy as np
        if isinstance(X, FeatureSet):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        y = np.asarray(y)
        if self.classes_ is None:
            if classes is None:
                raise ValueError('classes must be provided for the first call to partial_fit')
            self.classes_ = np.unique(classes)
        else:
            new_classes = np.unique(y)
            all_classes = np.unique(np.concatenate([self.classes_, new_classes]))
            if not np.array_equal(all_classes, self.classes_):
                old_n_classes = len(self.classes_)
                self.classes_ = all_classes
                n_classes = len(self.classes_)
                if self.coef_ is not None:
                    if n_classes > old_n_classes:
                        new_coef = np.zeros((n_classes, self.coef_.shape[1])) if n_classes > 2 else np.zeros((1, self.coef_.shape[1]))
                        new_intercept = np.zeros(n_classes) if n_classes > 2 else np.zeros(1)
                        if old_n_classes == 2:
                            new_coef[0] = self.coef_[0] if n_classes > 2 else self.coef_[0]
                        else:
                            new_coef[:old_n_classes] = self.coef_[:old_n_classes] if n_classes > 2 else self.coef_[:old_n_classes].mean(axis=0, keepdims=True)
                        new_intercept[:old_n_classes] = self.intercept_[:old_n_classes] if n_classes > 2 else self.intercept_[:old_n_classes].mean()
                        self.coef_ = new_coef
                        self.intercept_ = new_intercept
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            n_classes = len(self.classes_)
            self.coef_ = np.zeros((1, self.n_features_in_)) if n_classes == 2 else np.zeros((n_classes, self.n_features_in_))
            self.intercept_ = np.zeros(1) if n_classes == 2 else np.zeros(n_classes)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError('Number of features in X does not match the number of features used in the model')
        n_classes = len(self.classes_)
        if n_classes == 2:
            y_binary = np.where(y == self.classes_[1], 1, -1)
            for i in range(len(X)):
                self._update_weights_single(X[i], y_binary[i], 0)
        else:
            y_binary = np.zeros((len(y), n_classes))
            for (i, cls) in enumerate(self.classes_):
                y_binary[:, i] = np.where(y == cls, 1, -1)
            for i in range(len(X)):
                for cls_idx in range(n_classes):
                    self._update_weights_single(X[i], y_binary[i, cls_idx], cls_idx)
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, FeatureSet], **kwargs) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : numpy.ndarray or FeatureSet
            Samples.
        **kwargs : dict
            Additional prediction parameters.
            
        Returns
        -------
        numpy.ndarray
            Predicted class labels for samples in X.
        """
        if not self.is_fitted:
            raise ValueError("This PassiveAggressiveClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        decision = self.decision_function(X)
        if len(self.classes_) == 2:
            predictions = np.where(decision >= 0, self.classes_[1], self.classes_[0])
        else:
            predictions = self.classes_[np.argmax(decision, axis=1)]
        return predictions

    def score(self, X: Union[np.ndarray, FeatureSet], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : numpy.ndarray or FeatureSet
            Test samples.
        y : numpy.ndarray or list
            True labels for X.
        **kwargs : dict
            Additional scoring parameters.
            
        Returns
        -------
        float
            Mean accuracy of self.predict(X) wrt. y.
        """
        if not self.is_fitted:
            raise ValueError("This PassiveAggressiveClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        predictions = self.predict(X)
        y_true = np.asarray(y)
        return np.mean(predictions == y_true)

    def decision_function(self, X: Union[np.ndarray, FeatureSet], **kwargs) -> np.ndarray:
        """
        Predict confidence scores for samples.
        
        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.
        
        Parameters
        ----------
        X : numpy.ndarray or FeatureSet
            Samples.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        numpy.ndarray
            Confidence scores per sample.
        """
        if not self.is_fitted:
            raise ValueError("This PassiveAggressiveClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(X, FeatureSet):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        if len(self.classes_) == 2:
            scores = np.dot(X, self.coef_[0]) + self.intercept_[0]
        else:
            scores = np.dot(X, self.coef_.T) + self.intercept_
        return scores

    def _update_weights_single(self, x: np.ndarray, y: int, cls_idx: int):
        """
        Update weights for a single sample using the Passive-Aggressive update rule.
        
        Parameters
        ----------
        x : numpy.ndarray
            Single sample feature vector.
        y : int
            Binary label (-1 or 1) for the sample.
        cls_idx : int
            Class index for multiclass case.
        """
        if len(self.classes_) == 2:
            decision = np.dot(x, self.coef_[0]) + self.intercept_[0]
            margin = y * decision
            if margin < 1.0:
                loss = 1.0 - margin
                if self.loss == 'hinge':
                    tau = min(self.C, loss / np.dot(x, x) if np.dot(x, x) > 0 else self.C)
                else:
                    denominator = np.dot(x, x) + 1.0 / (2 * self.C) if np.dot(x, x) > 0 else 1.0 / (2 * self.C)
                    tau = loss / denominator
                update = tau * y
                self.coef_[0] += update * x
                if self.fit_intercept:
                    self.intercept_[0] += update
        else:
            decision = np.dot(x, self.coef_[cls_idx]) + self.intercept_[cls_idx]
            margin = y * decision
            if margin < 1.0:
                loss = 1.0 - margin
                if self.loss == 'hinge':
                    tau = min(self.C, loss / np.dot(x, x) if np.dot(x, x) > 0 else self.C)
                else:
                    denominator = np.dot(x, x) + 1.0 / (2 * self.C) if np.dot(x, x) > 0 else 1.0 / (2 * self.C)
                    tau = loss / denominator
                update = tau * y
                self.coef_[cls_idx] += update * x
                if self.fit_intercept:
                    self.intercept_[cls_idx] += update

    def _compute_score(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Compute validation score.
        
        Parameters
        ----------
        X_val : numpy.ndarray
            Validation features.
        y_val : numpy.ndarray
            Validation labels.
            
        Returns
        -------
        float
            Validation accuracy score.
        """
        if not self.is_fitted or self.coef_ is None:
            return -np.inf
        if len(self.classes_) == 2:
            decision = np.dot(X_val, self.coef_[0]) + self.intercept_[0]
            predictions = np.where(decision >= 0, self.classes_[1], self.classes_[0])
        else:
            decision = np.dot(X_val, self.coef_.T) + self.intercept_
            predictions = self.classes_[np.argmax(decision, axis=1)]
        return np.mean(predictions == y_val)