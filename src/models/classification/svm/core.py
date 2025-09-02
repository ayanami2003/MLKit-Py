from general.base_classes.model_base import BaseModel
from typing import Optional, Union, Callable
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

class LinearSVMClassifier(BaseModel):

    def __init__(self, C: float=1.0, tol: float=0.0001, max_iter: int=1000, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Linear SVM Classifier.
        
        Args:
            C (float): Regularization parameter. Must be positive. Default is 1.0.
            tol (float): Tolerance for stopping criterion. Default is 1e-4.
            max_iter (int): Maximum number of iterations. Default is 1000.
            random_state (Optional[int]): Random seed for reproducibility. Default is None.
            name (Optional[str]): Name of the model instance. Default is class name.
        """
        super().__init__(name=name)
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> 'LinearSVMClassifier':
        """
        Fit the linear SVM classifier according to the given training data.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training data features.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
            y (Union[np.ndarray, list]): Target values (class labels) for training.
                Shape should be (n_samples,) for binary classification.
                
        Returns:
            LinearSVMClassifier: Fitted classifier instance.
            
        Raises:
            ValueError: If input dimensions don't match or invalid parameters.
        """
        if self.C <= 0:
            raise ValueError('C must be positive')
        if self.tol <= 0:
            raise ValueError('tol must be positive')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        if y_array.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        unique_labels = np.unique(y_array)
        if len(unique_labels) != 2:
            raise ValueError('Only binary classification is supported')
        self.classes_ = unique_labels
        y_mapped = np.where(y_array == unique_labels[0], -1, 1)
        self.n_features_in_ = X_array.shape[1]
        (n_samples, n_features) = X_array.shape
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        errors = np.zeros(n_samples)
        for iteration in range(self.max_iter):
            num_changed = 0
            for i in range(n_samples):
                Ei = self._compute_error(i, X_array, y_mapped, errors)
                errors[i] = Ei
                if y_mapped[i] * Ei < -self.tol and self.alphas[i] < self.C or (y_mapped[i] * Ei > self.tol and self.alphas[i] > 0):
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    Ej = self._compute_error(j, X_array, y_mapped, errors)
                    errors[j] = Ej
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    if y_mapped[i] != y_mapped[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if L == H:
                        continue
                    eta = 2 * np.dot(X_array[i], X_array[j]) - np.dot(X_array[i], X_array[i]) - np.dot(X_array[j], X_array[j])
                    if eta >= 0:
                        continue
                    self.alphas[j] = alpha_j_old - y_mapped[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-05:
                        continue
                    self.alphas[i] = alpha_i_old + y_mapped[i] * y_mapped[j] * (alpha_j_old - self.alphas[j])
                    b1 = self.b - Ei - y_mapped[i] * (self.alphas[i] - alpha_i_old) * np.dot(X_array[i], X_array[i]) - y_mapped[j] * (self.alphas[j] - alpha_j_old) * np.dot(X_array[i], X_array[j])
                    b2 = self.b - Ej - y_mapped[i] * (self.alphas[i] - alpha_i_old) * np.dot(X_array[i], X_array[j]) - y_mapped[j] * (self.alphas[j] - alpha_j_old) * np.dot(X_array[j], X_array[j])
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    num_changed += 1
                    errors[i] = self._compute_error(i, X_array, y_mapped, errors)
                    errors[j] = self._compute_error(j, X_array, y_mapped, errors)
            if num_changed == 0:
                break
        sv_indices = np.where(self.alphas > 1e-06)[0]
        self.support_ = sv_indices
        self.support_vectors_ = X_array[sv_indices]
        self.dual_coef_ = (self.alphas[sv_indices] * y_mapped[sv_indices]).reshape(1, -1)
        self.coef_ = np.sum((self.alphas * y_mapped).reshape(-1, 1) * X_array, axis=0)
        self._is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Perform classification on samples in X.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Input data for prediction.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
                
        Returns:
            np.ndarray: Predicted class labels for X with shape (n_samples,).
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise RuntimeError('Model must be fitted before making predictions')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X_array.shape[1]} features, but {self.n_features_in_} were expected')
        decision_values = np.dot(X_array, self.coef_) + self.b
        predictions = np.where(decision_values >= 0, self.classes_[1], self.classes_[0])
        return predictions

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Input data for probability prediction.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
                
        Returns:
            np.ndarray: Predicted class probabilities for X with shape (n_samples, n_classes).
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise RuntimeError('Model must be fitted before making predictions')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X_array.shape[1]} features, but {self.n_features_in_} were expected')
        decision_values = np.dot(X_array, self.coef_) + self.b
        prob_positive = 1 / (1 + np.exp(-decision_values))
        prob_negative = 1 - prob_positive
        probabilities = np.column_stack([prob_negative, prob_positive])
        return probabilities

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Test samples.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
            y (Union[np.ndarray, list]): True labels for X.
                Shape should be (n_samples,).
                
        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise RuntimeError('Model must be fitted before scoring')
        predictions = self.predict(X)
        y_array = np.asarray(y)
        accuracy = np.mean(predictions == y_array)
        return float(accuracy)

    def get_support_vectors(self) -> np.ndarray:
        """
        Return the support vectors.
        
        Returns:
            np.ndarray: Support vectors used in decision function.
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise RuntimeError('Model must be fitted before retrieving support vectors')
        return self.support_vectors_.copy()

    def _compute_error(self, i, X, y, errors):
        """Helper method to compute error for i-th sample"""
        decision_value = np.sum(self.alphas * y * np.dot(X, X[i])) + self.b
        return decision_value - y[i]


# ...(code omitted)...


class SVMClassifier(BaseModel):
    """
    Support Vector Machine classifier with configurable kernel selection.
    
    This classifier implements a support vector machine for classification tasks with flexible
    kernel options. Users can choose from predefined kernels including linear and RBF kernels.
    
    Supported kernels:
    - 'linear': Linear kernel
    - 'rbf': Radial Basis Function kernel
    - 'poly': Polynomial kernel
    - 'sigmoid': Sigmoid kernel
    - callable: Custom kernel function
    
    Attributes:
        C (float): Regularization parameter.
        kernel (Union[str, Callable]): Kernel function to use.
        degree (int): Degree for poly kernel.
        gamma (Union[float, str]): Kernel coefficient for rbf, poly and sigmoid.
        coef0 (float): Independent term in poly and sigmoid kernels.
        tol (float): Tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.
        random_state (Optional[int]): Random seed for reproducibility.
        
    Methods:
        fit: Train the classifier on provided data
        predict: Make predictions on new data
        predict_proba: Predict class probabilities
        score: Evaluate model performance
        get_support_vectors: Retrieve the support vectors
    """

    def __init__(self, C: float=1.0, kernel: Union[str, Callable]='rbf', degree: int=3, gamma: Union[float, str]='scale', coef0: float=0.0, tol: float=0.0001, max_iter: int=1000, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the SVM Classifier.
        
        Args:
            C (float): Regularization parameter. Must be positive. Default is 1.0.
            kernel (Union[str, Callable]): Kernel function to use. Can be 'linear', 'rbf', 'poly', 
                'sigmoid', or a callable that takes two arrays and returns a kernel matrix.
                Default is 'rbf'.
            degree (int): Degree for poly kernel. Ignored by other kernels. Default is 3.
            gamma (Union[float, str]): Kernel coefficient for rbf, poly and sigmoid kernels.
                If 'scale', gamma = 1 / (n_features * X.var()) is used.
                If 'auto', gamma = 1 / n_features is used.
                Default is 'scale'.
            coef0 (float): Independent term in poly and sigmoid kernels. 
                It is only significant in poly and sigmoid. Default is 0.0.
            tol (float): Tolerance for stopping criterion. Default is 1e-4.
            max_iter (int): Maximum number of iterations. Default is 1000.
            random_state (Optional[int]): Random seed for reproducibility. Default is None.
            name (Optional[str]): Name of the model instance. Default is class name.
        """
        super().__init__(name=name)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self._fitted = False

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> 'SVMClassifier':
        """
        Fit the SVM classifier according to the given training data.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training data features.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
            y (Union[np.ndarray, list]): Target values (class labels) for training.
                Shape should be (n_samples,) for binary classification.
                
        Returns:
            SVMClassifier: Fitted classifier instance.
            
        Raises:
            ValueError: If input dimensions don't match or invalid parameters.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        if isinstance(y, list):
            y_array = np.array(y)
        else:
            y_array = y
        if len(X_array) != len(y_array):
            raise ValueError('X and y must have the same number of samples')
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_array)
        self.classes_ = self._label_encoder.classes_
        self._svm_model = SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, tol=self.tol, max_iter=self.max_iter, random_state=self.random_state, probability=True)
        self._svm_model.fit(X_array, y_encoded)
        self._fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Perform classification on samples in X.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Input data for prediction.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
                
        Returns:
            np.ndarray: Predicted class labels for X with shape (n_samples,).
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise RuntimeError('Model must be fitted before making predictions')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        predictions = self._svm_model.predict(X_array)
        return self._label_encoder.inverse_transform(predictions)

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Input data for probability prediction.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
                
        Returns:
            np.ndarray: Predicted class probabilities for X with shape (n_samples, n_classes).
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise RuntimeError('Model must be fitted before predicting probabilities')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        return self._svm_model.predict_proba(X_array)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Test samples.
                If FeatureSet, uses the features attribute.
                If ndarray, expects shape (n_samples, n_features).
            y (Union[np.ndarray, list]): True labels for X.
                Shape should be (n_samples,).
                
        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise RuntimeError('Model must be fitted before scoring')
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        if isinstance(y, list):
            y_array = np.array(y)
        else:
            y_array = y
        predictions = self.predict(X_array)
        return float(np.mean(predictions == y_array))

    def get_support_vectors(self) -> np.ndarray:
        """
        Retrieve the support vectors identified during training.
        
        Returns:
            np.ndarray: Support vectors with shape (n_support_vectors, n_features).
            
        Raises:
            RuntimeError: If the model hasn't been fitted yet.
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise RuntimeError('Model must be fitted before retrieving support vectors')
        return self._svm_model.support_vectors_