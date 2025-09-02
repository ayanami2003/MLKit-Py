from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from general.base_classes.model_base import BaseModel
from typing import Optional, List, Union
import numpy as np
from scipy.optimize import minimize
from collections import Counter

class SupportVectorMachineClassifier(BaseModel):

    def __init__(self, kernel: str='rbf', C: float=1.0, gamma: Union[str, float]='scale', degree: int=3, coef0: float=0.0, probability: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Support Vector Machine classifier.
        
        Args:
            kernel (str): Specifies the kernel type to be used in the algorithm.
                         Options include 'linear', 'poly', 'rbf', 'sigmoid'. Defaults to 'rbf'.
            C (float): Regularization parameter. The strength of the regularization is
                      inversely proportional to C. Must be strictly positive. Defaults to 1.0.
            gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
                                 If 'scale', gamma = 1 / (n_features * X.var())
                                 If 'auto', gamma = 1 / n_features
                                 If float, must be non-negative. Defaults to 'scale'.
            degree (int): Degree of the polynomial kernel function ('poly').
                         Ignored by all other kernels. Must be non-negative. Defaults to 3.
            coef0 (float): Independent term in kernel function.
                          It is only significant in 'poly' and 'sigmoid'. Defaults to 0.0.
            probability (bool): Whether to enable probability estimates.
                               Defaults to False.
            random_state (Optional[int]): Random state for reproducibility.
                                         Defaults to None.
            name (Optional[str]): Name of the model. Defaults to class name.
        """
        super().__init__(name=name)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.probability = probability
        self.random_state = random_state
        if self.C <= 0:
            raise ValueError('C must be strictly positive')
        if self.kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            raise ValueError("kernel must be one of 'linear', 'poly', 'rbf', 'sigmoid'")
        if isinstance(self.gamma, str) and self.gamma not in ['scale', 'auto']:
            raise ValueError("gamma must be 'scale', 'auto', or a non-negative float")
        if isinstance(self.gamma, float) and self.gamma < 0:
            raise ValueError('gamma must be non-negative if float')
        if self.degree < 0:
            raise ValueError('degree must be non-negative')

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix between two datasets.
        
        Args:
            X1 (np.ndarray): First dataset of shape (n_samples1, n_features).
            X2 (np.ndarray): Second dataset of shape (n_samples2, n_features).
            
        Returns:
            np.ndarray: Kernel matrix of shape (n_samples1, n_samples2).
        """
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            gamma = self._get_gamma_value(X1.shape[1])
            return (np.dot(X1, X2.T) * gamma + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            gamma = self._get_gamma_value(X1.shape[1])
            return np.exp(-gamma * dist_sq)
        elif self.kernel == 'sigmoid':
            gamma = self._get_gamma_value(X1.shape[1])
            return np.tanh(np.dot(X1, X2.T) * gamma + self.coef0)
        else:
            raise ValueError(f'Unsupported kernel: {self.kernel}')

    def fit(self, X: FeatureSet, y: DataBatch, **kwargs) -> 'SupportVectorMachineClassifier':
        """
        Train the SVM classifier on the provided data.
        
        Args:
            X (FeatureSet): Training feature set containing feature matrix and metadata.
            y (DataBatch): Training labels corresponding to the feature set.
            **kwargs: Additional fitting parameters.
            
        Returns:
            SupportVectorMachineClassifier: Fitted classifier instance.
            
        Raises:
            ValueError: If the dimensions of X and y do not match.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_data = X.features
        y_data = np.array(y.labels)
        if hasattr(X_data, 'values'):
            X_data = X_data.values
        if hasattr(y_data, 'values'):
            y_data = y_data.values
        if X_data.shape[0] != len(y_data):
            raise ValueError(f'X and y have incompatible shapes: X has {X_data.shape[0]} samples, y has {len(y_data)} samples')
        (self.n_samples_, self.n_features_) = X_data.shape
        if self.gamma == 'scale':
            self.gamma_ = 1.0 / (self.n_features_ * X_data.var())
        elif self.gamma == 'auto':
            self.gamma_ = 1.0 / self.n_features_
        else:
            self.gamma_ = self.gamma
        self.classes_ = np.unique(y_data)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ == 2:
            y_encoded = np.where(y_data == self.classes_[0], -1, 1)
            self._solve_dual_problem(X_data, y_encoded)
            self.class_mapping_ = {self.classes_[0]: -1, self.classes_[1]: 1}
            self.inverse_class_mapping_ = {-1: self.classes_[0], 1: self.classes_[1]}
        else:
            self._fit_multiclass(X_data, y_data)
        if self.probability:
            self._fit_probability(X_data, y_data)
        return self

    def _solve_dual_problem(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Solve the dual optimization problem for SVM.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Encoded labels of shape (n_samples,).
        """
        K = self._compute_kernel(X, X)

        def objective(alpha):
            return 0.5 * np.sum(alpha[:, None] * alpha[None, :] * y[:, None] * y[None, :] * K) - np.sum(alpha)

        def gradient(alpha):
            return np.sum(alpha[:, None] * y[:, None] * y[None, :] * K, axis=1) - 1
        constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y), 'jac': lambda alpha: y}
        bounds = [(0, self.C) for _ in range(len(y))]
        alpha0 = np.zeros(len(y))
        res = minimize(objective, alpha0, method='SLSQP', jac=gradient, bounds=bounds, constraints=constraints, options={'ftol': 1e-09})
        self.alpha_ = res.x
        sv_indices = self.alpha_ > 1e-06
        self.support_vectors_ = X[sv_indices]
        self.support_vector_labels_ = y[sv_indices]
        self.support_vector_alphas_ = self.alpha_[sv_indices]
        self.bias_ = 0
        n_sv = 0
        for i in range(len(sv_indices)):
            if sv_indices[i]:
                self.bias_ += y[i] - np.sum(self.alpha_ * y * K[i, :])
                n_sv += 1
        if n_sv > 0:
            self.bias_ /= n_sv

    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit SVM for multi-class classification using one-vs-one approach.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Labels of shape (n_samples,).
        """
        self.binary_classifiers_ = {}
        for i in range(self.n_classes_):
            for j in range(i + 1, self.n_classes_):
                (class_i, class_j) = (self.classes_[i], self.classes_[j])
                mask = (y == class_i) | (y == class_j)
                X_binary = X[mask]
                y_binary = y[mask]
                y_encoded = np.where(y_binary == class_i, -1, 1)
                classifier = SupportVectorMachineClassifier(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, coef0=self.coef0, random_state=self.random_state)
                classifier._solve_dual_problem(X_binary, y_encoded)
                self.binary_classifiers_[class_i, class_j] = classifier

    def _fit_probability(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit probability model using Platt scaling.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Labels of shape (n_samples,).
        """
        if self.n_classes_ == 2:
            decision_values = self.decision_function(X)
            targets = np.where(y == self.classes_[1], 1, 0)
            X_platt = np.vstack([decision_values, np.ones(len(decision_values))]).T
            try:
                coeffs = np.linalg.lstsq(X_platt, targets, rcond=None)[0]
                self.platt_a_ = coeffs[0]
                self.platt_b_ = coeffs[1]
            except np.linalg.LinAlgError:
                self.platt_a_ = 0
                self.platt_b_ = 0
        else:
            pass

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values for samples in X.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Decision function values of shape (n_samples,) for binary classification
                       or (n_samples, n_classes*(n_classes-1)/2) for multi-class.
        """
        if self.n_classes_ == 2:
            K = self._compute_kernel(X, self.support_vectors_)
            decisions = np.sum(self.support_vector_alphas_ * self.support_vector_labels_ * K, axis=1) + self.bias_
            return decisions
        else:
            n_samples = X.shape[0]
            n_binary = len(self.binary_classifiers_)
            decisions = np.zeros((n_samples, n_binary))
            for (idx, ((class_i, class_j), classifier)) in enumerate(self.binary_classifiers_.items()):
                K = classifier._compute_kernel(X, classifier.support_vectors_)
                dec = np.sum(classifier.support_vector_alphas_ * classifier.support_vector_labels_ * K, axis=1) + classifier.bias_
                decisions[:, idx] = dec
            return decisions

    def predict(self, X: FeatureSet, **kwargs) -> List[int]:
        """
        Make predictions on new data.
        
        Args:
            X (FeatureSet): Feature set to make predictions on.
            **kwargs: Additional prediction parameters.
            
        Returns:
            List[int]: Predicted class labels for each sample.
        """
        X_data = X.get_feature_matrix()
        if hasattr(X_data, 'values'):
            X_data = X_data.values
        if self.n_classes_ == 2:
            decisions = self.decision_function(X_data)
            predictions = np.where(decisions >= 0, self.classes_[1], self.classes_[0])
        else:
            n_samples = X_data.shape[0]
            votes = np.zeros((n_samples, self.n_classes_))
            decisions = self.decision_function(X_data)
            for (idx, ((class_i, class_j), _)) in enumerate(self.binary_classifiers_.items()):
                mask_i = decisions[:, idx] < 0
                mask_j = decisions[:, idx] >= 0
                i_idx = np.where(self.classes_ == class_i)[0][0]
                j_idx = np.where(self.classes_ == class_j)[0][0]
                votes[mask_i, i_idx] += 1
                votes[mask_j, j_idx] += 1
            predicted_indices = np.argmax(votes, axis=1)
            predictions = self.classes_[predicted_indices]
        return predictions.tolist()

    def predict_proba(self, X: FeatureSet, **kwargs) -> List[List[float]]:
        """
        Output probabilities for each class.
        
        Args:
            X (FeatureSet): Feature set to make predictions on.
            **kwargs: Additional prediction parameters.
            
        Returns:
            List[List[float]]: Probability estimates for each class.
            
        Note:
            Requires probability=True during initialization.
        """
        if not self.probability:
            raise ValueError('Probability estimates are not available. Set probability=True during initialization.')
        X_data = X.get_feature_matrix()
        if hasattr(X_data, 'values'):
            X_data = X_data.values
        if self.n_classes_ == 2:
            decisions = self.decision_function(X_data)
            prob_positive = 1.0 / (1.0 + np.exp(self.platt_a_ * decisions + self.platt_b_))
            prob_negative = 1.0 - prob_positive
            probabilities = np.column_stack([prob_negative, prob_positive])
        else:
            decisions = self.decision_function(X_data)
            binary_probs = 1.0 / (1.0 + np.exp(-decisions))
            n_samples = X_data.shape[0]
            probabilities = np.zeros((n_samples, self.n_classes_))
            for (idx, ((class_i, class_j), _)) in enumerate(self.binary_classifiers_.items()):
                i_idx = np.where(self.classes_ == class_i)[0][0]
                j_idx = np.where(self.classes_ == class_j)[0][0]
                probabilities[:, i_idx] += 1 - binary_probs[:, idx]
                probabilities[:, j_idx] += binary_probs[:, idx]
            probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        return probabilities.tolist()

    def score(self, X: FeatureSet, y: DataBatch, **kwargs) -> float:
        """
        Calculate accuracy score on test data.
        
        Args:
            X (FeatureSet): Test feature set.
            y (DataBatch): True labels for the test set.
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: Accuracy score (fraction of correctly classified samples).
        """
        y_pred = self.predict(X)
        y_true = y.get_labels()
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        correct = np.sum(np.array(y_pred) == np.array(y_true))
        return float(correct) / len(y_true)