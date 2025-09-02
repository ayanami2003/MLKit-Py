from typing import Optional
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import numpy as np

class RidgeClassifierModel(BaseModel):
    """
    Ridge Classifier for binary or multiclass classification tasks.
    
    This classifier uses Ridge regression with built-in L2 regularization to perform classification.
    It converts binary targets to {-1, 1} and solves the regression problem for each class
    against all others in the multiclass case.
    
    Attributes:
        alpha (float): Regularization strength. Must be a positive float.
        fit_intercept (bool): Whether to calculate the intercept for this model.
        normalize (bool): This parameter is ignored when fit_intercept is set to False.
        copy_X (bool): If True, X will be copied; else, it may be overwritten.
        max_iter (Optional[int]): Maximum number of iterations for conjugate gradient solver.
        tol (float): Precision of the solution.
        class_weight (Optional[dict]): Weights associated with classes.
        solver (str): Solver to use in the computational routines.
        random_state (Optional[int]): Used when solver == 'saga' to shuffle the data.
    
    Methods:
        fit: Fit the ridge classifier according to the given training data.
        predict: Predict class labels for samples in X.
        score: Return the mean accuracy on the given test data and labels.
        decision_function: Predict confidence scores for samples.
    """

    def __init__(self, alpha: float=1.0, fit_intercept: bool=True, normalize: bool=False, copy_X: bool=True, max_iter: Optional[int]=None, tol: float=0.0001, class_weight: Optional[dict]=None, solver: str='auto', random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the RidgeClassifierModel.
        
        Args:
            alpha: Regularization strength. Must be a positive float.
            fit_intercept: Whether to calculate the intercept for this model.
            normalize: This parameter is ignored when fit_intercept is set to False.
            copy_X: If True, X will be copied; else, it may be overwritten.
            max_iter: Maximum number of iterations for conjugate gradient solver.
            tol: Precision of the solution.
            class_weight: Weights associated with classes.
            solver: Solver to use in the computational routines.
            random_state: Used when solver == 'saga' to shuffle the data.
            name: Optional name for the model.
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.solver = solver
        self.random_state = random_state

    def fit(self, X: FeatureSet, y: DataBatch, **kwargs) -> 'RidgeClassifierModel':
        """
        Fit the ridge classifier according to the given training data.
        
        Args:
            X: Training feature set.
            y: Training labels.
            **kwargs: Additional fitting parameters.
            
        Returns:
            RidgeClassifierModel: Fitted classifier.
        """
        X_data = X.features.copy() if self.copy_X else X.features
        y_data = y.data
        if not isinstance(X_data, np.ndarray):
            X_data = np.array(X_data)
        if not isinstance(y_data, np.ndarray):
            y_data = np.array(y_data)
        (n_samples, n_features) = X_data.shape
        if self.normalize and self.fit_intercept:
            self._X_mean = np.mean(X_data, axis=0)
            self._X_std = np.std(X_data, axis=0)
            self._X_std = np.where(self._X_std == 0, 1.0, self._X_std)
            X_data = (X_data - self._X_mean) / self._X_std
        else:
            self._X_mean = None
            self._X_std = None
        self.classes_ = np.unique(y_data)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('At least two classes are needed for classification')
        if n_classes == 2:
            pos_class = self.classes_[1]
            neg_class = self.classes_[0]
            y_mapped = np.where(y_data == pos_class, 1, -1)
            (coef, intercept) = self._solve_ridge(X_data, y_mapped)
            self.coef_ = coef
            self.intercept_ = intercept
        else:
            self.coef_ = np.empty((n_classes, n_features))
            self.intercept_ = np.zeros((n_classes,))
            for (i, class_label) in enumerate(self.classes_):
                y_binary = np.where(y_data == class_label, 1, -1)
                (coef, intercept) = self._solve_ridge(X_data, y_binary)
                self.coef_[i] = coef
                self.intercept_[i] = intercept
        self.is_fitted = True
        return self

    def _solve_ridge(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Solve the ridge regression problem: min ||Xw - y||^2 + alpha*||w||^2
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            
        Returns:
            tuple: (coef, intercept) - coefficient vector and intercept
        """
        (n_samples, n_features) = X.shape
        if self.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(n_samples), X])
            n_features_with_intercept = n_features + 1
        else:
            X_with_intercept = X
            n_features_with_intercept = n_features
        Id = np.identity(n_features_with_intercept)
        if self.fit_intercept:
            Id[0, 0] = 0
        XtX = np.dot(X_with_intercept.T, X_with_intercept)
        reg_matrix = XtX + self.alpha * Id
        Xty = np.dot(X_with_intercept.T, y)
        try:
            coef_with_intercept = np.linalg.solve(reg_matrix, Xty)
        except np.linalg.LinAlgError:
            (coef_with_intercept, _, _, _) = np.linalg.lstsq(reg_matrix, Xty, rcond=None)
        if self.fit_intercept:
            intercept = coef_with_intercept[0]
            coef = coef_with_intercept[1:]
        else:
            intercept = 0.0
            coef = coef_with_intercept
        return (coef, intercept)

    def predict(self, X: FeatureSet, **kwargs) -> list:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Feature set for which to make predictions.
            **kwargs: Additional prediction parameters.
            
        Returns:
            list: Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        decision_vals = self.decision_function(X)
        decision_vals = np.array(decision_vals)
        if len(self.classes_) == 2:
            predictions = np.where(decision_vals > 0, self.classes_[1], self.classes_[0])
        else:
            indices = np.argmax(decision_vals, axis=1)
            predictions = self.classes_[indices]
        return predictions.tolist()

    def score(self, X: FeatureSet, y: DataBatch, **kwargs) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X: Test feature set.
            y: True labels for X.
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        y_pred = np.array(self.predict(X))
        y_true = np.array(y.data)
        accuracy = np.mean(y_pred == y_true)
        return float(accuracy)

    def decision_function(self, X: FeatureSet, **kwargs) -> list:
        """
        Predict confidence scores for samples.
        
        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.
        
        Args:
            X: Feature set for which to compute confidence scores.
            **kwargs: Additional parameters.
            
        Returns:
            list: Confidence scores per sample.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        X_data = X.features
        if not isinstance(X_data, np.ndarray):
            X_data = np.array(X_data)
        if self._X_mean is not None and self._X_std is not None:
            X_data = (X_data - self._X_mean) / self._X_std
        if len(self.classes_) == 2:
            decision_vals = np.dot(X_data, self.coef_) + self.intercept_
        else:
            decision_vals = np.dot(X_data, self.coef_.T) + self.intercept_
        return decision_vals.tolist()