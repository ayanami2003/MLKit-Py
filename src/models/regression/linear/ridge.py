from typing import Optional, Union, Dict, Any, List
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from scipy import linalg, sparse
from src.models.evaluation.performance_metrics import regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import warnings
from src.models.evaluation.performance_metrics.regression import RegressionErrorMetrics, RSquaredMetrics

class RidgeRegressor(BaseModel):

    def __init__(self, alpha: float=1.0, fit_intercept: bool=True, solver: str='auto', max_iter: Optional[int]=None, tol: float=0.0001, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize Ridge Regressor.
        
        Args:
            alpha (float): Regularization strength. Must be positive.
            fit_intercept (bool): Whether to fit the intercept.
            solver (str): Algorithm to use for solving the optimization problem.
            max_iter (Optional[int]): Maximum iterations for iterative solvers.
            tol (float): Tolerance for stopping criteria.
            random_state (Optional[int]): Seed for random number generator.
            name (Optional[str]): Custom name for the model.
        """
        super().__init__(name=name)
        if alpha < 0:
            raise ValueError('alpha must be non-negative')
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> 'RidgeRegressor':
        """
        Fit the ridge regression model according to the given training data.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training vectors with shape (n_samples, n_features).
            y (Union[np.ndarray, list]): Target values with shape (n_samples,).
            **kwargs: Additional fitting parameters.
            
        Returns:
            RidgeRegressor: Fitted estimator.
        """
        if isinstance(X, FeatureSet):
            X = X.features
        else:
            X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2 or y.shape[1] != 1:
            raise ValueError('y must be a 1D array or 2D column vector')
        (n_samples, n_features) = X.shape
        X_mean = np.zeros(n_features)
        y_mean = 0.0
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y, axis=0)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
        solver = self.solver
        if solver == 'auto':
            solver = 'cholesky' if n_samples <= n_features else 'lsqr'
        if solver == 'cholesky':
            A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
            b = X_centered.T @ y_centered
            coef = np.linalg.solve(A, b)
        elif solver in ['lsqr', 'sparse_cg']:
            A = X_centered.T @ X_centered + self.alpha * np.eye(n_features)
            b = X_centered.T @ y_centered
            (coef, _, _, _) = np.linalg.lstsq(A, b, rcond=None)
        else:
            raise ValueError(f'Solver {solver} is not supported')
        self.coef_ = coef.flatten()
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Samples with shape (n_samples, n_features).
            **kwargs: Additional prediction parameters.
            
        Returns:
            np.ndarray: Predicted values with shape (n_samples,).
        """
        if not self.is_fitted:
            raise ValueError("This RidgeRegressor instance is not fitted yet. Call 'fit' before using this estimator.")
        if isinstance(X, FeatureSet):
            X = X.features
        else:
            X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array or FeatureSet')
        predictions = X @ self.coef_ + self.intercept_
        return predictions.flatten()

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Test samples with shape (n_samples, n_features).
            y (Union[np.ndarray, list]): True values with shape (n_samples,).
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: R^2 score of the prediction.
        """
        if not self.is_fitted:
            raise ValueError("This RidgeRegressor instance is not fitted yet. Call 'fit' before using this estimator.")
        y_pred = self.predict(X)
        y_true = np.asarray(y).flatten()
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError('Length of y_true and y_pred must match.')
        y_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        if ss_tot == 0:
            return 0.0 if ss_res > 0 else 1.0
        r2 = 1.0 - ss_res / ss_tot
        return float(r2)

class RidgeRegressionModel(BaseModel):

    def __init__(self, alpha: float=1.0, fit_intercept: bool=True, solver: str='auto', max_iter: Optional[int]=None, tol: float=0.0001, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize Ridge Regression model.
        
        Args:
            alpha (float): Regularization strength. Must be positive.
            fit_intercept (bool): Whether to fit the intercept.
            solver (str): Algorithm to use for solving the optimization problem.
            max_iter (Optional[int]): Maximum iterations for iterative solvers.
            tol (float): Tolerance for stopping criteria.
            random_state (Optional[int]): Seed for random number generator.
            name (Optional[str]): Custom name for the model.
        """
        super().__init__(name=name)
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> 'RidgeRegressionModel':
        """
        Fit the ridge regression model according to the given training data.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training vectors with shape (n_samples, n_features).
            y (Union[np.ndarray, list]): Target values with shape (n_samples,) for regression or
                                         (n_samples,) for classification with discrete labels.
            **kwargs: Additional fitting parameters.
            
        Returns:
            RidgeRegressionModel: Fitted estimator.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = np.asarray(X)
        y_array = np.asarray(y)
        if y_array.ndim == 1:
            unique_vals = np.unique(y_array)
            if y_array.dtype.kind in 'SU' or (len(unique_vals) <= max(10, 0.1 * len(y_array)) and np.issubdtype(y_array.dtype, np.integer)):
                self._is_classification = True
            else:
                self._is_classification = False
        else:
            self._is_classification = False
        if self._is_classification:
            self.classes_ = np.unique(y_array)
            if len(self.classes_) == 2:
                y_processed = np.where(y_array == self.classes_[1], 1, 0)
            else:
                y_processed = y_array.astype(float)
        else:
            y_processed = y_array.astype(float)
        if self.fit_intercept:
            if sparse.issparse(X_array):
                ones = sparse.csr_matrix(np.ones((X_array.shape[0], 1)))
                X_with_intercept = sparse.hstack([X_array, ones])
            else:
                ones = np.ones((X_array.shape[0], 1))
                X_with_intercept = np.hstack([X_array, ones])
        else:
            X_with_intercept = X_array
        if self.random_state is not None:
            np.random.seed(self.random_state)
        (n_samples, n_features) = X_with_intercept.shape
        if self.solver == 'auto':
            if sparse.issparse(X_with_intercept):
                solver = 'sparse_cg'
            elif n_samples > n_features:
                solver = 'cholesky'
            else:
                solver = 'svd'
        else:
            solver = self.solver
        if solver in ['svd', 'cholesky']:
            if sparse.issparse(X_with_intercept):
                XtX = X_with_intercept.T.dot(X_with_intercept)
                Xty = X_with_intercept.T.dot(y_processed)
            else:
                XtX = X_with_intercept.T @ X_with_intercept
                Xty = X_with_intercept.T @ y_processed
            if sparse.issparse(XtX):
                diag_indices = np.arange(min(XtX.shape))
                XtX = XtX.tolil()
                XtX[diag_indices, diag_indices] += self.alpha
                XtX = XtX.tocsr()
            else:
                np.fill_diagonal(XtX, XtX.diagonal() + self.alpha)
            if solver == 'cholesky':
                try:
                    self.coef_ = linalg.solve(XtX, Xty, assume_a='pos')
                except linalg.LinAlgError:
                    self.coef_ = linalg.lstsq(XtX, Xty)[0]
            else:
                self.coef_ = linalg.lstsq(XtX, Xty)[0]
        elif solver == 'lsqr':
            from scipy.sparse.linalg import lsqr
            if sparse.issparse(X_with_intercept):
                reg_matrix = sparse.diags([np.sqrt(self.alpha)] * n_features)
                augmented_X = sparse.vstack([X_with_intercept, reg_matrix])
                augmented_y = np.concatenate([y_processed, np.zeros(n_features)])
            else:
                reg_matrix = np.eye(n_features) * np.sqrt(self.alpha)
                augmented_X = np.vstack([X_with_intercept, reg_matrix])
                augmented_y = np.concatenate([y_processed, np.zeros(n_features)])
            result = lsqr(augmented_X, augmented_y, atol=self.tol, btol=self.tol, iter_lim=self.max_iter)
            self.coef_ = result[0]
        elif solver == 'sparse_cg':
            from scipy.sparse.linalg import cg
            if sparse.issparse(X_with_intercept):
                A = X_with_intercept.T.dot(X_with_intercept)
                b = X_with_intercept.T.dot(y_processed)
            else:
                A = X_with_intercept.T @ X_with_intercept
                b = X_with_intercept.T @ y_processed
            if sparse.issparse(A):
                A = A + sparse.diags([self.alpha] * A.shape[0])
            else:
                A = A + np.eye(A.shape[0]) * self.alpha
            x0 = np.zeros(A.shape[0])
            (self.coef_, info) = cg(A, b, x0=x0, tol=self.tol, maxiter=self.max_iter)
            if info > 0:
                raise RuntimeError(f'CG failed to converge after {info} iterations')
            elif info < 0:
                raise RuntimeError('CG failed with illegal input or breakdown')
        else:
            raise ValueError(f"Unknown solver '{solver}'")
        if self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.intercept_ = 0.0
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Samples with shape (n_samples, n_features).
            **kwargs: Additional prediction parameters.
            
        Returns:
            np.ndarray: Predicted values with shape (n_samples,).
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError('Model must be fitted before making predictions')
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = np.asarray(X)
        if sparse.issparse(X_array):
            predictions = X_array.dot(self.coef_)
        else:
            predictions = X_array @ self.coef_
        predictions += self.intercept_
        if self._is_classification:
            if len(self.classes_) == 2:
                binary_preds = (predictions > 0).astype(int)
                return self.classes_[binary_preds]
            else:
                class_indices = np.clip(np.round(predictions).astype(int), 0, len(self.classes_) - 1)
                return self.classes_[class_indices]
        return predictions

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        For classification tasks, this returns the mean accuracy.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Test samples with shape (n_samples, n_features).
            y (Union[np.ndarray, list]): True values with shape (n_samples,).
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: R^2 score for regression or accuracy for classification.
        """
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        if self._is_classification:
            return np.mean(y_true == y_pred)
        else:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict class probabilities for samples in X (for classification tasks).
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Samples with shape (n_samples, n_features).
            **kwargs: Additional prediction parameters.
            
        Returns:
            np.ndarray: Predicted class probabilities with shape (n_samples, n_classes).
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError('Model must be fitted before making predictions')
        if not self._is_classification:
            raise RuntimeError('predict_proba is only available for classification tasks')
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = np.asarray(X)
        if sparse.issparse(X_array):
            raw_predictions = X_array.dot(self.coef_)
        else:
            raw_predictions = X_array @ self.coef_
        raw_predictions += self.intercept_
        if len(self.classes_) == 2:
            clipped_predictions = np.clip(raw_predictions, -250, 250)
            probs_positive = 1 / (1 + np.exp(-clipped_predictions))
            probs_negative = 1 - probs_positive
            return np.column_stack([probs_negative, probs_positive])
        else:
            if raw_predictions.ndim == 1:
                raw_predictions = raw_predictions.reshape(-1, 1)
            if raw_predictions.shape[1] == 1:
                n_samples = raw_predictions.shape[0]
                n_classes = len(self.classes_)
                scores = np.zeros((n_samples, n_classes))
                for i in range(n_classes):
                    scores[:, i] = raw_predictions.flatten() - (i - (n_classes - 1) / 2)
            else:
                scores = raw_predictions
            shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(shifted_scores)
            sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
            sum_exp = np.where(sum_exp == 0, 1, sum_exp)
            return exp_scores / sum_exp

class RidgeRegressionAnalyzer:

    def __init__(self, model: 'RidgeRegressor', feature_names: Optional[List[str]]=None):
        """
        Initialize Ridge Regression Analyzer.
        
        Args:
            model (RidgeRegressor): Fitted ridge regression model to analyze.
            feature_names (Optional[List[str]]): Optional list of feature names for interpretation.
        """
        self.model = model
        self.feature_names = feature_names

    def get_coefficient_path(self, alphas: Union[List[float], np.ndarray], X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list]) -> Dict[str, np.ndarray]:
        """
        Compute coefficient paths for different regularization strengths.
        
        Args:
            alphas (Union[List[float], np.ndarray]): List of alpha values to evaluate.
            X (Union[FeatureSet, np.ndarray]): Training data.
            y (Union[np.ndarray, list]): Target values.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing 'alphas' and 'coefficients' arrays.
        """
        if isinstance(X, FeatureSet):
            X_data = X.features
            if self.feature_names is None:
                self.feature_names = X.feature_names
        else:
            X_data = np.asarray(X)
        y_data = np.asarray(y).ravel()
        original_alpha = self.model.alpha
        original_is_fitted = self.model.is_fitted
        coefficients = []
        for alpha in alphas:
            self.model.alpha = alpha
            self.model.fit(X_data, y_data)
            coefficients.append(self.model.coef_.copy())
        self.model.alpha = original_alpha
        self.model.is_fitted = original_is_fitted
        coefficients = np.array(coefficients)
        return {'alphas': np.array(alphas), 'coefficients': coefficients}

    def analyze_regularization_effect(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], alpha_range: tuple=(0.001, 1000), num_points: int=100) -> Dict[str, Any]:
        """
        Analyze how regularization affects model coefficients and performance.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training data.
            y (Union[np.ndarray, list]): Target values.
            alpha_range (tuple): Min and max alpha values to examine.
            num_points (int): Number of points in the regularization path.
            
        Returns:
            Dict[str, Any]: Analysis results including coefficient paths and metrics.
        """
        if isinstance(X, FeatureSet):
            X_data = X.features
            if self.feature_names is None:
                self.feature_names = X.feature_names
        else:
            X_data = np.asarray(X)
        y_data = np.asarray(y).ravel()
        alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), num_points)
        coef_path_result = self.get_coefficient_path(alphas, X_data, y_data)
        r2_scores = []
        mse_scores = []
        mae_scores = []
        original_alpha = self.model.alpha
        original_is_fitted = self.model.is_fitted
        metrics_calculator = RegressionErrorMetrics()
        r2_calculator = RSquaredMetrics()
        for alpha in alphas:
            self.model.alpha = alpha
            self.model.fit(X_data, y_data)
            y_pred = self.model.predict(X_data)
            r2 = r2_calculator.r_squared_score(y_data, y_pred)
            mse = metrics_calculator.mean_squared_error(y_data, y_pred)
            mae = metrics_calculator.mean_absolute_error(y_data, y_pred)
            r2_scores.append(r2)
            mse_scores.append(mse)
            mae_scores.append(mae)
        self.model.alpha = original_alpha
        self.model.is_fitted = original_is_fitted
        return {'alphas': alphas, 'coefficients': coef_path_result['coefficients'], 'train_scores': r2_scores, 'r2_scores': np.array(r2_scores), 'mse_scores': np.array(mse_scores), 'mae_scores': np.array(mae_scores), 'feature_names': self.feature_names}

    def get_feature_importance(self, method: str='coefficient_magnitude') -> Union[np.ndarray, Dict[str, float]]:
        """
        Calculate feature importance based on model coefficients.
        
        Args:
            method (str): Method to compute importance ('coefficient_magnitude' or 'normalized').
            
        Returns:
            Union[np.ndarray, Dict[str, float]]: Feature importance values.
        """
        if not hasattr(self.model, 'coef_') or self.model.coef_ is None:
            raise ValueError("Model must be fitted before calculating feature importance. Call 'fit' on the model first.")
        if method == 'coefficient_magnitude':
            importance = np.abs(self.model.coef_)
        elif method == 'normalized':
            coef_abs = np.abs(self.model.coef_)
            importance = coef_abs / np.sum(coef_abs) if np.sum(coef_abs) > 0 else coef_abs
        else:
            raise ValueError("Method must be either 'coefficient_magnitude' or 'normalized'")
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        else:
            return importance

    def perform_cross_validation_analysis(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], cv_folds: int=5, alphas: Optional[Union[List[float], np.ndarray]]=None) -> Dict[str, Any]:
        """
        Perform cross-validation analysis to evaluate model performance across alpha values.
        
        Args:
            X (Union[FeatureSet, np.ndarray]): Training data.
            y (Union[np.ndarray, list]): Target values.
            cv_folds (int): Number of cross-validation folds.
            alphas (Optional[Union[List[float], np.ndarray]]): Alpha values to test.
            
        Returns:
            Dict[str, Any]: Cross-validation results including mean scores and standard deviations.
        """
        if alphas is None:
            alphas = np.logspace(-4, 2, 20)
        if isinstance(X, FeatureSet):
            X_data = X.features
            if self.feature_names is None:
                self.feature_names = X.feature_names
        else:
            X_data = np.asarray(X)
        y_data = np.asarray(y).ravel()
        original_alpha = self.model.alpha
        original_is_fitted = self.model.is_fitted
        cv_results = {'alphas': [], 'mean_scores': [], 'std_scores': [], 'all_scores': []}
        for alpha in alphas:
            self.model.alpha = alpha
            scores = []
            n_samples = X_data.shape[0]
            indices = np.arange(n_samples)
            np.random.seed(42)
            np.random.shuffle(indices)
            fold_size = n_samples // cv_folds
            for fold in range(cv_folds):
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < cv_folds - 1 else n_samples
                test_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                X_train_fold = X_data[train_indices]
                y_train_fold = y_data[train_indices]
                X_test_fold = X_data[test_indices]
                y_test_fold = y_data[test_indices]
                self.model.fit(X_train_fold, y_train_fold)
                score = self.model.score(X_test_fold, y_test_fold)
                scores.append(score)
            cv_results['alphas'].append(alpha)
            cv_results['mean_scores'].append(np.mean(scores))
            cv_results['std_scores'].append(np.std(scores))
            cv_results['all_scores'].append(scores)
        self.model.alpha = original_alpha
        self.model.is_fitted = original_is_fitted
        cv_results['alphas'] = np.array(cv_results['alphas'])
        cv_results['mean_scores'] = np.array(cv_results['mean_scores'])
        cv_results['std_scores'] = np.array(cv_results['std_scores'])
        best_idx = np.argmax(cv_results['mean_scores'])
        cv_results['best_alpha'] = cv_results['alphas'][best_idx]
        cv_results['best_score'] = cv_results['mean_scores'][best_idx]
        return cv_results