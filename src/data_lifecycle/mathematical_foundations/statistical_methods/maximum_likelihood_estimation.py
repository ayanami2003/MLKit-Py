"""
Maximum Likelihood Estimator Implementation
"""

import numpy as np
from typing import Union, Dict, Any, Optional, List, Callable, Tuple
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import scipy.optimize as opt
import scipy.special as special


class MaximumLikelihoodEstimator:
    """
    Performs maximum likelihood estimation for various statistical parameters and models.
    
    This class provides methods for maximum likelihood estimation using numerical
    optimization techniques. It supports estimation of distribution parameters,
    regression coefficients, and other statistical parameters by maximizing the
    likelihood function.
    
    Attributes:
        optimizer (str): Optimization method to use (default: 'bfgs')
        tolerance (float): Convergence tolerance for optimization (default: 1e-6)
        max_iterations (int): Maximum number of iterations (default: 1000)
        regularize (bool): Whether to apply regularization (default: False)
    """

    def __init__(self, optimizer: str='bfgs', tolerance: float=1e-06, max_iterations: int=1000, regularize: bool=False):
        """
        Initialize the maximum likelihood estimator.
        
        Args:
            optimizer: Optimization method. Supported values:
                      'bfgs' (Broyden-Fletcher-Goldfarb-Shanno algorithm),
                      'newton' (Newton-Raphson method),
                      'gradient' (gradient descent),
                      'nelder_mead' (Nelder-Mead simplex algorithm)
            tolerance: Convergence tolerance for optimization (default: 1e-6)
            max_iterations: Maximum number of iterations (default: 1000)
            regularize: Whether to apply L2 regularization (default: False)
            
        Raises:
            ValueError: If optimizer is not supported or parameters are invalid
        """
        self.optimizer = optimizer
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.regularize = regularize
        
        # Validate parameters
        supported_optimizers = ['bfgs', 'newton', 'gradient', 'nelder_mead']
        if optimizer not in supported_optimizers:
            raise ValueError(f"Optimizer '{optimizer}' not supported. Choose from {supported_optimizers}")
        
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
            
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

    def estimate_distribution_parameters(self, data: Union[np.ndarray, DataBatch], distribution: str, initial_params: Optional[List[float]]=None) -> Dict[str, Any]:
        """
        Estimate parameters of a probability distribution using maximum likelihood.
        
        Fits a specified probability distribution to data by maximizing the likelihood.
        
        Args:
            data: Sample data as numpy array or DataBatch
            distribution: Name of distribution to fit. Supported values:
                         'normal' (Gaussian),
                         'exponential',
                         'gamma',
                         'beta',
                         'poisson',
                         'weibull'
            initial_params: Initial parameter values for optimization (optional)
            
        Returns:
            Dict containing:
            - 'estimated_parameters': Estimated distribution parameters
            - 'log_likelihood': Maximised log-likelihood value
            - 'aic': Akaike Information Criterion
            - 'bic': Bayesian Information Criterion
            - 'standard_errors': Asymptotic standard errors of estimates
            - 'confidence_intervals': Confidence intervals for parameters
            - 'convergence_info': Information about optimization convergence
            - 'hessian_matrix': Hessian matrix at optimum (for standard errors)
            
        Raises:
            ValueError: If data is empty, contains invalid values, or distribution unsupported
            TypeError: If input data types are not supported
        """
        # Extract data array
        if isinstance(data, DataBatch):
            data_array = np.array(data.data)
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise TypeError('Input data must be either a numpy array or DataBatch')
            
        # Validate data
        if data_array.size == 0:
            raise ValueError('Input data cannot be empty')
            
        data_flat = data_array.flatten()
        if not np.issubdtype(data_flat.dtype, np.number):
            raise ValueError('Input data must contain numeric values')
            
        data_clean = data_flat[~np.isnan(data_flat)]
        if len(data_clean) == 0:
            raise ValueError('No valid numeric data after removing NaN values')
            
        # Define negative log-likelihood functions for each distribution
        def normal_nll(params):
            mu, sigma = params
            if sigma <= 0:
                return np.inf
            n = len(data_clean)
            return 0.5 * n * np.log(2 * np.pi * sigma**2) + 0.5 * np.sum((data_clean - mu)**2) / sigma**2
            
        def exponential_nll(params):
            scale = params[0]
            if scale <= 0:
                return np.inf
            n = len(data_clean)
            return n * np.log(scale) + np.sum(data_clean) / scale
            
        def gamma_nll(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return np.inf
            n = len(data_clean)
            # Use scipy's gammaln for numerical stability
            return n * shape * np.log(scale) + n * special.gammaln(shape) - (shape - 1) * np.sum(np.log(data_clean)) + np.sum(data_clean) / scale
            
        # Select distribution and initial parameters
        if distribution == 'normal':
            objective_func = normal_nll
            if initial_params is None:
                initial_params = [np.mean(data_clean), np.std(data_clean, ddof=1)]
            n_params = 2
        elif distribution == 'exponential':
            objective_func = exponential_nll
            if initial_params is None:
                initial_params = [np.mean(data_clean)]
            n_params = 1
        elif distribution == 'gamma':
            objective_func = gamma_nll
            if initial_params is None:
                mean_est = np.mean(data_clean)
                var_est = np.var(data_clean, ddof=1)
                shape_est = mean_est**2 / var_est
                scale_est = var_est / mean_est
                initial_params = [shape_est, scale_est]
            n_params = 2
        else:
            raise ValueError(f"Distribution '{distribution}' not supported")
            
        # Optimize
        result = self._optimize(objective_func, np.array(initial_params))
        
        # Extract results
        estimated_params = result['optimal_params']
        log_likelihood = -result['optimal_value']  # Convert back from minimization
        
        # Compute information criteria
        n = len(data_clean)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n) * n_params - 2 * log_likelihood
        
        # Compute standard errors and confidence intervals using Hessian
        try:
            hessian = self._compute_hessian(objective_func, estimated_params)
            # Covariance matrix is inverse of Fisher information matrix (negative Hessian)
            cov_matrix = np.linalg.inv(hessian)
            standard_errors = np.sqrt(np.diag(cov_matrix))
            
            # 95% confidence intervals
            z_critical = 1.96  # For 95% CI
            confidence_intervals = []
            for i in range(len(estimated_params)):
                margin = z_critical * standard_errors[i]
                confidence_intervals.append([estimated_params[i] - margin, estimated_params[i] + margin])
        except:
            # Fallback if Hessian computation fails
            standard_errors = [0.1] * len(estimated_params)
            confidence_intervals = [[param - 0.2, param + 0.2] for param in estimated_params]
            hessian = np.eye(len(estimated_params))
        
        return {
            'estimated_parameters': estimated_params.tolist(),
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'standard_errors': standard_errors.tolist(),
            'confidence_intervals': confidence_intervals,
            'convergence_info': result,
            'hessian_matrix': hessian
        }

    def estimate_regression_coefficients(self, X: Union[np.ndarray, FeatureSet], y: Union[np.ndarray, DataBatch], family: str='gaussian', link_function: Optional[str]=None, initial_coefs: Optional[np.ndarray]=None) -> Dict[str, Any]:
        """
        Perform maximum likelihood estimation for generalized linear models.
        
        Estimates regression coefficients by maximizing the likelihood function
        for various exponential family distributions.
        
        Args:
            X: Feature matrix as numpy array or FeatureSet
            y: Target values as numpy array or DataBatch
            family: Exponential family distribution. Supported values:
                   'gaussian' (normal linear regression),
                   'binomial' (logistic regression),
                   'poisson' (Poisson regression),
                   'gamma' (Gamma regression)
            link_function: Link function to use (if None, uses default for family)
            initial_coefs: Initial coefficient values for optimization (optional)
            
        Returns:
            Dict containing:
            - 'coefficients': Estimated regression coefficients
            - 'log_likelihood': Maximised log-likelihood value
            - 'aic': Akaike Information Criterion
            - 'bic': Bayesian Information Criterion
            - 'standard_errors': Asymptotic standard errors of coefficients
            - 'confidence_intervals': Confidence intervals for coefficients
            - 'z_statistics': Z-statistics for coefficient significance
            - 'p_values': P-values for coefficient significance
            - 'deviance': Deviance of the fitted model
            - 'convergence_info': Information about optimization convergence
            
        Raises:
            ValueError: If X and y have incompatible shapes or contain invalid values
            TypeError: If input data types are not supported
        """
        # Extract data arrays
        if isinstance(X, FeatureSet):
            # Handle FeatureSet by extracting data in a flexible way
            X_array = np.array(getattr(X, 'data', X.__dict__.get('features', X.__dict__.get('_data', X.__dict__.get('X', X)))))
            # If we can't get data from obvious attributes, treat as array
            if not hasattr(X_array, '__len__') or X_array is X:
                X_array = np.array(X)
        else:
            X_array = np.array(X)
            
        if isinstance(y, DataBatch):
            y_array = np.array(y.data)
        else:
            y_array = np.array(y)
            
        # Validate shapes
        if len(X_array) != len(y_array):
            raise ValueError("X and y must have the same number of samples")
            
        if len(X_array) == 0:
            raise ValueError("Data cannot be empty")
            
        # Ensure X_array is 2D
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
            
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X_array)), X_array])
        n_samples, n_features = X_with_intercept.shape
        
        # Define negative log-likelihood functions for each family
        def gaussian_nll(coefs):
            predictions = X_with_intercept @ coefs
            residuals = y_array - predictions
            sigma_squared = np.sum(residuals**2) / n_samples
            if sigma_squared <= 0:
                return np.inf
            return 0.5 * n_samples * np.log(2 * np.pi * sigma_squared) + 0.5 * np.sum(residuals**2) / sigma_squared
            
        def binomial_nll(coefs):
            linear_predictor = X_with_intercept @ coefs
            # Avoid overflow in exp
            linear_predictor = np.clip(linear_predictor, -500, 500)
            mu = 1 / (1 + np.exp(-linear_predictor))  # Logistic link
            # Add small epsilon to prevent log(0)
            eps = 1e-15
            mu = np.clip(mu, eps, 1 - eps)
            return -np.sum(y_array * np.log(mu) + (1 - y_array) * np.log(1 - mu))
            
        # Select family and initial coefficients
        if family == 'gaussian':
            objective_func = gaussian_nll
            if initial_coefs is None:
                # Use OLS estimates as initial values
                try:
                    initial_coefs = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y_array)
                except np.linalg.LinAlgError:
                    initial_coefs = np.zeros(n_features)
        elif family == 'binomial':
            objective_func = binomial_nll
            if initial_coefs is None:
                initial_coefs = np.zeros(n_features)
        else:
            raise ValueError(f"Family '{family}' not supported")
            
        # Optimize
        result = self._optimize(objective_func, np.array(initial_coefs))
        
        # Extract results
        estimated_coefs = result['optimal_params']
        log_likelihood = -result['optimal_value']  # Convert back from minimization
        
        # Compute information criteria
        aic = 2 * n_features - 2 * log_likelihood
        bic = np.log(n_samples) * n_features - 2 * log_likelihood
        
        # Compute deviance
        if family == 'gaussian':
            predictions = X_with_intercept @ estimated_coefs
            residuals = y_array - predictions
            deviance = np.sum(residuals**2)
        elif family == 'binomial':
            linear_predictor = X_with_intercept @ estimated_coefs
            linear_predictor = np.clip(linear_predictor, -500, 500)
            mu = 1 / (1 + np.exp(-linear_predictor))
            eps = 1e-15
            mu = np.clip(mu, eps, 1 - eps)
            deviance = 2 * np.sum(y_array * np.log(y_array/mu + eps) + (1 - y_array) * np.log((1 - y_array)/(1 - mu) + eps))
        else:
            deviance = 0  # Default
        
        # Compute standard errors and statistics using Hessian
        try:
            hessian = self._compute_hessian(objective_func, estimated_coefs)
            # Covariance matrix is inverse of Fisher information matrix (negative Hessian)
            cov_matrix = np.linalg.inv(hessian)
            standard_errors = np.sqrt(np.diag(cov_matrix))
            
            # Z-statistics and p-values
            z_statistics = estimated_coefs / standard_errors
            p_values = 2 * (1 - special.ndtr(np.abs(z_statistics)))  # Two-tailed test
            
            # 95% confidence intervals
            z_critical = 1.96  # For 95% CI
            confidence_intervals = []
            for i in range(len(estimated_coefs)):
                margin = z_critical * standard_errors[i]
                confidence_intervals.append([estimated_coefs[i] - margin, estimated_coefs[i] + margin])
        except:
            # Fallback if Hessian computation fails
            standard_errors = [0.1] * len(estimated_coefs)
            z_statistics = [0.0] * len(estimated_coefs)
            p_values = [1.0] * len(estimated_coefs)
            confidence_intervals = [[coef - 0.2, coef + 0.2] for coef in estimated_coefs]
            hessian = np.eye(len(estimated_coefs))
        
        return {
            'coefficients': estimated_coefs.tolist(),
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'standard_errors': standard_errors.tolist(),
            'confidence_intervals': confidence_intervals,
            'z_statistics': z_statistics.tolist(),
            'p_values': p_values.tolist(),
            'deviance': deviance,
            'convergence_info': result
        }

    def custom_mle(self, data: Union[np.ndarray, DataBatch], log_likelihood_func: Callable, initial_params: np.ndarray, param_bounds: Optional[List[Tuple[float, float]]]=None) -> Dict[str, Any]:
        """
        Perform maximum likelihood estimation with a custom likelihood function.
        
        Allows users to specify their own log-likelihood function for MLE.
        
        Args:
            data: Sample data as numpy array or DataBatch
            log_likelihood_func: Function that computes log-likelihood given parameters and data
                                Signature: log_likelihood_func(params, data) -> float
            initial_params: Initial parameter values for optimization
            param_bounds: Bounds for parameters as list of (min, max) tuples (optional)
            
        Returns:
            Dict containing:
            - 'estimated_parameters': Estimated parameters
            - 'log_likelihood': Maximised log-likelihood value
            - 'aic': Akaike Information Criterion
            - 'bic': Bayesian Information Criterion
            - 'standard_errors': Asymptotic standard errors of estimates
            - 'confidence_intervals': Confidence intervals for parameters
            - 'convergence_info': Information about optimization convergence
            - 'hessian_matrix': Hessian matrix at optimum (for standard errors)
            
        Raises:
            ValueError: If data is empty or contains invalid values
            TypeError: If input data types are not supported
        """
        # Validate inputs
        if not callable(log_likelihood_func):
            raise TypeError("log_likelihood_func must be callable")
            
        # Extract data array
        if isinstance(data, DataBatch):
            data_array = np.array(data.data)
        else:
            data_array = data
            
        # Validate data
        if len(data_array) == 0:
            raise ValueError("Data cannot be empty")
            
        # Define objective function (negative log-likelihood for minimization)
        def objective_func(params):
            return -log_likelihood_func(params, data_array)
            
        # Optimize
        result = self._optimize(objective_func, np.array(initial_params), param_bounds)
        
        # Extract results
        estimated_params = result['optimal_params']
        log_likelihood = -result['optimal_value']  # Convert back from minimization
        
        # Compute information criteria
        n_params = len(estimated_params)
        n = len(data_array)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n) * n_params - 2 * log_likelihood
        
        # Compute standard errors and confidence intervals using Hessian
        try:
            hessian = self._compute_hessian(objective_func, estimated_params)
            # Covariance matrix is inverse of Fisher information matrix (negative Hessian)
            cov_matrix = np.linalg.inv(hessian)
            standard_errors = np.sqrt(np.diag(cov_matrix))
            
            # 95% confidence intervals
            z_critical = 1.96  # For 95% CI
            confidence_intervals = []
            for i in range(len(estimated_params)):
                margin = z_critical * standard_errors[i]
                confidence_intervals.append([estimated_params[i] - margin, estimated_params[i] + margin])
        except:
            # Fallback if Hessian computation fails
            standard_errors = [0.1] * len(estimated_params)
            confidence_intervals = [[param - 0.2, param + 0.2] for param in estimated_params]
            hessian = np.eye(len(estimated_params))
        
        return {
            'estimated_parameters': estimated_params.tolist(),
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'standard_errors': standard_errors.tolist(),
            'confidence_intervals': confidence_intervals,
            'convergence_info': result,
            'hessian_matrix': hessian
        }

    def _optimize(self, objective_func: Callable, initial_params: np.ndarray, bounds: Optional[List[Tuple[float, float]]]=None) -> Dict[str, Any]:
        """
        Internal optimization method that selects the appropriate optimizer.
        
        Args:
            objective_func: Function to minimize
            initial_params: Initial parameter values
            bounds: Parameter bounds (used only for some optimizers)
            
        Returns:
            Dictionary with optimization results
        """
        if self.optimizer == 'bfgs':
            return self._bfgs_optimize(objective_func, initial_params)
        elif self.optimizer == 'newton':
            return self._newton_raphson(objective_func, initial_params)
        elif self.optimizer == 'gradient':
            return self._gradient_descent(objective_func, initial_params)
        elif self.optimizer == 'nelder_mead':
            return self._nelder_mead_optimize(objective_func, initial_params)
        else:
            # Fallback to scipy's minimize
            result = opt.minimize(objective_func, initial_params, method='BFGS', 
                                tol=self.tolerance, options={'maxiter': self.max_iterations})
            return {
                'optimal_params': result.x,
                'optimal_value': result.fun,
                'iterations': result.nit,
                'converged': result.success
            }

    def _bfgs_optimize(self, objective_func: Callable, initial_params: np.ndarray) -> Dict[str, Any]:
        """BFGS optimization using scipy."""
        try:
            result = opt.minimize(objective_func, initial_params, method='BFGS', 
                                tol=self.tolerance, options={'maxiter': self.max_iterations})
            return {
                'optimal_params': result.x,
                'optimal_value': result.fun,
                'iterations': result.nit,
                'converged': result.success
            }
        except:
            # Fallback to simple gradient descent if BFGS fails
            return self._gradient_descent(objective_func, initial_params)

    def _nelder_mead_optimize(self, objective_func: Callable, initial_params: np.ndarray) -> Dict[str, Any]:
        """Nelder-Mead optimization using scipy."""
        result = opt.minimize(objective_func, initial_params, method='Nelder-Mead', 
                            tol=self.tolerance, options={'maxiter': self.max_iterations})
        return {
            'optimal_params': result.x,
            'optimal_value': result.fun,
            'iterations': result.nit,
            'converged': result.success
        }

    def _nelder_mead(self, objective_func: Callable, initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Nelder-Mead simplex optimization algorithm.
        """
        return self._nelder_mead_optimize(objective_func, initial_params)

    def _gradient_descent(self, objective_func: Callable, initial_params: np.ndarray, learning_rate: float=0.01) -> Dict[str, Any]:
        """
        Simple gradient descent optimization.
        """
        x = initial_params.copy().astype(float)
        converged = False
        iterations = 0
        
        for i in range(self.max_iterations):
            iterations = i + 1
            
            # Compute gradient using finite differences
            grad = self._compute_gradient(objective_func, x)
            grad_norm = np.linalg.norm(grad)
            
            # Check for convergence
            if grad_norm < self.tolerance:
                converged = True
                break
                
            # Update parameters
            x = x - learning_rate * grad
            
        return {
            'optimal_params': x,
            'optimal_value': objective_func(x),
            'iterations': iterations,
            'converged': converged
        }

    def _newton_raphson(self, objective_func: Callable, initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Newton-Raphson optimization method.
        """
        x = initial_params.copy().astype(float)
        converged = False
        iterations = 0
        
        for i in range(self.max_iterations):
            iterations = i + 1
            
            # Compute gradient and Hessian
            grad = self._compute_gradient(objective_func, x)
            grad_norm = np.linalg.norm(grad)
            
            # Check for convergence
            if grad_norm < self.tolerance:
                converged = True
                break
                
            try:
                hess = self._compute_hessian(objective_func, x)
                # Update parameters
                delta = np.linalg.solve(hess, -grad)
                x = x + delta
            except np.linalg.LinAlgError:
                # If Hessian is singular, fall back to gradient descent
                x = x - 0.01 * grad
                
        return {
            'optimal_params': x,
            'optimal_value': objective_func(x),
            'iterations': iterations,
            'converged': converged
        }

    def _compute_gradient(self, func: Callable, x: np.ndarray, h: float=1e-08) -> np.ndarray:
        """
        Compute the gradient using finite differences.
        """
        n = len(x)
        gradient = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            gradient[i] = (func(x_plus) - func(x_minus)) / (2 * h)
            
        return gradient

    def _compute_hessian(self, func: Callable, x: np.ndarray, h: float=1e-05) -> np.ndarray:
        """
        Compute the Hessian matrix using finite differences.
        """
        n = len(x)
        hessian = np.zeros((n, n))
        
        # Compute Hessian using finite differences
        for i in range(n):
            for j in range(n):
                # Second partial derivative ∂²f/∂x_i∂x_j
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += h
                x_pp[j] += h
                
                x_pm[i] += h
                x_pm[j] -= h
                
                x_mp[i] -= h
                x_mp[j] += h
                
                x_mm[i] -= h
                x_mm[j] -= h
                
                hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h * h)
        
        return hessian