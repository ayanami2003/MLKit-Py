from typing import Callable, Optional, Union, Tuple, List, Any
from dataclasses import dataclass
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

# ...(code omitted)...

class OptimizationResult:
    """Container for optimization results."""
    x: np.ndarray
    fun: float
    success: bool = False
    message: Optional[str] = None
    nit: int = 0
    jac: Optional[np.ndarray] = None
    hess_inv: Optional[np.ndarray] = None
    nfev: Optional[int] = None
    njev: Optional[int] = None

    def __post_init__(self):
        """Validate inputs after initialization."""
        if not isinstance(self.x, np.ndarray):
            raise TypeError('x must be a numpy array')
        if not isinstance(self.fun, (int, float)):
            raise TypeError('fun must be a numeric value')
        if not isinstance(self.success, bool):
            raise TypeError('success must be a boolean')
        if self.message is not None and (not isinstance(self.message, str)):
            raise TypeError('message must be a string or None')
        if not isinstance(self.nit, int) or self.nit < 0:
            raise TypeError('nit must be a non-negative integer')
        if self.jac is not None and (not isinstance(self.jac, np.ndarray)):
            raise TypeError('jac must be a numpy array or None')
        if self.hess_inv is not None and (not isinstance(self.hess_inv, np.ndarray)):
            raise TypeError('hess_inv must be a numpy array or None')
        if self.nfev is not None and (not isinstance(self.nfev, int) or self.nfev < 0):
            raise TypeError('nfev must be a non-negative integer or None')
        if self.njev is not None and (not isinstance(self.njev, int) or self.njev < 0):
            raise TypeError('njev must be a non-negative integer or None')
        if isinstance(self.x, np.ndarray):
            object.__setattr__(self, 'x', self.x.copy())
        if isinstance(self.jac, np.ndarray):
            object.__setattr__(self, 'jac', self.jac.copy())
        if isinstance(self.hess_inv, np.ndarray):
            object.__setattr__(self, 'hess_inv', self.hess_inv.copy())

class BFGSOptimizer(BaseModel):

    def __init__(self, learning_rate: float=1.0, max_iterations: int=1000, tolerance: float=1e-06, verbose: bool=False, name: Optional[str]=None):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

    def fit(self, objective_function: Callable[[np.ndarray], float], gradient_function: Callable[[np.ndarray], np.ndarray], initial_point: np.ndarray, **kwargs) -> 'BFGSOptimizer':
        """
        Fit the optimizer to find the minimum of the objective function.
        
        Args:
            objective_function: Function to minimize, taking parameters as input and returning scalar value.
            gradient_function: Gradient of the objective function, returning gradient vector.
            initial_point: Starting point for optimization as numpy array.
            **kwargs: Additional keyword arguments for fitting.
            
        Returns:
            BFGSOptimizer: Self instance for method chaining.
        """
        if not callable(objective_function):
            raise TypeError('objective_function must be callable')
        if not callable(gradient_function):
            raise TypeError('gradient_function must be callable')
        if not isinstance(initial_point, np.ndarray):
            raise TypeError('initial_point must be a numpy array')
        x = initial_point.copy().astype(float)
        n = len(x)
        I = np.eye(n)
        H = I.copy()
        self._objective_function = objective_function
        self._gradient_function = gradient_function
        for iteration in range(self.max_iterations):
            grad = gradient_function(x)
            grad_norm = np.linalg.norm(grad)
            if self.verbose:
                obj_val = objective_function(x)
                print(f'Iteration {iteration}: f(x) = {obj_val:.6f}, ||grad|| = {grad_norm:.6f}')
            if grad_norm <= self.tolerance:
                if self.verbose:
                    print(f'Converged after {iteration} iterations')
                self.is_fitted = True
                self._optimal_x = x.copy()
                self._optimal_value = objective_function(x)
                self._final_gradient = grad.copy()
                self._final_hess_inv = H.copy()
                return self
            p = -H @ grad
            alpha = 1.0
            c1 = 0.0001
            beta = 0.5
            max_linesearch_iter = 20
            obj_val_current = objective_function(x)
            for _ in range(max_linesearch_iter):
                x_new = x + alpha * p
                obj_val_new = objective_function(x_new)
                obj_val_linear_approx = obj_val_current + c1 * alpha * np.dot(grad, p)
                if obj_val_new <= obj_val_linear_approx:
                    break
                alpha *= beta
            x_new = x + alpha * p
            grad_new = gradient_function(x_new)
            s = x_new - x
            y = grad_new - grad
            sy = np.dot(s, y)
            if sy > 1e-10:
                rho = 1.0 / sy
                V = I - rho * np.outer(y, s)
                H = V @ H @ V.T + rho * np.outer(s, s)
            x = x_new
        if self.verbose:
            print(f'Maximum iterations ({self.max_iterations}) reached')
        self.is_fitted = True
        self._optimal_x = x.copy()
        self._optimal_value = objective_function(x)
        self._final_gradient = gradient_function(x).copy()
        self._final_hess_inv = H.copy()
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Prediction is not supported for optimization algorithms.
        
        Args:
            X: Input data (ignored).
            **kwargs: Additional parameters (ignored).
            
        Raises:
            NotImplementedError: Always raised as prediction is not applicable.
        """
        raise NotImplementedError('Prediction is not supported for optimization algorithms.')

    def score(self, X: np.ndarray=None, y: Optional[np.ndarray]=None, objective_function: Optional[Callable[[np.ndarray], float]]=None, **kwargs) -> float:
        """
        Return the objective function value at the optimized parameters.
        
        Args:
            X: Ignored - kept for interface compatibility.
            y: Ignored in optimization context.
            objective_function: Optional function to evaluate; uses fitted function if not provided.
            **kwargs: Additional parameters.
            
        Returns:
            float: Objective function value at optimum.
        """
        if objective_function is not None:
            if X is None:
                raise ValueError('A point (X) must be provided to evaluate the objective function.')
            if not isinstance(X, np.ndarray):
                raise TypeError('X must be a numpy array.')
            return objective_function(X)
        if not self.is_fitted or not hasattr(self, '_optimal_x'):
            raise ValueError("Optimizer has no optimal point stored. Call 'fit' first or provide a valid point.")
        optimal_x = self._optimal_x
        return self._objective_function(optimal_x)


# ...(code omitted)...


def stratified_sample(data: Union[np.ndarray, DataBatch], target_column: Union[int, str], sample_size: Union[int, float], random_state: Optional[int]=None, replacement: bool=False) -> Union[np.ndarray, DataBatch]:
    """
    Perform stratified sampling to maintain class distribution in the sample.
    
    This function creates a sample from the input data where the proportion of 
    each class in the target column is preserved. It supports both array and 
    DataBatch inputs, returning the same type as the input.
    
    Stratified sampling ensures that each class is represented proportionally
    in the sample, which is especially important for imbalanced datasets.
    
    Args:
        data: Input data as numpy array or DataBatch object.
        target_column: Index (int) or name (str) of the target column for stratification.
        sample_size: Absolute number of samples (int) or fraction of data (float, 0-1).
        random_state: Random seed for reproducibility.
        replacement: Whether to sample with replacement.
        
    Returns:
        Union[np.ndarray, DataBatch]: Sampled data maintaining the same type as input.
        
    Raises:
        ValueError: If sample_size is invalid or target_column is not found.
        TypeError: If data type is unsupported.
        
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 0, 1, 1])
        >>> data = DataBatch(data=X, labels=y)
        >>> sampled = stratified_sample(data, 'labels', 0.5, random_state=42)
        >>> print(sampled.data.shape)  # (2, 2)
    """
    if not isinstance(data, (np.ndarray, DataBatch)):
        raise TypeError('Data must be either numpy array or DataBatch object.')
    rng = np.random.default_rng(random_state)
    if isinstance(data, np.ndarray):
        if data.ndim < 2:
            raise ValueError('Data array must be at least 2-dimensional.')
        array_data = data
        if isinstance(target_column, int):
            if target_column < 0 or target_column >= data.shape[1]:
                raise ValueError(f'Target column index {target_column} out of bounds.')
            target_values = data[:, target_column]
            feature_indices = [i for i in range(data.shape[1]) if i != target_column]
        elif isinstance(target_column, str):
            raise ValueError('String target_column is not supported for numpy arrays.')
        else:
            raise TypeError('target_column must be either int or str.')
    else:
        array_data = np.asarray(data.data)
        if data.labels is not None:
            target_values = np.asarray(data.labels)
        elif isinstance(target_column, int):
            if target_column < 0 or target_column >= array_data.shape[1]:
                raise ValueError(f'Target column index {target_column} out of bounds.')
            target_values = array_data[:, target_column]
            feature_indices = [i for i in range(array_data.shape[1]) if i != target_column]
        elif isinstance(target_column, str):
            if data.feature_names is None:
                raise ValueError('Feature names not available in DataBatch.')
            try:
                target_index = data.feature_names.index(target_column)
            except ValueError:
                raise ValueError(f'Target column "{target_column}" not found in feature names.')
            target_values = array_data[:, target_index]
            feature_indices = [i for i in range(array_data.shape[1]) if i != target_index]
        else:
            raise TypeError('target_column must be either int or str.')
    n_samples_total = len(target_values)
    if isinstance(sample_size, float):
        if sample_size <= 0 or sample_size > 1:
            raise ValueError('Fractional sample_size must be between 0 and 1.')
        n_samples = int(np.round(sample_size * n_samples_total))
        if n_samples == 0:
            n_samples = 1
    elif isinstance(sample_size, int):
        if sample_size <= 0:
            raise ValueError('Absolute sample_size must be positive.')
        if not replacement and sample_size > n_samples_total:
            raise ValueError(f'Cannot sample {sample_size} samples without replacement from a population of size {n_samples_total}.')
        n_samples = sample_size
    else:
        raise TypeError('sample_size must be either int or float.')
    (unique_classes, class_counts) = np.unique(target_values, return_counts=True)
    if replacement:
        class_proportions = class_counts / n_samples_total
        samples_per_class = np.round(class_proportions * n_samples).astype(int)
        diff = n_samples - samples_per_class.sum()
        if diff != 0:
            idx = np.argmax(class_counts)
            samples_per_class[idx] += diff
    else:
        class_proportions = class_counts / n_samples_total
        samples_per_class = np.round(class_proportions * n_samples).astype(int)
        samples_per_class = np.minimum(samples_per_class, class_counts)
        diff = n_samples - samples_per_class.sum()
        while diff != 0:
            if diff > 0:
                eligible = np.where(samples_per_class < class_counts)[0]
                if len(eligible) == 0:
                    break
                deficits = class_counts - samples_per_class
                idx = eligible[np.argmax(deficits[eligible])]
                samples_per_class[idx] += 1
                diff -= 1
            else:
                eligible = np.where(samples_per_class > 0)[0]
                if len(eligible) == 0:
                    break
                deficits = class_counts - samples_per_class
                idx = eligible[np.argmin(deficits[eligible])]
                samples_per_class[idx] -= 1
                diff += 1
    selected_indices = []
    for (cls, n_samples_cls) in zip(unique_classes, samples_per_class):
        class_indices = np.where(target_values == cls)[0]
        if replacement:
            sampled_indices = rng.choice(class_indices, size=n_samples_cls, replace=True)
        else:
            sampled_indices = rng.choice(class_indices, size=n_samples_cls, replace=False)
        selected_indices.extend(sampled_indices)
    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)
    if isinstance(data, np.ndarray):
        return data[selected_indices]
    else:
        if isinstance(data.data, np.ndarray):
            sampled_data = data.data[selected_indices]
        else:
            sampled_data = [data.data[i] for i in selected_indices]
        sampled_labels = None
        if data.labels is not None:
            if isinstance(data.labels, np.ndarray):
                sampled_labels = data.labels[selected_indices]
            else:
                sampled_labels = [data.labels[i] for i in selected_indices]
        sampled_sample_ids = None
        if data.sample_ids is not None:
            sampled_sample_ids = [data.sample_ids[i] for i in selected_indices]
        return DataBatch(data=sampled_data, labels=sampled_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=sampled_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)