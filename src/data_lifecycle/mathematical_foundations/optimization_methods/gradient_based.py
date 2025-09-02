from general.structures.data_batch import DataBatch
from typing import Callable, Union, Optional, Iterator, Tuple
import numpy as np
from general.base_classes.model_base import BaseModel

class AdaGradOptimizer(BaseModel):
    """
    Adaptive Gradient Algorithm (AdaGrad) optimizer for gradient-based optimization.

    This optimizer adapts the learning rate for each parameter based on the historical
    squared gradients, making it particularly effective for sparse data.

    Attributes:
        learning_rate (float): Initial learning rate for the optimizer.
        epsilon (float): Small constant to prevent division by zero.
    """

    def __init__(self, learning_rate: float=0.01, epsilon: float=1e-08, name: Optional[str]=None):
        """
        Initialize the AdaGrad optimizer.

        Args:
            learning_rate (float): Initial learning rate. Defaults to 0.01.
            epsilon (float): Small constant for numerical stability. Defaults to 1e-8.
            name (Optional[str]): Name of the optimizer instance.
        """
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self._history = None

    def fit(self, objective_function: Callable[[np.ndarray], float], gradient_function: Callable[[np.ndarray], np.ndarray], initial_point: np.ndarray, max_iterations: int=1000, tolerance: float=1e-06, **kwargs) -> 'AdaGradOptimizer':
        """
        Optimize the objective function using AdaGrad algorithm.

        Args:
            objective_function (Callable[[np.ndarray], float]): Function to minimize.
            gradient_function (Callable[[np.ndarray], np.ndarray]): Gradient of the objective function.
            initial_point (np.ndarray): Starting point for optimization.
            max_iterations (int): Maximum number of iterations. Defaults to 1000.
            tolerance (float): Convergence tolerance. Defaults to 1e-6.
            **kwargs: Additional parameters for optimization.

        Returns:
            AdaGradOptimizer: Self instance for method chaining.
        """
        theta = np.array(initial_point, dtype=np.float64)
        n_params = len(theta)
        self._history = np.zeros(n_params, dtype=np.float64)
        for i in range(max_iterations):
            grad = gradient_function(theta)
            self._history += grad ** 2
            adaptive_lr = self.learning_rate / (np.sqrt(self._history) + self.epsilon)
            theta_new = theta - adaptive_lr * grad
            if np.linalg.norm(theta_new - theta) < tolerance:
                break
            theta = theta_new
        self._optimal_params = theta
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get optimized parameters.

        Args:
            X (np.ndarray): Input array (not used in this context, but required by BaseModel).
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: Optimized parameters.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        return self._optimal_params

    def score(self, X: np.ndarray, y: Optional[np.ndarray]=None, objective_function: Optional[Callable[[np.ndarray], float]]=None, **kwargs) -> float:
        """
        Evaluate the optimized solution using the objective function.

        Args:
            X (np.ndarray): Input array (not used in this context).
            y (Optional[np.ndarray]): Target values (not used in this context).
            objective_function (Optional[Callable[[np.ndarray], float]]): Function to evaluate.
            **kwargs: Additional parameters.

        Returns:
            float: Value of the objective function at the optimized point.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'score'.")
        if objective_function is None:
            raise ValueError('objective_function must be provided for scoring.')
        return objective_function(self._optimal_params)

class RMSPropOptimizer(BaseModel):
    """
    RMSProp optimizer for gradient-based optimization.

    This optimizer uses a moving average of squared gradients to adapt the learning rate,
    addressing the diminishing learning rates problem found in AdaGrad.

    Attributes:
        learning_rate (float): Initial learning rate for the optimizer.
        decay_rate (float): Decay rate for the moving average of squared gradients.
        epsilon (float): Small constant to prevent division by zero.
    """

    def __init__(self, learning_rate: float=0.001, decay_rate: float=0.9, epsilon: float=1e-08, name: Optional[str]=None):
        """
        Initialize the RMSProp optimizer.

        Args:
            learning_rate (float): Initial learning rate. Defaults to 0.001.
            decay_rate (float): Decay rate for squared gradient moving average. Defaults to 0.9.
            epsilon (float): Small constant for numerical stability. Defaults to 1e-8.
            name (Optional[str]): Name of the optimizer instance.
        """
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self._history = None

    def fit(self, objective_function: Callable[[np.ndarray], float], gradient_function: Callable[[np.ndarray], np.ndarray], initial_point: np.ndarray, max_iterations: int=1000, tolerance: float=1e-06, **kwargs) -> 'RMSPropOptimizer':
        """
        Optimize the objective function using RMSProp algorithm.

        Args:
            objective_function (Callable[[np.ndarray], float]): Function to minimize.
            gradient_function (Callable[[np.ndarray], np.ndarray]): Gradient of the objective function.
            initial_point (np.ndarray): Starting point for optimization.
            max_iterations (int): Maximum number of iterations. Defaults to 1000.
            tolerance (float): Convergence tolerance. Defaults to 1e-6.
            **kwargs: Additional parameters for optimization.

        Returns:
            RMSPropOptimizer: Self instance for method chaining.
        """
        theta = np.array(initial_point, dtype=np.float64)
        n_params = len(theta)
        self._history = np.zeros(n_params, dtype=np.float64)
        for i in range(max_iterations):
            grad = gradient_function(theta)
            self._history = self.decay_rate * self._history + (1 - self.decay_rate) * grad ** 2
            adaptive_lr = self.learning_rate / (np.sqrt(self._history) + self.epsilon)
            theta_new = theta - adaptive_lr * grad
            if np.linalg.norm(theta_new - theta) < tolerance:
                break
            theta = theta_new
        self._optimal_params = theta
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get optimized parameters.

        Args:
            X (np.ndarray): Input array (not used in this context, but required by BaseModel).
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: Optimized parameters.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        return self._optimal_params

    def score(self, X: np.ndarray, y: Optional[np.ndarray]=None, objective_function: Optional[Callable[[np.ndarray], float]]=None, **kwargs) -> float:
        """
        Evaluate the optimized solution using the objective function.

        Args:
            X (np.ndarray): Input array (not used in this context).
            y (Optional[np.ndarray]): Target values (not used in this context).
            objective_function (Optional[Callable[[np.ndarray], float]]): Function to evaluate.
            **kwargs: Additional parameters.

        Returns:
            float: Value of the objective function at the optimized point.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'score'.")
        if objective_function is None:
            raise ValueError('objective_function must be provided for scoring.')
        return objective_function(self._optimal_params)

def compute_gradient(func: Callable[[np.ndarray], float], x: np.ndarray, method: str='central', h: float=1e-05) -> np.ndarray:
    """
    Compute the gradient of a scalar-valued function at a given point using finite differences.

    This function numerically approximates the gradient of a function using either forward,
    backward, or central difference methods. It supports multi-dimensional inputs and is
    suitable for gradient-based optimization routines.

    Args:
        func (Callable[[np.ndarray], float]): The scalar-valued function to differentiate.
            Must accept a numpy array and return a float.
        x (np.ndarray): The point at which to compute the gradient. Must be a 1D array.
        method (str, optional): The finite difference method to use. Options are 'forward',
            'backward', or 'central'. Defaults to 'central'.
        h (float, optional): The step size for finite differences. Defaults to 1e-5.

    Returns:
        np.ndarray: The gradient vector of the function at point x.

    Raises:
        ValueError: If method is not one of 'forward', 'backward', or 'central'.
        ValueError: If x is not a 1D numpy array.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError('x must be a numpy array')
    if x.ndim != 1:
        raise ValueError('x must be a 1D numpy array')
    if method not in ('forward', 'backward', 'central'):
        raise ValueError("method must be one of 'forward', 'backward', or 'central'")
    n = len(x)
    gradient = np.zeros(n)
    if method == 'forward':
        fx = func(x)
        for i in range(n):
            x_ph = x.copy()
            x_ph[i] += h
            gradient[i] = (func(x_ph) - fx) / h
    elif method == 'backward':
        fx = func(x)
        for i in range(n):
            x_mh = x.copy()
            x_mh[i] -= h
            gradient[i] = (fx - func(x_mh)) / h
    elif method == 'central':
        for i in range(n):
            x_ph = x.copy()
            x_ph[i] += h
            x_mh = x.copy()
            x_mh[i] -= h
            gradient[i] = (func(x_ph) - func(x_mh)) / (2 * h)
    return gradient

def create_mini_batches(data: Union[np.ndarray, DataBatch], batch_size: int, shuffle: bool=True, random_state: int=None) -> Iterator[Tuple[np.ndarray, Union[np.ndarray, None]]]:
    """
    Create mini-batches from input data for gradient descent optimization.

    This function partitions data into smaller batches, optionally shuffling them,
    to enable mini-batch gradient descent algorithms. It supports both raw numpy arrays
    and structured DataBatch objects.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data to batch. If numpy array, 
            assumes shape (n_samples, n_features). If DataBatch, uses its data and labels.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle data before batching. Defaults to True.
        random_state (int, optional): Random seed for shuffling. Defaults to None.

    Yields:
        Iterator[Tuple[np.ndarray, Union[np.ndarray, None]]]: Tuples of (batch_features, batch_labels).
            For unlabeled data, batch_labels will be None.

    Raises:
        ValueError: If batch_size is not positive or larger than the dataset.
        TypeError: If data is neither a numpy array nor a DataBatch.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('NumPy array data must be 2-dimensional')
        features = data
        labels = None
        n_samples = features.shape[0]
    elif isinstance(data, DataBatch):
        if isinstance(data.data, np.ndarray):
            if data.data.ndim != 2:
                raise ValueError("DataBatch data must be 2-dimensional when it's a numpy array")
            features = data.data
        else:
            features = np.array(data.data)
            if features.ndim != 2:
                raise ValueError('DataBatch data must be convertible to a 2-dimensional array')
        labels = data.labels
        if labels is not None and (not isinstance(labels, np.ndarray)):
            labels = np.array(labels)
        n_samples = features.shape[0]
    else:
        raise TypeError('data must be either a numpy array or a DataBatch instance')
    if batch_size > n_samples:
        raise ValueError('batch_size cannot be larger than the number of samples in the dataset')
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batch_features = features[batch_indices]
        batch_labels = None if labels is None else labels[batch_indices]
        yield (batch_features, batch_labels)