from typing import Callable, Optional, Any
import numpy as np

def rmsprop_optimizer(gradient_func: Callable[[np.ndarray], np.ndarray], initial_params: np.ndarray, learning_rate: float=0.001, decay_rate: float=0.9, epsilon: float=1e-08, max_iterations: int=1000, tolerance: float=1e-06, callback: Optional[Callable[[int, np.ndarray, float], Any]]=None) -> np.ndarray:
    """
    Optimize parameters using the RMSProp optimization algorithm.

    This function implements the RMSProp (Root Mean Square Propagation) optimization technique,
    which adapts the learning rate for each parameter based on the moving average of squared gradients.
    It is particularly effective for non-stationary objectives and noisy gradients.

    Args:
        gradient_func (Callable[[np.ndarray], np.ndarray]): 
            Function that computes the gradient of the objective with respect to the parameters.
        initial_params (np.ndarray): 
            Initial parameter values to start optimization.
        learning_rate (float, optional): 
            Base learning rate for parameter updates. Defaults to 0.001.
        decay_rate (float, optional): 
            Decay rate for the moving average of squared gradients. Defaults to 0.9.
        epsilon (float, optional): 
            Small constant added to denominator for numerical stability. Defaults to 1e-8.
        max_iterations (int, optional): 
            Maximum number of optimization iterations. Defaults to 1000.
        tolerance (float, optional): 
            Convergence tolerance for early stopping. Defaults to 1e-6.
        callback (Optional[Callable[[int, np.ndarray, float], Any]], optional): 
            Function called after each iteration with iteration number, current parameters, and loss value.
            Can be used for logging or early stopping. Defaults to None.

    Returns:
        np.ndarray: Optimized parameter values.

    Raises:
        ValueError: If learning_rate, decay_rate, or epsilon are not positive.
        RuntimeError: If optimization fails to converge within max_iterations.
    """
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    if decay_rate <= 0 or decay_rate >= 1:
        raise ValueError('decay_rate must be in the range (0, 1)')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    params = initial_params.copy()
    avg_squared_grad = np.zeros_like(params)
    for i in range(max_iterations):
        grad = gradient_func(params)
        avg_squared_grad = decay_rate * avg_squared_grad + (1 - decay_rate) * grad ** 2
        delta = learning_rate * grad / (np.sqrt(avg_squared_grad) + epsilon)
        params -= delta
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tolerance:
            if callback is not None:
                callback(i, params, grad_norm)
            return params
        if callback is not None:
            callback(i, params, grad_norm)
    raise RuntimeError(f'RMSProp failed to converge within {max_iterations} iterations')