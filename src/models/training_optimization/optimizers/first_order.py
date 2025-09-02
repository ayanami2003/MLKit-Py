from typing import Optional, Any, List, Tuple, Callable
import numpy as np
from general.base_classes.model_base import BaseModel
from typing import Optional, Any, List, Tuple, Dict

class SGDOptimizer:
    """
    Stochastic Gradient Descent optimizer with momentum and Nesterov acceleration.
    
    This optimizer implements classical SGD with optional momentum terms to accelerate
    convergence and reduce oscillations. It supports both standard momentum and
    Nesterov accelerated gradient methods.
    
    Args:
        learning_rate (float): Initial learning rate (default: 0.01).
        momentum (float): Momentum factor (default: 0.0).
        nesterov (bool): Whether to apply Nesterov momentum (default: False).
        weight_decay (float): Weight decay coefficient (L2 regularization) (default: 0.0).
        name (Optional[str]): Name of the optimizer instance.
        
    Attributes:
        learning_rate (float): Current learning rate.
        momentum (float): Momentum factor.
        nesterov (bool): Nesterov momentum flag.
        weight_decay (float): L2 penalty coefficient.
        name (str): Optimizer name.
    """

    def __init__(self, learning_rate: float=0.01, momentum: float=0.0, nesterov: bool=False, weight_decay: float=0.0, name: Optional[str]=None):
        if learning_rate <= 0.0:
            raise ValueError('Learning rate must be positive')
        if momentum < 0.0:
            raise ValueError('Momentum must be non-negative')
        if weight_decay < 0.0:
            raise ValueError('Weight decay must be non-negative')
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.name = name
        self._velocities: dict = {}

    def apply_gradients(self, grads_and_vars: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Apply gradients to variables using SGD with optional momentum.
        
        Updates variables by applying computed gradients with optional momentum
        accumulation and Nesterov acceleration. Maintains velocity buffers for
        each variable when momentum is enabled.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs where gradient and variable
                          are numpy arrays of the same shape.
                          
        Raises:
            ValueError: If gradient and variable shapes do not match.
            RuntimeError: If optimizer has not been initialized properly.
        """
        if not hasattr(self, 'learning_rate'):
            raise RuntimeError('Optimizer not properly initialized')
        for (grad, var) in grads_and_vars:
            if grad.shape != var.shape:
                raise ValueError(f'Gradient shape {grad.shape} does not match variable shape {var.shape}')
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * var
            var_id = id(var)
            if self.momentum == 0.0:
                var -= self.learning_rate * grad
            else:
                if var_id not in self._velocities:
                    self._velocities[var_id] = np.zeros_like(var)
                velocity = self._velocities[var_id]
                if self.nesterov:
                    velocity_prev = velocity.copy()
                    velocity *= self.momentum
                    velocity += grad
                    var -= self.learning_rate * (self.momentum * velocity_prev + grad)
                else:
                    velocity *= self.momentum
                    velocity += grad
                    var -= self.learning_rate * velocity

    def get_config(self) -> dict:
        """
        Get optimizer configuration as a dictionary.
        
        Returns:
            Dictionary containing all optimizer hyperparameters and state information needed
            to reconstruct the optimizer. Keys include 'learning_rate', 'momentum', 'nesterov',
            'weight_decay', and 'name'.
        """
        return {'learning_rate': self.learning_rate, 'momentum': self.momentum, 'nesterov': self.nesterov, 'weight_decay': self.weight_decay, 'name': self.name}

    @classmethod
    def from_config(cls, config: dict) -> 'SGDOptimizer':
        """
        Create optimizer instance from configuration dictionary.
        
        Args:
            config: Dictionary containing optimizer configuration parameters.
            
        Returns:
            New optimizer instance with parameters restored from config.
            
        Raises:
            KeyError: If required configuration keys are missing.
        """
        return cls(learning_rate=config['learning_rate'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'], name=config.get('name'))

class AdagradOptimizer:
    """
    Adagrad optimizer with adaptive learning rates per parameter.
    
    This optimizer adapts the learning rate for each parameter based on the historical
    sum of squared gradients. Parameters with large gradients will have smaller learning
    rates, while parameters with small gradients will have larger learning rates.
    Particularly effective for sparse data and natural language processing tasks.
    
    Args:
        learning_rate (float): Initial learning rate (default: 0.001).
        initial_accumulator_value (float): Initial value for gradient accumulators (default: 0.1).
        epsilon (float): Small constant for numerical stability (default: 1e-7).
        weight_decay (float): Weight decay coefficient (L2 regularization) (default: 0.0).
        name (Optional[str]): Name of the optimizer instance.
        
    Attributes:
        learning_rate (float): Current learning rate.
        initial_accumulator_value (float): Starting value for accumulators.
        epsilon (float): Numerical stability constant.
        weight_decay (float): L2 penalty coefficient.
        name (str): Optimizer name.
    """

    def __init__(self, learning_rate: float=0.001, initial_accumulator_value: float=0.1, epsilon: float=1e-07, weight_decay: float=0.0, name: Optional[str]=None):
        if learning_rate <= 0.0:
            raise ValueError('Learning rate must be positive')
        if initial_accumulator_value < 0.0:
            raise ValueError('Initial accumulator value must be non-negative')
        if epsilon <= 0.0:
            raise ValueError('Epsilon must be positive')
        if weight_decay < 0.0:
            raise ValueError('Weight decay must be non-negative')
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.name = name
        self._accumulators: dict = {}

    def apply_gradients(self, grads_and_vars: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Apply gradients to variables using Adagrad algorithm.
        
        Updates variables by applying computed gradients with Adagrad normalization.
        Maintains accumulator buffers for squared gradients for each variable.
        Accumulators are initialized to initial_accumulator_value.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs where gradient and variable
                          are numpy arrays of the same shape.
                          
        Raises:
            ValueError: If gradient and variable shapes do not match.
            RuntimeError: If optimizer has not been initialized properly.
        """
        if not hasattr(self, 'learning_rate'):
            raise RuntimeError('Optimizer not properly initialized')
        for (grad, var) in grads_and_vars:
            if grad.shape != var.shape:
                raise ValueError(f'Gradient shape {grad.shape} does not match variable shape {var.shape}')
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * var
            var_id = id(var)
            if var_id not in self._accumulators:
                self._accumulators[var_id] = np.full_like(var, self.initial_accumulator_value)
            accumulator = self._accumulators[var_id]
            accumulator += grad ** 2
            var -= self.learning_rate / (np.sqrt(accumulator) + self.epsilon) * grad

    def get_config(self) -> dict[str, Any]:
        """
        Get optimizer configuration as a dictionary.
        
        Returns:
            Dictionary containing all optimizer hyperparameters and state information needed
            to reconstruct the optimizer. Keys include 'learning_rate', 'initial_accumulator_value',
            'epsilon', 'weight_decay', and 'name'.
        """
        return {'learning_rate': self.learning_rate, 'initial_accumulator_value': self.initial_accumulator_value, 'epsilon': self.epsilon, 'weight_decay': self.weight_decay, 'name': self.name}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'AdagradOptimizer':
        """
        Create optimizer instance from configuration dictionary.
        
        Args:
            config: Dictionary containing optimizer configuration parameters.
            
        Returns:
            New optimizer instance with parameters restored from config.
            
        Raises:
            KeyError: If required configuration keys are missing.
        """
        return cls(learning_rate=config['learning_rate'], initial_accumulator_value=config['initial_accumulator_value'], epsilon=config['epsilon'], weight_decay=config['weight_decay'], name=config.get('name'))

class RMSpropOptimizer:
    """
    RMSprop optimizer with optional centered variant.
    
    This optimizer uses a moving average of squared gradients to normalize the gradient.
    It is particularly effective for non-stationary objectives and noisy data. The
    centered variant additionally maintains a moving average of gradients to estimate
    their variance, improving performance on problems with high curvature.
    
    Args:
        learning_rate (float): Initial learning rate (default: 0.001).
        rho (float): Discounting factor for the history/coming gradient (default: 0.9).
        momentum (float): Momentum factor (default: 0.0).
        epsilon (float): Small constant for numerical stability (default: 1e-7).
        centered (bool): If True, gradients are normalized by the estimated variance
                        of the gradient (default: False).
        weight_decay (float): Weight decay coefficient (L2 regularization) (default: 0.0).
        name (Optional[str]): Name of the optimizer instance.
        
    Attributes:
        learning_rate (float): Current learning rate.
        rho (float): Gradient discounting factor.
        momentum (float): Momentum factor.
        epsilon (float): Numerical stability constant.
        centered (bool): Centered RMSprop flag.
        weight_decay (float): L2 penalty coefficient.
        name (str): Optimizer name.
    """

    def __init__(self, learning_rate: float=0.001, rho: float=0.9, momentum: float=0.0, epsilon: float=1e-07, centered: bool=False, weight_decay: float=0.0, nesterov: bool=False, name: Optional[str]=None):
        if learning_rate <= 0.0:
            raise ValueError('Learning rate must be positive')
        if rho < 0.0 or rho >= 1.0:
            raise ValueError('Rho must be in the range [0.0, 1.0)')
        if momentum < 0.0:
            raise ValueError('Momentum must be non-negative')
        if epsilon <= 0.0:
            raise ValueError('Epsilon must be positive')
        if weight_decay < 0.0:
            raise ValueError('Weight decay must be non-negative')
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.name = name
        self._accumulators: dict = {}
        self._means: dict = {}
        self._velocities: dict = {}

    def apply_gradients(self, grads_and_vars: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Apply gradients to variables using RMSprop algorithm.
        
        Updates variables by applying computed gradients with RMSprop normalization.
        When centered=True, uses the centered variant that normalizes by estimated
        gradient variance. Maintains accumulator buffers for squared gradients and
        optionally gradient means and velocities.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs where gradient and variable
                          are numpy arrays of the same shape.
                          
        Raises:
            ValueError: If gradient and variable shapes do not match.
            RuntimeError: If optimizer has not been initialized properly.
        """
        if not hasattr(self, 'learning_rate'):
            raise RuntimeError('Optimizer not properly initialized')
        for (grad, var) in grads_and_vars:
            if grad.shape != var.shape:
                raise ValueError(f'Gradient shape {grad.shape} does not match variable shape {var.shape}')
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * var
            var_id = id(var)
            if var_id not in self._accumulators:
                self._accumulators[var_id] = np.zeros_like(var)
            accumulator = self._accumulators[var_id]
            accumulator *= self.rho
            accumulator += (1.0 - self.rho) * np.square(grad)
            if self.centered:
                if var_id not in self._means:
                    self._means[var_id] = np.zeros_like(var)
                mean = self._means[var_id]
                mean *= self.rho
                mean += (1.0 - self.rho) * grad
                denom = np.sqrt(accumulator - np.square(mean) + self.epsilon)
            else:
                denom = np.sqrt(accumulator + self.epsilon)
            normalized_grad = grad / denom
            if self.momentum > 0.0:
                if var_id not in self._velocities:
                    self._velocities[var_id] = np.zeros_like(var)
                velocity = self._velocities[var_id]
                if self.nesterov:
                    velocity *= self.momentum
                    velocity += normalized_grad
                    var -= self.learning_rate * (normalized_grad + self.momentum * velocity)
                else:
                    velocity *= self.momentum
                    velocity += normalized_grad
                    var -= self.learning_rate * velocity
            else:
                var -= self.learning_rate * normalized_grad

    def get_config(self) -> dict[str, Any]:
        """
        Get optimizer configuration as a dictionary.
        
        Returns:
            Dictionary containing all optimizer hyperparameters and state information needed
            to reconstruct the optimizer. Keys include 'learning_rate', 'rho', 'momentum',
            'epsilon', 'centered', 'weight_decay', 'nesterov', and 'name'.
        """
        return {'learning_rate': self.learning_rate, 'rho': self.rho, 'momentum': self.momentum, 'epsilon': self.epsilon, 'centered': self.centered, 'weight_decay': self.weight_decay, 'nesterov': self.nesterov, 'name': self.name}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'RMSpropOptimizer':
        """
        Create optimizer instance from configuration dictionary.
        
        Args:
            config: Dictionary containing optimizer configuration parameters.
            
        Returns:
            New optimizer instance with parameters restored from config.
            
        Raises:
            KeyError: If required configuration keys are missing.
        """
        return cls(learning_rate=config['learning_rate'], rho=config['rho'], momentum=config.get('momentum', 0.0), epsilon=config.get('epsilon', 1e-07), centered=config.get('centered', False), weight_decay=config.get('weight_decay', 0.0), nesterov=config.get('nesterov', False), name=config.get('name', None))

class AdamOptimizer:
    """
    Adam optimizer implementing adaptive moment estimation for gradient-based optimization.
    
    This optimizer combines the advantages of AdaGrad and RMSProp by computing adaptive learning rates
    for each parameter using estimates of first and second moments of the gradients. It is well-suited
    for problems with large datasets or parameters with varying curvature.
    
    Args:
        learning_rate (float): Initial learning rate (default: 0.001).
        beta1 (float): Exponential decay rate for the first moment estimates (default: 0.9).
        beta2 (float): Exponential decay rate for the second moment estimates (default: 0.999).
        epsilon (float): Small constant for numerical stability (default: 1e-8).
        weight_decay (float): Weight decay coefficient (L2 regularization) (default: 0).
        amsgrad (bool): Whether to use the AMSGrad variant (default: False).
        name (Optional[str]): Name of the optimizer instance.
        
    Attributes:
        learning_rate (float): Current learning rate.
        beta1 (float): Decay rate for first moment.
        beta2 (float): Decay rate for second moment.
        epsilon (float): Numerical stability constant.
        weight_decay (float): L2 penalty coefficient.
        amsgrad (bool): AMSGrad flag.
        name (str): Optimizer name.
    """

    def __init__(self, learning_rate: float=0.001, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-08, weight_decay: float=0.0, amsgrad: bool=False, name: Optional[str]=None):
        if learning_rate <= 0.0:
            raise ValueError('Learning rate must be positive')
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError('Beta1 must be in the range [0.0, 1.0)')
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError('Beta2 must be in the range [0.0, 1.0)')
        if epsilon <= 0.0:
            raise ValueError('Epsilon must be positive')
        if weight_decay < 0.0:
            raise ValueError('Weight decay must be non-negative')
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.name = name
        self._step = 0
        self._moments_first: dict = {}
        self._moments_second: dict = {}
        self._max_moments_second: dict = {}

    def apply_gradients(self, grads_and_vars: list[tuple[Any, Any]]) -> None:
        """
        Apply gradients to variables.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs.
        """
        if not hasattr(self, 'learning_rate'):
            raise RuntimeError('Optimizer not properly initialized')
        self._step += 1
        for (grad, var) in grads_and_vars:
            if grad.shape != var.shape:
                raise ValueError(f'Gradient shape {grad.shape} does not match variable shape {var.shape}')
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * var
            var_id = id(var)
            if var_id not in self._moments_first:
                self._moments_first[var_id] = np.zeros_like(var)
                self._moments_second[var_id] = np.zeros_like(var)
                if self.amsgrad:
                    self._max_moments_second[var_id] = np.zeros_like(var)
            self._moments_first[var_id] = self.beta1 * self._moments_first[var_id] + (1.0 - self.beta1) * grad
            self._moments_second[var_id] = self.beta2 * self._moments_second[var_id] + (1.0 - self.beta2) * np.square(grad)
            bias_correction1 = 1.0 - self.beta1 ** self._step
            bias_correction2 = 1.0 - self.beta2 ** self._step
            if self.amsgrad:
                self._max_moments_second[var_id] = np.maximum(self._max_moments_second[var_id], self._moments_second[var_id])
                denom = np.sqrt(self._max_moments_second[var_id]) / np.sqrt(bias_correction2) + self.epsilon
            else:
                denom = np.sqrt(self._moments_second[var_id]) / np.sqrt(bias_correction2) + self.epsilon
            step_size = self.learning_rate / bias_correction1
            var -= step_size * self._moments_first[var_id] / denom

    def get_config(self) -> dict[str, Any]:
        """
        Get optimizer configuration.
        
        Returns:
            Dictionary containing optimizer configuration parameters.
        """
        return {'learning_rate': self.learning_rate, 'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon, 'weight_decay': self.weight_decay, 'amsgrad': self.amsgrad, 'name': self.name}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'AdamOptimizer':
        """
        Create optimizer from configuration.
        
        Args:
            config: Dictionary containing optimizer configuration.
            
        Returns:
            New optimizer instance.
        """
        return cls(learning_rate=config['learning_rate'], beta1=config['beta1'], beta2=config['beta2'], epsilon=config['epsilon'], weight_decay=config.get('weight_decay', 0.0), amsgrad=config.get('amsgrad', False), name=config.get('name', None))

class AdaptiveMomentEstimationOptimizer:
    """
    Unified optimizer for adaptive moment estimation algorithms including Adam, Adamax, and Nadam.
    
    This optimizer implements adaptive learning rate methods that compute individual learning rates
    for different parameters based on estimates of first and second moments of the gradients.
    
    Supported methods:
    - Adam: Adaptive Moment Estimation
    - Adamax: Adam with infinity norm
    - Nadam: Nesterov-accelerated Adam
    
    Args:
        learning_rate (float): Initial learning rate (default: 0.001).
        beta1 (float): Exponential decay rate for the first moment estimates (default: 0.9).
        beta2 (float): Exponential decay rate for the second moment estimates (default: 0.999).
        epsilon (float): Small constant for numerical stability (default: 1e-8).
        weight_decay (float): Weight decay coefficient (L2 regularization) (default: 0).
        amsgrad (bool): Whether to use the AMSGrad variant (default: False).
        method (str): Optimization method ('adam', 'adamax', 'nadam') (default: 'adam').
        name (Optional[str]): Name of the optimizer instance.
        
    Attributes:
        learning_rate (float): Current learning rate.
        beta1 (float): Decay rate for first moment.
        beta2 (float): Decay rate for second moment.
        epsilon (float): Numerical stability constant.
        weight_decay (float): L2 penalty coefficient.
        amsgrad (bool): AMSGrad flag.
        method (str): Optimization method.
        name (str): Optimizer name.
    """

    def __init__(self, learning_rate: float=0.001, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-08, weight_decay: float=0.0, amsgrad: bool=False, method: str='adam', name: Optional[str]=None):
        if learning_rate <= 0.0:
            raise ValueError('Learning rate must be positive')
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError('Beta1 must be in the range [0.0, 1.0)')
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError('Beta2 must be in the range [0.0, 1.0)')
        if epsilon <= 0.0:
            raise ValueError('Epsilon must be positive')
        if weight_decay < 0.0:
            raise ValueError('Weight decay must be non-negative')
        if method not in ['adam', 'adamax', 'nadam']:
            raise ValueError("Method must be one of 'adam', 'adamax', 'nadam'")
        if amsgrad and method != 'adam':
            raise ValueError('AMSGrad can only be used with Adam method')
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.method = method
        self.name = name
        self._step = 0
        self._moments_first: dict = {}
        self._moments_second: dict = {}
        self._max_moments_second: dict = {}

    def apply_gradients(self, grads_and_vars: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Apply gradients to variables according to the selected adaptive moment estimation method.
        
        Updates the variables by applying the computed gradients using the specified optimization
        algorithm. Handles internal state updates for moment estimates and step counters.
        
        Args:
            grads_and_vars: List of (gradient, variable) pairs where gradient and variable
                          are numpy arrays of the same shape.
                          
        Raises:
            ValueError: If gradient and variable shapes do not match.
            RuntimeError: If optimizer has not been initialized properly.
        """
        if not hasattr(self, 'learning_rate'):
            raise RuntimeError('Optimizer not properly initialized')
        self._step += 1
        for (grad, var) in grads_and_vars:
            if grad.shape != var.shape:
                raise ValueError(f'Gradient shape {grad.shape} does not match variable shape {var.shape}')
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * var
            var_id = id(var)
            if var_id not in self._moments_first:
                self._moments_first[var_id] = np.zeros_like(var)
                self._moments_second[var_id] = np.zeros_like(var)
                if self.amsgrad:
                    self._max_moments_second[var_id] = np.zeros_like(var)
            if self.method == 'adam':
                self._apply_adam_update(grad, var, var_id)
            elif self.method == 'adamax':
                self._apply_adamax_update(grad, var, var_id)
            elif self.method == 'nadam':
                self._apply_nadam_update(grad, var, var_id)

    def _apply_adam_update(self, grad: np.ndarray, var: np.ndarray, var_id: int) -> None:
        """Apply Adam update rule."""
        self._moments_first[var_id] = self.beta1 * self._moments_first[var_id] + (1.0 - self.beta1) * grad
        self._moments_second[var_id] = self.beta2 * self._moments_second[var_id] + (1.0 - self.beta2) * np.square(grad)
        bias_correction1 = 1.0 - self.beta1 ** self._step
        bias_correction2 = 1.0 - self.beta2 ** self._step
        if self.amsgrad:
            self._max_moments_second[var_id] = np.maximum(self._max_moments_second[var_id], self._moments_second[var_id])
            denom = np.sqrt(self._max_moments_second[var_id]) / np.sqrt(bias_correction2) + self.epsilon
        else:
            denom = np.sqrt(self._moments_second[var_id]) / np.sqrt(bias_correction2) + self.epsilon
        step_size = self.learning_rate / bias_correction1
        var -= step_size * self._moments_first[var_id] / denom

    def _apply_adamax_update(self, grad: np.ndarray, var: np.ndarray, var_id: int) -> None:
        """Apply Adamax update rule."""
        self._moments_first[var_id] = self.beta1 * self._moments_first[var_id] + (1.0 - self.beta1) * grad
        self._moments_second[var_id] = np.maximum(self.beta2 * self._moments_second[var_id], np.abs(grad))
        bias_correction1 = 1.0 - self.beta1 ** self._step
        step_size = self.learning_rate / bias_correction1
        var -= step_size * self._moments_first[var_id] / (self._moments_second[var_id] + self.epsilon)

    def _apply_nadam_update(self, grad: np.ndarray, var: np.ndarray, var_id: int) -> None:
        """Apply Nadam update rule."""
        self._moments_first[var_id] = self.beta1 * self._moments_first[var_id] + (1.0 - self.beta1) * grad
        self._moments_second[var_id] = self.beta2 * self._moments_second[var_id] + (1.0 - self.beta2) * np.square(grad)
        bias_correction1 = 1.0 - self.beta1 ** self._step
        bias_correction2 = 1.0 - self.beta2 ** self._step
        momentum_grad = (self.beta1 * self._moments_first[var_id] + (1.0 - self.beta1) * grad) / bias_correction1
        denom = np.sqrt(self._moments_second[var_id]) / np.sqrt(bias_correction2) + self.epsilon
        var -= self.learning_rate * momentum_grad / denom

    def get_config(self) -> dict[str, Any]:
        """
        Get optimizer configuration as a dictionary.
        
        Returns:
            Dictionary containing all optimizer hyperparameters and state information needed
            to reconstruct the optimizer. Keys include 'learning_rate', 'beta1', 'beta2',
            'epsilon', 'weight_decay', 'amsgrad', 'method', and 'name'.
        """
        return {'learning_rate': self.learning_rate, 'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon, 'weight_decay': self.weight_decay, 'amsgrad': self.amsgrad, 'method': self.method, 'name': self.name}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'AdaptiveMomentEstimationOptimizer':
        """
        Create optimizer instance from configuration dictionary.
        
        Args:
            config: Dictionary containing optimizer configuration parameters.
            
        Returns:
            New optimizer instance with parameters restored from config.
            
        Raises:
            KeyError: If required configuration keys are missing.
        """
        return cls(learning_rate=config['learning_rate'], beta1=config['beta1'], beta2=config['beta2'], epsilon=config['epsilon'], weight_decay=config.get('weight_decay', 0.0), amsgrad=config.get('amsgrad', False), method=config.get('method', 'adam'), name=config.get('name', None))

class LBFGSOptimizer:
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimizer.
    
    This optimizer is a quasi-Newton method that approximates the Hessian matrix
    using gradient information from past iterations. It is particularly effective
    for smooth, unconstrained optimization problems and can converge faster than
    first-order methods for suitable problems. The limited-memory variant stores
    only a few vectors representing approximations to the Hessian, making it
    memory-efficient for large-scale problems.
    
    Args:
        max_iterations (int): Maximum number of iterations (default: 100).
        tolerance (float): Convergence tolerance (default: 1e-5).
        history_size (int): Number of updates to store for Hessian approximation (default: 10).
        line_search_fn (Optional[str]): Line search function ('strong_wolfe' or None) (default: 'strong_wolfe').
        name (Optional[str]): Name of the optimizer instance.
        
    Attributes:
        max_iterations (int): Maximum iteration limit.
        tolerance (float): Convergence threshold.
        history_size (int): Size of L-BFGS history buffer.
        line_search_fn (Optional[str]): Selected line search strategy.
        name (str): Optimizer name.
    """

    def __init__(self, max_iterations: int=100, tolerance: float=1e-05, history_size: int=10, line_search_fn: Optional[str]='strong_wolfe', name: Optional[str]=None):
        if max_iterations <= 0:
            raise ValueError('max_iterations must be positive')
        if tolerance <= 0:
            raise ValueError('tolerance must be positive')
        if history_size <= 0:
            raise ValueError('history_size must be positive')
        if line_search_fn is not None and line_search_fn != 'strong_wolfe':
            raise ValueError("line_search_fn must be 'strong_wolfe' or None")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        self.name = name

    def minimize(self, objective_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]], initial_params: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Minimize an objective function using L-BFGS algorithm.
        
        Performs optimization by iteratively updating parameters using L-BFGS updates.
        Requires access to both function values and gradients through the objective_fn.
        Implements two-loop recursion for efficient Hessian-vector products.
        
        Args:
            objective_fn: Callable that takes parameters and returns (loss, gradient) tuple.
                         Loss is a scalar float, gradient is numpy array matching params shape.
            initial_params: Initial parameter values as numpy array.
            
        Returns:
            Tuple of (optimized_parameters, final_loss_value).
            
        Raises:
            ValueError: If initial parameters are incompatible with objective function.
            RuntimeError: If optimization fails to converge within max_iterations.
        """
        if not callable(objective_fn):
            raise ValueError('objective_fn must be callable')
        if not isinstance(initial_params, np.ndarray):
            raise ValueError('initial_params must be a numpy array')
        x = initial_params.astype(np.float64).flatten()
        (f, g) = objective_fn(x)
        if not isinstance(f, (int, float)) or not isinstance(g, np.ndarray):
            raise ValueError('objective_fn must return (float, numpy array) tuple')
        if g.shape != x.shape:
            raise ValueError('Gradient shape must match parameter shape')
        s_history = []
        y_history = []
        rho_history = []
        k = 0
        converged = False
        nfev = 1
        njev = 1
        while k < self.max_iterations:
            grad_norm = np.linalg.norm(g)
            if grad_norm <= self.tolerance:
                converged = True
                break
            q = g.copy()
            alpha = np.zeros(len(s_history))
            for i in range(len(s_history) - 1, -1, -1):
                rho_i = rho_history[i]
                s_i = s_history[i]
                y_i = y_history[i]
                alpha[i] = rho_i * np.dot(s_i, q)
                q = q - alpha[i] * y_i
            r = q.copy()
            for i in range(len(s_history)):
                rho_i = rho_history[i]
                s_i = s_history[i]
                y_i = y_history[i]
                beta = rho_i * np.dot(y_i, r)
                r = r + s_i * (alpha[i] - beta)
            p = -r
            if self.line_search_fn == 'strong_wolfe':
                alpha_step = 1.0
                c1 = 0.0001
                c2 = 0.9
                max_ls_iter = 20
                for _ in range(max_ls_iter):
                    x_new = x + alpha_step * p
                    (f_new, g_new) = objective_fn(x_new)
                    nfev += 1
                    njev += 1
                    if f_new <= f + c1 * alpha_step * np.dot(g, p):
                        if abs(np.dot(g_new, p)) <= c2 * abs(np.dot(g, p)):
                            x = x_new
                            f = f_new
                            g = g_new
                            break
                    alpha_step *= 0.5
                else:
                    x_new = x + alpha_step * p
                    (f_new, g_new) = objective_fn(x_new)
                    nfev += 1
                    njev += 1
                    x = x_new
                    f = f_new
                    g = g_new
            else:
                x_new = x + p
                (f_new, g_new) = objective_fn(x_new)
                nfev += 1
                njev += 1
                x = x_new
                f = f_new
                g = g_new
            if k > 0:
                s_k = x - x_prev
                y_k = g - g_prev
                sy = np.dot(y_k, s_k)
                if sy > 1e-10:
                    rho_k = 1.0 / sy
                    s_history.append(s_k)
                    y_history.append(y_k)
                    rho_history.append(rho_k)
                    if len(s_history) > self.history_size:
                        s_history.pop(0)
                        y_history.pop(0)
                        rho_history.pop(0)
            x_prev = x.copy()
            g_prev = g.copy()
            k += 1
        if not converged:
            raise RuntimeError(f'L-BFGS failed to converge within {self.max_iterations} iterations')
        return (x, f)

    def get_config(self) -> dict[str, Any]:
        """
        Get optimizer configuration as a dictionary.
        
        Returns:
            Dictionary containing all optimizer hyperparameters and state information needed
            to reconstruct the optimizer. Keys include 'max_iterations', 'tolerance',
            'history_size', 'line_search_fn', and 'name'.
        """
        return {'max_iterations': self.max_iterations, 'tolerance': self.tolerance, 'history_size': self.history_size, 'line_search_fn': self.line_search_fn, 'name': self.name}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'LBFGSOptimizer':
        """
        Create optimizer instance from configuration dictionary.
        
        Args:
            config: Dictionary containing optimizer configuration parameters.
            
        Returns:
            New optimizer instance with parameters restored from config.
            
        Raises:
            KeyError: If required configuration keys are missing.
        """
        required_keys = ['max_iterations', 'tolerance', 'history_size']
        for key in required_keys:
            if key not in config:
                raise KeyError(f'Missing required configuration key: {key}')
        return cls(max_iterations=config['max_iterations'], tolerance=config['tolerance'], history_size=config['history_size'], line_search_fn=config.get('line_search_fn'), name=config.get('name'))