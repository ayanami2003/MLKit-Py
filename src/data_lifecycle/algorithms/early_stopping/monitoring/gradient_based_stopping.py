from typing import Optional, Union, Callable
import numpy as np
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch

class GradientBasedStoppingCriterion(BaseValidator):

    def __init__(self, threshold: float=1e-06, patience: int=10, norm_order: Union[int, str]='l2', min_delta: float=1e-08, restore_best_weights: bool=False, name: Optional[str]=None):
        """
        Initialize the gradient-based stopping criterion.
        
        Args:
            threshold: Minimum gradient norm value that triggers stopping.
                      Training continues while the gradient norm exceeds this value.
            patience: Number of iterations with no significant gradient improvement
                     before stopping. Helps avoid premature stopping due to noise.
            norm_order: Type of norm to compute for gradient magnitude. Supports
                       'l1', 'l2', or any integer order (e.g., 1, 2).
            min_delta: Minimum decrease in gradient norm considered an improvement.
                      Prevents stopping on minor fluctuations.
            restore_best_weights: If True, restores model weights from the iteration
                                with the smallest gradient norm.
            name: Optional custom name for the validator instance.
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.patience = patience
        self.norm_order = norm_order
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self._best_grad_norm = float('inf')
        self._patience_counter = 0
        self._best_weights = None
        self._current_iteration = 0

    @property
    def patience_counter(self) -> int:
        """int: Current patience counter value."""
        return self._patience_counter

    @property
    def best_norm(self) -> float:
        """float: Best gradient norm observed so far."""
        return self._best_grad_norm

    def validate(self, data: Union[DataBatch, np.ndarray], **kwargs) -> bool:
        """
        Check if training should stop based on current gradient information.
        
        Evaluates the gradient norm against configured stopping criteria.
        Updates internal tracking of best gradient norms and patience counters.
        
        Args:
            data: Gradient values from current training iteration. Expected to be
                  a numpy array of gradient vectors or a DataBatch containing gradients.
            **kwargs: Additional parameters including 'model_weights' for weight restoration.
            
        Returns:
            bool: True if training should stop, False otherwise.
            
        Raises:
            ValueError: If gradient data format is unsupported.
        """
        if isinstance(data, DataBatch):
            if data.metadata is not None and 'gradients' in data.metadata and (data.metadata['gradients'] is not None):
                gradients = data.metadata['gradients']
            elif isinstance(data.data, np.ndarray):
                gradients = data.data
            else:
                raise ValueError("DataBatch must contain 'gradients' in metadata or have array-like data")
        elif isinstance(data, np.ndarray):
            gradients = data
        else:
            raise ValueError(f'Unsupported data type: {type(data)}. Expected DataBatch or numpy array')
        current_norm = self._compute_gradient_norm(gradients)
        if current_norm <= self.threshold:
            return True
        self._update_patience_counter(current_norm, **kwargs)
        self._current_iteration += 1
        if self._patience_counter >= self.patience:
            return True
        return False

    def reset_monitoring_state(self) -> None:
        """
        Reset all internal tracking variables to initial state.
        
        Clears best gradient norms, patience counters, and stored weights.
        Should be called before starting a new training session.
        """
        self._best_grad_norm = float('inf')
        self._patience_counter = 0
        self._best_weights = None
        self._current_iteration = 0

    def _compute_gradient_norm(self, gradients: np.ndarray) -> float:
        """
        Compute the norm of gradient vectors.
        
        Args:
            gradients: Array of gradient values.
            
        Returns:
            float: Computed norm value according to configured norm_order.
        """
        if self.norm_order == 'l1':
            ord_val = 1
        elif self.norm_order == 'l2':
            ord_val = 2
        else:
            ord_val = self.norm_order
        return float(np.linalg.norm(gradients, ord=ord_val))

    def _update_patience_counter(self, current_norm: float, **kwargs) -> None:
        """
        Update patience counter based on current gradient norm.
        
        Args:
            current_norm: Current computed gradient norm.
            **kwargs: Additional parameters including 'model_weights'.
        """
        if current_norm < self._best_grad_norm - self.min_delta:
            self._best_grad_norm = current_norm
            self._patience_counter = 0
            if self.restore_best_weights:
                if 'model_weights' in kwargs:
                    self._best_weights = kwargs['model_weights']
        else:
            self._patience_counter += 1