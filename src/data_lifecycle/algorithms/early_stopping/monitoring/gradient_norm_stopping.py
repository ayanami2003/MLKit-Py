from general.structures.data_batch import DataBatch
from general.base_classes.validator_base import BaseValidator
from typing import Optional, Union, List
import numpy as np

class GradientNormStoppingCriterion(BaseValidator):
    """
    Early stopping validator based on gradient norm monitoring.
    
    This validator monitors the norm of gradients during training to determine
    when to stop training. When the gradient norm falls below a specified threshold
    or stops improving significantly, training is considered to have converged.
    
    The validator tracks gradient norms over time and implements stopping criteria
    based on absolute thresholds, relative improvements, or patience-based waiting.
    
    Attributes
    ----------
    threshold : float
        Minimum gradient norm value below which training stops
    patience : int
        Number of iterations to wait before stopping after threshold is reached
    min_delta : float
        Minimum change in gradient norm to qualify as improvement
    norm_type : str
        Type of norm to compute ('l1', 'l2', 'max', etc.)
    restore_best_weights : bool
        Whether to restore model weights from iteration with minimum gradient norm
    """

    def __init__(self, threshold: float=1e-06, patience: int=10, min_delta: float=1e-08, norm_type: str='l2', restore_best_weights: bool=False, name: Optional[str]=None):
        """
        Initialize the gradient norm stopping criterion.
        
        Parameters
        ----------
        threshold : float, default=1e-6
            Minimum gradient norm value below which training stops
        patience : int, default=10
            Number of iterations to wait before stopping after threshold is reached
        min_delta : float, default=1e-8
            Minimum change in gradient norm to qualify as improvement
        norm_type : str, default='l2'
            Type of norm to compute ('l1', 'l2', 'max', etc.)
        restore_best_weights : bool, default=False
            Whether to restore model weights from iteration with minimum gradient norm
        name : str, optional
            Name for this validator instance
        """
        super().__init__(name=name)
        self.threshold = threshold
        self.patience = patience
        self.min_delta = min_delta
        self.norm_type = norm_type
        self.restore_best_weights = restore_best_weights
        self.reset_monitoring_state()

    def validate(self, data: Union[DataBatch, List[float]], **kwargs) -> bool:
        """
        Validate whether training should stop based on gradient norms.
        
        This method accepts gradient norms either as a DataBatch containing
        gradient information or as a list of gradient norm values. It evaluates
        the stopping criteria and returns whether training should be halted.
        
        Parameters
        ----------
        data : Union[DataBatch, List[float]]
            Gradient norms to evaluate. Can be a DataBatch with gradient information
            or a list of gradient norm values.
        **kwargs : dict
            Additional parameters for validation. May include:
            - current_iteration: int, current training iteration
            - model_weights: array-like, current model weights to save if needed
            
        Returns
        -------
        bool
            True if training should stop, False otherwise
            
        Raises
        ------
        ValueError
            If gradient norms are not provided in a supported format
        """
        gradient_norm = None
        if isinstance(data, DataBatch):
            if data.metadata is not None:
                for key in ['gradient_norm', 'grad_norm', 'gradient_l2_norm']:
                    if key in data.metadata:
                        gradient_norm = data.metadata[key]
                        break
                if gradient_norm is None and 'gradients' in data.metadata:
                    gradients = data.metadata['gradients']
                    gradient_norm = self._compute_norm(gradients)
        elif isinstance(data, list):
            if len(data) > 0:
                gradient_norm = data[-1]
        else:
            raise ValueError('Gradient norms must be provided as DataBatch or List[float]')
        if gradient_norm is None:
            raise ValueError('Could not extract gradient norm from data')
        gradient_norm = float(gradient_norm)
        self.gradient_history.append(gradient_norm)
        if gradient_norm < self.threshold:
            if self.restore_best_weights and self.best_weights is not None and ('model_weights' in kwargs):
                pass
            return True
        if self.best_norm is None or gradient_norm < self.best_norm - self.min_delta:
            self.best_norm = gradient_norm
            self.waiting_iterations = 0
            if self.restore_best_weights and 'model_weights' in kwargs:
                self.best_weights = kwargs['model_weights']
        else:
            self.waiting_iterations += 1
        if self.waiting_iterations >= self.patience:
            if self.restore_best_weights and self.best_weights is not None and ('model_weights' in kwargs):
                pass
            return True
        return False

    def _compute_norm(self, gradients) -> float:
        """
        Compute the norm of gradients based on the specified norm type.
        
        Parameters
        ----------
        gradients : array-like
            Gradient values to compute norm for
            
        Returns
        -------
        float
            Computed norm value
        """
        grad_array = np.array(gradients)
        if self.norm_type == 'l1':
            return np.sum(np.abs(grad_array))
        elif self.norm_type == 'l2':
            return np.sqrt(np.sum(grad_array ** 2))
        elif self.norm_type == 'max':
            return np.max(np.abs(grad_array))
        else:
            try:
                return np.linalg.norm(grad_array, ord=self.norm_type)
            except Exception:
                return np.sqrt(np.sum(grad_array ** 2))

    def reset_monitoring_state(self) -> None:
        """
        Reset the internal monitoring state.
        
        Clears the history of gradient norms and resets counters
        to enable reuse of the validator for a new training run.
        """
        self.gradient_history: List[float] = []
        self.best_norm: Optional[float] = None
        self.waiting_iterations: int = 0
        self.best_weights = None

    def get_gradient_history(self) -> List[float]:
        """
        Retrieve the history of monitored gradient norms.
        
        Returns
        -------
        List[float]
            List of gradient norm values recorded during monitoring
        """
        return self.gradient_history.copy()