from typing import Optional, Union, List
from general.structures.data_batch import DataBatch
from general.base_classes.validator_base import BaseValidator

class ImprovementValidator(BaseValidator):

    def __init__(self, patience: int=5, min_delta: float=0.0001, restore_best_weights: bool=True, name: Optional[str]=None):
        """
        Initialize the ImprovementValidator.

        Args:
            patience (int): Number of iterations with no improvement after which
                            training will be stopped. Defaults to 5.
            min_delta (float): Minimum change in the monitored quantity to qualify
                               as improvement. Defaults to 0.0001.
            restore_best_weights (bool): Whether to restore model weights from the
                                         iteration with the best value. Defaults to True.
            name (Optional[str]): Optional name for the validator instance.
        """
        super().__init__(name=name)
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.reset_monitoring_state()

    def validate(self, data: Union[DataBatch, List[float]], **kwargs) -> bool:
        """
        Validate if sufficient improvement has occurred in the monitored metric.

        This method evaluates whether the current metric value represents a significant
        improvement over previous values. It maintains internal state to track the
        best observed value and the number of iterations since last improvement.

        Args:
            data (Union[DataBatch, List[float]]): Current metric values to evaluate.
                                                Can be a list of floats representing
                                                metric values per epoch, or a DataBatch
                                                containing relevant metrics.
            **kwargs: Additional parameters for validation (e.g., metric name).

        Returns:
            bool: True if training should continue (improvement occurred or patience
                  not exceeded), False if training should stop (no improvement within
                  patience limit).
        """
        current_value = None
        if isinstance(data, DataBatch):
            metric_name = kwargs.get('metric_name')
            if metric_name and data.metadata and (metric_name in data.metadata):
                current_value = data.metadata[metric_name]
            elif data.metadata:
                for name in ['accuracy', 'loss', 'val_accuracy', 'val_loss']:
                    if name in data.metadata:
                        current_value = data.metadata[name]
                        break
        elif isinstance(data, list):
            if data:
                current_value = data[-1]
        if current_value is None:
            raise ValueError('Could not extract metric value from data')
        mode = kwargs.get('mode', 'min')
        if self.best_value is None:
            self.best_value = current_value
            self.waiting_epochs = 0
            if self.restore_best_weights and 'model' in kwargs:
                model = kwargs['model']
                if hasattr(model, 'get_weights'):
                    self.best_weights = model.get_weights()
                else:
                    self.best_weights = model
            return True
        improved = False
        if mode == 'min':
            if current_value <= self.best_value - self.min_delta:
                improved = True
        elif mode == 'max':
            if current_value >= self.best_value + self.min_delta:
                improved = True
        else:
            raise ValueError("Mode must be either 'min' or 'max'")
        if improved:
            self.best_value = current_value
            self.waiting_epochs = 0
            if self.restore_best_weights and 'model' in kwargs:
                model = kwargs['model']
                if hasattr(model, 'get_weights'):
                    self.best_weights = model.get_weights()
                else:
                    self.best_weights = model
            return True
        else:
            self.waiting_epochs += 1
            if self.waiting_epochs > self.patience:
                if self.restore_best_weights and self.best_weights is not None and ('model' in kwargs):
                    model = kwargs['model']
                    if hasattr(model, 'set_weights') and hasattr(self.best_weights, '__iter__'):
                        model.set_weights(self.best_weights)
                return False
            return True

    def reset_monitoring_state(self) -> None:
        """
        Reset the internal monitoring state.

        Clears tracked metrics, best values, and patience counters to their initial
        states. Useful when starting a new training run or reinitializing monitoring.
        """
        self.best_value = None
        self.waiting_epochs = 0
        self.best_weights = None