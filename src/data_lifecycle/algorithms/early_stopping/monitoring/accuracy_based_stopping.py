from typing import Optional, Union, List
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch

class AccuracyBasedStoppingCriterion(BaseValidator):

    def __init__(self, patience: int=5, min_delta: float=0.001, restore_best_weights: bool=True, name: Optional[str]=None):
        """
        Initialize the accuracy-based stopping criterion.

        Args:
            patience (int): Number of epochs with no improvement after which training stops. Defaults to 5.
            min_delta (float): Minimum accuracy improvement required to reset patience counter. Defaults to 0.001.
            restore_best_weights (bool): If True, restores model weights from the best accuracy epoch. Defaults to True.
            name (Optional[str]): Optional name for the validator instance.
        """
        super().__init__(name=name)
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.reset_monitoring_state()

    def validate(self, data: Union[DataBatch, List[float]], **kwargs) -> bool:
        """
        Assess whether training should be stopped based on accuracy metrics.

        This method evaluates the provided accuracy values (either from a DataBatch or a list of floats)
        and determines if the stopping condition has been met. It maintains internal state to track
        consecutive epochs without improvement.

        Args:
            data (Union[DataBatch, List[float]]): Accuracy values for recent training epochs.
                If DataBatch, expects accuracy values in metadata or labels.
                If List[float], directly uses the values for evaluation.
            **kwargs: Additional parameters for validation (not used in current implementation).

        Returns:
            bool: True if training should continue; False if training should stop.

        Raises:
            ValueError: If provided data is malformed or incompatible.
        """
        current_accuracy = None
        if isinstance(data, DataBatch):
            if data.metadata is not None:
                for key in ['accuracy', 'val_accuracy', 'training_accuracy']:
                    if key in data.metadata:
                        current_accuracy = data.metadata[key]
                        break
            if current_accuracy is None and data.labels is not None:
                if isinstance(data.labels, (list, tuple)) and len(data.labels) > 0:
                    try:
                        current_accuracy = float(data.labels[-1])
                    except (ValueError, TypeError):
                        pass
        elif isinstance(data, list):
            if not data:
                raise ValueError('Accuracy list cannot be empty')
            current_accuracy = data[-1]
        else:
            raise TypeError('Data must be either a DataBatch or a list of floats')
        if current_accuracy is None:
            raise ValueError('Could not extract accuracy value from data')
        current_accuracy = float(current_accuracy)
        if self.best_accuracy is None:
            self.best_accuracy = current_accuracy
            self.waiting_epochs = 0
            if self.restore_best_weights and 'model' in kwargs:
                model = kwargs['model']
                if hasattr(model, 'get_weights'):
                    self.best_weights = model.get_weights()
                else:
                    self.best_weights = getattr(model, 'weights', None)
            return True
        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.waiting_epochs = 0
            if self.restore_best_weights and 'model' in kwargs:
                model = kwargs['model']
                if hasattr(model, 'get_weights'):
                    self.best_weights = model.get_weights()
                else:
                    self.best_weights = getattr(model, 'weights', None)
            return True
        else:
            self.waiting_epochs += 1
            if self.waiting_epochs >= self.patience:
                if self.restore_best_weights and self.best_weights is not None and ('model' in kwargs):
                    model = kwargs['model']
                    if hasattr(model, 'set_weights') and hasattr(self.best_weights, '__iter__'):
                        model.set_weights(self.best_weights)
                    elif hasattr(model, 'weights'):
                        setattr(model, 'weights', self.best_weights)
                return False
            return True

    def reset_monitoring_state(self) -> None:
        """
        Reset internal tracking variables for accuracy monitoring.

        This includes counters for patience, best accuracy observed, and any stored model weights.
        Typically called at the beginning of a new training cycle.
        """
        self.best_accuracy = None
        self.waiting_epochs = 0
        self.best_weights = None