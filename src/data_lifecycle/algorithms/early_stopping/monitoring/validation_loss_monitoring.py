from typing import Optional, List, Union
from general.structures.data_batch import DataBatch
from general.base_classes.validator_base import BaseValidator

class ValidationLossMonitor(BaseValidator):

    def __init__(self, patience: int=5, min_delta: float=0.0001, restore_best_weights: bool=True, name: Optional[str]=None):
        """
        Initialize the ValidationLossMonitor.
        
        Parameters
        ----------
        patience : int, default=5
            Number of epochs with no improvement after which training will be stopped.
        min_delta : float, default=1e-4
            Minimum decrease in validation loss considered as improvement.
        restore_best_weights : bool, default=True
            Whether to restore model weights from the epoch with the best validation loss.
        name : str, optional
            Name of the monitor instance.
        """
        super().__init__(name=name)
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.waiting_epochs = 0
        self.best_weights = None

    def validate(self, data: Union[DataBatch, List[float]], **kwargs) -> bool:
        """
        Check if training should continue based on validation loss.
        
        Parameters
        ----------
        data : Union[DataBatch, List[float]]
            Either a DataBatch containing validation loss in metadata or a list of validation losses.
        **kwargs : dict
            Additional parameters. May include 'model' for weight restoration.
            
        Returns
        -------
        bool
            True if training should continue, False if it should stop.
        """
        current_loss = None
        if isinstance(data, DataBatch):
            if data.metadata is not None:
                for key in ['loss', 'validation_loss', 'val_loss']:
                    if key in data.metadata:
                        current_loss = data.metadata[key]
                        break
        elif isinstance(data, list):
            if not data:
                raise ValueError('Validation loss list cannot be empty')
            current_loss = data[-1]
        else:
            raise TypeError('Data must be either a DataBatch or a list of floats')
        if current_loss is None:
            raise ValueError("DataBatch must contain 'loss' in metadata for validation monitoring")
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.waiting_epochs = 0
            if self.restore_best_weights and 'model' in kwargs:
                model = kwargs['model']
                if hasattr(model, 'get_weights'):
                    self.best_weights = model.get_weights()
                else:
                    self.best_weights = model
        else:
            self.waiting_epochs += 1
            if self.waiting_epochs >= self.patience:
                if self.restore_best_weights and self.best_weights is not None and ('model' in kwargs):
                    model = kwargs['model']
                    if hasattr(model, 'set_weights') and hasattr(self.best_weights, '__iter__'):
                        model.set_weights(self.best_weights)
                return False
        return True

    def reset_monitoring_state(self) -> None:
        """
        Reset the monitoring state to initial conditions.
        
        This clears the best loss, waiting epochs count, and stored weights.
        """
        self.best_loss = float('inf')
        self.waiting_epochs = 0
        self.best_weights = None