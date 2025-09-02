from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import numpy as np

@dataclass
class DataBatch:
    """
    Standardized structure for passing batched data between components.
    
    Provides a consistent container for data, labels, metadata, and
    associated information that flows through the ML pipeline.
    
    Attributes
    ----------
    data : Union[np.ndarray, List]
        Primary data content (features, samples, etc.)
    labels : Optional[Union[np.ndarray, List]]
        Associated target values or labels
    metadata : Optional[Dict[str, Any]]
        Additional information about the batch
    sample_ids : Optional[List[str]]
        Identifiers for individual samples
    feature_names : Optional[List[str]]
        Names of features/columns when applicable
    batch_id : Optional[str]
        Identifier for this batch
    """
    data: Union[np.ndarray, List]
    labels: Optional[Union[np.ndarray, List]] = None
    metadata: Optional[Dict[str, Any]] = None
    sample_ids: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    batch_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.sample_ids is not None:
            data_length = len(self.data) if hasattr(self.data, '__len__') else 0
            if len(self.sample_ids) != data_length:
                raise ValueError('Length of sample_ids must match data length')
        if self.labels is not None:
            data_length = len(self.data) if hasattr(self.data, '__len__') else 0
            labels_length = len(self.labels) if hasattr(self.labels, '__len__') else 0
            if data_length != labels_length:
                raise ValueError('Length of data and labels must match')

    def get_shape(self) -> tuple:
        """Get the shape of the primary data."""
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        elif hasattr(self.data, '__len__'):
            if len(self.data) > 0 and hasattr(self.data[0], '__len__'):
                return (len(self.data), len(self.data[0]))
            else:
                return (len(self.data),)
        else:
            return (1,)

    def is_labeled(self) -> bool:
        """Check if this batch contains labels."""
        return self.labels is not None