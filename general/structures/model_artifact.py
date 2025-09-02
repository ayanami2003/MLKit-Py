from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
import json
from datetime import datetime

@dataclass
class ModelArtifact:
    """
    Standardized structure for model artifacts and metadata.
    
    Container for trained models, their parameters, performance metrics,
    and associated metadata for storage, deployment, and versioning.
    
    Attributes
    ----------
    model : Any
        The trained model object
    model_type : str
        Type/class of the model
    hyperparameters : Optional[Dict[str, Any]]
        Model hyperparameters used during training
    metrics : Optional[Dict[str, float]]
        Performance metrics on training/validation data
    training_metadata : Optional[Dict[str, Any]]
        Information about training process (duration, hardware, etc.)
    feature_names : Optional[List[str]]
        Names of features used during training
    version : Optional[str]
        Model version identifier
    timestamp : datetime
        When the model was created/trained
    tags : Optional[List[str]]
        User-defined tags for categorization
    """
    model: Any
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    training_metadata: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None
    version: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.metrics is None:
            self.metrics = {}
        if self.training_metadata is None:
            self.training_metadata = {}
        if self.tags is None:
            self.tags = []
        if self.feature_names is None:
            self.feature_names = []

    def add_metric(self, name: str, value: float) -> None:
        """
        Add a performance metric to the metrics dictionary.
        
        Parameters
        ----------
        name : str
            Name of the metric
        value : float
            Value of the metric
        """
        if not isinstance(name, str):
            raise TypeError('Metric name must be a string.')
        if not isinstance(value, (int, float)):
            raise TypeError('Metric value must be a number.')
        self.metrics[name] = value

    def add_hyperparameter(self, name: str, value: Any) -> None:
        """
        Add a hyperparameter to the hyperparameters dictionary.
        
        Parameters
        ----------
        name : str
            Name of the hyperparameter
        value : Any
            Value of the hyperparameter
        """
        if not isinstance(name, str):
            raise TypeError('Hyperparameter name must be a string.')
        self.hyperparameters[name] = value

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the tags list if it doesn't already exist.
        
        Parameters
        ----------
        tag : str
            Tag to add
        """
        if not isinstance(tag, str):
            raise TypeError('Tag must be a string.')
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation (model excluded).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation excluding the model object
        """
        return {'model_type': self.model_type, 'hyperparameters': self.hyperparameters, 'metrics': self.metrics, 'training_metadata': self.training_metadata, 'feature_names': self.feature_names.copy() if self.feature_names else [], 'version': self.version, 'timestamp': self.timestamp.isoformat(), 'tags': self.tags.copy() if self.tags else []}