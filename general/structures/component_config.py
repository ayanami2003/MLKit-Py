from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

@dataclass
class ComponentConfig:
    """
    Standardized structure for component configuration parameters.
    
    Provides a consistent way to define, store, and pass configuration
    parameters across all ML components in the system.
    
    Attributes
    ----------
    component_name : str
        Name of the component this configuration is for
    component_type : str
        Type/category of the component
    parameters : Dict[str, Any]
        Configuration parameters as key-value pairs
    version : Optional[str]
        Configuration version identifier
    description : Optional[str]
        Human-readable description of this configuration
    created_at : datetime
        When this configuration was created
    tags : List[str]
        Tags for categorizing this configuration
    metadata : Dict[str, Any]
        Additional metadata about the configuration
    """
    component_name: str
    component_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_parameter(self, name: str, value: Any) -> None:
        """
        Add a configuration parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        value : Any
            Parameter value
        """
        self.parameters[name] = value

    def get_parameter(self, name: str, default: Any=None) -> Any:
        """
        Get a configuration parameter value.
        
        Parameters
        ----------
        name : str
            Parameter name
        default : Any
            Default value if parameter not found
            
        Returns
        -------
        Any
            Parameter value or default
        """
        return self.parameters.get(name, default)

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update multiple parameters at once.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters to update
        """
        self.parameters.update(params)

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to this configuration.
        
        Parameters
        ----------
        tag : str
            Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration
        """
        return {'component_name': self.component_name, 'component_type': self.component_type, 'parameters': self.parameters.copy(), 'version': self.version, 'description': self.description, 'created_at': self.created_at.isoformat(), 'tags': self.tags.copy(), 'metadata': self.metadata.copy()}

    def to_json(self) -> str:
        """
        Convert to JSON string.
        
        Returns
        -------
        str
            JSON representation of the configuration
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentConfig':
        """
        Create ComponentConfig from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation
            
        Returns
        -------
        ComponentConfig
            New ComponentConfig instance
        """
        data = data.copy()
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)