from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BasePipelineComponent(ABC):
    """
    Abstract base class for pipeline components.
    
    Defines a standard interface for components that can be chained together
    in ML pipelines, including transformers, models, and validators.
    """

    def __init__(self, name: Optional[str]=None):
        self.name = name or self.__class__.__name__
        self.component_config: Optional[Dict[str, Any]] = None

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process input data through this pipeline component.
        
        Parameters
        ----------
        data : Any
            Input data to process
        **kwargs : dict
            Additional processing parameters
            
        Returns
        -------
        Any
            Processed data
        """
        raise NotImplementedError('Subclasses must implement the process method')

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set configuration parameters for this component.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration parameters
        """
        self.component_config = config

    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get current configuration parameters.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Current configuration or None if not set
        """
        return self.component_config

class PipelineStep(BasePipelineComponent):

    def __init__(self, name: Optional[str]=None, step_order: int=0):
        super().__init__(name)
        self.step_order = step_order
        self.previous_step: Optional['PipelineStep'] = None
        self.next_step: Optional['PipelineStep'] = None

    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Any:
        """
        Execute this pipeline step.
        
        Parameters
        ----------
        data : Any
            Input data for this step
        **kwargs : dict
            Additional execution parameters
            
        Returns
        -------
        Any
            Output data from this step
        """
        pass

    def process(self, data: Any, **kwargs) -> Any:
        """
        Process data through this pipeline step.
        
        This is the implementation of the BasePipelineComponent interface.
        
        Parameters
        ----------
        data : Any
            Input data to process
        **kwargs : dict
            Additional processing parameters
            
        Returns
        -------
        Any
            Processed data
        """
        return self.execute(data, **kwargs)