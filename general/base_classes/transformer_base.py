from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseTransformer(ABC):
    """
    Abstract base class for data transformation components.
    
    Defines a standard interface for components that transform input data
    into a different format or representation.
    """

    def __init__(self, name: Optional[str]=None):
        self.name = name

    @abstractmethod
    def fit(self, data: Any, **kwargs) -> 'BaseTransformer':
        """
        Fit the transformer to the input data.
        
        Parameters
        ----------
        data : Any
            Input data to fit the transformer on
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        BaseTransformer
            Self instance for method chaining
        """
        pass

    @abstractmethod
    def transform(self, data: Any, **kwargs) -> Any:
        """
        Apply the transformation to input data.
        
        Parameters
        ----------
        data : Any
            Input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        Any
            Transformed data
        """
        pass

    def fit_transform(self, data: Any, **kwargs) -> Any:
        """
        Convenience method to fit and transform in one step.
        
        Parameters
        ----------
        data : Any
            Input data to fit and transform
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Any
            Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    @abstractmethod
    def inverse_transform(self, data: Any, **kwargs) -> Any:
        """
        Apply the inverse transformation if possible.
        
        Parameters
        ----------
        data : Any
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Any
            Original data format
        """
        pass