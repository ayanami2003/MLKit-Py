from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for machine learning models.
    
    Defines a standard interface for model training, prediction,
    and evaluation components across different model types.
    """

    def __init__(self, name: Optional[str]=None):
        self.name = name or self.__class__.__name__
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: Any, y: Optional[Any]=None, **kwargs) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Parameters
        ----------
        X : Any
            Training features
        y : Any, optional
            Target values
        **kwargs : dict
            Additional training parameters
            
        Returns
        -------
        BaseModel
            Self instance for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : Any
            Input features for prediction
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        Any
            Model predictions
        """
        pass

    @abstractmethod
    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Evaluate model performance on test data.
        
        Parameters
        ----------
        X : Any
            Test features
        y : Any
            True target values
        **kwargs : dict
            Additional evaluation parameters
            
        Returns
        -------
        float
            Model performance score
        """
        pass

    def fit_predict(self, X: Any, y: Optional[Any]=None, **kwargs) -> Any:
        """
        Convenience method to fit and predict in one step.
        
        Parameters
        ----------
        X : Any
            Training features
        y : Any, optional
            Target values
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Any
            Model predictions
        """
        return self.fit(X, y, **kwargs).predict(X, **kwargs)