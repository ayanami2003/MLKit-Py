from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import warnings

class BaseValidator(ABC):
    """
    Abstract base class for data and model validation components.
    
    Defines a standard interface for validation operations that check
    data quality, model assumptions, or system constraints.
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> bool:
        """
        Perform validation checks on the input data.
        
        Parameters
        ----------
        data : Any
            Data to validate
        **kwargs : dict
            Additional validation parameters
            
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        pass
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing validation results, errors, and warnings
        """
        return {
            'name': self.name,
            'passed': len(self.validation_errors) == 0,
            'errors': self.validation_errors.copy(),
            'warnings': self.validation_warnings.copy(),
            'error_count': len(self.validation_errors),
            'warning_count': len(self.validation_warnings)
        }
    
    def reset_validation_state(self) -> None:
        """Clear previous validation results."""
        self.validation_errors.clear()
        self.validation_warnings.clear()
    
    def add_error(self, message: str) -> None:
        """Add a validation error message."""
        self.validation_errors.append(message)
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning message."""
        self.validation_warnings.append(message)
        warnings.warn(message, UserWarning)