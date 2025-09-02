from general.base_classes.validator_base import BaseValidator
from general.structures.component_config import ComponentConfig
from typing import Any, Dict, List, Optional, Union

class HyperparameterValidator(BaseValidator):

    def __init__(self, expected_params: Optional[Dict[str, Dict[str, Any]]]=None, strict_mode: bool=True, name: Optional[str]=None):
        """
        Initialize the HyperparameterValidator.
        
        Parameters
        ----------
        expected_params : Optional[Dict[str, Dict[str, Any]]]
            Dictionary defining expected parameters and their constraints.
            Each parameter entry can contain keys like 'type', 'range', 'choices', 'required'.
        strict_mode : bool, default=True
            If True, raises errors for unexpected parameters; if False, only warns.
        name : Optional[str]
            Name for the validator instance.
        """
        super().__init__(name=name)
        self.expected_params = expected_params or {}
        self.strict_mode = strict_mode

    def validate(self, data: Union[Dict[str, Any], ComponentConfig], **kwargs) -> bool:
        """
        Validate hyperparameter settings against expected constraints.
        
        Parameters
        ----------
        data : Union[Dict[str, Any], ComponentConfig]
            Hyperparameter dictionary or ComponentConfig object to validate
        **kwargs : dict
            Additional validation parameters
            
        Returns
        -------
        bool
            True if validation passes, False otherwise
            
        Raises
        ------
        TypeError
            If data is neither a dict nor ComponentConfig
        """
        self.reset_validation_state()
        if isinstance(data, ComponentConfig):
            params = data.parameters
        elif isinstance(data, dict):
            params = data
        else:
            raise TypeError(f'Data must be a dict or ComponentConfig, got {type(data)}')
        for (param_name, constraints) in self.expected_params.items():
            if constraints.get('required', False) and param_name not in params:
                self.add_error(f"Required parameter '{param_name}' is missing")
        for (param_name, param_value) in params.items():
            if param_name not in self.expected_params:
                if self.strict_mode:
                    self.add_error(f"Unexpected parameter '{param_name}' in strict mode")
                else:
                    self.add_warning(f"Unexpected parameter '{param_name}' encountered")
                continue
            constraints = self.expected_params[param_name]
            expected_type = constraints.get('type')
            if expected_type and (not isinstance(param_value, expected_type)):
                self.add_error(f"Parameter '{param_name}' has type {type(param_value).__name__}, expected {expected_type.__name__}")
            choices = constraints.get('choices')
            if choices and param_value not in choices:
                self.add_error(f"Parameter '{param_name}' has value {param_value}, which is not in allowed choices {choices}")
            value_range = constraints.get('range')
            if value_range and isinstance(param_value, (int, float)):
                (min_val, max_val) = value_range
                if not min_val <= param_value <= max_val:
                    self.add_error(f"Parameter '{param_name}' has value {param_value}, which is outside allowed range [{min_val}, {max_val}]")
        return len(self.validation_errors) == 0

    def add_parameter_constraint(self, param_name: str, param_type: Optional[type]=None, value_range: Optional[tuple]=None, choices: Optional[List[Any]]=None, required: bool=False, description: Optional[str]=None) -> None:
        """
        Add a constraint for a specific hyperparameter.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter
        param_type : Optional[type]
            Expected type of the parameter value
        value_range : Optional[tuple]
            Valid range as (min, max) for numeric parameters
        choices : Optional[List[Any]]
            List of valid choices for categorical parameters
        required : bool, default=False
            Whether this parameter is required
        description : Optional[str]
            Description of the parameter's purpose
        """
        constraint = {}
        if param_type is not None:
            constraint['type'] = param_type
        if value_range is not None:
            constraint['range'] = value_range
        if choices is not None:
            constraint['choices'] = choices
        if required:
            constraint['required'] = True
        if description is not None:
            constraint['description'] = description
        self.expected_params[param_name] = constraint

    def remove_parameter_constraint(self, param_name: str) -> None:
        """
        Remove a parameter constraint.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter to remove
        """
        pass

    def set_expected_parameters(self, params: Dict[str, Dict[str, Any]]) -> None:
        """
        Set all expected parameter constraints at once.
        
        Parameters
        ----------
        params : Dict[str, Dict[str, Any]]
            Dictionary of parameter constraints
        """
        pass