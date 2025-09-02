import re
from general.structures.data_batch import DataBatch
from general.base_classes.validator_base import BaseValidator
from typing import Union, List, Optional, Any
from typing import Union, List, Optional, Any, Dict
import numpy as np

class GenericFormatValidator(BaseValidator):

    def __init__(self, format_rules: Optional[dict]=None, strict: bool=True, name: Optional[str]=None):
        """
        Initialize the generic format validator.
        
        Args:
            format_rules: Dictionary of format validation rules to apply
            strict: Whether to treat warnings as errors
            name: Optional name for the validator instance
        """
        if name is not None:
            super().__init__(name=name)
        else:
            super().__init__()
            self.name = None
        self.format_rules = format_rules or {}
        self.strict = strict

    def validate(self, data: Union[DataBatch, Any], **kwargs) -> bool:
        """
        Validate that input data conforms to specified format rules.
        
        This method applies configured format validation rules to check if
        the input data meets structural and type requirements.
        
        Args:
            data: Data to validate in the form of DataBatch or other structured format
            **kwargs: Additional keyword arguments for validation configuration
            
        Returns:
            bool: True if data passes all format validations, False otherwise
            
        Raises:
            TypeError: If data type is incompatible with validation rules
            ValueError: If data structure is invalid or malformed
        """
        self.reset_validation_state()
        if isinstance(data, DataBatch):
            data_to_validate = data.data
        else:
            data_to_validate = data
        for (rule_name, rule_definition) in self.format_rules.items():
            try:
                self._apply_format_rule(rule_name, rule_definition, data_to_validate)
            except Exception as e:
                self.add_error(f"Error applying rule '{rule_name}': {str(e)}")
        if self.strict and self.validation_warnings:
            self.validation_errors.extend(self.validation_warnings)
            self.validation_warnings.clear()
        return len(self.validation_errors) == 0

    def _apply_format_rule(self, rule_name: str, rule_definition: dict, data: Any) -> None:
        """
        Apply a specific format rule to the data.
        
        Args:
            rule_name: Name of the rule being applied
            rule_definition: Dictionary defining the validation rule
            data: Data to validate against the rule
        """
        rule_type = rule_definition.get('type')
        if rule_type == 'structure':
            self._validate_structure(rule_definition, data, rule_name)
        elif rule_type == 'type':
            self._validate_type(rule_definition, data, rule_name)
        elif rule_type == 'shape':
            self._validate_shape(rule_definition, data, rule_name)
        elif rule_type == 'range':
            self._validate_range(rule_definition, data, rule_name)
        else:
            self.add_warning(f"Unknown rule type '{rule_type}' for rule '{rule_name}'")

    def _validate_structure(self, rule: dict, data: Any, rule_name: str) -> None:
        """Validate data structure according to rule."""
        expected_keys = rule.get('required_keys')
        target_path = rule.get('target')
        if target_path:
            try:
                target_data = self._get_nested_attribute(data, target_path)
                if target_data is None:
                    self.add_error(f"Rule '{rule_name}': Target path '{target_path}' not found")
                    return
                data = target_data
            except (AttributeError, KeyError, TypeError) as e:
                self.add_error(f"Rule '{rule_name}': Cannot access target '{target_path}' - {str(e)}")
                return
        if expected_keys:
            if not hasattr(data, '__getitem__'):
                self.add_error(f"Rule '{rule_name}': Data does not support key-based access")
                return
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                self.add_error(f"Rule '{rule_name}': Missing required keys: {missing_keys}")

    def _validate_type(self, rule: dict, data: Any, rule_name: str) -> None:
        """Validate data type according to rule."""
        expected_type = rule.get('expected_type')
        if expected_type:
            if isinstance(expected_type, str):
                type_mapping = {'int': int, 'float': float, 'str': str, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'numeric_array': 'numeric_array'}
                if expected_type in type_mapping:
                    expected_type = type_mapping[expected_type]
                else:
                    self.add_warning(f"Rule '{rule_name}': Unknown type '{expected_type}'")
                    return
            if expected_type == 'numeric_array' and isinstance(data, np.ndarray):
                if not np.issubdtype(data.dtype, np.number):
                    self.add_error(f"Rule '{rule_name}': Array is not numeric")
                return
            if isinstance(expected_type, type) or (isinstance(expected_type, tuple) and all((isinstance(t, type) for t in expected_type))):
                if not isinstance(data, expected_type):
                    type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                    self.add_error(f"Rule '{rule_name}': Expected type {type_name}, got {type(data).__name__}")
            else:
                self.add_warning(f"Rule '{rule_name}': Invalid expected_type specification")

    def _validate_shape(self, rule: dict, data: Any, rule_name: str) -> None:
        """Validate data shape according to rule."""
        expected_shape = rule.get('expected_shape')
        if expected_shape:
            target_path = rule.get('target')
            if target_path:
                try:
                    data = self._get_nested_attribute(data, target_path)
                    if data is None:
                        self.add_error(f"Rule '{rule_name}': Target path '{target_path}' not found")
                        return
                except (AttributeError, KeyError, TypeError) as e:
                    self.add_error(f"Rule '{rule_name}': Cannot access target '{target_path}' - {str(e)}")
                    return
            if hasattr(data, 'shape'):
                actual_shape = data.shape
            elif hasattr(data, '__len__'):
                actual_shape = self._get_nested_shape(data)
            else:
                actual_shape = (1,)
            if actual_shape != expected_shape:
                self.add_error(f"Rule '{rule_name}': Expected shape {expected_shape}, got {actual_shape}")

    def _validate_range(self, rule: dict, data: Any, rule_name: str) -> None:
        """Validate data value ranges according to rule."""
        min_val = rule.get('min_value')
        max_val = rule.get('max_value')
        if isinstance(data, np.ndarray):
            if min_val is not None and np.any(data < min_val):
                self.add_error(f"Rule '{rule_name}': Values below minimum {min_val} found")
            if max_val is not None and np.any(data > max_val):
                self.add_error(f"Rule '{rule_name}': Values above maximum {max_val} found")
        elif hasattr(data, '__iter__') and (not isinstance(data, str)):
            try:
                flat_data = np.array(list(data)).flatten()
                if min_val is not None and np.any(flat_data < min_val):
                    self.add_error(f"Rule '{rule_name}': Values below minimum {min_val} found")
                if max_val is not None and np.any(flat_data > max_val):
                    self.add_error(f"Rule '{rule_name}': Values above maximum {max_val} found")
            except (TypeError, ValueError):
                self.add_warning(f"Rule '{rule_name}': Could not validate range for iterable data")
        else:
            try:
                val = float(data)
                if min_val is not None and val < min_val:
                    self.add_error(f"Rule '{rule_name}': Value {val} below minimum {min_val}")
                if max_val is not None and val > max_val:
                    self.add_error(f"Rule '{rule_name}': Value {val} above maximum {max_val}")
            except (TypeError, ValueError):
                self.add_warning(f"Rule '{rule_name}': Could not convert value to numeric for range check")

    def _get_nested_shape(self, data: Any) -> tuple:
        """Recursively determine shape of nested lists/tuples."""
        if not hasattr(data, '__len__') or isinstance(data, str):
            return ()
        if len(data) == 0:
            return (0,)
        inner_shape = self._get_nested_shape(data[0])
        return (len(data),) + inner_shape

    def add_format_rule(self, rule_name: str, rule_definition: dict) -> None:
        """
        Add a new format validation rule.
        
        Args:
            rule_name: Unique identifier for the validation rule
            rule_definition: Dictionary defining the validation rule parameters
        """
        if not isinstance(rule_definition, dict):
            raise TypeError('Rule definition must be a dictionary')
        if 'type' not in rule_definition:
            raise ValueError("Rule definition must include a 'type' field")
        self.format_rules[rule_name] = rule_definition

    def _get_nested_attribute(self, obj: Any, path: str) -> Any:
        """Get nested attribute using dot notation path."""
        attributes = path.split('.')
        for attr in attributes:
            if isinstance(obj, dict):
                try:
                    obj = obj[attr]
                except KeyError:
                    return None
            else:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    return None
        return obj

class CSVFormatValidator(BaseValidator):

    def __init__(self, delimiter: str=',', quotechar: str='"', strict: bool=True, name: Optional[str]=None):
        """
        Initialize the CSV format validator.
        
        Args:
            delimiter: Character used to separate fields in the CSV
            quotechar: Character used to quote fields containing special characters
            strict: Whether to enforce strict CSV format compliance
            name: Optional name for the validator instance
        """
        super().__init__(name=name)
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.strict = strict

    def validate(self, data: Union[DataBatch, List[str]], **kwargs) -> bool:
        """
        Validate that input data conforms to expected CSV format.
        
        This method checks if the input data follows proper CSV formatting rules,
        including consistent column counts across rows, correct quoting, and
        appropriate use of delimiters.
        
        Args:
            data: Either a DataBatch containing raw CSV text lines or a list of CSV row strings
                  to validate. When using DataBatch, data.data should contain the CSV lines.
            **kwargs: Additional keyword arguments (reserved for future extensions)
            
        Returns:
            bool: True if all rows pass CSV format validation, False otherwise
            
        Raises:
            TypeError: If data is neither a DataBatch nor a list of strings
            ValueError: If data is empty or has incompatible structure
        """
        self.reset_validation_state()
        if isinstance(data, DataBatch):
            lines = data.data
        elif isinstance(data, list):
            lines = data
        else:
            raise TypeError('Data must be either a DataBatch or a list of strings')
        if not lines:
            self.add_error('No data provided for validation')
            return False
        if not all((isinstance(line, str) for line in lines)):
            self.add_error('All data rows must be strings')
            return False
        import csv
        from io import StringIO
        csv_params = {'delimiter': self.delimiter, 'quotechar': self.quotechar}
        if self.strict:
            csv_params['strict'] = True
        validated_rows = []
        for (i, line) in enumerate(lines):
            try:
                reader = csv.reader(StringIO(line), **csv_params)
                rows = list(reader)
                if len(rows) != 1:
                    self.add_error(f'Line {i} produced {len(rows)} rows instead of 1')
                    return False
                validated_rows.append(rows[0])
                common_delimiters = [',', ';', '\t', '|']
                if self.delimiter in common_delimiters:
                    current_field_count = len(rows[0])
                    for other_delim in common_delimiters:
                        if other_delim != self.delimiter and other_delim in line:
                            try:
                                other_reader = csv.reader(StringIO(line), delimiter=other_delim, quotechar=self.quotechar)
                                other_rows = list(other_reader)
                                if len(other_rows) == 1 and len(other_rows[0]) > current_field_count:
                                    self.add_error(f'Line {i} appears to use delimiter "{other_delim}" instead of expected "{self.delimiter}"')
                                    return False
                            except:
                                continue
            except Exception as e:
                self.add_error(f'CSV parsing failed for line {i}: {str(e)}')
                return False
        if validated_rows:
            expected_columns = len(validated_rows[0])
            for (i, row) in enumerate(validated_rows):
                if len(row) != expected_columns:
                    self.add_error(f'Row {i} has {len(row)} columns, expected {expected_columns}')
                    return False
        return True

class EmailFormatValidator(BaseValidator):
    """
    Validator for ensuring email addresses conform to standard formatting rules.
    
    This validator checks whether email addresses in input data conform to 
    standard email format specifications, including proper structure with
    local and domain parts, valid characters, and appropriate domain formats.
    It can validate individual emails or columns of emails in batch data.
    
    Attributes:
        email_column (Optional[str]): Name of column containing emails to validate (for batch data)
        strict (bool): Whether to enforce strict RFC-compliant email validation (default: False)
        
    Methods:
        validate: Performs email format validation on input data
        get_validation_report: Returns detailed validation results
    """

    def __init__(self, email_column: Optional[str]=None, strict: bool=False, name: Optional[str]=None):
        """
        Initialize the email format validator.
        
        Args:
            email_column: Name of column containing emails to validate (required when validating DataBatch)
            strict: Whether to enforce strict RFC-compliant email validation
            name: Optional name for the validator instance
        """
        super().__init__(name=name)
        self.email_column = email_column
        self.strict = strict

    def validate(self, data: Union[DataBatch, List[str], str], **kwargs) -> bool:
        """
        Validate that email addresses conform to expected format standards.
        
        This method checks if email addresses follow proper formatting rules,
        including valid local and domain parts, appropriate use of special
        characters, and valid domain structures.
        
        Args:
            data: Email data to validate - can be:
                  - A single email string
                  - A list of email strings
                  - A DataBatch containing a column of emails (requires email_column to be set)
            **kwargs: Additional keyword arguments (reserved for future extensions)
            
        Returns:
            bool: True if all email addresses pass format validation, False otherwise
            
        Raises:
            TypeError: If data type is incompatible with email validation
            ValueError: If email_column is not specified when validating DataBatch
        """
        self.reset_validation_state()
        if isinstance(data, DataBatch):
            if self.email_column is None:
                self.add_error('email_column must be specified when validating DataBatch')
                return False
            if data.feature_names is None or self.email_column not in data.feature_names:
                self.add_error(f"Column '{self.email_column}' not found in DataBatch")
                return False
            column_index = data.feature_names.index(self.email_column)
            if isinstance(data.data, np.ndarray):
                email_data = data.data[:, column_index].tolist()
            else:
                email_data = [row[column_index] for row in data.data]
        elif isinstance(data, list):
            email_data = data
        elif isinstance(data, str):
            email_data = [data]
        else:
            self.add_error(f'Unsupported data type: {type(data)}')
            return False
        for (i, email) in enumerate(email_data):
            if not isinstance(email, str):
                self.add_error(f'Email at index {i} is not a string: {type(email)}')
                continue
            if not self._is_valid_email(email):
                self.add_error(f"Invalid email format at index {i}: '{email}'")
        return len(self.validation_errors) == 0

    def _is_valid_email(self, email: str) -> bool:
        """
        Check if an email address is valid according to formatting rules.
        
        Args:
            email: Email address to validate
            
        Returns:
            bool: True if email is valid, False otherwise
        """
        if self.strict:
            pattern = "^[a-zA-Z0-9.!#$%&\\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        else:
            pattern = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None