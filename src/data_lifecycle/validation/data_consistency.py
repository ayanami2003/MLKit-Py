from datetime import datetime
from typing import Dict, List, Optional, Union
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch

class UniqueFieldValidator(BaseValidator):

    def __init__(self, fields: List[str], name: Optional[str]=None):
        """
        Initialize the UniqueFieldValidator.
        
        Parameters
        ----------
        fields : List[str]
            List of column names to validate for uniqueness
        name : Optional[str]
            Name identifier for the validator instance
        """
        super().__init__(name)
        self.fields = fields

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that specified fields contain only unique values.
        
        Parameters
        ----------
        data : DataBatch
            Data batch containing the fields to validate
        **kwargs : dict
            Additional parameters (unused)
            
        Returns
        -------
        bool
            True if all specified fields contain unique values, False otherwise
        """
        self.reset_validation_state()
        if hasattr(data, 'data') and hasattr(data.data, '__array__'):
            import pandas as pd
            df = pd.DataFrame(data.data, columns=data.feature_names) if data.feature_names else pd.DataFrame(data.data)
        elif hasattr(data, 'data') and hasattr(data.data, 'columns'):
            df = data.data
        else:
            self.add_error('Data format not supported for uniqueness validation')
            return False
        for field in self.fields:
            if field not in df.columns:
                self.add_error(f"Field '{field}' not found in data")
                continue
            column_values = df[field]
            if len(column_values) == 0:
                continue
            if column_values.duplicated().any():
                self.add_error(f"Field '{field}' contains duplicate values")
                return False
        return len(self.validation_errors) == 0

class CompositeUniqueConstraintValidator(BaseValidator):
    """
    Validator to ensure specified field combinations are unique across the dataset.
    
    This validator checks that the combination of values across multiple fields is unique
    for each row in the dataset. Unlike UniqueFieldValidator which validates individual
    fields, this validator ensures that the tuple of values across specified fields is
    unique.
    
    Attributes
    ----------
    field_sets : List[List[str]]
        List of field combinations to validate for composite uniqueness
    name : Optional[str]
        Name identifier for the validator instance
        
    Methods
    -------
    validate() -> bool
        Perform the composite uniqueness validation on provided data
    """

    def __init__(self, field_sets: List[List[str]], name: Optional[str]=None):
        """
        Initialize the CompositeUniqueConstraintValidator.
        
        Parameters
        ----------
        field_sets : List[List[str]]
            List of field combinations where each combination must be unique
        name : Optional[str]
            Name identifier for the validator instance
        """
        super().__init__(name)
        self.field_sets = field_sets

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that specified field combinations contain only unique value tuples.
        
        Parameters
        ----------
        data : DataBatch
            Data batch containing the fields to validate
        **kwargs : dict
            Additional parameters (unused)
            
        Returns
        -------
        bool
            True if all specified field combinations contain unique value tuples, False otherwise
        """
        self.reset_validation_state()
        if hasattr(data, 'data') and hasattr(data.data, '__array__'):
            import pandas as pd
            df = pd.DataFrame(data.data, columns=data.feature_names) if data.feature_names else pd.DataFrame(data.data)
        elif hasattr(data, 'data') and hasattr(data.data, 'columns'):
            df = data.data
        else:
            self.add_error('Data format not supported for composite uniqueness validation')
            return False
        if len(df) == 0:
            return True
        for field_set in self.field_sets:
            missing_fields = [field for field in field_set if field not in df.columns]
            if missing_fields:
                self.add_error(f'Fields {missing_fields} not found in data')
                return False
            subset_df = df[field_set]
            if subset_df.duplicated().any():
                self.add_error(f'Duplicate composite values found for fields {field_set}')
                return False
        return len(self.validation_errors) == 0

class NumericRangeValidator(BaseValidator):
    """
    Validator to ensure numeric fields fall within specified ranges.
    
    This validator checks that values in specified numeric columns fall within
    defined minimum and maximum bounds. It supports both inclusive and exclusive
    range boundaries.
    
    Attributes
    ----------
    column_ranges : Dict[str, Dict[str, Union[float, int, bool]]]
        Dictionary mapping column names to their range constraints with keys:
        - 'min': Minimum allowed value
        - 'max': Maximum allowed value
        - 'inclusive_min': Whether minimum is inclusive (default True)
        - 'inclusive_max': Whether maximum is inclusive (default True)
    name : Optional[str]
        Name identifier for the validator instance
        
    Methods
    -------
    validate() -> bool
        Perform the numeric range validation on provided data
    set_column_range()
        Define range constraints for a specific column
    """

    def __init__(self, column_ranges: Optional[Dict[str, Dict[str, Union[float, int, bool]]]]=None, name: Optional[str]=None):
        """
        Initialize the NumericRangeValidator.
        
        Parameters
        ----------
        column_ranges : Optional[Dict[str, Dict[str, Union[float, int, bool]]]]
            Dictionary mapping column names to their range constraints
        name : Optional[str]
            Name identifier for the validator instance
        """
        super().__init__(name)
        if column_ranges is not None and (not isinstance(column_ranges, dict)):
            raise TypeError('column_ranges must be a dictionary or None')
        self.column_ranges = column_ranges or {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that numeric fields fall within specified ranges.
        
        Parameters
        ----------
        data : DataBatch
            Data batch containing the fields to validate
        **kwargs : dict
            Additional parameters (unused)
            
        Returns
        -------
        bool
            True if all values in specified numeric fields are within their defined ranges, False otherwise
        """
        import numpy as np
        self.reset_validation_state()
        if isinstance(data.data, np.ndarray):
            arr = data.data
        elif isinstance(data.data, list):
            arr = np.array(data.data)
        else:
            raise TypeError('Data must be a numpy array or list')
        if self.column_ranges and data.feature_names is None and (arr.ndim > 1):
            raise ValueError('Column ranges specified but data has no feature names')
        for (column, constraints) in self.column_ranges.items():
            if data.feature_names is not None:
                if column not in data.feature_names:
                    continue
                col_idx = data.feature_names.index(column)
            else:
                if column != '0' and len(self.column_ranges) > 1:
                    continue
                col_idx = 0
            if arr.ndim > 1:
                col_data = arr[:, col_idx]
            else:
                col_data = arr
            if not np.issubdtype(col_data.dtype, np.number):
                self.add_warning(f"Column '{column}' is not numeric, skipping range validation")
                continue
            valid_mask = ~np.isnan(col_data)
            clean_col_data = col_data[valid_mask]
            if len(clean_col_data) == 0:
                continue
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            inclusive_min = constraints.get('inclusive_min', True)
            inclusive_max = constraints.get('inclusive_max', True)
            if min_val is not None:
                if inclusive_min:
                    violations = clean_col_data < min_val
                else:
                    violations = clean_col_data <= min_val
                if np.any(violations):
                    self.add_error(f"Column '{column}' has {np.sum(violations)} values below {('inclusive' if inclusive_min else 'exclusive')} minimum {min_val}")
            if max_val is not None:
                if inclusive_max:
                    violations = clean_col_data > max_val
                else:
                    violations = clean_col_data >= max_val
                if np.any(violations):
                    self.add_error(f"Column '{column}' has {np.sum(violations)} values above {('inclusive' if inclusive_max else 'exclusive')} maximum {max_val}")
        return len(self.validation_errors) == 0

    def set_column_range(self, column: str, min_val: Optional[Union[float, int]]=None, max_val: Optional[Union[float, int]]=None, inclusive_min: bool=True, inclusive_max: bool=True) -> None:
        """
        Define range constraints for a specific column.
        
        Parameters
        ----------
        column : str
            Name of the column to set range constraints for
        min_val : Optional[Union[float, int]]
            Minimum allowed value (None means no minimum)
        max_val : Optional[Union[float, int]]
            Maximum allowed value (None means no maximum)
        inclusive_min : bool
            Whether the minimum value is inclusive (default True)
        inclusive_max : bool
            Whether the maximum value is inclusive (default True)
        """
        constraints = {}
        if min_val is not None:
            constraints['min'] = min_val
            constraints['inclusive_min'] = inclusive_min
        if max_val is not None:
            constraints['max'] = max_val
            constraints['inclusive_max'] = inclusive_max
        self.column_ranges[column] = constraints