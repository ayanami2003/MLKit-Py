from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
from typing import Any, Dict, List, Union, Optional, Callable
from datetime import datetime
import numpy as np


# ...(code omitted)...


class NumericalRangeValidator(BaseValidator):

    def __init__(self, column_ranges: Dict[str, Dict[str, Union[float, int]]]=None, name: str=None):
        """
        Initialize the NumericalRangeValidator with column range specifications.
        
        Parameters
        ----------
        column_ranges : Dict[str, Dict[str, Union[float, int]]], optional
            Dictionary mapping column names to range constraints.
            Each constraint should specify 'min' and/or 'max' values,
            with optional 'inclusive_min' (default True) and 'inclusive_max' (default True) flags.
            Example: {'age': {'min': 0, 'max': 120, 'inclusive_min': True, 'inclusive_max': False}}
        name : str, optional
            Name identifier for the validator instance
        """
        super().__init__(name)
        self.column_ranges = column_ranges or {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that numerical values in the data batch fall within specified ranges.
        
        Checks each specified column against its configured range constraints.
        Records validation errors for values outside acceptable bounds.
        
        Parameters
        ----------
        data : DataBatch
            Data batch to validate, containing numerical columns to check
        **kwargs : dict
            Additional parameters for validation (not used in this implementation)
            
        Returns
        -------
        bool
            True if all numerical values are within specified ranges, False otherwise
            
        Raises
        ------
        TypeError
            If data is not a DataBatch instance
        ValueError
            If range specifications are invalid
        """
        if not isinstance(data, DataBatch):
            raise TypeError('Data must be a DataBatch instance')
        self.reset_validation_state()
        import pandas as pd
        import numpy as np
        if isinstance(data.data, pd.DataFrame):
            df = data.data
        else:
            try:
                if data.feature_names:
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                else:
                    df = pd.DataFrame(data.data)
            except Exception as e:
                self.add_error(f'Failed to convert data to DataFrame: {str(e)}')
                return False
        is_valid = True
        for (column, constraints) in self.column_ranges.items():
            if column not in df.columns:
                continue
            series = pd.to_numeric(df[column], errors='coerce')
            nan_mask = series.isna()
            nan_count = nan_mask.sum()
            if nan_count > 0 and (not df[column].isna().all()):
                if nan_count < len(series):
                    self.add_error(f"Column '{column}' contains {nan_count} NaN values which violate range constraints")
                    is_valid = False
                    continue
            if nan_mask.all():
                continue
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            inclusive_min = constraints.get('inclusive_min', True)
            inclusive_max = constraints.get('inclusive_max', True)
            if min_val is not None:
                if inclusive_min:
                    min_violations = series < min_val
                else:
                    min_violations = series <= min_val
                if min_violations.any():
                    violating_indices = series[min_violations].index.tolist()
                    self.add_error(f"Column '{column}' has {min_violations.sum()} values below minimum {min_val} at indices {violating_indices}")
                    is_valid = False
            if max_val is not None:
                if inclusive_max:
                    max_violations = series > max_val
                else:
                    max_violations = series >= max_val
                if max_violations.any():
                    violating_indices = series[max_violations].index.tolist()
                    self.add_error(f"Column '{column}' has {max_violations.sum()} values above maximum {max_val} at indices {violating_indices}")
                    is_valid = False
        return is_valid

    def set_column_range(self, column: str, min_val: Union[float, int, None]=None, max_val: Union[float, int, None]=None, inclusive_min: bool=True, inclusive_max: bool=True) -> None:
        """
        Set or update range constraints for a specific column.
        
        Parameters
        ----------
        column : str
            Name of the column to constrain
        min_val : float, int, or None, optional
            Minimum allowed value (None means no lower bound)
        max_val : float, int, or None, optional
            Maximum allowed value (None means no upper bound)
        inclusive_min : bool, default True
            Whether the minimum bound is inclusive
        inclusive_max : bool, default True
            Whether the maximum bound is inclusive
            
        Raises
        ------
        ValueError
            If both min_val and max_val are None, or if min_val > max_val
        TypeError
            If min_val or max_val are not numeric when provided
        """
        if min_val is None and max_val is None:
            raise ValueError('At least one of min_val or max_val must be specified')
        if min_val is not None and (not isinstance(min_val, (int, float))):
            raise TypeError('min_val must be numeric or None')
        if max_val is not None and (not isinstance(max_val, (int, float))):
            raise TypeError('max_val must be numeric or None')
        if min_val is not None and max_val is not None and (min_val > max_val):
            raise ValueError('min_val cannot be greater than max_val')
        self.column_ranges[column] = {'min': min_val, 'max': max_val, 'inclusive_min': inclusive_min, 'inclusive_max': inclusive_max}

class DateRangeValidator(BaseValidator):
    """
    Validator that checks if date/datetime values fall within specified ranges.
    
    This validator ensures that date and datetime columns in a dataset conform to
    predefined minimum and maximum bounds. It handles various date formats and
    can validate both date-only and datetime values.
    
    Attributes
    ----------
    column_ranges : Dict[str, Dict[str, Union[datetime, str]]]
        Mapping of column names to their allowed date range specifications
        Format: {'column_name': {'min': datetime_or_string, 'max': datetime_or_string, 'format': str}}
    name : str
        Name identifier for the validator
        
    Methods
    -------
    validate() : Validates date ranges in a DataBatch
    set_column_date_range() : Sets date range constraints for a specific column
    """

    def __init__(self, column_ranges: Dict[str, Dict[str, Union[datetime, str]]]=None, name: str=None):
        """
        Initialize the DateRangeValidator with column date range specifications.
        
        Parameters
        ----------
        column_ranges : Dict[str, Dict[str, Union[datetime, str]]], optional
            Dictionary mapping column names to date range constraints.
            Each constraint should specify 'min' and/or 'max' dates (as datetime objects or parseable strings),
            with an optional 'format' string for parsing string dates.
            Example: {'birth_date': {'min': '1900-01-01', 'max': '2023-12-31', 'format': '%Y-%m-%d'}}
        name : str, optional
            Name identifier for the validator instance
        """
        super().__init__(name)
        self.column_ranges = column_ranges or {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that date values in the data batch fall within specified ranges.
        
        Checks each specified date/datetime column against its configured range constraints.
        Records validation errors for values outside acceptable bounds.
        
        Parameters
        ----------
        data : DataBatch
            Data batch to validate, containing date/datetime columns to check
        **kwargs : dict
            Additional parameters for validation (not used in this implementation)
            
        Returns
        -------
        bool
            True if all date values are within specified ranges, False otherwise
            
        Raises
        ------
        TypeError
            If data is not a DataBatch instance
        ValueError
            If date range specifications are invalid or dates cannot be parsed
        """
        if not isinstance(data, DataBatch):
            raise TypeError('Data must be a DataBatch instance')
        self.reset_validation_state()
        if data.feature_names is None:
            self.add_warning('No feature names provided in DataBatch, skipping date range validation')
            return True
        try:
            if isinstance(data.data, list):
                data_array = np.array(data.data)
            else:
                data_array = data.data
        except Exception as e:
            self.add_error(f'Could not convert data to array for validation: {str(e)}')
            return False
        for (column, range_spec) in self.column_ranges.items():
            if column not in data.feature_names:
                self.add_warning(f"Column '{column}' specified in ranges but not found in data")
                continue
            try:
                col_idx = data.feature_names.index(column)
            except ValueError:
                self.add_warning(f"Column '{column}' not found in feature names")
                continue
            try:
                min_date = self._parse_date(range_spec.get('min'), range_spec.get('format'))
                max_date = self._parse_date(range_spec.get('max'), range_spec.get('format'))
            except ValueError as e:
                self.add_error(f"Invalid date specification for column '{column}': {str(e)}")
                continue
            if isinstance(data_array, np.ndarray):
                if data_array.ndim == 1:
                    values = data_array
                else:
                    values = data_array[:, col_idx]
            else:
                values = [row[col_idx] for row in data_array]
            for (i, value) in enumerate(values):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                try:
                    parsed_value = self._parse_date(value, range_spec.get('format'))
                    if min_date is not None and parsed_value < min_date:
                        self.add_error(f"Column '{column}', row {i}: Value {value} is before minimum allowed date {min_date}")
                    if max_date is not None and parsed_value > max_date:
                        self.add_error(f"Column '{column}', row {i}: Value {value} is after maximum allowed date {max_date}")
                except ValueError as e:
                    self.add_error(f"Column '{column}', row {i}: Could not parse date value '{value}': {str(e)}")
        return len(self.validation_errors) == 0

    def set_column_date_range(self, column: str, min_date: Union[datetime, str]=None, max_date: Union[datetime, str]=None, date_format: str=None) -> None:
        """
        Set date range constraints for a specific column.
        
        Parameters
        ----------
        column : str
            Name of the column to apply date range constraints to
        min_date : Union[datetime, str], optional
            Minimum allowed date (as datetime object or parseable string)
        max_date : Union[datetime, str], optional
            Maximum allowed date (as datetime object or parseable string)
        date_format : str, optional
            Format string for parsing date strings (e.g., '%Y-%m-%d')
        """
        if column not in self.column_ranges:
            self.column_ranges[column] = {}
        if min_date is not None:
            self.column_ranges[column]['min'] = min_date
        if max_date is not None:
            self.column_ranges[column]['max'] = max_date
        if date_format is not None:
            self.column_ranges[column]['format'] = date_format

    def _parse_date(self, date_value: Union[datetime, str], date_format: str=None) -> datetime:
        """
        Parse a date value into a datetime object.
        
        Parameters
        ----------
        date_value : Union[datetime, str]
            Date value to parse
        date_format : str, optional
            Format string for parsing date strings
            
        Returns
        -------
        datetime
            Parsed datetime object
            
        Raises
        ------
        ValueError
            If date cannot be parsed
        """
        if date_value is None:
            return None
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, str):
            if date_format:
                try:
                    return datetime.strptime(date_value, date_format)
                except ValueError:
                    raise ValueError(f"Could not parse date '{date_value}' with format '{date_format}'")
            else:
                formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S.%f', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
                for fmt in formats:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Could not parse date string '{date_value}' with any known format")
        raise ValueError(f'Unsupported date value type: {type(date_value)}')