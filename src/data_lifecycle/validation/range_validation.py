from typing import Union, Optional, Dict, List, Tuple
from general.structures.data_batch import DataBatch
import numpy as np
from general.base_classes.validator_base import BaseValidator

class NumericRangeValidator(BaseValidator):
    """
    Validates that numeric data falls within specified ranges.
    
    This validator checks whether all numeric values in the provided data batch
    conform to user-defined minimum and maximum bounds. It supports validation
    on specific columns or globally across all numeric data.
    
    Attributes:
        min_value (Optional[Union[float, List[float]]]): Minimum allowed value(s). 
            If a list, must match the number of validated columns.
        max_value (Optional[Union[float, List[float]]]): Maximum allowed value(s).
            If a list, must match the number of validated columns.
        columns (Optional[List[str]]): Specific column names to validate. 
            If None, validates all numeric columns.
        strict (bool): If True, raises errors for out-of-bound values. 
            If False, issues warnings only.
    
    Methods:
        validate: Performs the range validation on the input data.
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
        if data.data is None or len(data.data) == 0:
            return True
        if isinstance(data.data, np.ndarray):
            arr = data.data
        elif isinstance(data.data, list):
            arr = np.array(data.data)
        else:
            raise TypeError('Data must be a numpy array or list')
        if not self.column_ranges:
            return True
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

class UpperBoundValidator(BaseValidator):

    def __init__(self, upper_bound: Union[float, List[float]], columns: Optional[List[str]]=None, strict: bool=True, name: Optional[str]=None):
        """
        Initializes the UpperBoundValidator with specified bounds and options.
        
        Args:
            upper_bound: Maximum allowed value(s). Can be a scalar or per-column list.
            columns: Column names to apply validation to. If None, uses all numeric columns.
            strict: Whether to treat violations as errors (True) or warnings (False).
            name: Optional name for the validator instance.
        """
        super().__init__(name=name)
        self.upper_bound = upper_bound
        self.columns = columns
        self.strict = strict

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validates that data values do not exceed the specified upper bounds.
        
        Args:
            data: The DataBatch containing data to validate.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            bool: True if all values are within upper bounds, False otherwise.
        """
        self.reset_validation_state()
        if isinstance(data.data, np.ndarray):
            arr = data.data
        elif isinstance(data.data, list):
            arr = np.array(data.data)
        else:
            raise TypeError('Data must be a numpy array or list')
        if self.columns is not None:
            if data.feature_names is None:
                raise ValueError('Column names specified but data has no feature names')
            try:
                column_indices = [data.feature_names.index(col) for col in self.columns]
            except ValueError as e:
                raise ValueError(f'Column not found in data: {e}')
        elif data.feature_names is not None:
            column_indices = []
            for i in range(arr.shape[1] if arr.ndim > 1 else 1):
                col_data = arr[:, i] if arr.ndim > 1 else arr
                if np.issubdtype(col_data.dtype, np.number):
                    column_indices.append(i)
        else:
            column_indices = list(range(arr.shape[1])) if arr.ndim > 1 else [0]
        if isinstance(self.upper_bound, list):
            if len(self.upper_bound) != len(column_indices):
                raise ValueError('Length of upper_bound list must match number of validated columns')
            upper_bounds = self.upper_bound
        else:
            upper_bounds = [self.upper_bound] * len(column_indices)
        all_valid = True
        for (i, col_idx) in enumerate(column_indices):
            if arr.ndim > 1:
                col_data = arr[:, col_idx]
            else:
                col_data = arr
            if not np.issubdtype(col_data.dtype, np.number):
                continue
            upper_val = upper_bounds[i]
            if upper_val is not None and np.any(col_data > upper_val):
                col_name = data.feature_names[col_idx] if data.feature_names else col_idx
                msg = f"Column '{col_name}' has values above maximum bound {upper_val}"
                if self.strict:
                    self.add_error(msg)
                    return False
                else:
                    self.add_warning(msg)
                all_valid = False
        return True if not self.strict else all_valid

class BoundaryValueValidator(BaseValidator):
    """
    Validates data against boundary value constraints including edge cases.
    
    This validator performs comprehensive boundary checks, ensuring data points
    lie within specified limits and correctly handles edge cases like minimum
    and maximum allowable values. It supports both inclusive and exclusive
    boundary definitions.
    
    Attributes:
        lower_bound (Optional[float]): Minimum acceptable value (inclusive by default).
        upper_bound (Optional[float]): Maximum acceptable value (inclusive by default).
        inclusive_lower (bool): Whether lower bound is inclusive.
        inclusive_upper (bool): Whether upper bound is inclusive.
        columns (Optional[List[str]]): Column names to validate.
            If None, validates all numeric columns.
        check_edge_cases (bool): Whether to perform special handling for boundary values.
    
    Methods:
        validate: Executes boundary validation including edge case analysis.
    """

    def __init__(self, lower_bound: Optional[float]=None, upper_bound: Optional[float]=None, inclusive_lower: bool=True, inclusive_upper: bool=True, columns: Optional[List[str]]=None, check_edge_cases: bool=True, name: Optional[str]=None):
        """
        Initializes the BoundaryValueValidator with boundary rules and options.
        
        Args:
            lower_bound: Minimum acceptable value.
            upper_bound: Maximum acceptable value.
            inclusive_lower: Whether lower bound is inclusive.
            inclusive_upper: Whether upper bound is inclusive.
            columns: Column names to validate. If None, uses all numeric columns.
            check_edge_cases: Whether to specially validate boundary values.
            name: Optional name for the validator instance.
        """
        super().__init__(name=name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.inclusive_lower = inclusive_lower
        self.inclusive_upper = inclusive_upper
        self.columns = columns
        self.check_edge_cases = check_edge_cases

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validates that data points conform to specified boundary constraints.
        
        Args:
            data: The DataBatch containing data to validate.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            bool: True if all values satisfy boundary conditions, False otherwise.
        """
        self.reset_validation_state()
        if isinstance(data.data, np.ndarray):
            arr = data.data
        elif isinstance(data.data, list):
            arr = np.array(data.data)
        else:
            raise TypeError('Data must be a numpy array or list')
        if self.columns is not None:
            if data.feature_names is None:
                raise ValueError('Column names specified but data has no feature names')
            try:
                column_indices = [data.feature_names.index(col) for col in self.columns]
            except ValueError as e:
                raise ValueError(f'Column not found in data: {e}')
        elif data.feature_names is not None:
            column_indices = list(range(arr.shape[1])) if arr.ndim > 1 else [0]
        else:
            column_indices = list(range(arr.shape[1])) if arr.ndim > 1 else [0]
        all_valid = True
        for col_idx in column_indices:
            if arr.ndim > 1:
                col_data = arr[:, col_idx]
            else:
                col_data = arr
            if not np.issubdtype(col_data.dtype, np.number):
                continue
            valid_mask = ~np.isnan(col_data)
            clean_col_data = col_data[valid_mask]
            if len(clean_col_data) == 0:
                continue
            if self.lower_bound is not None:
                if self.inclusive_lower:
                    lower_violations = clean_col_data < self.lower_bound
                else:
                    lower_violations = clean_col_data <= self.lower_bound
                if np.any(lower_violations):
                    col_name = data.feature_names[col_idx] if data.feature_names else col_idx
                    msg = f"Column '{col_name}' has values below lower bound {self.lower_bound}"
                    self.add_error(msg)
                    all_valid = False
                if self.check_edge_cases and np.any(clean_col_data == self.lower_bound):
                    if not self.inclusive_lower:
                        col_name = data.feature_names[col_idx] if data.feature_names else col_idx
                        msg = f"Column '{col_name}' has values exactly at lower bound {self.lower_bound} (exclusive)"
                        self.add_error(msg)
                        all_valid = False
            if self.upper_bound is not None:
                if self.inclusive_upper:
                    upper_violations = clean_col_data > self.upper_bound
                else:
                    upper_violations = clean_col_data >= self.upper_bound
                if np.any(upper_violations):
                    col_name = data.feature_names[col_idx] if data.feature_names else col_idx
                    msg = f"Column '{col_name}' has values above upper bound {self.upper_bound}"
                    self.add_error(msg)
                    all_valid = False
                if self.check_edge_cases and np.any(clean_col_data == self.upper_bound):
                    if not self.inclusive_upper:
                        col_name = data.feature_names[col_idx] if data.feature_names else col_idx
                        msg = f"Column '{col_name}' has values exactly at upper bound {self.upper_bound} (exclusive)"
                        self.add_error(msg)
                        all_valid = False
        return all_valid

class IQROutlierValidator(BaseValidator):
    """
    Detects outliers in data based on the Interquartile Range (IQR) method.
    
    This validator identifies data points that fall outside the range defined by
    Q1 - (multiplier * IQR) and Q3 + (multiplier * IQR), where Q1 and Q3 are
    the first and third quartiles respectively, and IQR is their difference.
    
    Attributes:
        multiplier (float): Factor multiplied by IQR to determine outlier bounds.
            Defaults to 1.5, which is common for identifying mild outliers.
        columns (Optional[List[str]]): Column names to check for outliers.
            If None, checks all numeric columns.
        store_iqr_bounds (bool): Whether to store computed IQR bounds for later access.
    
    Methods:
        validate: Checks for presence of outliers in the data.
        get_iqr_bounds: Retrieves computed IQR bounds for validated columns.
    """

    def __init__(self, multiplier: float=1.5, columns: Optional[List[str]]=None, store_iqr_bounds: bool=False, name: Optional[str]=None):
        """
        Initializes the IQROutlierValidator with specified parameters.
        
        Args:
            multiplier: IQR scaling factor for determining outlier bounds.
            columns: Column names to validate. If None, uses all numeric columns.
            store_iqr_bounds: Whether to retain computed IQR bounds after validation.
            name: Optional name for the validator instance.
        """
        super().__init__(name=name)
        self.multiplier = multiplier
        self.columns = columns
        self.store_iqr_bounds = store_iqr_bounds
        self._iqr_bounds: Dict[str, Tuple[float, float]] = {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validates that no data points fall outside computed IQR-based bounds.
        
        Args:
            data: The DataBatch containing data to validate.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            bool: True if no outliers detected, False otherwise.
        """
        self.reset_validation_state()
        if self.store_iqr_bounds:
            self._iqr_bounds = {}
        if isinstance(data.data, np.ndarray):
            array_data = data.data
        else:
            try:
                array_data = np.asarray(data.data)
            except Exception as e:
                self.add_error(f'Failed to convert data to numpy array: {e}')
                return False
        if array_data.ndim == 1:
            array_data = array_data.reshape(-1, 1)
        (n_rows, n_cols) = array_data.shape
        if self.columns is not None:
            if data.feature_names:
                available_columns = data.feature_names
            else:
                available_columns = [f'col_{i}' for i in range(n_cols)]
            col_indices = []
            for col_name in self.columns:
                try:
                    idx = available_columns.index(col_name)
                    col_indices.append(idx)
                except ValueError:
                    self.add_warning(f"Column '{col_name}' not found in data. Skipping.")
            if not col_indices:
                self.add_error('None of the specified columns were found in the data.')
                return False
        else:
            col_indices = []
            if data.feature_names:
                available_columns = data.feature_names
            else:
                available_columns = [f'col_{i}' for i in range(n_cols)]
            for i in range(n_cols):
                if np.issubdtype(array_data[:, i].dtype, np.number):
                    col_indices.append(i)
            if not col_indices:
                self.add_error('No numeric columns found in the data.')
                return False
        has_outliers = False
        for idx in col_indices:
            col_data = array_data[:, idx]
            if not np.issubdtype(col_data.dtype, np.number):
                continue
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) == 0:
                continue
            q1 = np.percentile(valid_data, 25)
            q3 = np.percentile(valid_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.multiplier * iqr
            upper_bound = q3 + self.multiplier * iqr
            if self.store_iqr_bounds:
                col_name = available_columns[idx] if idx < len(available_columns) else f'col_{idx}'
                self._iqr_bounds[col_name] = (lower_bound, upper_bound)
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
            if np.any(outliers):
                has_outliers = True
                col_name = available_columns[idx] if idx < len(available_columns) else f'col_{idx}'
                outlier_count = np.sum(outliers)
                self.add_warning(f"Column '{col_name}' contains {outlier_count} outlier(s) outside bounds [{lower_bound:.4f}, {upper_bound:.4f}]")
        return not has_outliers

    def get_iqr_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Gets the computed IQR bounds for validated columns.
        
        Returns:
            Dict mapping column names to (lower_bound, upper_bound) tuples.
        """
        return self._iqr_bounds.copy()

def compute_maximum_values(data: Union[np.ndarray, DataBatch], axis: Optional[int]=None) -> Union[float, np.ndarray]:
    """
    Computes the maximum values across specified axes of the input data.
    
    This function calculates the largest value(s) present in the data, either
    globally or along specific dimensions. It supports both raw NumPy arrays
    and DataBatch structures.
    
    Args:
        data: Input data as a NumPy array or DataBatch object.
        axis: Axis or axes along which to compute maxima.
              If None, computes the global maximum.
              
    Returns:
        Maximum value(s) as a scalar or array depending on axis parameter.
        
    Raises:
        ValueError: If the data structure is unsupported or axis is invalid.
    """
    if isinstance(data, DataBatch):
        data = data.data
    elif not isinstance(data, np.ndarray):
        raise ValueError('Unsupported data type. Expected numpy.ndarray or DataBatch.')
    return np.nanmax(data, axis=axis)