import numpy as np
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch

class DataIntegrityValidator(BaseValidator):

    def __init__(self, check_duplicates: bool=True, check_ranges: bool=True, check_structure: bool=True):
        """
        Initialize the data integrity validator.
        
        Parameters
        ----------
        check_duplicates : bool, optional
            Whether to check for duplicate records (default is True)
        check_ranges : bool, optional
            Whether to validate value ranges (default is True)
        check_structure : bool, optional
            Whether to verify structural consistency (default is True)
        """
        super().__init__(name='DataIntegrityValidator')
        self.check_duplicates = check_duplicates
        self.check_ranges = check_ranges
        self.check_structure = check_structure
        self.custom_checks: List[Callable[[DataBatch], bool]] = []

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Perform comprehensive integrity checks on the data batch.
        
        Executes various integrity validations based on configuration:
        - Duplicate record detection
        - Value range validation
        - Structural consistency checks
        - Custom integrity checks if provided
        
        Parameters
        ----------
        data : DataBatch
            Data batch to validate for integrity
        **kwargs : dict
            Additional validation parameters (unused)
            
        Returns
        -------
        bool
            True if all integrity checks pass, False otherwise
        """
        if self.check_duplicates:
            if not self._check_duplicates(data):
                return False
        if self.check_ranges:
            if not self._check_value_ranges(data):
                return False
        if self.check_structure:
            if not self._check_structure(data):
                return False
        for check_func in self.custom_checks:
            try:
                if not check_func(data):
                    return False
            except Exception:
                return False
        return True

    def _check_duplicates(self, data: DataBatch) -> bool:
        """Check for duplicate records in the data batch."""
        try:
            if isinstance(data.data, np.ndarray):
                if data.data.ndim == 0:
                    return True
                elif data.data.ndim == 1:
                    unique_vals = np.unique(data.data)
                    return unique_vals.shape[0] == data.data.shape[0]
                else:
                    unique_rows = np.unique(data.data, axis=0)
                    return unique_rows.shape[0] == data.data.shape[0]
            elif hasattr(data.data, '__len__') and len(data.data) > 0:
                try:
                    if hasattr(data.data[0], '__iter__') and (not isinstance(data.data[0], (str, bytes))):
                        tuples = [tuple(row) for row in data.data]
                        return len(tuples) == len(set(tuples))
                    else:
                        return len(data.data) == len(set(data.data))
                except (TypeError, IndexError):
                    return True
            else:
                return True
        except Exception:
            return False

    def _check_value_ranges(self, data: DataBatch) -> bool:
        """Validate value ranges in the data batch."""
        try:
            if isinstance(data.data, np.ndarray):
                return np.isfinite(data.data).all()
            elif isinstance(data.data, list) and len(data.data) > 0:
                for row in data.data:
                    if hasattr(row, '__iter__') and (not isinstance(row, (str, bytes))):
                        for val in row:
                            if isinstance(val, (int, float)) and (not np.isfinite(val)):
                                return False
                    elif isinstance(row, (int, float)) and (not np.isfinite(row)):
                        return False
                return True
            else:
                return True
        except Exception:
            return False

    def _check_structure(self, data: DataBatch) -> bool:
        """Check structural consistency of the data batch."""
        try:
            data_length = len(data.data) if hasattr(data.data, '__len__') else 0
            if data.labels is not None:
                labels_length = len(data.labels) if hasattr(data.labels, '__len__') else 0
                if data_length != labels_length:
                    return False
            if data.sample_ids is not None:
                if len(data.sample_ids) != data_length:
                    return False
            if isinstance(data.data, np.ndarray):
                if data.data.ndim > 2:
                    return False
            if isinstance(data.data, list) and len(data.data) > 0:
                if hasattr(data.data[0], '__iter__') and (not isinstance(data.data[0], (str, bytes))):
                    try:
                        first_row_len = len(data.data[0])
                        for row in data.data[1:]:
                            if not hasattr(row, '__len__') or len(row) != first_row_len:
                                return False
                    except (TypeError, AttributeError):
                        return False
            return True
        except Exception:
            return False

    def add_custom_check(self, check_func: Callable[[DataBatch], bool]) -> None:
        """
        Add a custom integrity check function.
        
        Parameters
        ----------
        check_func : Callable[[DataBatch], bool]
            Function that takes a DataBatch and returns bool
        """
        if not callable(check_func):
            raise TypeError('check_func must be callable')
        self.custom_checks.append(check_func)


# ...(code omitted)...


class DataFreshnessValidator(BaseValidator):

    def __init__(self, max_age_hours: int=24, timestamp_field: str='timestamp'):
        """
        Initialize the data freshness validator.
        
        Parameters
        ----------
        max_age_hours : int, optional
            Maximum acceptable age of data in hours (default is 24)
        timestamp_field : str, optional
            Name of the field containing timestamp information (default is "timestamp")
        """
        super().__init__(name='DataFreshnessValidator')
        self.max_age_hours = max_age_hours
        self.timestamp_field = timestamp_field

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that the data is fresh according to configured thresholds.
        
        Examines the timestamp metadata in the data batch to determine if all
        records are within the acceptable age limit. Records without timestamps
        or with invalid timestamps will generate warnings.
        
        Parameters
        ----------
        data : DataBatch
            Data batch to validate for freshness
        **kwargs : dict
            Additional validation parameters (unused)
            
        Returns
        -------
        bool
            True if all data is fresh enough, False otherwise
            
        Raises
        ------
        ValueError
            If timestamp field is not found in metadata when required
        """
        if not data.metadata:
            raise ValueError(f"Timestamp field '{self.timestamp_field}' not found in metadata")
        if self.timestamp_field not in data.metadata:
            raise ValueError(f"Timestamp field '{self.timestamp_field}' not found in metadata")
        timestamps = data.metadata[self.timestamp_field]
        if not timestamps:
            return False
        current_time = datetime.now()
        max_age_seconds = self.max_age_hours * 3600
        for ts in timestamps:
            if ts is None:
                return False
            try:
                if isinstance(ts, str):
                    record_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    record_time = datetime.fromtimestamp(ts)
                elif isinstance(ts, datetime):
                    record_time = ts
                else:
                    return False
                age_seconds = (current_time - record_time).total_seconds()
                if age_seconds > max_age_seconds or age_seconds < 0:
                    return False
            except (ValueError, TypeError):
                return False
        return True

def check_data_freshness(data: DataBatch, max_age_hours: int=24, timestamp_field: str='timestamp') -> bool:
    """
    Check if the data in a batch is fresh based on timestamp metadata.
    
    This function evaluates whether the data is fresh enough for use based on
    configurable age thresholds. It examines timestamp fields in the data
    metadata to determine if the data meets freshness requirements.
    
    Parameters
    ----------
    data : DataBatch
        Data batch to validate for freshness
    max_age_hours : int, optional
        Maximum acceptable age of data in hours (default is 24)
    timestamp_field : str, optional
        Name of the field containing timestamp information (default is "timestamp")
        
    Returns
    -------
    bool
        True if all data is fresh enough, False otherwise
        
    Raises
    ------
    ValueError
        If timestamp field is not found in metadata when required
    """
    if not data.metadata:
        raise ValueError(f"Timestamp field '{timestamp_field}' not found in metadata")
    if timestamp_field not in data.metadata:
        raise ValueError(f"Timestamp field '{timestamp_field}' not found in metadata")
    timestamps = data.metadata[timestamp_field]
    if not isinstance(timestamps, (list, tuple)):
        timestamps = [timestamps]
    if len(timestamps) == 0:
        return True
    current_time = datetime.now()
    max_age_seconds = max_age_hours * 3600
    for ts in timestamps:
        if ts is None:
            return False
        try:
            if isinstance(ts, str):
                parsed_ts = None
                formats = ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']
                for fmt in formats:
                    try:
                        parsed_ts = datetime.strptime(ts, fmt)
                        break
                    except ValueError:
                        continue
                if parsed_ts is None:
                    try:
                        if ts.endswith('Z'):
                            ts_iso = ts.replace('Z', '+00:00')
                        else:
                            ts_iso = ts
                        parsed_ts = datetime.fromisoformat(ts_iso)
                    except (ValueError, AttributeError):
                        raise ValueError(f'Unable to parse timestamp string: {ts}')
                ts = parsed_ts
            elif isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts)
            if not isinstance(ts, datetime):
                raise ValueError(f'Invalid timestamp type: {type(ts)}')
            age_seconds = (current_time - ts).total_seconds()
            if age_seconds < 0:
                continue
            if age_seconds > max_age_seconds:
                return False
        except Exception:
            return False
    return True