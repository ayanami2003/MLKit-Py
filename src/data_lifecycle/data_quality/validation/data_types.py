import numpy as np
import pandas as pd
from typing import Union, List
from general.structures.data_batch import DataBatch
from general.base_classes.validator_base import BaseValidator

class NumericTypeValidator(BaseValidator):

    def __init__(self, columns: Union[List[str], None]=None, strict: bool=True, name: str=None):
        """
        Initialize the NumericTypeValidator.
        
        Parameters
        ----------
        columns : List[str], optional
            Specific column names to validate. If None, validates all columns.
        strict : bool, default=True
            If True, requires all values to be strictly numeric. If False, allows NaN values.
        name : str, optional
            Name for the validator instance.
        """
        super().__init__(name=name)
        self.columns = columns
        self.strict = strict

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that specified columns in the data batch contain only numeric values.
        
        Parameters
        ----------
        data : DataBatch
            The data batch to validate, expected to have structured data with feature names.
        **kwargs : dict
            Additional parameters for validation (not used in this implementation).
            
        Returns
        -------
        bool
            True if all specified columns contain only numeric values, False otherwise.
            
        Raises
        ------
        ValueError
            If specified columns are not found in the data.
        """
        if self.columns is not None:
            columns_to_validate = self.columns
        elif data.feature_names is not None:
            columns_to_validate = data.feature_names
        elif isinstance(data.data, np.ndarray) and data.data.ndim == 2:
            columns_to_validate = [f'col_{i}' for i in range(data.data.shape[1])]
        else:
            raise ValueError('Cannot infer columns to validate. Please specify column names or provide feature names in DataBatch.')
        if data.feature_names is not None:
            missing_columns = set(columns_to_validate) - set(data.feature_names)
            if missing_columns:
                raise ValueError(f'Specified columns not found in data: {missing_columns}')
            col_indices = {name: idx for (idx, name) in enumerate(data.feature_names)}
        elif isinstance(data.data, np.ndarray) and data.data.ndim == 2:
            n_cols = data.data.shape[1]
            available_cols = [f'col_{i}' for i in range(n_cols)]
            missing_columns = set(columns_to_validate) - set(available_cols)
            if missing_columns:
                raise ValueError(f'Specified columns not found in data: {missing_columns}')
            col_indices = {f'col_{i}': i for i in range(n_cols)}
        else:
            raise ValueError('Data must be a 2D numpy array when feature names are not provided.')
        if not isinstance(data.data, np.ndarray):
            try:
                array_data = np.array(data.data)
            except Exception as e:
                raise ValueError(f'Failed to convert data to numpy array: {e}')
        else:
            array_data = data.data
        for col_name in columns_to_validate:
            col_idx = col_indices[col_name]
            col_data = array_data[:, col_idx] if array_data.ndim == 2 else array_data
            try:
                numeric_col = np.asarray(col_data, dtype=float)
            except (ValueError, TypeError):
                return False
            if self.strict and np.isnan(numeric_col).any():
                return False
        return True

class DateTypeValidator(BaseValidator):
    """
    Validator to ensure that specified columns in a dataset contain only date/datetime data.
    
    This validator checks whether the data in specified columns can be interpreted as date or datetime values.
    It supports various date formats and can parse strings, timestamps, and other representations of dates.
    
    Attributes
    ----------
    columns : List[str], optional
        Specific column names to validate. If None, validates all columns.
    date_format : str, optional
        Expected date format string (e.g., '%Y-%m-%d'). If None, tries to infer the format.
    strict : bool, default=True
        If True, requires all values to be strictly date/datetime. If False, allows NaN values.
    """

    def __init__(self, columns: Union[List[str], None]=None, date_format: Union[str, None]=None, strict: bool=True, name: str=None):
        """
        Initialize the DateTypeValidator.
        
        Parameters
        ----------
        columns : List[str], optional
            Specific column names to validate. If None, validates all columns.
        date_format : str, optional
            Expected date format string (e.g., '%Y-%m-%d'). If None, tries to infer the format.
        strict : bool, default=True
            If True, requires all values to be strictly date/datetime. If False, allows NaN values.
        name : str, optional
            Name for the validator instance.
        """
        super().__init__(name=name)
        self.columns = columns
        self.date_format = date_format
        self.strict = strict

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that specified columns in the data batch contain only date/datetime values.
        
        Parameters
        ----------
        data : DataBatch
            The data batch to validate, expected to have structured data with feature names.
        **kwargs : dict
            Additional parameters for validation (not used in this implementation).
            
        Returns
        -------
        bool
            True if all specified columns contain only date/datetime values, False otherwise.
            
        Raises
        ------
        ValueError
            If specified columns are not found in the data.
        """
        if self.columns is not None:
            columns_to_validate = self.columns
        elif data.feature_names is not None:
            columns_to_validate = data.feature_names
        elif isinstance(data.data, np.ndarray) and data.data.ndim == 2:
            columns_to_validate = [f'col_{i}' for i in range(data.data.shape[1])]
        else:
            raise ValueError('Cannot infer columns to validate. Please specify column names or provide feature names in DataBatch.')
        if data.feature_names is not None:
            missing_columns = set(columns_to_validate) - set(data.feature_names)
            if missing_columns:
                raise ValueError(f'Specified columns not found in data: {missing_columns}')
        elif isinstance(data.data, np.ndarray) and data.data.ndim == 2:
            n_cols = data.data.shape[1]
            available_cols = [f'col_{i}' for i in range(n_cols)]
            missing_columns = set(columns_to_validate) - set(available_cols)
            if missing_columns:
                raise ValueError(f'Specified columns not found in data: {missing_columns}')
        else:
            raise ValueError('Data must be a 2D numpy array when feature names are not provided.')
        if isinstance(data.data, np.ndarray):
            if data.feature_names is not None:
                df = pd.DataFrame(data.data, columns=data.feature_names)
            else:
                df = pd.DataFrame(data.data, columns=[f'col_{i}' for i in range(data.data.shape[1])])
        else:
            df = data.data
        for col_name in columns_to_validate:
            if col_name not in df.columns:
                raise ValueError(f"Column '{col_name}' not found in data.")
            col_data = df[col_name]
            if not self.strict:
                mask = pd.isna(col_data)
                filtered_data = col_data[~mask]
                if len(filtered_data) == 0:
                    continue
            else:
                filtered_data = col_data
                if pd.isna(filtered_data).any():
                    return False
            try:
                if self.date_format:
                    converted = pd.to_datetime(filtered_data, format=self.date_format, errors='raise')
                else:
                    converted = pd.to_datetime(filtered_data, infer_datetime_format=True, errors='raise')
            except (ValueError, TypeError):
                return False
            if converted.isna().any():
                return False
        return True