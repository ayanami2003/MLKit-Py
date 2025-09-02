from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch

class NullConstraintValidator(BaseValidator):

    def __init__(self, columns: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the NullConstraintValidator.
        
        Parameters
        ----------
        columns : list of str, optional
            List of column names to check for null constraints.
            If None, all columns will be checked.
        name : str, optional
            Name for this validator instance
        """
        super().__init__(name=name)
        self.columns = columns

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that specified columns do not contain null values.
        
        Checks the provided data batch to ensure that specified columns
        (or all columns if none specified) do not contain null values.
        
        Parameters
        ----------
        data : DataBatch
            The data to validate for null constraints
        **kwargs : dict
            Additional parameters for validation
            
        Returns
        -------
        bool
            True if null constraints are satisfied, False otherwise
        """
        raw_data = data.data
        if len(raw_data) == 0:
            return True
        if not isinstance(raw_data, pd.DataFrame):
            if hasattr(data, 'feature_names') and data.feature_names:
                df = pd.DataFrame(raw_data, columns=data.feature_names)
            else:
                df = pd.DataFrame(raw_data)
        else:
            df = raw_data
        if self.columns is None:
            columns_to_check = df.columns.tolist()
        else:
            missing_columns = set(self.columns) - set(df.columns)
            if missing_columns:
                self.add_warning(f'Columns {list(missing_columns)} not found in data')
            columns_to_check = [col for col in self.columns if col in df.columns]
        if not columns_to_check:
            return True
        null_check = df[columns_to_check].isnull().any().any()
        return not null_check

class DomainConstraintValidator(BaseValidator):

    def __init__(self, domain_rules: Dict[str, Dict[str, Any]], name: Optional[str]=None):
        """
        Initialize the DomainConstraintValidator.
        
        Parameters
        ----------
        domain_rules : dict of str to dict
            Dictionary mapping column names to their domain constraint specifications.
            Supported constraints:
            - 'allowed_values': list of permitted values
            - 'min_value': minimum permitted numeric value
            - 'max_value': maximum permitted numeric value
            - 'data_type': required data type as string
            - 'pattern': regex pattern for string validation
        name : str, optional
            Name for this validator instance
        """
        super().__init__(name=name)
        self.domain_rules = domain_rules

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that data adheres to specified domain constraints.
        
        Checks each column in the data batch against its defined domain constraints,
        including value ranges, allowed categories, data types, and pattern matching.
        
        Parameters
        ----------
        data : DataBatch
            The data to validate against domain constraints
        **kwargs : dict
            Additional parameters for validation
            
        Returns
        -------
        bool
            True if all domain constraints are satisfied, False otherwise
        """
        df = data.data if isinstance(data.data, pd.DataFrame) else pd.DataFrame(data.data)
        for (column, rules) in self.domain_rules.items():
            if column not in df.columns:
                self.add_warning(f"Column '{column}' not found in data")
                continue
            series = df[column]
            if 'allowed_values' in rules:
                allowed = set(rules['allowed_values'])
                invalid = series.dropna()[~series.dropna().isin(allowed)]
                if not invalid.empty:
                    self.add_error(f"Column '{column}' contains disallowed values: {invalid.unique()}")
                    return False
            if 'min_value' in rules or 'max_value' in rules:
                try:
                    numeric_series = pd.to_numeric(series, errors='raise')
                    if 'min_value' in rules and (numeric_series < rules['min_value']).any():
                        self.add_error(f"Column '{column}' has values below minimum {rules['min_value']}")
                        return False
                    if 'max_value' in rules and (numeric_series > rules['max_value']).any():
                        self.add_error(f"Column '{column}' has values above maximum {rules['max_value']}")
                        return False
                except (ValueError, TypeError):
                    self.add_error(f"Column '{column}' cannot be converted to numeric for range check")
                    return False
            if 'data_type' in rules:
                expected_type = rules['data_type']
                type_map = {'int': 'integer', 'float': 'floating', 'str': 'object', 'bool': 'boolean'}
                if expected_type in type_map:
                    expected_dtype = type_map[expected_type]
                    if expected_dtype == 'integer':
                        non_ints = series.dropna()[~series.dropna().apply(lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()))]
                        if not non_ints.empty:
                            self.add_error(f"Column '{column}' contains non-integer values")
                            return False
                    elif expected_dtype == 'boolean':
                        non_bools = series.dropna()[~series.dropna().apply(lambda x: isinstance(x, (bool, np.bool_)))]
                        if not non_bools.empty:
                            self.add_error(f"Column '{column}' contains non-boolean values")
                            return False
                    elif expected_dtype == 'floating':
                        try:
                            pd.to_numeric(series, errors='raise')
                        except (ValueError, TypeError):
                            self.add_error(f"Column '{column}' cannot be converted to float")
                            return False
                    elif not all(df[column].dropna().apply(lambda x: isinstance(x, str))):
                        self.add_error(f"Column '{column}' contains non-string values")
                        return False
                else:
                    self.add_warning(f"Unknown data type '{expected_type}' for column '{column}'")
            if 'pattern' in rules:
                if not all(series.dropna().apply(lambda x: isinstance(x, str))):
                    self.add_error(f"Column '{column}' must contain strings for pattern matching")
                    return False
                pattern = rules['pattern']
                mismatches = series.dropna()[~series.dropna().str.match(pattern)]
                if not mismatches.empty:
                    self.add_error(f"Column '{column}' has values not matching pattern '{pattern}': {mismatches.unique()}")
                    return False
        return len(self.validation_errors) == 0


# ...(code omitted)...


class OnUpdateConstraintValidator(BaseValidator):

    def __init__(self, immutable_fields: Optional[list]=None, monotonic_fields: Optional[Dict[str, str]]=None, custom_rules: Optional[Dict[str, callable]]=None, name: Optional[str]=None):
        """
        Initialize the OnUpdateConstraintValidator.
        
        Parameters
        ----------
        immutable_fields : list of str, optional
            List of field names that must remain unchanged during updates
        monotonic_fields : dict of str to str, optional
            Dictionary mapping field names to their monotonicity requirement
            ('increasing' or 'decreasing')
        custom_rules : dict of str to callable, optional
            Custom validation functions that take (old_value, new_value) and
            return a boolean indicating validity
        name : str, optional
            Name for this validator instance
        """
        super().__init__(name=name)
        self.immutable_fields = immutable_fields or []
        self.monotonic_fields = monotonic_fields or {}
        self.custom_rules = custom_rules or {}

    def validate(self, old_data: DataBatch, new_data: DataBatch, **kwargs) -> bool:
        """
        Validate that update constraints are satisfied.
        
        Compares old and new data batches to ensure all update-specific
        integrity constraints are maintained.
        
        Parameters
        ----------
        old_data : DataBatch
            The original data before update
        new_data : DataBatch
            The updated data to validate
        **kwargs : dict
            Additional parameters for validation
            
        Returns
        -------
        bool
            True if all update constraints are satisfied, False otherwise
            
        Raises
        ------
        ValueError
            If old_data and new_data have mismatched shapes or incompatible structures
        """
        old_df = old_data.data if isinstance(old_data.data, pd.DataFrame) else pd.DataFrame(old_data.data)
        new_df = new_data.data if isinstance(new_data.data, pd.DataFrame) else pd.DataFrame(new_data.data)
        if old_df.shape != new_df.shape:
            self.add_error('Shape mismatch between old and new data')
            return False
        for field in self.immutable_fields:
            if field in old_df.columns and field in new_df.columns:
                if not old_df[field].equals(new_df[field]):
                    self.add_error(f"Immutable field '{field}' has been modified")
                    return False
            elif field not in old_df.columns and field in new_df.columns:
                self.add_error(f"Immutable field '{field}' has been added")
                return False
            elif field in old_df.columns and field not in new_df.columns:
                self.add_error(f"Immutable field '{field}' has been removed")
                return False
        for (field, trend) in self.monotonic_fields.items():
            if field in old_df.columns and field in new_df.columns:
                old_values = old_df[field]
                new_values = new_df[field]
                if trend == 'increasing':
                    if not (new_values >= old_values).all():
                        self.add_error(f"Field '{field}' violates increasing monotonicity constraint")
                        return False
                elif trend == 'decreasing':
                    if not (new_values <= old_values).all():
                        self.add_error(f"Field '{field}' violates decreasing monotonicity constraint")
                        return False
        for (field, rule_func) in self.custom_rules.items():
            if field in old_df.columns and field in new_df.columns:
                old_values = old_df[field]
                new_values = new_df[field]
                for (old_val, new_val) in zip(old_values, new_values):
                    try:
                        if not rule_func(old_val, new_val):
                            self.add_error(f"Custom rule failed for field '{field}' with values ({old_val}, {new_val})")
                            return False
                    except Exception as e:
                        self.add_error(f"Error applying custom rule for field '{field}': {str(e)}")
                        return False
        return len(self.validation_errors) == 0