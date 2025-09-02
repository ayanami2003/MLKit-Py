from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
import numpy as np
from typing import Optional, List, Union
import pandas as pd

class DatePartExtractor(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, date_parts: Optional[List[str]]=None, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the DatePartExtractor.
        
        Parameters
        ----------
        datetime_column : Optional[str], default=None
            Name of the datetime column to process. If None, will attempt to
            automatically identify datetime columns in the data.
        date_parts : Optional[List[str]], default=None
            Specific date components to extract. If None, extracts common components
            ['year', 'month', 'day', 'weekday']. Valid components:
            ['year', 'quarter', 'month', 'week', 'day', 'weekday', 'hour', 'minute', 'second']
        drop_original : bool, default=True
            Whether to remove the original datetime column from the output
        name : Optional[str], default=None
            Name identifier for the transformer
        """
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.date_parts = date_parts or ['year', 'month', 'day', 'weekday']
        self.drop_original = drop_original

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'DatePartExtractor':
        """
        Fit the transformer to the input data.
        
        This method identifies the datetime column if not specified and validates
        that requested date parts are supported.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime information
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        DatePartExtractor
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If specified datetime column is not found or contains invalid data
        """
        supported_parts = ['year', 'quarter', 'month', 'week', 'day', 'weekday', 'hour', 'minute', 'second']
        invalid_parts = [part for part in self.date_parts if part not in supported_parts]
        if invalid_parts:
            raise ValueError(f'Unsupported date parts: {invalid_parts}. Supported parts are: {supported_parts}')
        if isinstance(data, DataBatch):
            df = pd.DataFrame(data.data, columns=data.feature_names)
            self._input_type = 'databatch'
            self._original_labels = data.labels
            self._original_metadata = data.metadata
            self._original_sample_ids = data.sample_ids
            self._original_batch_id = data.batch_id
        elif isinstance(data, np.ndarray):
            if hasattr(self, '_original_feature_names') and self._original_feature_names:
                feature_names = self._original_feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(data.shape[1])] if len(data.shape) > 1 else ['feature_0']
            df = pd.DataFrame(data, columns=feature_names)
            self._input_type = 'numpy'
        else:
            raise TypeError('Input data must be a DataBatch or numpy array')
        if self.datetime_column is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    self.datetime_column = col
                    break
            if self.datetime_column is None:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            sample_data = df[col].dropna().head(10)
                            if len(sample_data) > 0:
                                pd.to_datetime(sample_data)
                                self.datetime_column = col
                                break
                        except (ValueError, TypeError):
                            continue
        if self.datetime_column is None:
            raise ValueError('No datetime column found in the data')
        if self.datetime_column not in df.columns:
            raise ValueError(f"Specified datetime column '{self.datetime_column}' not found in data")
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_column]):
            try:
                df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
            except Exception as e:
                raise ValueError(f"Could not convert column '{self.datetime_column}' to datetime: {str(e)}")
        self._original_feature_names = df.columns.tolist()
        return self

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Extract date parts from the datetime column.
        
        Creates new columns for each requested date component and adds them to
        the dataset. Optionally removes the original datetime column.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data with datetime column
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Transformed data with extracted date part columns
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or datetime column is missing
        """
        if not hasattr(self, 'datetime_column') or self.datetime_column is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, DataBatch):
            df = pd.DataFrame(data.data, columns=data.feature_names)
            is_data_batch = True
            original_labels = data.labels
            original_metadata = data.metadata
            original_sample_ids = data.sample_ids
            original_batch_id = data.batch_id
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=self._original_feature_names)
            is_data_batch = False
            original_labels = getattr(self, '_original_labels', None)
            original_metadata = getattr(self, '_original_metadata', None)
            original_sample_ids = getattr(self, '_original_sample_ids', None)
            original_batch_id = getattr(self, '_original_batch_id', None)
        else:
            raise TypeError('Input data must be a DataBatch or numpy array')
        if self.datetime_column not in df.columns:
            raise ValueError(f"Datetime column '{self.datetime_column}' not found in input data")
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_column]):
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        for part in self.date_parts:
            new_col_name = f'{self.datetime_column}_{part}'
            if part == 'year':
                df[new_col_name] = df[self.datetime_column].dt.year.astype(int)
            elif part == 'quarter':
                df[new_col_name] = df[self.datetime_column].dt.quarter.astype(int)
            elif part == 'month':
                df[new_col_name] = df[self.datetime_column].dt.month.astype(int)
            elif part == 'week':
                df[new_col_name] = df[self.datetime_column].dt.isocalendar().week.astype(int)
            elif part == 'day':
                df[new_col_name] = df[self.datetime_column].dt.day.astype(int)
            elif part == 'weekday':
                df[new_col_name] = df[self.datetime_column].dt.weekday.astype(int)
            elif part == 'hour':
                df[new_col_name] = df[self.datetime_column].dt.hour.astype(int)
            elif part == 'minute':
                df[new_col_name] = df[self.datetime_column].dt.minute.astype(int)
            elif part == 'second':
                df[new_col_name] = df[self.datetime_column].dt.second.astype(int)
        if self.drop_original:
            df = df.drop(columns=[self.datetime_column])
        if is_data_batch:
            new_feature_names = df.columns.tolist()
            return DataBatch(data=df.values, labels=original_labels, metadata=original_metadata, sample_ids=original_sample_ids, feature_names=new_feature_names, batch_id=original_batch_id)
        else:
            return df.values

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Reverse the transformation by removing extracted date part columns.
        
        Removes the date part columns that were added during transformation,
        optionally restoring the original datetime column.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Transformed data with date part columns
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with date part columns removed
        """
        if not hasattr(self, '_original_feature_names'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, DataBatch):
            df = pd.DataFrame(data.data, columns=data.feature_names)
            is_data_batch = True
            original_labels = data.labels
            original_metadata = data.metadata
            original_sample_ids = data.sample_ids
            original_batch_id = data.batch_id
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])
            is_data_batch = False
            original_labels = getattr(self, '_original_labels', None)
            original_metadata = getattr(self, '_original_metadata', None)
            original_sample_ids = getattr(self, '_original_sample_ids', None)
            original_batch_id = getattr(self, '_original_batch_id', None)
        else:
            raise TypeError('Input data must be a DataBatch or numpy array')
        prefix = f'{self.datetime_column}_'
        cols_to_drop = [col for col in df.columns if col.startswith(prefix) and col[len(prefix):] in self.date_parts]
        df = df.drop(columns=cols_to_drop)
        if hasattr(self, '_original_feature_names'):
            df = df.reindex(columns=self._original_feature_names)
        if is_data_batch:
            new_feature_names = df.columns.tolist()
            return DataBatch(data=df.values, labels=original_labels, metadata=original_metadata, sample_ids=original_sample_ids, feature_names=new_feature_names, batch_id=original_batch_id)
        else:
            return df.values