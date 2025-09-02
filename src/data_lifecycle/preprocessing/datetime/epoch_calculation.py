from typing import Optional, List, Union
from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
import numpy as np
import pandas as pd
from datetime import datetime

class EpochCalculator(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, unit: str='seconds', drop_original: bool=True, name: Optional[str]=None):
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.unit = unit
        self.drop_original = drop_original
        self._validate_unit()

    def _validate_unit(self) -> None:
        """Validate that the unit is supported."""
        valid_units = ['seconds', 'milliseconds', 'minutes', 'hours', 'days']
        if self.unit not in valid_units:
            raise ValueError(f"Unit '{self.unit}' is not supported. Valid units are: {valid_units}")

    def _detect_datetime_column(self, data: Union[DataBatch, np.ndarray]) -> str:
        """
        Detect datetime column in the data.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data to search for datetime columns.
            
        Returns
        -------
        str
            Name of the detected datetime column.
            
        Raises
        ------
        ValueError
            If no datetime column is found or multiple datetime columns exist.
        """
        if isinstance(data, DataBatch):
            if data.feature_names is None:
                raise ValueError('DataBatch must have feature_names to detect datetime columns')
            datetime_columns = []
            for (i, name) in enumerate(data.feature_names):
                column_data = data.data[:, i] if isinstance(data.data, np.ndarray) else np.array([row[i] for row in data.data])
                if self._is_datetime_column(column_data):
                    datetime_columns.append(name)
            if len(datetime_columns) == 0:
                raise ValueError('No datetime columns detected in the data')
            elif len(datetime_columns) > 1:
                raise ValueError(f'Multiple datetime columns detected: {datetime_columns}. Please specify datetime_column explicitly.')
            else:
                return datetime_columns[0]
        elif self.datetime_column is not None:
            return self.datetime_column
        else:
            raise ValueError('datetime_column must be specified when using numpy arrays')

    def _is_datetime_column(self, column_data: np.ndarray) -> bool:
        """
        Check if a column contains datetime data.
        
        Parameters
        ----------
        column_data : np.ndarray
            Column data to check.
            
        Returns
        -------
        bool
            True if column contains datetime data, False otherwise.
        """
        try:
            sample_values = column_data[:min(5, len(column_data))]
            parsed_count = 0
            for val in sample_values:
                if pd.isna(val):
                    continue
                if isinstance(val, (pd.Timestamp, datetime, np.datetime64)):
                    parsed_count += 1
                elif isinstance(val, (str, int, float)):
                    pd.to_datetime(val)
                    parsed_count += 1
            non_null_count = len([v for v in sample_values if not pd.isna(v)])
            return parsed_count >= max(1, non_null_count // 2) if non_null_count > 0 else False
        except (ValueError, TypeError):
            return False

    def _convert_to_datetime(self, data: Union[DataBatch, np.ndarray]) -> Union[DataBatch, np.ndarray]:
        """
        Convert the specified column to datetime format.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data.
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with datetime column converted to proper datetime format.
        """
        if isinstance(data, DataBatch):
            if data.feature_names is None:
                raise ValueError('DataBatch must have feature_names')
            col_idx = data.feature_names.index(self.datetime_column)
            new_data = data.data.copy() if isinstance(data.data, np.ndarray) else [row[:] for row in data.data]
            if isinstance(new_data, np.ndarray):
                try:
                    new_data[:, col_idx] = pd.to_datetime(new_data[:, col_idx], errors='coerce')
                except Exception:
                    for i in range(new_data.shape[0]):
                        new_data[i, col_idx] = pd.to_datetime(new_data[i, col_idx], errors='coerce')
            else:
                for row in new_data:
                    row[col_idx] = pd.to_datetime(row[col_idx], errors='coerce')
            return DataBatch(data=new_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        else:
            return data

    def _calculate_epoch_time(self, dt_series: np.ndarray) -> np.ndarray:
        """
        Calculate epoch time for datetime series.
        
        Parameters
        ----------
        dt_series : np.ndarray
            Array of datetime values.
            
        Returns
        -------
        np.ndarray
            Array of epoch times in the specified unit.
        """
        if not isinstance(dt_series, pd.Series):
            pd_series = pd.Series(dt_series)
        else:
            pd_series = dt_series
        if not pd.api.types.is_datetime64_any_dtype(pd_series):
            pd_series = pd.to_datetime(pd_series, errors='coerce')
        if self.unit == 'seconds':
            return (pd_series.astype('int64') // 10 ** 9).values
        elif self.unit == 'milliseconds':
            return (pd_series.astype('int64') // 10 ** 6).values
        elif self.unit == 'minutes':
            return (pd_series.astype('int64') // (10 ** 9 * 60)).values
        elif self.unit == 'hours':
            return (pd_series.astype('int64') // (10 ** 9 * 3600)).values
        elif self.unit == 'days':
            return (pd_series.astype('int64') // (10 ** 9 * 86400)).values

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'EpochCalculator':
        """
        Fit the transformer to the input data.
        
        For this transformer, fitting simply identifies the datetime column if not specified.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime columns.
        **kwargs : dict
            Additional parameters (not used).
            
        Returns
        -------
        EpochCalculator
            Self instance for method chaining.
        """
        self._validate_unit()
        if self.datetime_column is None:
            self.datetime_column = self._detect_datetime_column(data)
        return self

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Transform the input data by calculating time since epoch.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime columns.
        **kwargs : dict
            Additional parameters (not used).
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Transformed data with time since epoch features.
        """
        if not hasattr(self, 'datetime_column') or self.datetime_column is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, np.ndarray):
            if self.datetime_column is None:
                raise ValueError('datetime_column must be specified for numpy arrays')
            try:
                col_idx = int(self.datetime_column)
            except (ValueError, TypeError):
                raise ValueError('For numpy arrays, datetime_column must be an integer index')
            dt_column = data[:, col_idx]
            dt_column_converted = pd.to_datetime(dt_column, errors='coerce')
            epoch_times = self._calculate_epoch_time(dt_column_converted)
            epoch_column = epoch_times.reshape(-1, 1)
            if self.drop_original:
                new_data = np.delete(data, col_idx, axis=1)
                new_data = np.hstack([new_data, epoch_column])
            else:
                new_data = np.hstack([data, epoch_column])
            return new_data
        converted_data = self._convert_to_datetime(data)
        if isinstance(converted_data, DataBatch):
            if converted_data.feature_names is None:
                raise ValueError('DataBatch must have feature_names')
            col_idx = converted_data.feature_names.index(self.datetime_column)
            if isinstance(converted_data.data, np.ndarray):
                dt_column = converted_data.data[:, col_idx]
            else:
                dt_column = np.array([row[col_idx] for row in converted_data.data])
            epoch_times = self._calculate_epoch_time(dt_column)
            new_feature_name = f'{self.datetime_column}_epoch_{self.unit}'
            if isinstance(converted_data.data, np.ndarray):
                epoch_column = epoch_times.reshape(-1, 1)
                new_data = np.hstack([converted_data.data, epoch_column])
                new_feature_names = converted_data.feature_names + [new_feature_name]
            else:
                new_data = [row + [epoch_times[i]] for (i, row) in enumerate(converted_data.data)]
                new_feature_names = converted_data.feature_names + [new_feature_name]
            if self.drop_original:
                orig_idx = new_feature_names.index(self.datetime_column)
                if isinstance(new_data, np.ndarray):
                    new_data = np.delete(new_data, orig_idx, axis=1)
                else:
                    new_data = [row[:orig_idx] + row[orig_idx + 1:] for row in new_data]
                new_feature_names = [name for name in new_feature_names if name != self.datetime_column]
            return DataBatch(data=new_data, labels=converted_data.labels, metadata=converted_data.metadata, sample_ids=converted_data.sample_ids, feature_names=new_feature_names, batch_id=converted_data.batch_id)
        else:
            return converted_data

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Inverse transform is not supported for this transformer as the original datetime
        information is lost after conversion to epoch time.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Transformed data.
        **kwargs : dict
            Additional parameters (not used).
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Original data (raises NotImplementedError).
        """
        raise NotImplementedError('Inverse transformation is not supported for epoch time conversion.')