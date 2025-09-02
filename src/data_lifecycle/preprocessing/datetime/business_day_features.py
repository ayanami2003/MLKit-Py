from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
import numpy as np
import pandas as pd

class BusinessDayFeatureGenerator(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, feature_types: List[str]=None, holidays: Optional[List[str]]=None, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the BusinessDayFeatureGenerator.
        
        Parameters
        ----------
        datetime_column : Optional[str], default=None
            Name of the datetime column to process. If None, will attempt to detect.
        feature_types : List[str], default=None
            Types of business day features to generate. Defaults to all available features.
        holidays : Optional[List[str]], default=None
            List of holiday dates in 'YYYY-MM-DD' format to exclude from business days
        drop_original : bool, default=True
            Whether to remove the original datetime column after feature generation
        name : Optional[str], default=None
            Name for the transformer instance
        """
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.feature_types = feature_types or ['is_business_day', 'is_week_start', 'is_week_end', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end', 'days_to_week_start', 'days_from_week_end']
        self.holidays = holidays or []
        self.drop_original = drop_original
        self._fitted = False
        self._original_column_data = None

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'BusinessDayFeatureGenerator':
        """
        Fit the transformer to the input data.
        
        This method identifies the datetime column if not specified and prepares
        internal state for transformation.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime information
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        BusinessDayFeatureGenerator
            Self instance for method chaining
        """
        if isinstance(data, np.ndarray):
            data_batch = DataBatch(data=data)
        else:
            data_batch = data
        if self.datetime_column is None:
            if hasattr(data_batch, 'feature_names') and data_batch.feature_names:
                datetime_indicators = ['date', 'datetime', 'time', 'timestamp']
                for name in data_batch.feature_names:
                    if any((indicator in name.lower() for indicator in datetime_indicators)):
                        self.datetime_column = name
                        break
            if self.datetime_column is None:
                raise ValueError('No datetime column specified and none could be detected automatically')
        elif hasattr(data_batch, 'feature_names') and data_batch.feature_names:
            if self.datetime_column not in data_batch.feature_names:
                raise ValueError(f"Specified datetime column '{self.datetime_column}' not found in data")
        self._holiday_dates = set()
        if self.holidays:
            for holiday_str in self.holidays:
                try:
                    holiday_date = pd.to_datetime(holiday_str).date()
                    self._holiday_dates.add(holiday_date)
                except Exception:
                    raise ValueError(f"Invalid holiday date format: {holiday_str}. Expected 'YYYY-MM-DD'")
        self._fitted = True
        return self

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Generate business day features from datetime data.
        
        Adds new columns to the data based on the specified feature_types, representing
        various business day characteristics of the datetime column.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data with datetime column
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with added business day features
            
        Raises
        ------
        ValueError
            If datetime column is not found or specified
        """
        if not self._fitted:
            raise ValueError('Transformer must be fitted before transform')
        if isinstance(data, DataBatch):
            data_batch = data
            is_databatch = True
        else:
            data_batch = DataBatch(data=data)
            is_databatch = False
        if isinstance(data_batch.data, np.ndarray):
            if hasattr(data_batch, 'feature_names') and data_batch.feature_names:
                df = pd.DataFrame(data_batch.data, columns=data_batch.feature_names)
            else:
                df = pd.DataFrame(data_batch.data)
        else:
            df = data_batch.data.copy()
        if self.datetime_column is None or self.datetime_column not in df.columns:
            datetime_indicators = ['date', 'datetime', 'time', 'timestamp']
            for col in df.columns:
                col_name = str(col) if not isinstance(col, str) else col
                if any((indicator in col_name.lower() for indicator in datetime_indicators)):
                    self.datetime_column = col_name
                    break
        if self.datetime_column not in df.columns:
            raise ValueError(f"Datetime column '{self.datetime_column}' not found")
        if self.drop_original:
            self._original_column_data = df[self.datetime_column].copy()
        datetime_series = pd.to_datetime(df[self.datetime_column])
        feature_prefix = f'{self.datetime_column}_'
        for feature_type in self.feature_types:
            if feature_type == 'is_business_day':
                is_weekday = datetime_series.dt.weekday < 5
                is_holiday = datetime_series.dt.date.isin(self._holiday_dates)
                feature_values = is_weekday & ~is_holiday
                df[f'{feature_prefix}is_business_day'] = feature_values.astype(int)
            elif feature_type == 'is_week_start':
                feature_values = datetime_series.dt.weekday == 0
                df[f'{feature_prefix}is_week_start'] = feature_values.astype(int)
            elif feature_type == 'is_week_end':
                feature_values = datetime_series.dt.weekday == 4
                df[f'{feature_prefix}is_week_end'] = feature_values.astype(int)
            elif feature_type == 'is_month_start':
                feature_values = datetime_series.dt.is_month_start
                df[f'{feature_prefix}is_month_start'] = feature_values.astype(int)
            elif feature_type == 'is_month_end':
                feature_values = datetime_series.dt.is_month_end
                df[f'{feature_prefix}is_month_end'] = feature_values.astype(int)
            elif feature_type == 'is_quarter_start':
                feature_values = datetime_series.dt.is_quarter_start
                df[f'{feature_prefix}is_quarter_start'] = feature_values.astype(int)
            elif feature_type == 'is_quarter_end':
                feature_values = datetime_series.dt.is_quarter_end
                df[f'{feature_prefix}is_quarter_end'] = feature_values.astype(int)
            elif feature_type == 'is_year_start':
                feature_values = datetime_series.dt.is_year_start
                df[f'{feature_prefix}is_year_start'] = feature_values.astype(int)
            elif feature_type == 'is_year_end':
                feature_values = datetime_series.dt.is_year_end
                df[f'{feature_prefix}is_year_end'] = feature_values.astype(int)
            elif feature_type == 'days_to_week_start':
                days_to_monday = (7 - datetime_series.dt.weekday) % 7
                df[f'{feature_prefix}days_to_week_start'] = days_to_monday
            elif feature_type == 'days_from_week_end':
                days_from_friday = (datetime_series.dt.weekday - 4) % 7
                df[f'{feature_prefix}days_from_week_end'] = days_from_friday
        if self.drop_original and self.datetime_column in df.columns:
            df = df.drop(columns=[self.datetime_column])
        if is_databatch:
            if isinstance(data.data, np.ndarray):
                result_data = df.values
                result_feature_names = df.columns.tolist() if hasattr(df, 'columns') else None
                return DataBatch(data=result_data, feature_names=result_feature_names)
            else:
                data_copy = data.copy() if hasattr(data, 'copy') else DataBatch(data=df)
                data_copy.data = df
                if hasattr(data_copy, 'feature_names'):
                    data_copy.feature_names = df.columns.tolist()
                return data_copy
        elif isinstance(data, np.ndarray):
            return df.values
        else:
            return df

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Reverse the transformation by removing generated features.
        
        If drop_original was True, this will restore the original datetime column.
        Otherwise, it removes the generated business day features.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Transformed data with business day features
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with business day features removed
        """
        if not self._fitted:
            raise ValueError('Transformer must be fitted before inverse_transform')
        if isinstance(data, DataBatch):
            data_batch = data
            is_databatch = True
        else:
            data_batch = DataBatch(data=data)
            is_databatch = False
        if isinstance(data_batch.data, np.ndarray):
            if hasattr(data_batch, 'feature_names') and data_batch.feature_names:
                df = pd.DataFrame(data_batch.data, columns=data_batch.feature_names)
            else:
                df = pd.DataFrame(data_batch.data)
        else:
            df = data_batch.data.copy()
        feature_prefix = f'{self.datetime_column}_'
        columns_to_drop = [col for col in df.columns if str(col).startswith(feature_prefix)]
        df = df.drop(columns=columns_to_drop)
        if self.drop_original and self._original_column_data is not None:
            df[self.datetime_column] = self._original_column_data.reset_index(drop=True)
        if is_databatch:
            if isinstance(data.data, np.ndarray):
                result_data = df.values
                result_feature_names = df.columns.tolist() if hasattr(df, 'columns') else None
                return DataBatch(data=result_data, feature_names=result_feature_names)
            else:
                data_copy = data.copy() if hasattr(data, 'copy') else DataBatch(data=df)
                data_copy.data = df
                if hasattr(data_copy, 'feature_names'):
                    data_copy.feature_names = df.columns.tolist()
                return data_copy
        elif isinstance(data, np.ndarray):
            return df.values
        else:
            return df

    def fit_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Convenience method to fit and transform in one step.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data to fit and transform
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)