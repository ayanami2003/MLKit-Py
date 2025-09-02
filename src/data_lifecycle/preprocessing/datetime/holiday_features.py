from typing import List, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import holidays
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch


# ...(code omitted)...


class HolidayFeatureCreator(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, country_codes: Optional[List[str]]=None, custom_dates: Optional[List[str]]=None, include_weekends: bool=False, feature_prefix: str='holiday_created', drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the HolidayFeatureCreator.
        
        Parameters
        ----------
        datetime_column : Optional[str], default=None
            Name of the datetime column to process. If None, attempts to detect datetime columns.
        country_codes : Optional[List[str]], default=None
            ISO country codes for which to create holiday features (e.g., ['US', 'CA']).
        custom_dates : Optional[List[str]], default=None
            List of custom holiday dates in 'YYYY-MM-DD' format.
        include_weekends : bool, default=False
            Whether to include weekends as part of holiday features.
        feature_prefix : str, default="holiday_created"
            Prefix for created feature names.
        drop_original : bool, default=True
            Whether to remove the original datetime column after feature creation.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.country_codes = country_codes or []
        self.custom_dates = custom_dates or []
        self.include_weekends = include_weekends
        self.feature_prefix = feature_prefix
        self.drop_original = drop_original

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'HolidayFeatureCreator':
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime information.
        **kwargs : dict
            Additional parameters (ignored).

        Returns
        -------
        HolidayFeatureCreator
            Self instance for method chaining.
        """
        is_databatch = isinstance(data, DataBatch)
        if is_databatch:
            df = data.data.copy() if hasattr(data.data, 'copy') else pd.DataFrame(data.data)
            self._stored_meta = data.metadata
        else:
            df = data.copy() if hasattr(data, 'copy') else pd.DataFrame(data)
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        if self.datetime_column is None:
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                else:
                    try:
                        pd.to_datetime(df[col].iloc[:min(5, len(df))])
                        datetime_cols.append(col)
                    except:
                        pass
            if len(datetime_cols) == 1:
                self.datetime_column = datetime_cols[0]
            elif len(datetime_cols) > 1:
                raise ValueError(f'Multiple datetime columns detected: {datetime_cols}. Please specify datetime_column explicitly.')
            else:
                raise ValueError('No datetime column detected. Please specify datetime_column explicitly.')
        elif isinstance(self.datetime_column, int):
            if self.datetime_column < len(df.columns):
                self.datetime_column = df.columns[self.datetime_column]
            else:
                raise ValueError(f"Specified datetime column index '{self.datetime_column}' out of range.")
        elif self.datetime_column not in df.columns:
            raise ValueError(f"Specified datetime column '{self.datetime_column}' not found in data.")
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        self._original_columns = list(df.columns)
        if self.drop_original:
            self._original_datetime_data = df[self.datetime_column].copy()
        self._years = sorted(df[self.datetime_column].dt.year.unique())
        self._holiday_calendars = {}
        for country in self.country_codes:
            try:
                country_holidays = holidays.country_holidays(country, years=self._years)
                self._holiday_calendars[country] = country_holidays
            except Exception as e:
                raise ValueError(f"Failed to create holiday calendar for country '{country}': {str(e)}")
        self._custom_date_set = set()
        for date_str in self.custom_dates:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                self._custom_date_set.add(date_obj)
            except ValueError:
                raise ValueError(f'Invalid date format in custom_dates: {date_str}. Expected format: YYYY-MM-DD')
        return self

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Generate holiday features from datetime data.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime information.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with added holiday features.
        """
        is_databatch = isinstance(data, DataBatch)
        if is_databatch:
            df = data.data.copy() if hasattr(data.data, 'copy') else pd.DataFrame(data.data)
            meta = data.metadata
        else:
            df = data.copy() if hasattr(data, 'copy') else pd.DataFrame(data)
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
        dates = df[self.datetime_column].dt.date
        for country in self.country_codes:
            feature_name = f'{self.feature_prefix}_{country}_holiday'
            df[feature_name] = dates.isin(self._holiday_calendars[country]).astype(int)
        if self.custom_dates:
            feature_name = f'{self.feature_prefix}_custom'
            df[feature_name] = dates.isin(self._custom_date_set).astype(int)
        if self.include_weekends:
            feature_name = f'{self.feature_prefix}_weekend'
            df[feature_name] = df[self.datetime_column].dt.weekday.isin([5, 6]).astype(int)
        if self.drop_original:
            df = df.drop(columns=[self.datetime_column])
        if is_databatch:
            return DataBatch(data=df, metadata=meta)
        else:
            return df

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Reverse the holiday feature generation transformation.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Data with holiday features to be removed.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with holiday features removed.
        """
        is_databatch = isinstance(data, DataBatch)
        if is_databatch:
            df = data.data.copy() if hasattr(data.data, 'copy') else pd.DataFrame(data.data)
            meta = data.metadata
        else:
            df = data.copy() if hasattr(data, 'copy') else pd.DataFrame(data)
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        columns_to_remove = []
        for col in df.columns:
            if col.startswith(self.feature_prefix):
                columns_to_remove.append(col)
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
        if self.drop_original and hasattr(self, '_original_datetime_data'):
            df[self.datetime_column] = self._original_datetime_data.reset_index(drop=True)
            original_cols_without_datetime = [col for col in self._original_columns if col != self.datetime_column]
            new_column_order = original_cols_without_datetime + [self.datetime_column]
            final_column_order = [col for col in new_column_order if col in df.columns or col == self.datetime_column]
            if final_column_order:
                df = df[final_column_order]
        if is_databatch:
            return DataBatch(data=df, metadata=meta)
        else:
            return df