from typing import Union, Optional, List
from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
import numpy as np

class TimestampExtractor(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, extract_features: Optional[List[str]]=None, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the TimestampExtractor.
        
        Parameters
        ----------
        datetime_column : str, optional
            Name of the column containing datetime information. If None, the transformer
            will attempt to automatically detect datetime columns.
        extract_features : list of str, optional
            List of feature names to extract. If None, extracts common features.
            Possible values include:
            ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 
             'dayofyear', 'quarter', 'week', 'is_month_start', 'is_month_end',
             'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']
        drop_original : bool, default=True
            Whether to drop the original datetime column after feature extraction.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.extract_features = extract_features or ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 'dayofyear', 'quarter', 'week', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']
        self.drop_original = drop_original

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'TimestampExtractor':
        """
        Fit the transformer to the input data.
        
        This method identifies the datetime column if not specified and prepares
        for feature extraction.
        
        Parameters
        ----------
        data : DataBatch or numpy.ndarray
            Input data containing datetime information.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        TimestampExtractor
            Self instance for method chaining.
        """
        supported_features = ['year', 'month', 'day', 'hour', 'minute', 'second', 'weekday', 'dayofyear', 'quarter', 'week', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']
        if self.extract_features:
            invalid_features = [f for f in self.extract_features if f not in supported_features]
            if invalid_features:
                raise ValueError(f'Unsupported features: {invalid_features}. Supported features are: {supported_features}')
        if isinstance(data, DataBatch):
            if data.feature_names is None:
                if self.datetime_column is None:
                    raise ValueError('DataBatch must contain feature_names for TimestampExtractor when auto-detection is needed')
                else:
                    self._is_fitted = True
                    return self
            df_data = data.data
            feature_names = data.feature_names
            if self.datetime_column is None:
                datetime_cols = []
                for (i, col) in enumerate(feature_names):
                    if isinstance(df_data, np.ndarray):
                        col_data = df_data[:, i]
                    else:
                        col_data = df_data[col] if hasattr(df_data, '__getitem__') else None
                    if col_data is not None:
                        try:
                            import pandas as pd
                            sample_size = min(100, len(col_data)) if hasattr(col_data, '__len__') else 1
                            sample_data = col_data[:sample_size] if hasattr(col_data, '__getitem__') else col_data
                            pd_series = pd.Series(sample_data)
                            converted = pd.to_datetime(pd_series, errors='coerce')
                            non_null_ratio = converted.notna().sum() / len(converted) if len(converted) > 0 else 0
                            if non_null_ratio > 0.95 and len(converted) > 1 and (pd.api.types.is_datetime64_any_dtype(converted) or non_null_ratio > 0.99):
                                datetime_cols.append(col)
                        except (ValueError, TypeError):
                            continue
                if len(datetime_cols) == 1:
                    self.datetime_column = datetime_cols[0]
                elif len(datetime_cols) > 1:
                    likely_datetime_cols = []
                    for col in datetime_cols:
                        col_idx = feature_names.index(col)
                        if isinstance(df_data, np.ndarray):
                            col_data = df_data[:, col_idx]
                        else:
                            col_data = df_data[col]
                        try:
                            import pandas as pd
                            pd_series = pd.Series(col_data)
                            converted = pd.to_datetime(pd_series, errors='coerce')
                            non_null_ratio = converted.notna().sum() / len(converted) if len(converted) > 0 else 0
                            if pd.api.types.is_datetime64_any_dtype(converted) or non_null_ratio > 0.99:
                                likely_datetime_cols.append(col)
                        except:
                            continue
                    if len(likely_datetime_cols) == 1:
                        self.datetime_column = likely_datetime_cols[0]
                    else:
                        raise ValueError(f'Multiple datetime columns detected: {datetime_cols}. Please specify datetime_column.')
                else:
                    raise ValueError('No datetime column detected. Please specify datetime_column.')
            self._is_fitted = True
            return self
        elif self.datetime_column is None:
            raise ValueError('datetime_column must be specified when using numpy arrays')
        self._is_fitted = True
        return self

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Extract timestamp features from the input data.
        
        Parameters
        ----------
        data : DataBatch or numpy.ndarray
            Input data containing datetime information.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        DataBatch or numpy.ndarray
            Transformed data with extracted timestamp features.
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        import pandas as pd
        if isinstance(data, DataBatch):
            df_data = data.data
            feature_names = data.feature_names
            if isinstance(df_data, np.ndarray):
                if feature_names is None:
                    raise ValueError('DataBatch with numpy array must contain feature_names')
                df = pd.DataFrame(df_data, columns=feature_names)
            else:
                df = pd.DataFrame(df_data)
                if feature_names is not None:
                    df.columns = feature_names
            if self.datetime_column not in df.columns:
                raise ValueError(f"Specified datetime column '{self.datetime_column}' not found in data")
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])
            dt_series = df[self.datetime_column]
            new_columns = {}
            for feature in self.extract_features:
                if feature == 'year':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.year
                elif feature == 'month':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.month
                elif feature == 'day':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.day
                elif feature == 'hour':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.hour
                elif feature == 'minute':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.minute
                elif feature == 'second':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.second
                elif feature == 'weekday':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.weekday
                elif feature == 'dayofyear':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.dayofyear
                elif feature == 'quarter':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.quarter
                elif feature == 'week':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.isocalendar().week
                elif feature == 'is_month_start':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.is_month_start.astype(int)
                elif feature == 'is_month_end':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.is_month_end.astype(int)
                elif feature == 'is_quarter_start':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.is_quarter_start.astype(int)
                elif feature == 'is_quarter_end':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.is_quarter_end.astype(int)
                elif feature == 'is_year_start':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.is_year_start.astype(int)
                elif feature == 'is_year_end':
                    new_columns[f'{self.datetime_column}_{feature}'] = dt_series.dt.is_year_end.astype(int)
            for (col_name, col_data) in new_columns.items():
                df[col_name] = col_data
            if self.drop_original:
                df = df.drop(columns=[self.datetime_column])
            new_feature_names = list(df.columns)
            return DataBatch(data=df.values, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=new_feature_names, batch_id=data.batch_id)
        else:
            if self.datetime_column is None:
                raise ValueError('datetime_column must be specified when using numpy arrays')
            if isinstance(self.datetime_column, int):
                datetime_col_idx = self.datetime_column
            elif isinstance(self.datetime_column, str) and self.datetime_column.isdigit():
                datetime_col_idx = int(self.datetime_column)
            else:
                raise ValueError('When using numpy arrays, datetime_column must be an integer index or string representation of an integer')
            datetime_data = pd.to_datetime(data[:, datetime_col_idx])
            if not isinstance(datetime_data, pd.Series):
                datetime_data = pd.Series(datetime_data)
            extracted_features = []
            feature_names = []
            for feature in self.extract_features:
                if feature == 'year':
                    extracted_features.append(datetime_data.dt.year.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'month':
                    extracted_features.append(datetime_data.dt.month.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'day':
                    extracted_features.append(datetime_data.dt.day.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'hour':
                    extracted_features.append(datetime_data.dt.hour.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'minute':
                    extracted_features.append(datetime_data.dt.minute.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'second':
                    extracted_features.append(datetime_data.dt.second.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'weekday':
                    extracted_features.append(datetime_data.dt.weekday.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'dayofyear':
                    extracted_features.append(datetime_data.dt.dayofyear.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'quarter':
                    extracted_features.append(datetime_data.dt.quarter.values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'week':
                    week_values = datetime_data.dt.isocalendar().week.values
                    extracted_features.append(week_values)
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'is_month_start':
                    extracted_features.append(datetime_data.dt.is_month_start.values.astype(int))
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'is_month_end':
                    extracted_features.append(datetime_data.dt.is_month_end.values.astype(int))
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'is_quarter_start':
                    extracted_features.append(datetime_data.dt.is_quarter_start.values.astype(int))
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'is_quarter_end':
                    extracted_features.append(datetime_data.dt.is_quarter_end.values.astype(int))
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'is_year_start':
                    extracted_features.append(datetime_data.dt.is_year_start.values.astype(int))
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
                elif feature == 'is_year_end':
                    extracted_features.append(datetime_data.dt.is_year_end.values.astype(int))
                    feature_names.append(f'{self.datetime_column}_{feature}' if isinstance(self.datetime_column, str) and (not self.datetime_column.isdigit()) else f'datetime_{feature}')
            extracted_features = np.column_stack(extracted_features)
            if self.drop_original:
                remaining_indices = [i for i in range(data.shape[1]) if i != datetime_col_idx]
                if remaining_indices:
                    remaining_data = data[:, remaining_indices]
                    result = np.column_stack([remaining_data, extracted_features])
                else:
                    result = extracted_features
            else:
                result = np.column_stack([data, extracted_features])
            return result

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Inverse transform is not supported for this transformer.
        
        Parameters
        ----------
        data : DataBatch or numpy.ndarray
            Transformed data.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        DataBatch or numpy.ndarray
            Original data (identity operation).
        """
        return data