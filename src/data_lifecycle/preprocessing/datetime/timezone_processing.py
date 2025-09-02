from typing import Optional, List, Union
from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
import numpy as np

class TimezoneAwareDatetimeProcessor(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, source_tz: Optional[str]=None, target_tz: Optional[str]=None, extract_features: Optional[List[str]]=None, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the TimezoneAwareDatetimeProcessor.
        
        Parameters
        ----------
        datetime_column : Optional[str]
            Name of the column containing datetime values
        source_tz : Optional[str]
            Source timezone for naive datetime conversion
        target_tz : Optional[str]
            Target timezone for datetime conversion
        extract_features : Optional[List[str]]
            List of timezone-related features to extract
        drop_original : bool
            Whether to drop the original datetime column after processing
        name : Optional[str]
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.source_tz = source_tz
        self.target_tz = target_tz
        self.extract_features = extract_features or []
        self.drop_original = drop_original

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'TimezoneAwareDatetimeProcessor':
        """
        Fit the transformer to the input data.
        
        This method identifies datetime columns if not explicitly specified
        and validates timezone parameters.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime features
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        TimezoneAwareDatetimeProcessor
            Self instance for method chaining
        """
        import pandas as pd
        import pytz
        self._input_type = type(data).__name__
        if isinstance(data, DataBatch):
            if data.feature_names is not None:
                df = pd.DataFrame(data.data, columns=data.feature_names)
            else:
                df = pd.DataFrame(data.data)
        else:
            df = pd.DataFrame(data)
        if self.datetime_column is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or (df[col].dtype == object and pd.to_datetime(df[col], errors='coerce').notna().all()):
                    self.datetime_column = col
                    break
            if self.datetime_column is None:
                raise ValueError('No datetime column found. Please specify datetime_column parameter.')
        elif self.datetime_column not in df.columns:
            raise ValueError(f"Specified datetime_column '{self.datetime_column}' not found in data.")
        if self.source_tz is not None:
            try:
                pytz.timezone(self.source_tz)
            except Exception as e:
                raise ValueError(f"Invalid source_tz '{self.source_tz}': {str(e)}")
        if self.target_tz is not None:
            try:
                pytz.timezone(self.target_tz)
            except Exception as e:
                raise ValueError(f"Invalid target_tz '{self.target_tz}': {str(e)}")
        self._feature_names = df.columns.tolist() if hasattr(data, 'feature_names') and data.feature_names else None
        if self.datetime_column in df.columns:
            dt_series = pd.to_datetime(df[self.datetime_column])
            self._original_tz = dt_series.dt.tz
        self._original_columns = df.columns.tolist()
        if self.datetime_column in df.columns:
            self._original_datetime_series = pd.to_datetime(df[self.datetime_column]).copy()
        return self

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Apply timezone-aware datetime processing to input data.
        
        This method performs timezone localization and conversion, and extracts
        requested timezone-related features.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Transformed data with timezone-aware datetime features
        """
        import pandas as pd
        if isinstance(data, DataBatch):
            if data.feature_names is not None:
                df = pd.DataFrame(data.data, columns=data.feature_names)
            else:
                df = pd.DataFrame(data.data)
        else:
            df = pd.DataFrame(data)
        if self.datetime_column not in df.columns:
            raise ValueError(f"Fitted datetime_column '{self.datetime_column}' not found in transform data.")
        dt_series = pd.to_datetime(df[self.datetime_column])
        if dt_series.dt.tz is None and self.source_tz is not None:
            dt_series = dt_series.dt.tz_localize(self.source_tz)
        if self.target_tz is not None:
            dt_series = dt_series.dt.tz_convert(self.target_tz)
        result_df = df.copy()
        result_df[self.datetime_column] = dt_series
        new_features = {}
        if 'timezone_offset' in self.extract_features:
            new_features[f'{self.datetime_column}_timezone_offset'] = dt_series.apply(lambda x: x.strftime('%z') if pd.notna(x) else None)
        if 'timezone_name' in self.extract_features:
            new_features[f'{self.datetime_column}_timezone_name'] = dt_series.apply(lambda x: str(x.tz) if pd.notna(x) and x.tz else None)
        if 'is_dst' in self.extract_features:
            new_features[f'{self.datetime_column}_is_dst'] = dt_series.apply(lambda x: x.dst() != pd.Timedelta(0) if pd.notna(x) and x.tz else False)
        if 'utc_timestamp' in self.extract_features:
            new_features[f'{self.datetime_column}_utc_timestamp'] = dt_series.apply(lambda x: x.timestamp() if pd.notna(x) else None)
        for (feat_name, feat_values) in new_features.items():
            result_df[feat_name] = feat_values
        if self.drop_original:
            result_df = result_df.drop(columns=[self.datetime_column])
        if isinstance(data, DataBatch):
            return DataBatch(data=result_df.values, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=result_df.columns.tolist(), batch_id=data.batch_id)
        else:
            return result_df.values

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Apply the inverse transformation if possible.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with inverse transformation applied
        """
        import pandas as pd
        if isinstance(data, DataBatch):
            if data.feature_names is not None:
                df = pd.DataFrame(data.data, columns=data.feature_names)
            else:
                df = pd.DataFrame(data.data)
            if hasattr(self, '_original_tz') and self.datetime_column in df.columns:
                dt_series = pd.to_datetime(df[self.datetime_column])
                if self.target_tz is not None and self.source_tz is not None:
                    dt_series = dt_series.dt.tz_convert(self.source_tz)
                elif self.source_tz is not None and dt_series.dt.tz is not None:
                    dt_series = dt_series.dt.tz_localize(None)
                df[self.datetime_column] = dt_series
            extracted_features = [f'{self.datetime_column}_timezone_offset', f'{self.datetime_column}_timezone_name', f'{self.datetime_column}_is_dst', f'{self.datetime_column}_utc_timestamp']
            cols_to_drop = [col for col in extracted_features if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
            return DataBatch(data=df.values, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=df.columns.tolist(), batch_id=data.batch_id)
        elif isinstance(data, np.ndarray):
            return data
        return None