from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
from typing import List, Optional, Union
import numpy as np
import pandas as pd

class CyclicalTimeEncoder(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, cyclical_components: Optional[List[str]]=None, periods: Optional[dict]=None, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the CyclicalTimeEncoder.
        
        Parameters
        ----------
        datetime_column : Optional[str], default=None
            Name of the column containing datetime information. If None, assumes
            the input data has datetime features already extracted.
        cyclical_components : Optional[List[str]], default=None
            List of cyclical components to encode. Common options include:
            'second', 'minute', 'hour', 'day_of_week', 'day_of_month', 'day_of_year',
            'week_of_year', 'month', 'quarter'. If None, defaults to ['hour', 'day_of_week', 'month'].
        periods : Optional[dict], default=None
            Custom periods for cyclical components. If None, uses default periods:
            {
                'second': 60, 'minute': 60, 'hour': 24, 'day_of_week': 7,
                'day_of_month': 31, 'day_of_year': 365, 'week_of_year': 52,
                'month': 12, 'quarter': 4
            }
        drop_original : bool, default=True
            Whether to drop the original datetime column after encoding
        name : Optional[str], default=None
            Name of the transformer
        """
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.cyclical_components = cyclical_components or ['hour', 'day_of_week', 'month']
        self.periods = periods or {'second': 60, 'minute': 60, 'hour': 24, 'day_of_week': 7, 'day_of_month': 31, 'day_of_year': 365, 'week_of_year': 52, 'month': 12, 'quarter': 4}
        self.drop_original = drop_original
        self._fitted = False

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'CyclicalTimeEncoder':
        """
        Fit the encoder to the input data.
        
        This method validates the input data and prepares the encoder for transformation.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime information
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        CyclicalTimeEncoder
            Self instance for method chaining
        """
        if not isinstance(data, (DataBatch, np.ndarray)):
            raise TypeError('Input data must be either DataBatch or numpy array')
        if isinstance(data, DataBatch):
            if self.datetime_column and data.feature_names and (self.datetime_column not in data.feature_names):
                raise ValueError(f"Datetime column '{self.datetime_column}' not found in DataBatch")
        supported_components = {'second', 'minute', 'hour', 'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'month', 'quarter'}
        invalid_components = set(self.cyclical_components) - supported_components
        if invalid_components:
            raise ValueError(f'Unsupported cyclical components: {invalid_components}')
        for component in self.cyclical_components:
            if component not in self.periods:
                raise ValueError(f"Period not defined for component '{component}'")
        self._fitted = True
        return self

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Apply cyclical encoding to the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Transformed data with cyclical features encoded
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise RuntimeError('Transformer must be fitted before transform')
        if isinstance(data, DataBatch):
            return self._transform_databatch(data)
        elif isinstance(data, np.ndarray):
            return self._transform_array(data)
        else:
            raise TypeError('Input data must be either DataBatch or numpy array')

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Apply the inverse transformation to recover original features.
        
        Note: Since sine and cosine transformations lose some information,
        the inverse transformation may not perfectly recover original values.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Transformed data to inverse transform
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data with cyclical encodings converted back to approximate original features
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise RuntimeError('Transformer must be fitted before inverse_transform')
        if isinstance(data, DataBatch):
            return self._inverse_transform_databatch(data)
        elif isinstance(data, np.ndarray):
            return self._inverse_transform_array(data)
        else:
            raise TypeError('Input data must be either DataBatch or numpy array')

    def _transform_databatch(self, data: DataBatch) -> DataBatch:
        """Transform DataBatch input."""
        if isinstance(data.data, list):
            X = np.array(data.data)
        else:
            X = data.data.copy()
        feature_names = data.feature_names or []
        if self.datetime_column:
            if self.datetime_column not in feature_names:
                raise ValueError(f"Datetime column '{self.datetime_column}' not found")
            dt_col_idx = feature_names.index(self.datetime_column)
            dt_values = X[:, dt_col_idx]
            encoded_features = self._encode_cyclical_features(dt_values)
            if self.drop_original:
                X = np.delete(X, dt_col_idx, axis=1)
                new_feature_names = [name for name in feature_names if name != self.datetime_column]
            else:
                new_feature_names = feature_names.copy()
            X = np.hstack([X, encoded_features])
            new_feature_names.extend(self._get_encoded_feature_names())
        else:
            encoded_features = self._encode_from_components(X, feature_names)
            X = np.hstack([X, encoded_features]) if X.size > 0 else encoded_features
            new_feature_names = feature_names + self._get_encoded_feature_names()
        return DataBatch(data=X, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=new_feature_names, batch_id=data.batch_id)

    def _transform_array(self, data: np.ndarray) -> np.ndarray:
        """Transform numpy array input."""
        if self.datetime_column:
            raise ValueError('Cannot specify datetime_column when transforming numpy arrays directly')
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        feature_names = [f'component_{i}' for i in range(data.shape[1])] if data.shape[1] > 0 else []
        encoded_features = self._encode_from_components(data, feature_names)
        return np.hstack([data, encoded_features]) if data.size > 0 else encoded_features

    def _encode_cyclical_features(self, dt_values) -> np.ndarray:
        """Encode datetime values into cyclical features."""
        if isinstance(dt_values, pd.Series):
            dt_series = pd.to_datetime(dt_values)
        elif isinstance(dt_values, np.ndarray):
            if dt_values.ndim > 1:
                dt_values = dt_values.flatten()
            dt_series = pd.to_datetime(pd.Series(dt_values))
        elif isinstance(dt_values, list):
            dt_series = pd.to_datetime(pd.Series(dt_values))
        else:
            dt_series = pd.to_datetime(pd.Series([dt_values]))
        encoded_features = []
        for component in self.cyclical_components:
            if component == 'second':
                values = dt_series.dt.second
            elif component == 'minute':
                values = dt_series.dt.minute
            elif component == 'hour':
                values = dt_series.dt.hour
            elif component == 'day_of_week':
                values = dt_series.dt.dayofweek
            elif component == 'day_of_month':
                values = dt_series.dt.day
            elif component == 'day_of_year':
                values = dt_series.dt.dayofyear
            elif component == 'week_of_year':
                values = dt_series.dt.isocalendar().week
            elif component == 'month':
                values = dt_series.dt.month
            elif component == 'quarter':
                values = dt_series.dt.quarter
            else:
                raise ValueError(f'Unsupported component: {component}')
            period = self.periods[component]
            nan_mask = values.isna()
            values_filled = values.fillna(0)
            sin_vals = np.sin(2 * np.pi * values_filled / period)
            cos_vals = np.cos(2 * np.pi * values_filled / period)
            sin_vals = np.where(nan_mask, np.nan, sin_vals)
            cos_vals = np.where(nan_mask, np.nan, cos_vals)
            encoded_features.append(sin_vals)
            encoded_features.append(cos_vals)
        return np.column_stack(encoded_features)

    def _encode_from_components(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Encode from pre-extracted datetime components."""
        encoded_features = []
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        for component in self.cyclical_components:
            if component in feature_names:
                col_idx = feature_names.index(component)
                values = data[:, col_idx].astype(float)
            elif len(feature_names) == 0 or (len(feature_names) == data.shape[1] and all((name.startswith('component_') for name in feature_names))):
                try:
                    col_idx = self.cyclical_components.index(component)
                    if col_idx < data.shape[1]:
                        values = data[:, col_idx].astype(float)
                    else:
                        raise ValueError(f"Component '{component}' not found in input data")
                except (ValueError, IndexError):
                    raise ValueError(f"Component '{component}' not found in input data")
            else:
                raise ValueError(f"Component '{component}' not found in input data")
            period = self.periods[component]
            nan_mask = np.isnan(values)
            values_filled = np.where(nan_mask, 0, values)
            sin_vals = np.sin(2 * np.pi * values_filled / period)
            cos_vals = np.cos(2 * np.pi * values_filled / period)
            sin_vals = np.where(nan_mask, np.nan, sin_vals)
            cos_vals = np.where(nan_mask, np.nan, cos_vals)
            encoded_features.append(sin_vals)
            encoded_features.append(cos_vals)
        return np.column_stack(encoded_features) if encoded_features else np.array([]).reshape(data.shape[0], 0)

    def _get_encoded_feature_names(self) -> List[str]:
        """Generate names for encoded features."""
        names = []
        for component in self.cyclical_components:
            names.append(f'{component}_sin')
            names.append(f'{component}_cos')
        return names

    def _inverse_transform_databatch(self, data: DataBatch) -> DataBatch:
        """Inverse transform DataBatch input."""
        if isinstance(data.data, list):
            X = np.array(data.data)
        else:
            X = data.data.copy()
        feature_names = data.feature_names or []
        encoded_feature_names = self._get_encoded_feature_names()
        encoded_indices = []
        remaining_features = []
        remaining_feature_names = []
        for (i, name) in enumerate(feature_names):
            if name in encoded_feature_names:
                encoded_indices.append(i)
            else:
                remaining_features.append(X[:, i])
                remaining_feature_names.append(name)
        if remaining_features:
            remaining_X = np.column_stack(remaining_features)
        else:
            remaining_X = np.array([]).reshape(X.shape[0], 0)
        reconstructed_features = []
        reconstructed_names = []
        for component in self.cyclical_components:
            sin_name = f'{component}_sin'
            cos_name = f'{component}_cos'
            if sin_name in feature_names and cos_name in feature_names:
                sin_idx = feature_names.index(sin_name)
                cos_idx = feature_names.index(cos_name)
                sin_vals = X[:, sin_idx]
                cos_vals = X[:, cos_idx]
                period = self.periods[component]
                reconstructed = period * (np.arctan2(sin_vals, cos_vals) + np.pi) / (2 * np.pi)
                reconstructed_features.append(reconstructed)
                reconstructed_names.append(component)
        if reconstructed_features:
            reconstructed_X = np.column_stack(reconstructed_features)
            new_X = np.hstack([remaining_X, reconstructed_X]) if remaining_X.size > 0 else reconstructed_X
            new_feature_names = remaining_feature_names + reconstructed_names
        else:
            new_X = remaining_X
            new_feature_names = remaining_feature_names
        return DataBatch(data=new_X, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=new_feature_names, batch_id=data.batch_id)

    def _inverse_transform_array(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform numpy array input."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        num_encoded_cols = len(self.cyclical_components) * 2
        if data.shape[1] < num_encoded_cols:
            raise ValueError('Not enough columns for inverse transformation')
        remaining_X = data[:, :-num_encoded_cols] if data.shape[1] > num_encoded_cols else np.array([]).reshape(data.shape[0], 0)
        reconstructed_features = []
        encoded_data = data[:, -num_encoded_cols:]
        for (i, component) in enumerate(self.cyclical_components):
            sin_idx = i * 2
            cos_idx = i * 2 + 1
            sin_vals = encoded_data[:, sin_idx]
            cos_vals = encoded_data[:, cos_idx]
            period = self.periods[component]
            reconstructed = period * (np.arctan2(sin_vals, cos_vals) + np.pi) / (2 * np.pi)
            reconstructed_features.append(reconstructed)
        if reconstructed_features:
            reconstructed_X = np.column_stack(reconstructed_features)
            return np.hstack([remaining_X, reconstructed_X]) if remaining_X.size > 0 else reconstructed_X
        else:
            return remaining_X