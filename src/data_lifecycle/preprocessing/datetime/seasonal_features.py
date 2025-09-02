import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from scipy.signal import periodogram
import pandas as pd

class SeasonalDecompositionExtractor(BaseTransformer):

    def __init__(self, datetime_column: Optional[str]=None, period: Optional[int]=None, model: str='additive', extract_features: Optional[List[str]]=None, drop_original: bool=True, name: Optional[str]=None):
        """
        Initialize the SeasonalDecompositionExtractor.
        
        Parameters
        ----------
        datetime_column : Optional[str], default=None
            Name of the datetime column to use for decomposition. If None, uses index.
        period : Optional[int], default=None
            Period of the seasonal component. If None, inferred from data frequency.
        model : str, default='additive'
            Type of seasonal decomposition ('additive' or 'multiplicative').
        extract_features : Optional[List[str]], default=None
            Components to extract. Options: 'trend', 'seasonal', 'residual', 'all'.
            If None, extracts all components.
        drop_original : bool, default=True
            Whether to remove the original datetime column after transformation.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.datetime_column = datetime_column
        self.period = period
        if model not in ['additive', 'multiplicative']:
            raise ValueError("model must be either 'additive' or 'multiplicative'")
        self.model = model
        if extract_features is None:
            self.extract_features = None
        else:
            valid_features = ['trend', 'seasonal', 'residual', 'all']
            if not isinstance(extract_features, list):
                raise ValueError('extract_features must be a list or None')
            if not all((f in valid_features for f in extract_features)):
                raise ValueError(f'extract_features must contain only {valid_features}')
            self.extract_features = extract_features
        self.drop_original = drop_original

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'SeasonalDecompositionExtractor':
        """
        Fit the seasonal decomposition extractor to the input data.
        
        This method analyzes the datetime information and determines the appropriate
        seasonal decomposition parameters.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data containing datetime information and time series values
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        SeasonalDecompositionExtractor
            Self instance for method chaining
        """
        if self.model not in ['additive', 'multiplicative']:
            raise ValueError("model must be either 'additive' or 'multiplicative'")
        valid_features = ['trend', 'seasonal', 'residual', 'all']
        if self.extract_features is not None:
            if not all((f in valid_features for f in self.extract_features)):
                raise ValueError(f'extract_features must contain only {valid_features}')
        if isinstance(data, DataBatch):
            if data.data is not None and data.data.ndim == 2:
                ts_data = data.data
            else:
                raise ValueError('DataBatch must contain 2D data array')
            if self.datetime_column:
                if hasattr(data, 'feature_names') and data.feature_names and (self.datetime_column in data.feature_names):
                    dt_col_idx = data.feature_names.index(self.datetime_column)
                    dt_series = ts_data[:, dt_col_idx]
                else:
                    raise ValueError(f"Datetime column '{self.datetime_column}' not found in DataBatch")
            else:
                dt_series = np.arange(len(ts_data))
        else:
            if data.ndim == 1:
                ts_data = data.reshape(-1, 1)
            elif data.ndim == 2:
                ts_data = data
            else:
                raise ValueError('Input numpy array must be 1D or 2D')
            if self.datetime_column:
                raise ValueError('datetime_column cannot be specified for numpy array input')
            dt_series = np.arange(len(ts_data))
        if not isinstance(dt_series, pd.Series):
            dt_series = pd.Series(dt_series)
        if self.period is None:
            if isinstance(dt_series.iloc[0], (pd.Timestamp, np.datetime64)):
                inferred_freq = pd.infer_freq(dt_series)
                if inferred_freq is None:
                    raise ValueError('Could not infer data frequency. Please specify period manually.')
                freq_map = {'Y': 1, 'A': 1, 'Q': 4, 'M': 12, 'W': 52, 'D': 365, 'H': 24}
                self.period = freq_map.get(inferred_freq[0], None)
                if self.period is None:
                    raise ValueError(f"Unsupported frequency '{inferred_freq}'. Please specify period manually.")
            else:
                n_samples = len(ts_data)
                if n_samples >= 24:
                    try:
                        if isinstance(data, DataBatch) and self.datetime_column:
                            ts_columns = [i for i in range(ts_data.shape[1]) if i != dt_col_idx]
                            if ts_columns:
                                ts_values = ts_data[:, ts_columns[0]]
                            else:
                                ts_values = ts_data[:, 0]
                        else:
                            ts_values = ts_data[:, 0]
                        detrended = ts_values - np.mean(ts_values)
                        (frequencies, power) = periodogram(detrended)
                        if len(power) > 10:
                            peak_idx = np.argmax(power[1:min(50, len(power) // 2)]) + 1
                            dominant_freq = frequencies[peak_idx]
                            if dominant_freq > 0:
                                inferred_period = int(round(1 / dominant_freq))
                                if 2 <= inferred_period <= min(n_samples // 2, 100):
                                    self.period = inferred_period
                        if self.period is None:
                            if n_samples >= 365:
                                self.period = 365
                            elif n_samples >= 180:
                                self.period = 180
                            elif n_samples >= 100:
                                self.period = 52
                            elif n_samples >= 50:
                                self.period = 24
                            elif n_samples >= 24:
                                self.period = 12
                    except Exception:
                        pass
                if self.period is None:
                    if n_samples >= 12:
                        self.period = 12
                    else:
                        self.period = max(2, n_samples // 2)
        self._fitted = True
        self._n_samples = ts_data.shape[0]
        self._ts_shape = ts_data.shape
        return self

    def _calculate_trend(self, ts_data: np.ndarray) -> np.ndarray:
        """
        Calculate trend component using moving average.
        
        Parameters
        ----------
        ts_data : np.ndarray
            Time series data
            
        Returns
        -------
        np.ndarray
            Trend component
        """
        if self.period is None:
            window_size = min(12, len(ts_data) // 2)
            if window_size % 2 == 0:
                window_size += 1
        else:
            window_size = self.period
        pad_width = window_size // 2
        padded_data = np.pad(ts_data, (pad_width, pad_width), mode='edge')
        trend = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
        if len(trend) > len(ts_data):
            trend = trend[:len(ts_data)]
        elif len(trend) < len(ts_data):
            trend = np.pad(trend, (0, len(ts_data) - len(trend)), mode='edge')
        return trend.copy()

    def _calculate_seasonal_additive(self, ts_data: np.ndarray, trend: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate seasonal component for additive model.
        
        Parameters
        ----------
        ts_data : np.ndarray
            Original time series data
        trend : np.ndarray
            Trend component
        period : int
            Seasonal period
            
        Returns
        -------
        np.ndarray
            Seasonal component
        """
        detrended = ts_data - trend
        seasonal = np.zeros_like(ts_data)
        for i in range(period):
            indices = np.arange(i, len(detrended), period)
            if len(indices) > 0:
                seasonal_mean = np.mean(detrended[indices])
                seasonal[indices] = seasonal_mean
        seasonal_adjustment = np.mean(seasonal)
        seasonal = seasonal - seasonal_adjustment
        return seasonal.copy()

    def _calculate_seasonal_multiplicative(self, ts_data: np.ndarray, trend_safe: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate seasonal component for multiplicative model.
        
        Parameters
        ----------
        ts_data : np.ndarray
            Original time series data
        trend_safe : np.ndarray
            Trend component (with zeros replaced by small values)
        period : int
            Seasonal period
            
        Returns
        -------
        np.ndarray
            Seasonal component
        """
        detrended = ts_data / trend_safe
        seasonal = np.zeros_like(ts_data)
        for i in range(period):
            indices = np.arange(i, len(detrended), period)
            if len(indices) > 0:
                seasonal_mean = np.mean(detrended[indices])
                seasonal[indices] = seasonal_mean
        seasonal_mean = np.mean(seasonal)
        if seasonal_mean != 0:
            seasonal = seasonal / seasonal_mean
        return seasonal.copy()

    def transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Apply seasonal decomposition feature extraction to the input data.
        
        Extracts trend, seasonal, and residual components from the time series data
        based on the fitted parameters.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Transformed data with seasonal decomposition features
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise ValueError('Transformer must be fitted before transform can be called')
        effective_extract_features = self.extract_features
        if effective_extract_features is None:
            effective_extract_features = ['all']
        if len(effective_extract_features) == 0 or (len(effective_extract_features) == 1 and effective_extract_features[0] == ''):
            if isinstance(data, DataBatch):
                result_data = np.empty((data.data.shape[0], 0))
                new_feature_names = []
                return DataBatch(data=result_data.copy(), labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=new_feature_names, batch_id=data.batch_id)
            else:
                return np.empty((data.shape[0], 0)).copy()
        if isinstance(data, DataBatch):
            original_data = data.data
            original_feature_names = data.feature_names or []
        else:
            if data.ndim == 1:
                original_data = data.reshape(-1, 1)
            elif data.ndim == 2:
                original_data = data
            else:
                raise ValueError('Input numpy array must be 1D or 2D')
            original_feature_names = []
        n_samples = original_data.shape[0]
        if isinstance(data, DataBatch) and self.datetime_column:
            if self.datetime_column in original_feature_names:
                dt_col_idx = original_feature_names.index(self.datetime_column)
                ts_columns = [i for i in range(original_data.shape[1]) if i != dt_col_idx]
            else:
                raise ValueError(f"Datetime column '{self.datetime_column}' not found in DataBatch")
        else:
            ts_columns = list(range(original_data.shape[1]))
        all_components = []
        component_names = []
        for col_idx in ts_columns:
            ts_data = original_data[:, col_idx]
            if self.model == 'additive':
                trend = self._calculate_trend(ts_data)
                if self.period and self.period < len(ts_data):
                    seasonal = self._calculate_seasonal_additive(ts_data, trend, self.period)
                else:
                    seasonal = np.zeros_like(ts_data)
                residual = ts_data - trend - seasonal
            else:
                trend = self._calculate_trend(ts_data)
                trend_safe = np.where(trend == 0, 1e-10, trend)
                if self.period and self.period < len(ts_data):
                    seasonal = self._calculate_seasonal_multiplicative(ts_data, trend_safe, self.period)
                else:
                    seasonal = np.ones_like(ts_data)
                seasonal_safe = np.where(seasonal == 0, 1e-10, seasonal)
                residual = ts_data / (trend_safe * seasonal_safe)
            components_to_extract = effective_extract_features
            if 'all' in components_to_extract:
                components_to_extract = ['trend', 'seasonal', 'residual']
            for component in components_to_extract:
                if component == 'trend':
                    all_components.append(trend)
                    if len(ts_columns) > 1:
                        component_names.append(f"{(original_feature_names[col_idx] if col_idx < len(original_feature_names) else f'col_{col_idx}')}_trend")
                    else:
                        component_names.append('trend')
                elif component == 'seasonal':
                    all_components.append(seasonal)
                    if len(ts_columns) > 1:
                        component_names.append(f"{(original_feature_names[col_idx] if col_idx < len(original_feature_names) else f'col_{col_idx}')}_seasonal")
                    else:
                        component_names.append('seasonal')
                elif component == 'residual':
                    all_components.append(residual)
                    if len(ts_columns) > 1:
                        component_names.append(f"{(original_feature_names[col_idx] if col_idx < len(original_feature_names) else f'col_{col_idx}')}_residual")
                    else:
                        component_names.append('residual')
        if all_components:
            result_array = np.column_stack(all_components)
        else:
            result_array = np.empty((n_samples, 0))
            component_names = []
        if isinstance(data, DataBatch):
            new_feature_names = component_names[:]
            if not self.drop_original and original_feature_names:
                if self.datetime_column and self.datetime_column in original_feature_names:
                    remaining_indices = [i for (i, name) in enumerate(original_feature_names) if name != self.datetime_column]
                else:
                    remaining_indices = list(range(len(original_feature_names)))
                if remaining_indices:
                    remaining_data = original_data[:, remaining_indices]
                    result_array = np.column_stack([result_array, remaining_data])
                    new_feature_names.extend([original_feature_names[i] for i in remaining_indices])
            return DataBatch(data=result_array.copy(), labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=new_feature_names, batch_id=data.batch_id)
        else:
            return result_array.copy()

    def inverse_transform(self, data: Union[DataBatch, np.ndarray], **kwargs) -> Union[DataBatch, np.ndarray]:
        """
        Reverse the seasonal decomposition transformation if possible.
        
        Note: Complete inversion may not be possible depending on what components
        were extracted and how the data was processed.
        
        Parameters
        ----------
        data : Union[DataBatch, np.ndarray]
            Transformed data to invert
        **kwargs : dict
            Additional inversion parameters
            
        Returns
        -------
        Union[DataBatch, np.ndarray]
            Data in original format (to the extent possible)
        """
        if not hasattr(self, '_fitted') or not self._fitted:
            raise ValueError('Transformer must be fitted before inverse_transform can be called')
        if isinstance(data, DataBatch):
            transformed_data = data.data
            original_labels = data.labels
            original_metadata = data.metadata.copy() if data.metadata else {}
            original_sample_ids = data.sample_ids
            original_batch_id = data.batch_id
            original_feature_names = data.feature_names[:] if data.feature_names else []
        elif data.ndim == 1:
            transformed_data = data.reshape(-1, 1)
        else:
            transformed_data = data
        if transformed_data.shape[1] == 3:
            trend = transformed_data[:, 0]
            seasonal = transformed_data[:, 1]
            residual = transformed_data[:, 2]
            if self.model == 'additive':
                reconstructed = trend + seasonal + residual
            else:
                reconstructed = trend * seasonal * residual
            reconstructed_data = reconstructed.reshape(-1, 1)
        elif transformed_data.shape[1] % 3 == 0 and transformed_data.shape[1] > 0:
            n_time_series = transformed_data.shape[1] // 3
            reconstructed_columns = []
            for i in range(n_time_series):
                start_idx = i * 3
                trend = transformed_data[:, start_idx]
                seasonal = transformed_data[:, start_idx + 1]
                residual = transformed_data[:, start_idx + 2]
                if self.model == 'additive':
                    reconstructed = trend + seasonal + residual
                else:
                    reconstructed = trend * seasonal * residual
                reconstructed_columns.append(reconstructed)
            if reconstructed_columns:
                reconstructed_data = np.column_stack(reconstructed_columns)
            else:
                reconstructed_data = np.empty((transformed_data.shape[0], 0))
        else:
            reconstructed_data = transformed_data
        if isinstance(data, DataBatch):
            feature_names = [f'reconstructed_col_{i}' for i in range(reconstructed_data.shape[1])] if reconstructed_data.shape[1] > 0 else []
            return DataBatch(data=reconstructed_data.copy(), labels=original_labels, metadata=original_metadata, sample_ids=original_sample_ids, feature_names=feature_names, batch_id=original_batch_id)
        else:
            return reconstructed_data.copy()