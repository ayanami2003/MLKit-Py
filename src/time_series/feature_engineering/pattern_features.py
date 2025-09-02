from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from scipy import stats


# ...(code omitted)...


class RegimeShiftDetector(BaseTransformer):

    def __init__(self, method: str='binary_segmentation', window_size: int=10, threshold: float=2.0, feature_columns: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the RegimeShiftDetector.
        
        Args:
            method (str): Detection method ('binary_segmentation', 'moving_window', 'cusum')
            window_size (int): Window size for moving window approaches
            threshold (float): Threshold for significant change detection
            feature_columns (Optional[List[str]]): Columns to analyze for regime shifts
            name (Optional[str]): Name for the transformer instance
        """
        super().__init__(name=name)
        if method not in ['binary_segmentation', 'moving_window', 'cusum']:
            raise ValueError("Method must be one of 'binary_segmentation', 'moving_window', 'cusum'")
        if window_size <= 0:
            raise ValueError('Window size must be positive')
        if threshold <= 0:
            raise ValueError('Threshold must be positive')
        self.method = method
        self.window_size = window_size
        self.threshold = threshold
        self.feature_columns = feature_columns
        self._is_fitted = False

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'RegimeShiftDetector':
        """
        Fit the regime shift detector to the input time series data.
        
        Args:
            data (Union[DataBatch, FeatureSet]): Input time series data
            **kwargs: Additional fitting parameters
            
        Returns:
            RegimeShiftDetector: Self instance for method chaining
        """
        if isinstance(data, DataBatch):
            feature_matrix = data.data
            if hasattr(data, 'feature_names') and data.feature_names is not None:
                feature_names = data.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])] if feature_matrix.ndim > 1 else ['feature_0']
        elif isinstance(data, FeatureSet):
            feature_matrix = data.features
            feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(feature_matrix.shape[1])]
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if not isinstance(feature_matrix, np.ndarray):
            feature_matrix = np.array(feature_matrix)
        if self.feature_columns is None:
            numeric_cols = []
            for (i, name) in enumerate(feature_names):
                if feature_matrix.ndim == 1:
                    if np.issubdtype(feature_matrix.dtype, np.number):
                        numeric_cols.append(0)
                    break
                elif np.issubdtype(feature_matrix[:, i].dtype, np.number):
                    numeric_cols.append(i)
            self._analyzed_columns = numeric_cols
            self._analyzed_column_names = [feature_names[i] for i in numeric_cols] if numeric_cols else []
        else:
            col_indices = []
            for col_name in self.feature_columns:
                if col_name in feature_names:
                    col_indices.append(feature_names.index(col_name))
                else:
                    raise ValueError(f"Column '{col_name}' not found in data")
            self._analyzed_columns = col_indices
            self._analyzed_column_names = self.feature_columns
        self._baseline_stats = {}
        for (i, col_idx) in enumerate(self._analyzed_columns):
            if feature_matrix.ndim == 1:
                col_data = feature_matrix
            else:
                col_data = feature_matrix[:, col_idx]
            self._baseline_stats[self._analyzed_column_names[i]] = {'mean': np.mean(col_data), 'std': np.std(col_data), 'var': np.var(col_data)}
        self._is_fitted = True
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Detect regime shifts and generate regime switching indicators.
        
        Args:
            data (Union[DataBatch, FeatureSet]): Input time series data
            **kwargs: Additional transformation parameters
            
        Returns:
            FeatureSet: Data with added regime shift indicators
        """
        if not self._is_fitted:
            raise ValueError('Transformer must be fitted before transform')
        if isinstance(data, DataBatch):
            feature_matrix = data.data
            if hasattr(data, 'feature_names') and data.feature_names is not None:
                feature_names = data.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])] if feature_matrix.ndim > 1 else ['feature_0']
            sample_ids = getattr(data, 'sample_ids', [f'sample_{i}' for i in range(feature_matrix.shape[0])])
        elif isinstance(data, FeatureSet):
            feature_matrix = data.features
            feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(feature_matrix.shape[1])]
            sample_ids = data.sample_ids
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if not isinstance(feature_matrix, np.ndarray):
            feature_matrix = np.array(feature_matrix)
        regime_indicators = []
        indicator_names = []
        for (i, col_idx) in enumerate(self._analyzed_columns):
            col_name = self._analyzed_column_names[i]
            if feature_matrix.ndim == 1:
                col_data = feature_matrix
            else:
                col_data = feature_matrix[:, col_idx]
            if self.method == 'binary_segmentation':
                shifts = self._binary_segmentation_detection(col_data, col_name)
            elif self.method == 'moving_window':
                shifts = self._moving_window_detection(col_data, col_name)
            elif self.method == 'cusum':
                shifts = self._cusum_detection(col_data, col_name)
            else:
                raise ValueError(f'Unsupported method: {self.method}')
            regime_indicators.append(shifts)
            indicator_names.append(f'{col_name}_regime_shift')
        regime_indicators = np.column_stack(regime_indicators) if len(regime_indicators) > 0 else np.array([]).reshape(feature_matrix.shape[0], 0)
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(-1, 1)
        if regime_indicators.size > 0:
            combined_features = np.column_stack([feature_matrix, regime_indicators])
        else:
            combined_features = feature_matrix
        combined_names = feature_names + indicator_names
        return FeatureSet(features=combined_features, feature_names=combined_names, sample_ids=sample_ids)

    def _binary_segmentation_detection(self, data: np.ndarray, col_name: str) -> np.ndarray:
        """
        Detect regime shifts using binary segmentation approach.
        
        Args:
            data (np.ndarray): Time series data for a single column
            col_name (str): Name of the column
            
        Returns:
            np.ndarray: Binary indicators for regime shifts
        """
        n = len(data)
        indicators = np.zeros(n)
        baseline_mean = self._baseline_stats[col_name]['mean']
        baseline_std = self._baseline_stats[col_name]['std']
        if baseline_std == 0:
            return indicators
        min_seg_size = max(5, self.window_size)

        def find_changepoints(start, end, depth=0, max_depth=3):
            if end - start < 2 * min_seg_size or depth >= max_depth:
                return []
            best_stat = 0
            best_idx = -1
            for i in range(start + min_seg_size, end - min_seg_size):
                left_data = data[start:i]
                right_data = data[i:end]
                if len(left_data) > 1 and len(right_data) > 1:
                    left_mean = np.mean(left_data)
                    right_mean = np.mean(right_data)
                    stat = abs(left_mean - right_mean) / (baseline_std + 1e-08) * np.sqrt(len(left_data) * len(right_data) / (len(left_data) + len(right_data) + 1e-08))
                    if stat > best_stat and stat > self.threshold * 0.5:
                        best_stat = stat
                        best_idx = i
            if best_idx != -1:
                changepoints = [best_idx]
                changepoints.extend(find_changepoints(start, best_idx, depth + 1, max_depth))
                changepoints.extend(find_changepoints(best_idx, end, depth + 1, max_depth))
                return changepoints
            else:
                return []
        changepoints = find_changepoints(0, n)
        for cp in changepoints:
            if 0 <= cp < n:
                indicators[cp] = 1
        return indicators

    def _moving_window_detection(self, data: np.ndarray, col_name: str) -> np.ndarray:
        """
        Detect regime shifts using moving window approach.
        
        Args:
            data (np.ndarray): Time series data for a single column
            col_name (str): Name of the column
            
        Returns:
            np.ndarray: Binary indicators for regime shifts
        """
        n = len(data)
        indicators = np.zeros(n)
        baseline_mean = self._baseline_stats[col_name]['mean']
        baseline_std = self._baseline_stats[col_name]['std']
        if baseline_std == 0 or n < 2 * self.window_size:
            return indicators
        for i in range(self.window_size, n - self.window_size):
            left_window = data[i - self.window_size:i]
            right_window = data[i:i + self.window_size]
            left_mean = np.mean(left_window)
            right_mean = np.mean(right_window)
            diff = abs(left_mean - right_mean)
            if diff > baseline_std * self.threshold * 0.5:
                indicators[i] = 1
        return indicators

    def _cusum_detection(self, data: np.ndarray, col_name: str) -> np.ndarray:
        """
        Detect regime shifts using CUSUM approach.
        
        Args:
            data (np.ndarray): Time series data for a single column
            col_name (str): Name of the column
            
        Returns:
            np.ndarray: Binary indicators for regime shifts
        """
        n = len(data)
        indicators = np.zeros(n)
        baseline_mean = self._baseline_stats[col_name]['mean']
        baseline_std = self._baseline_stats[col_name]['std']
        if baseline_std == 0:
            return indicators
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        k = 0.25 * self.threshold
        for i in range(1, n):
            deviation = (data[i] - baseline_mean) / (baseline_std + 1e-08)
            cusum_pos[i] = max(0, cusum_pos[i - 1] + deviation - k)
            cusum_neg[i] = min(0, cusum_neg[i - 1] + deviation + k)
            if cusum_pos[i] > self.threshold or abs(cusum_neg[i]) > self.threshold:
                indicators[i] = 1
                cusum_pos[i] = 0
                cusum_neg[i] = 0
        return indicators

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for regime shift detection.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data
            **kwargs: Additional parameters
            
        Raises:
            NotImplementedError: Always raised as inverse transform is not applicable
        """
        raise NotImplementedError('Inverse transform is not supported for regime shift detection')