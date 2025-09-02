from typing import Union, Optional, List, Dict, Any
from general.structures import DataBatch, FeatureSet
from general.base_classes import BaseTransformer
import numpy as np


# ...(code omitted)...


class LagFeatureGenerator(BaseTransformer):

    def __init__(self, lag_periods: Union[int, List[int]]=[1, 2, 3], feature_columns: Optional[List[str]]=None, drop_na: bool=True, name: Optional[str]=None):
        """
        Initialize the LagFeatureGenerator.

        Args:
            lag_periods (Union[int, List[int]]): Periods to lag features by.
                Can be a single integer or list of integers.
            feature_columns (List[str], optional): Columns to create lags for.
                If None, lags are created for all numeric columns.
            drop_na (bool): Remove rows with NaN values after lag creation.
            name (str, optional): Name of the transformer instance.
        """
        super().__init__(name=name)
        self.lag_periods = lag_periods if isinstance(lag_periods, list) else [lag_periods]
        self.feature_columns = feature_columns
        self.drop_na = drop_na

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'LagFeatureGenerator':
        """
        Fit the transformer to the input data.

        Validates the presence of required columns and sets up internal state
        for lag feature generation.

        Args:
            data (Union[DataBatch, FeatureSet]): Input time series data.
            **kwargs: Additional parameters (ignored).

        Returns:
            LagFeatureGenerator: Returns self for method chaining.
        """
        if not isinstance(data, (DataBatch, FeatureSet)):
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if isinstance(data, DataBatch):
            feature_names = data.feature_names
            features = data.data if isinstance(data.data, np.ndarray) else np.array(data.data)
        else:
            feature_names = data.feature_names
            features = data.features
        if self.feature_columns is not None:
            if feature_names is None:
                raise ValueError('feature_names must be provided in data when feature_columns is specified')
            missing_cols = set(self.feature_columns) - set(feature_names)
            if missing_cols:
                raise ValueError(f'Specified feature columns not found in data: {missing_cols}')
        elif feature_names is None:
            self.feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
        elif isinstance(data, FeatureSet) and data.feature_types is not None:
            numeric_indices = [i for (i, ftype) in enumerate(data.feature_types) if ftype in ['numeric', 'integer', 'float']]
            self.feature_columns = [feature_names[i] for i in numeric_indices]
        elif isinstance(data, DataBatch) and data.metadata.get('feature_types'):
            feature_types = data.metadata['feature_types']
            numeric_indices = [i for (i, ftype) in enumerate(feature_types) if ftype in ['numeric', 'integer', 'float']]
            self.feature_columns = [feature_names[i] for i in numeric_indices]
        else:
            self.feature_columns = feature_names.copy()
        if not self.lag_periods:
            raise ValueError('lag_periods cannot be empty')
        for period in self.lag_periods:
            if not isinstance(period, int) or period <= 0:
                raise ValueError('All lag periods must be positive integers')
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Generate lag features from the input data.

        Creates lagged versions of specified features for configured periods.

        Args:
            data (Union[DataBatch, FeatureSet]): Input time series data.
            **kwargs: Additional parameters (ignored).

        Returns:
            FeatureSet: New FeatureSet with lag features added.
        """
        if not isinstance(data, (DataBatch, FeatureSet)):
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if isinstance(data, DataBatch):
            feature_names = data.feature_names
            features = data.data if isinstance(data.data, np.ndarray) else np.array(data.data)
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        else:
            feature_names = data.feature_names
            features = data.features
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(feature_names)
            if missing_cols:
                raise ValueError(f'Specified feature columns not found in data: {missing_cols}')
        if self.feature_columns is None:
            column_indices = list(range(len(feature_names)))
            lag_columns = feature_names
        else:
            column_indices = [feature_names.index(col) for col in self.feature_columns]
            lag_columns = self.feature_columns
        (n_samples, n_features) = features.shape
        lag_features_list = []
        lag_feature_names = []
        lag_features_list.append(features)
        lag_feature_names.extend(feature_names)
        for period in self.lag_periods:
            for (col_idx, col_name) in zip(column_indices, lag_columns):
                lagged_col = np.full(n_samples, np.nan)
                if period < n_samples:
                    lagged_col[period:] = features[:-period, col_idx]
                lag_features_list.append(lagged_col.reshape(-1, 1))
                lag_feature_names.append(f'{col_name}_lag_{period}')
        new_features = np.hstack(lag_features_list)
        if self.drop_na:
            valid_rows = ~np.isnan(new_features).any(axis=1)
            new_features = new_features[valid_rows]
            if sample_ids is not None:
                sample_ids = [sample_ids[i] for i in range(len(sample_ids)) if valid_rows[i]]
        else:
            nan_rows = np.isnan(new_features).any(axis=1)
            metadata['nan_rows'] = nan_rows
        new_feature_names = lag_feature_names
        if isinstance(data, FeatureSet) and data.feature_types is not None:
            new_feature_types = data.feature_types.copy()
            for period in self.lag_periods:
                for col_name in lag_columns:
                    new_feature_types.append(data.feature_types[feature_names.index(col_name)])
        elif isinstance(data, DataBatch) and data.metadata.get('feature_types'):
            new_feature_types = data.metadata['feature_types'].copy()
            feature_types = data.metadata['feature_types']
            for period in self.lag_periods:
                for col_name in lag_columns:
                    new_feature_types.append(feature_types[feature_names.index(col_name)])
        else:
            new_feature_types = None
        return FeatureSet(features=new_features, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for this transformer.

        Args:
            data (Union[FeatureSet, DataBatch]): Transformed data.
            **kwargs: Additional parameters (ignored).

        Returns:
            FeatureSet: Original data without lag features.

        Raises:
            NotImplementedError: Always raised as inverse transform is not supported.
        """
        raise NotImplementedError('Inverse transform not supported for lag features')


# ...(code omitted)...


class RollingWindowStatistics(BaseTransformer):

    def __init__(self, window_size: int=5, statistics: Optional[List[str]]=None, feature_columns: Optional[List[str]]=None, center: bool=False, name: Optional[str]=None):
        """
        Initialize the RollingWindowStatistics transformer.

        Args:
            window_size (int): Size of the rolling window.
            statistics (List[str], optional): Statistics to compute.
                Options include 'mean', 'std', 'min', 'max', 'median', 'quantile'.
                If None, defaults to ['mean', 'std'].
            feature_columns (List[str], optional): Columns to compute statistics for.
                If None, statistics are computed for all numeric columns.
            center (bool): Center the window around the current observation.
            name (str, optional): Name of the transformer instance.
        """
        super().__init__(name=name)
        if window_size <= 0:
            raise ValueError('window_size must be positive')
        self.window_size = window_size
        self.statistics = statistics or ['mean', 'std']
        self.feature_columns = feature_columns
        self.center = center

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'RollingWindowStatistics':
        """
        Fit the transformer to the input data.

        Validates the presence of required columns and sets up internal state
        for rolling window statistic computation.

        Args:
            data (Union[DataBatch, FeatureSet]): Input time series data.
            **kwargs: Additional parameters (ignored).

        Returns:
            RollingWindowStatistics: Returns self for method chaining.
        """
        if self.window_size <= 0:
            raise ValueError('window_size must be positive')
        if isinstance(data, DataBatch):
            if data.feature_names is not None:
                available_columns = data.feature_names
            elif isinstance(data.data, np.ndarray) and data.data.ndim == 2:
                available_columns = [f'col_{i}' for i in range(data.data.shape[1])]
            else:
                raise ValueError('Cannot infer feature names from DataBatch data')
        elif isinstance(data, FeatureSet):
            if data.feature_names is not None:
                available_columns = data.feature_names
            else:
                available_columns = [f'col_{i}' for i in range(data.features.shape[1])]
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(available_columns)
            if missing_cols:
                raise ValueError(f'Specified feature columns not found in data: {missing_cols}')
        else:
            if isinstance(data, DataBatch):
                if isinstance(data.data, np.ndarray):
                    numeric_columns = available_columns
                else:
                    raise ValueError('DataBatch data must be numpy array for automatic column selection')
            else:
                numeric_columns = available_columns
            self.feature_columns = numeric_columns
        valid_stats = {'mean', 'std', 'min', 'max', 'median', 'quantile'}
        invalid_stats = set(self.statistics) - valid_stats
        if invalid_stats:
            raise ValueError(f'Invalid statistics requested: {invalid_stats}')
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Compute rolling window statistics from the input data.

        Calculates specified statistics over rolling windows for selected features.

        Args:
            data (Union[DataBatch, FeatureSet]): Input time series data.
            **kwargs: Additional parameters (ignored).

        Returns:
            FeatureSet: New FeatureSet with rolling window statistics added.
        """
        pass

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for this transformer.

        Args:
            data (Union[FeatureSet, DataBatch]): Transformed data.
            **kwargs: Additional parameters (ignored).

        Returns:
            FeatureSet: Original data without rolling window statistics.

        Raises:
            NotImplementedError: Always raised as inverse transform is not supported.
        """
        raise NotImplementedError('Inverse transform not supported for rolling window statistics')