from typing import Union, Optional, Callable, Dict
import numpy as np
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

def rolling_window_aggregation(data: Union[DataBatch, FeatureSet, np.ndarray], window_size: int, aggregation_func: Union[str, Callable]='mean', step_size: int=1, axis: Optional[int]=None, min_periods: Optional[int]=None, custom_aggregations: Optional[Dict[str, Callable]]=None) -> Union[np.ndarray, DataBatch, FeatureSet]:
    """
    Compute rolling window aggregations over input data.

    This function applies a specified aggregation function over a sliding window
    across the input data. It supports various data formats and allows customization
    of window parameters and aggregation methods.

    Args:
        data (Union[DataBatch, FeatureSet, np.ndarray]): Input data to aggregate.
            Can be a DataBatch, FeatureSet, or numpy array.
        window_size (int): Size of the rolling window. Must be positive.
        aggregation_func (Union[str, Callable]): Aggregation function to apply.
            Can be a string ('mean', 'sum', 'min', 'max', 'std', 'var', 'median')
            or a custom callable function that takes an array and returns a scalar.
        step_size (int): Step size for window movement. Defaults to 1.
        axis (Optional[int]): Axis along which to apply the rolling window.
            If None, defaults to the last axis for arrays or appropriate axis for structured data.
        min_periods (Optional[int]): Minimum number of observations in window
            required to have a value (otherwise result is NaN). Defaults to window_size.
        custom_aggregations (Optional[Dict[str, Callable]]): Dictionary mapping
            custom aggregation names to functions for use with string-based func specification.

    Returns:
        Union[np.ndarray, DataBatch, FeatureSet]: Aggregated data in the same
        format as input. For arrays, returns aggregated numpy array. For structured
        data types, returns object of same type with aggregated values.

    Raises:
        ValueError: If window_size is not positive, or if incompatible parameters are provided.
        TypeError: If data type is unsupported or aggregation_func is invalid.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError('window_size must be a positive integer')
    if min_periods is None:
        min_periods = window_size
    elif not isinstance(min_periods, int) or min_periods <= 0:
        raise ValueError('min_periods must be a positive integer if specified')
    if not isinstance(step_size, int) or step_size <= 0:
        raise ValueError('step_size must be a positive integer')
    builtin_aggregations = {'mean': np.mean, 'sum': np.sum, 'min': np.min, 'max': np.max, 'std': lambda x: np.std(x, ddof=1) if len(x) > 1 else np.nan, 'var': lambda x: np.var(x, ddof=1) if len(x) > 1 else np.nan, 'median': np.median}
    if isinstance(aggregation_func, str):
        if custom_aggregations and aggregation_func in custom_aggregations:
            agg_func = custom_aggregations[aggregation_func]
        elif aggregation_func in builtin_aggregations:
            agg_func = builtin_aggregations[aggregation_func]
        else:
            raise ValueError(f'Unknown aggregation function: {aggregation_func}')
    elif callable(aggregation_func):
        agg_func = aggregation_func
    else:
        raise TypeError('aggregation_func must be a string or callable')
    if isinstance(data, DataBatch):
        return _process_databatch(data, window_size, agg_func, step_size, axis, min_periods)
    elif isinstance(data, FeatureSet):
        return _process_featureset(data, window_size, agg_func, step_size, axis, min_periods)
    elif isinstance(data, np.ndarray):
        return _process_array(data, window_size, agg_func, step_size, axis, min_periods)
    else:
        raise TypeError(f'Unsupported data type: {type(data)}. Expected DataBatch, FeatureSet, or numpy array.')