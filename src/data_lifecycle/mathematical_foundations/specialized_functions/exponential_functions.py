from typing import Union, Optional
import numpy as np
from general.structures.data_batch import DataBatch

def apply_exponential_smoothing(data: Union[np.ndarray, DataBatch], alpha: float=0.3, beta: Optional[float]=None, gamma: Optional[float]=None, seasonal_periods: Optional[int]=None, initial_level: Optional[float]=None, initial_trend: Optional[float]=None, initial_seasonal: Optional[np.ndarray]=None, forecast_steps: int=0) -> Union[np.ndarray, DataBatch]:
    """
    Apply exponential smoothing to time series data.

    This function applies exponential smoothing to smooth time series data and optionally 
    generate forecasts. It supports simple, double (Holt's), and triple (Holt-Winters) 
    exponential smoothing methods based on the parameters provided.

    Args:
        data (Union[np.ndarray, DataBatch]): Input time series data to smooth. If DataBatch,
            the primary data attribute is used.
        alpha (float): Smoothing parameter for the level (0 < alpha < 1). 
            Higher values give more weight to recent observations.
        beta (Optional[float]): Smoothing parameter for the trend (0 < beta < 1).
            If provided, enables double exponential smoothing.
        gamma (Optional[float]): Smoothing parameter for seasonality (0 < gamma < 1).
            If provided, enables triple exponential smoothing.
        seasonal_periods (Optional[int]): Number of periods in a seasonal cycle.
            Required when gamma is provided.
        initial_level (Optional[float]): Initial value for the level component.
            If None, automatically computed from the data.
        initial_trend (Optional[float]): Initial value for the trend component.
            If None, automatically computed from the data.
        initial_seasonal (Optional[np.ndarray]): Initial values for seasonal components.
            Length should match seasonal_periods. If None, automatically computed.
        forecast_steps (int): Number of future time steps to forecast.
            If 0, only smoothed historical values are returned.

    Returns:
        Union[np.ndarray, DataBatch]: Smoothed time series data with the same type as input.
            If forecast_steps > 0, the result includes both smoothed historical values
            and forecasted future values.

    Raises:
        ValueError: If parameters are inconsistent (e.g., gamma provided without seasonal_periods)
            or if data is not one-dimensional.
    """
    if not 0 < alpha < 1:
        raise ValueError('Alpha must be between 0 and 1')
    if beta is not None and (not 0 < beta < 1):
        raise ValueError('Beta must be between 0 and 1')
    if gamma is not None and (not 0 < gamma < 1):
        raise ValueError('Gamma must be between 0 and 1')
    if gamma is not None and seasonal_periods is None:
        raise ValueError('seasonal_periods must be provided when gamma is specified')
    if seasonal_periods is not None and seasonal_periods <= 0:
        raise ValueError('seasonal_periods must be a positive integer')
    is_databatch = isinstance(data, DataBatch)
    if is_databatch:
        original_data = data.data
    else:
        original_data = data
    if not isinstance(original_data, np.ndarray):
        original_data = np.array(original_data)
    if original_data.ndim != 1:
        raise ValueError('Input data must be one-dimensional')
    n = len(original_data)
    if initial_level is None:
        initial_level = original_data[0]
    level = initial_level
    trend = 0.0
    if beta is not None and initial_trend is None:
        if n >= 2:
            initial_trend = (original_data[-1] - original_data[0]) / (n - 1)
        else:
            initial_trend = 0.0
    if beta is not None:
        trend = initial_trend
    seasonal = None
    if gamma is not None:
        if initial_seasonal is None:
            initial_seasonal = np.zeros(seasonal_periods)
            if n >= seasonal_periods:
                avg = np.mean(original_data[:seasonal_periods])
                for i in range(seasonal_periods):
                    initial_seasonal[i] = original_data[i] - avg
        seasonal = initial_seasonal.copy()
    method = 'simple'
    if beta is not None:
        method = 'double'
    if gamma is not None:
        method = 'triple'
    total_length = n + forecast_steps
    smoothed = np.zeros(total_length)
    for i in range(total_length):
        if i < n:
            observation = original_data[i]
        else:
            observation = np.nan
        if method == 'simple':
            if i == 0:
                smoothed_value = observation
            else:
                smoothed_value = alpha * observation + (1 - alpha) * level
            level = smoothed_value
            smoothed[i] = smoothed_value
        elif method == 'double':
            if i == 0:
                smoothed_value = observation
                level = observation
            else:
                smoothed_value = alpha * observation + (1 - alpha) * (level + trend)
                trend_new = beta * (smoothed_value - level) + (1 - beta) * trend
                level = smoothed_value
                trend = trend_new
            smoothed[i] = smoothed_value
        elif method == 'triple':
            seasonal_index = i % seasonal_periods
            if i == 0:
                smoothed_value = observation
                level = observation - seasonal[seasonal_index]
                smoothed[i] = observation
            elif i < n:
                smoothed_value = alpha * (observation - seasonal[seasonal_index]) + (1 - alpha) * (level + trend)
                trend_new = beta * (smoothed_value - level) + (1 - beta) * trend
                seasonal_new = gamma * (observation - smoothed_value) + (1 - gamma) * seasonal[seasonal_index]
                level = smoothed_value
                trend = trend_new
                seasonal[seasonal_index] = seasonal_new
                smoothed[i] = smoothed_value + seasonal[seasonal_index]
            else:
                smoothed_value = level + trend
                smoothed[i] = smoothed_value + seasonal[seasonal_index]
                level = smoothed_value
                trend = trend
    if is_databatch:
        result_data = DataBatch(data=smoothed, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        return result_data
    else:
        return smoothed