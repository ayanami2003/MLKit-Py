import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional

def plot_forecast_with_seasonal_decomposition(timestamps: Union[np.ndarray, List], observed_values: Union[np.ndarray, List[float]], trend_component: Union[np.ndarray, List[float]], seasonal_component: Union[np.ndarray, List[float]], residual_component: Union[np.ndarray, List[float]], forecast_values: Union[np.ndarray, List[float]], lower_bound: Union[np.ndarray, List[float]], upper_bound: Union[np.ndarray, List[float]], title: Optional[str]='Forecast with Seasonal Decomposition', x_label: Optional[str]='Time', y_label: Optional[str]='Values', observed_color: Optional[str]='black', forecast_color: Optional[str]='blue', interval_color: Optional[str]='lightblue', trend_color: Optional[str]='green', seasonal_color: Optional[str]='orange', residual_color: Optional[str]='red', alpha: Optional[float]=0.3) -> None:
    """
    Plot forecast results alongside seasonal decomposition components.

    This function creates a comprehensive visualization that displays the original time series
    decomposition (observed, trend, seasonal, and residual components) along with forecast
    predictions and their uncertainty intervals. It enables users to understand how well the
    forecast aligns with historical patterns and seasonal trends.

    Args:
        timestamps: Array or list of time points for the time series
        observed_values: Array or list of actual observed values
        trend_component: Array or list of trend component values from decomposition
        seasonal_component: Array or list of seasonal component values from decomposition
        residual_component: Array or list of residual component values from decomposition
        forecast_values: Array or list of forecasted values
        lower_bound: Array or list of lower bounds for prediction intervals
        upper_bound: Array or list of upper bounds for prediction intervals
        title: Title for the plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        observed_color: Color for observed values line
        forecast_color: Color for forecast line
        interval_color: Color for prediction interval shading
        trend_color: Color for trend component
        seasonal_color: Color for seasonal component
        residual_color: Color for residual component
        alpha: Transparency level for interval shading

    Returns:
        None: This function creates a plot but does not return any value

    Raises:
        ValueError: If input arrays have mismatched dimensions
        TypeError: If input types are not supported
    """
    try:
        timestamps = np.asarray(timestamps)
        observed_values = np.asarray(observed_values, dtype=float)
        trend_component = np.asarray(trend_component, dtype=float)
        seasonal_component = np.asarray(seasonal_component, dtype=float)
        residual_component = np.asarray(residual_component, dtype=float)
        forecast_values = np.asarray(forecast_values, dtype=float)
        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)
    except (TypeError, ValueError) as e:
        raise TypeError('All numeric inputs must be convertible to numpy arrays.') from e
    n_obs = len(timestamps)
    ts_components = [observed_values, trend_component, seasonal_component, residual_component]
    for comp in ts_components:
        if len(comp) != n_obs:
            raise ValueError('All time series components must have the same length as timestamps.')
    n_forecast = len(forecast_values)
    if not (len(lower_bound) == n_forecast and len(upper_bound) == n_forecast):
        raise ValueError('Forecast values, lower bound, and upper bound must all have the same length.')
    if n_forecast > 0:
        if n_forecast <= n_obs:
            forecast_timestamps = timestamps[-n_forecast:]
        elif n_obs > 1:
            step = timestamps[-1] - timestamps[-2]
            forecast_timestamps = np.arange(timestamps[-1] + step, timestamps[-1] + step * (n_forecast + 1), step)
            if len(forecast_timestamps) > n_forecast:
                forecast_timestamps = forecast_timestamps[:n_forecast]
        elif n_obs == 1:
            step = 1
            forecast_timestamps = np.arange(timestamps[0] + step, timestamps[0] + step * (n_forecast + 1), step)
            if len(forecast_timestamps) > n_forecast:
                forecast_timestamps = forecast_timestamps[:n_forecast]
        else:
            forecast_timestamps = np.arange(n_obs, n_obs + n_forecast)
    else:
        forecast_timestamps = np.array([])
    (fig, axes) = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    if title:
        fig.suptitle(title, fontsize=16)
    axes[0].plot(timestamps, observed_values, color=observed_color, label='Observed')
    if len(forecast_timestamps) > 0 and n_forecast > 0:
        axes[0].plot(forecast_timestamps, forecast_values, color=forecast_color, label='Forecast')
        axes[0].fill_between(forecast_timestamps, lower_bound, upper_bound, color=interval_color, alpha=alpha, label='Prediction Interval')
    axes[0].set_ylabel(y_label)
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(timestamps, trend_component, color=trend_color)
    axes[1].set_ylabel('Trend')
    axes[1].grid(True)
    axes[2].plot(timestamps, seasonal_component, color=seasonal_color)
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True)
    axes[3].plot(timestamps, residual_component, color=residual_color)
    axes[3].set_ylabel('Residual')
    axes[3].grid(True)
    axes[4].plot(timestamps, observed_values, color=observed_color, label='Observed')
    if len(forecast_timestamps) > 0 and n_forecast > 0:
        axes[4].plot(forecast_timestamps, forecast_values, color=forecast_color, label='Forecast')
        axes[4].fill_between(forecast_timestamps, lower_bound, upper_bound, color=interval_color, alpha=alpha, label='Prediction Interval')
    axes[4].set_xlabel(x_label)
    axes[4].set_ylabel(y_label)
    axes[4].legend()
    axes[4].grid(True)
    plt.tight_layout()
    plt.show()