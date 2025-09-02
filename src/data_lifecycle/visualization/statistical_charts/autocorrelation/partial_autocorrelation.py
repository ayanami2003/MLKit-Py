import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict, Any
from general.structures.data_batch import DataBatch

def partial_correlation_analysis(data: Union[DataBatch, np.ndarray], max_lags: int=20, figsize: tuple=(10, 6), title: Optional[str]=None, ax: Optional[plt.Axes]=None, conf_int: float=0.95, **plot_params: Dict[str, Any]) -> plt.Figure:
    """
    Perform partial autocorrelation analysis and generate a visualization plot.
    
    This function computes the partial autocorrelation function (PACF) for time series data
    and displays the results in a stem plot with confidence intervals. Partial autocorrelation
    measures the correlation between observations at different time lags, controlling for
    correlations at shorter lags.
    
    Args:
        data (Union[DataBatch, np.ndarray]): Input time series data. If DataBatch, expects
            a 1D array-like structure in the data attribute.
        max_lags (int): Maximum number of lags to compute partial autocorrelation for.
            Defaults to 20.
        figsize (tuple): Figure size as (width, height). Defaults to (10, 6).
        title (Optional[str]): Custom title for the plot. If None, a default title is used.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, a new figure is created.
        conf_int (float): Confidence interval level for significance bands. Defaults to 0.95.
        **plot_params (Dict[str, Any]): Additional parameters passed to the plotting function.
        
    Returns:
        plt.Figure: Matplotlib figure object containing the partial autocorrelation plot.
        
    Raises:
        ValueError: If data is empty, not one-dimensional, or max_lags is invalid.
        TypeError: If data is not a supported type.
    """
    if isinstance(data, DataBatch):
        values = np.array(data.data)
    elif isinstance(data, np.ndarray):
        values = data
    else:
        raise TypeError('Data must be either a DataBatch or numpy array')
    if values.ndim > 1:
        values = values.ravel()
    if values.size == 0:
        raise ValueError('Input data cannot be empty')
    if not isinstance(max_lags, int) or max_lags <= 0:
        raise ValueError('max_lags must be a positive integer')
    if max_lags >= len(values):
        raise ValueError('max_lags must be less than the length of the time series')
    if not 0 < conf_int < 1:
        raise ValueError('conf_int must be between 0 and 1')
    pacf_values = _compute_pacf_yule_walker(values, max_lags)
    critical_value = 1.96
    if conf_int != 0.95:
        from scipy.stats import norm
        critical_value = norm.ppf(0.5 + conf_int / 2)
    conf_interval = critical_value / np.sqrt(len(values))
    if ax is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    lags = np.arange(len(pacf_values))
    (markerline, stemlines, baseline) = ax.stem(lags, pacf_values, basefmt=' ', **plot_params)
    ax.axhline(y=conf_interval, color='r', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=-conf_interval, color='r', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    significant = np.abs(pacf_values) > conf_interval
    ax.scatter(lags[significant], pacf_values[significant], color='red', s=50, zorder=5, label='Significant')
    ax.set_xlim(-0.5, max_lags + 0.5)
    ax.set_ylim(min(-0.2, min(pacf_values) - 0.1), max(0.2, max(pacf_values) + 0.1))
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Autocorrelation')
    if title is None:
        title = f'Partial Autocorrelation Function (Confidence Interval: {conf_int:.0%})'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(['Confidence Interval', 'Zero Line', 'Significant'], loc='upper right')
    return fig