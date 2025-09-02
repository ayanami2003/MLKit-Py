from general.structures.data_batch import DataBatch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Optional, Union, Any

def qq_plot(data: Union[DataBatch, np.ndarray], distribution: str='norm', figsize: tuple=(8, 6), title: Optional[str]=None, ax: Optional[plt.Axes]=None, **distribution_params) -> plt.Figure:
    """
    Generate a Quantile-Quantile (Q-Q) plot to compare data distribution against a theoretical distribution.
    
    This function creates a Q-Q plot which is a graphical method for comparing two probability distributions
    by plotting their quantiles against each other. Points on the plot represent quantiles from the sample
    data versus quantiles from the theoretical distribution. If the points fall approximately along a straight
    line, it suggests the data follows the specified theoretical distribution.
    
    Args:
        data (Union[DataBatch, np.ndarray]): Input data to analyze. If DataBatch, uses the primary data attribute.
        distribution (str): Name of the theoretical distribution to compare against. Defaults to 'norm' (normal distribution).
                           Supported distributions include 'norm', 'expon', 'logistic', 'gamma', etc.
        figsize (tuple): Figure size as (width, height). Defaults to (8, 6).
        title (Optional[str]): Custom title for the plot. If None, a default title is generated.
        ax (Optional[plt.Axes]): Matplotlib axes object to plot on. If None, creates new figure and axes.
        **distribution_params: Additional parameters for the theoretical distribution (e.g., loc, scale for norm).
        
    Returns:
        plt.Figure: Matplotlib figure object containing the Q-Q plot.
        
    Raises:
        ValueError: If data is empty or distribution name is not supported.
        TypeError: If data is not a valid type.
    """
    supported_distributions = ['norm', 'expon', 'logistic', 'gamma']
    if distribution not in supported_distributions:
        raise ValueError(f"Distribution '{distribution}' is not supported. Supported distributions: {supported_distributions}")
    if isinstance(data, DataBatch):
        values = data.data
    elif isinstance(data, np.ndarray):
        values = data
    else:
        raise TypeError('Data must be either a DataBatch or numpy array')
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if values.ndim > 1:
        values = values.ravel()
    if values.size == 0:
        raise ValueError('Input data cannot be empty')
    if ax is None:
        (fig, ax) = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    try:
        dist_func = getattr(stats, distribution)
    except AttributeError:
        raise ValueError(f"Distribution '{distribution}' not found in scipy.stats")
    sorted_data = np.sort(values)
    n = len(sorted_data)
    quantiles = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)
    quantiles = np.clip(quantiles, 1e-10, 1 - 1e-10)
    theoretical_quantiles = dist_func.ppf(quantiles, **distribution_params)
    ax.scatter(theoretical_quantiles, sorted_data, alpha=0.6)
    min_val = min(np.min(theoretical_quantiles), np.min(sorted_data))
    max_val = max(np.max(theoretical_quantiles), np.max(sorted_data))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
    if title is None:
        title = f'Q-Q Plot vs {distribution.capitalize()} Distribution'
    ax.set_title(title)
    ax.set_xlabel(f'Theoretical Quantiles ({distribution})')
    ax.set_ylabel('Sample Quantiles')
    ax.grid(True, alpha=0.3)
    return fig