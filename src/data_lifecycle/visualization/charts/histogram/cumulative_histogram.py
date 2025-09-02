from general.structures.data_batch import DataBatch
import numpy as np
from typing import Optional, Union, List

def plot_cumulative_distribution(data: Union[np.ndarray, List[float], DataBatch], bins: Optional[Union[int, List[float]]]=None, title: Optional[str]=None, xlabel: Optional[str]=None, ylabel: Optional[str]='Cumulative Probability', figsize: Optional[tuple]=(8, 6), color: str='blue', alpha: float=0.7) -> None:
    """
    Plot a cumulative distribution function (CDF) histogram for the given data.

    This function generates a cumulative histogram where each bin represents the cumulative proportion
    of data points up to that bin. It supports both raw data arrays and DataBatch objects.
    
    The plot displays the empirical cumulative distribution function (ECDF), which shows the probability
    that a randomly selected value from the dataset will be less than or equal to a given value.

    Args:
        data (Union[np.ndarray, List[float], DataBatch]): Input data to visualize.
            If DataBatch, uses the 'data' attribute.
        bins (Optional[Union[int, List[float]]]): Number of bins or explicit bin edges.
            If None, defaults to 30 bins.
        title (Optional[str]): Title for the plot. Defaults to "Cumulative Distribution".
        xlabel (Optional[str]): Label for x-axis. If None, inferred from data.
        ylabel (Optional[str]): Label for y-axis. Defaults to "Cumulative Probability".
        figsize (Optional[tuple]): Figure size as (width, height). Defaults to (8, 6).
        color (str): Color of the histogram bars. Defaults to "blue".
        alpha (float): Transparency level of the bars. Defaults to 0.7.

    Returns:
        None: Displays the plot directly.

    Raises:
        ValueError: If data is empty or bins are invalid.
    """
    import matplotlib.pyplot as plt
    if isinstance(data, DataBatch):
        values = data.data
        if xlabel is None and data.feature_names and (len(data.feature_names) == 1):
            xlabel = data.feature_names[0]
    else:
        values = data
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if values.ndim > 1:
        values = values.ravel()
    if values.size == 0:
        raise ValueError('Input data cannot be empty')
    unique_values = np.unique(values)
    if len(unique_values) == 1:
        plt.figure(figsize=figsize)
        plt.bar(unique_values[0], 1.0, width=0.1 * abs(unique_values[0]) if unique_values[0] != 0 else 0.1, color=color, alpha=alpha, align='center')
        plt.title(title if title is not None else 'Cumulative Distribution')
        plt.xlabel(xlabel if xlabel is not None else 'Value' if not (isinstance(data, DataBatch) and data.feature_names and (len(data.feature_names) == 1)) else data.feature_names[0])
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.show()
        return
    if bins is None:
        bins = 30
    if isinstance(bins, int) and bins <= 0:
        raise ValueError('Number of bins must be positive')
    elif isinstance(bins, (list, np.ndarray)) and len(bins) < 2:
        raise ValueError('Bin edges list must contain at least two elements')
    if title is None:
        title = 'Cumulative Distribution'
    if xlabel is None:
        xlabel = 'Value'
    plt.figure(figsize=figsize)
    (counts, bin_edges) = np.histogram(values, bins=bins)
    cumulative_counts = np.cumsum(counts)
    cumulative_probabilities = cumulative_counts / len(values)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    plt.bar(bin_centers, cumulative_probabilities, width=bin_widths, color=color, alpha=alpha, align='center')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_step_cumulative_histogram(data: Union[np.ndarray, List[float], DataBatch], bins: Optional[Union[int, List[float]]]=None, title: Optional[str]=None, xlabel: Optional[str]=None, ylabel: Optional[str]='Cumulative Count', figsize: Optional[tuple]=(8, 6), color: str='green', linestyle: str='-', linewidth: float=1.5) -> None:
    """
    Plot a step-style cumulative histogram for the given data.

    This function generates a step plot representing cumulative counts or frequencies
    of data points up to each bin edge. Unlike the smooth CDF plot, this visualization
    emphasizes discrete jumps at each bin boundary using step transitions.
    
    The step plot is particularly useful for visualizing empirical cumulative distribution
    functions with clear delineation between bins.

    Args:
        data (Union[np.ndarray, List[float], DataBatch]): Input data to visualize.
            If DataBatch, uses the 'data' attribute.
        bins (Optional[Union[int, List[float]]]): Number of bins or explicit bin edges.
            If None, defaults to 30 bins.
        title (Optional[str]): Title for the plot. Defaults to "Step Cumulative Histogram".
        xlabel (Optional[str]): Label for x-axis. If None, inferred from data.
        ylabel (Optional[str]): Label for y-axis. Defaults to "Cumulative Count".
        figsize (Optional[tuple]): Figure size as (width, height). Defaults to (8, 6).
        color (str): Color of the step line. Defaults to "green".
        linestyle (str): Style of the line ('-', '--', '-.', ':'). Defaults to '-'.
        linewidth (float): Width of the step line. Defaults to 1.5.

    Returns:
        None: Displays the plot directly.

    Raises:
        ValueError: If data is empty or bins are invalid.
        TypeError: If data type is unsupported.
    """
    import matplotlib.pyplot as plt
    if isinstance(data, DataBatch):
        values = data.data
        if xlabel is None and data.feature_names and (len(data.feature_names) == 1):
            xlabel = data.feature_names[0]
    elif isinstance(data, (np.ndarray, list)):
        values = data
    else:
        raise TypeError('Unsupported data type. Expected numpy array, list, or DataBatch.')
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if values.ndim > 1:
        values = values.ravel()
    if values.size == 0:
        raise ValueError('Input data cannot be empty')
    unique_values = np.unique(values)
    if len(unique_values) == 1:
        plt.figure(figsize=figsize)
        plt.step([unique_values[0], unique_values[0]], [0, len(values)], where='post', color=color, linestyle=linestyle, linewidth=linewidth)
        plt.title(title if title is not None else 'Step Cumulative Histogram')
        plt.xlabel(xlabel if xlabel is not None else 'Value')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.show()
        return
    if bins is None:
        bins = 30
    if isinstance(bins, int) and bins <= 0:
        raise ValueError('Number of bins must be positive')
    elif isinstance(bins, (list, np.ndarray)) and len(bins) < 2:
        raise ValueError('Bin edges list must contain at least two elements')
    if title is None:
        title = 'Step Cumulative Histogram'
    if xlabel is None:
        xlabel = 'Value'
    plt.figure(figsize=figsize)
    (counts, bin_edges) = np.histogram(values, bins=bins)
    cumulative_counts = np.cumsum(counts)
    x_step = np.concatenate([[bin_edges[0]], np.repeat(bin_edges[1:-1], 2), [bin_edges[-1]]])
    y_step = np.concatenate([[0], np.repeat(cumulative_counts[:-1], 2), [cumulative_counts[-1]]])
    plt.step(x_step, y_step, where='post', color=color, linestyle=linestyle, linewidth=linewidth)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()