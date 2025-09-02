from general.structures.data_batch import DataBatch
import numpy as np
from typing import Union, Optional, List

def plot_variable_binning_histogram(data: Union[np.ndarray, List[float], DataBatch], bin_edges: Union[List[float], np.ndarray], title: Optional[str]=None, xlabel: Optional[str]=None, ylabel: Optional[str]=None, figsize: Optional[tuple]=(8, 6), color: str='steelblue', alpha: float=0.7, edgecolor: str='black') -> None:
    """
    Plot a histogram with variable bin widths using specified bin edges.

    This function creates a histogram where each bin can have a different width,
    as defined by the provided bin edges. It's useful for visualizing data distributions
    where uniform binning is not appropriate.

    Args:
        data (Union[np.ndarray, List[float], DataBatch]): The input data to plot.
            If DataBatch is provided, uses the primary data attribute.
        bin_edges (Union[List[float], np.ndarray]): The edges of the bins.
            Must be monotonically increasing and have at least two elements.
        title (Optional[str]): The title of the plot. If None, no title is set.
        xlabel (Optional[str]): Label for the x-axis. If None, no label is set.
        ylabel (Optional[str]): Label for the y-axis. Defaults to "Frequency".
        figsize (Optional[tuple]): Figure size as (width, height). Defaults to (8, 6).
        color (str): Color of the bars. Defaults to 'steelblue'.
        alpha (float): Transparency level of the bars (0-1). Defaults to 0.7.
        edgecolor (str): Color of the bar edges. Defaults to 'black'.

    Returns:
        None: Displays the plot but does not return any value.

    Raises:
        ValueError: If bin_edges has fewer than 2 elements or is not monotonically increasing.
        TypeError: If data is not a supported type.
    """
    import matplotlib.pyplot as plt
    if len(bin_edges) < 2:
        raise ValueError('bin_edges must contain at least two elements')
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError('bin_edges must be monotonically increasing')
    if isinstance(data, DataBatch):
        values = data.data
    elif isinstance(data, (np.ndarray, list)):
        values = data
    else:
        raise TypeError('data must be a numpy array, list, or DataBatch')
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if values.ndim > 1:
        values = values.ravel()
    plt.figure(figsize=figsize)
    (counts, _) = np.histogram(values, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2
    plt.bar(bin_centers, counts, width=bin_widths, color=color, alpha=alpha, edgecolor=edgecolor)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()