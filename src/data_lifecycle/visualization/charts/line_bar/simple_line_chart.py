from typing import Optional, Union, List
import numpy as np
from general.structures.data_batch import DataBatch

def plot_simple_line_chart(x_data: Union[List, np.ndarray], y_data: Union[List, np.ndarray], title: Optional[str]=None, x_label: Optional[str]=None, y_label: Optional[str]=None, line_color: str='blue', line_style: str='-', line_width: float=1.0, figsize: tuple=(10, 6)) -> None:
    """
    Plot a simple line chart using provided x and y data.

    This function generates a line chart visualization from two arrays of data points,
    representing the x-axis and y-axis values respectively. It supports customization
    of visual properties such as colors, labels, and figure size.

    Args:
        x_data (Union[List, np.ndarray]): Array-like data for the x-axis. Must be
            one-dimensional and have the same length as y_data.
        y_data (Union[List, np.ndarray]): Array-like data for the y-axis. Must be
            one-dimensional and have the same length as x_data.
        title (Optional[str]): Title for the chart. If None, no title is displayed.
        x_label (Optional[str]): Label for the x-axis. If None, no label is displayed.
        y_label (Optional[str]): Label for the y-axis. If None, no label is displayed.
        line_color (str): Color of the plotted line. Defaults to 'blue'.
        line_style (str): Style of the plotted line ('-', '--', '-.', ':'). Defaults to '-'.
        line_width (float): Width of the plotted line. Defaults to 1.0.
        figsize (tuple): Figure size as (width, height) in inches. Defaults to (10, 6).

    Returns:
        None: Displays the plot but does not return any value.

    Raises:
        ValueError: If x_data and y_data have different lengths.
        TypeError: If x_data or y_data are not array-like structures.
    """
    import matplotlib.pyplot as plt
    if not isinstance(x_data, (list, np.ndarray)):
        raise TypeError('x_data must be a list or numpy array')
    if not isinstance(y_data, (list, np.ndarray)):
        raise TypeError('y_data must be a list or numpy array')
    x_array = np.asarray(x_data)
    y_array = np.asarray(y_data)
    if x_array.ndim != 1:
        raise ValueError('x_data must be one-dimensional')
    if y_array.ndim != 1:
        raise ValueError('y_data must be one-dimensional')
    if len(x_array) != len(y_array):
        raise ValueError('x_data and y_data must have the same length')
    plt.figure(figsize=figsize)
    plt.plot(x_array, y_array, color=line_color, linestyle=line_style, linewidth=line_width)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.show()