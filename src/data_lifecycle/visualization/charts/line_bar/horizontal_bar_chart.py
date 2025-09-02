import numpy as np
from typing import Union, List, Optional, Tuple

def plot_horizontal_bar_chart(x_data: Union[List, np.ndarray], y_data: Union[List, np.ndarray], title: Optional[str]=None, x_label: Optional[str]=None, y_label: Optional[str]=None, bar_color: str='blue', figsize: Tuple[int, int]=(10, 6), sort_bars: bool=False, ascending: bool=False) -> None:
    """
    Plot a horizontal bar chart using the provided x and y data.

    This function creates a horizontal bar chart where categories are displayed along the y-axis
    and values are shown as bars extending horizontally along the x-axis. It supports customization
    of labels, colors, figure size, and sorting of bars by value.

    Args:
        x_data (Union[List, np.ndarray]): Values to be plotted as bar lengths.
        y_data (Union[List, np.ndarray]): Categories corresponding to each bar.
        title (Optional[str]): Title of the chart. Defaults to None.
        x_label (Optional[str]): Label for the x-axis. Defaults to None.
        y_label (Optional[str]): Label for the y-axis. Defaults to None.
        bar_color (str): Color of the bars. Defaults to 'blue'.
        figsize (Tuple[int, int]): Size of the figure (width, height) in inches. Defaults to (10, 6).
        sort_bars (bool): Whether to sort bars by value. Defaults to False.
        ascending (bool): Sort order when sort_bars is True. Defaults to False (descending).

    Returns:
        None: Displays the plot but does not return any value.

    Raises:
        ValueError: If x_data and y_data have mismatched lengths.
    """
    import matplotlib.pyplot as plt
    x_array = np.asarray(x_data)
    y_array = np.asarray(y_data)
    if x_array.ndim != 1:
        raise ValueError('x_data must be one-dimensional')
    if y_array.ndim != 1:
        raise ValueError('y_data must be one-dimensional')
    if len(x_array) != len(y_array):
        raise ValueError('x_data and y_data must have the same length')
    if sort_bars:
        sorted_indices = np.argsort(x_array)
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        x_array = x_array[sorted_indices]
        y_array = y_array[sorted_indices]
    plt.figure(figsize=figsize)
    plt.barh(y_array, x_array, color=bar_color)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()