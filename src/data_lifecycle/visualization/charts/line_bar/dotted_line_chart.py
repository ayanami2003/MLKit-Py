import numpy as np
from typing import Union, Optional, List

def plot_dotted_line_chart(x_data: Union[List, np.ndarray], y_data: Union[List, np.ndarray], title: Optional[str]=None, x_label: Optional[str]=None, y_label: Optional[str]=None, line_color: str='blue', line_width: float=1.0, figsize: tuple=(10, 6)) -> None:
    """
    Plot a dotted line chart using the provided x and y data.

    This function generates a line plot where the data points are connected by dotted lines. 
    It allows customization of the plot's title, axis labels, line color, line width, and figure size.

    Args:
        x_data (Union[List, np.ndarray]): The x-axis data points. Can be a list or numpy array.
        y_data (Union[List, np.ndarray]): The y-axis data points. Can be a list or numpy array.
        title (Optional[str]): The title of the plot. Defaults to None.
        x_label (Optional[str]): The label for the x-axis. Defaults to None.
        y_label (Optional[str]): The label for the y-axis. Defaults to None.
        line_color (str): The color of the dotted line. Defaults to 'blue'.
        line_width (float): The width of the dotted line. Defaults to 1.0.
        figsize (tuple): The size of the figure in inches as (width, height). Defaults to (10, 6).

    Returns:
        None: This function displays the plot but does not return any value.

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
    plt.figure(figsize=figsize)
    plt.plot(x_array, y_array, color=line_color, linewidth=line_width, linestyle=':')
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.show()