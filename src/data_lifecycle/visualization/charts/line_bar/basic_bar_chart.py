import numpy as np
from typing import Union, Optional, List

def plot_bar_chart(x_data: Union[List, np.ndarray], y_data: Union[List, np.ndarray], title: Optional[str]=None, x_label: Optional[str]=None, y_label: Optional[str]=None, bar_color: str='blue', bar_width: float=0.8, figsize: tuple=(10, 6), orientation: str='vertical') -> None:
    """
    Plot a basic bar chart with customizable appearance options.

    This function generates a bar chart from provided x and y data, supporting both vertical and horizontal
    orientations. It allows customization of colors, bar width, labels, and figure size to create publication-ready
    visualizations for data exploration and presentation purposes.

    Args:
        x_data (Union[List, np.ndarray]): Categories or labels for the bars. Can be strings or numeric values.
        y_data (Union[List, np.ndarray]): Heights/values for each bar corresponding to x_data.
        title (Optional[str]): Chart title displayed at the top of the plot.
        x_label (Optional[str]): Label for the x-axis.
        y_label (Optional[str]): Label for the y-axis.
        bar_color (str): Color of the bars. Accepts any matplotlib color specification.
        bar_width (float): Width of the bars as a fraction of available space (0.0 to 1.0).
        figsize (tuple): Figure size as (width, height) in inches.
        orientation (str): Orientation of bars - either 'vertical' or 'horizontal'.

    Returns:
        None: Displays the plot but does not return any value.

    Raises:
        ValueError: If x_data and y_data lengths do not match or if orientation is not 'vertical' or 'horizontal'.
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
    if orientation not in ['vertical', 'horizontal']:
        raise ValueError("orientation must be either 'vertical' or 'horizontal'")
    if not 0.0 < bar_width <= 1.0:
        raise ValueError('bar_width must be between 0.0 and 1.0')
    plt.figure(figsize=figsize)
    if orientation == 'vertical':
        plt.bar(x_array, y_array, color=bar_color, width=bar_width)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
    else:
        plt.barh(x_array, y_array, color=bar_color, height=bar_width)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
    if title:
        plt.title(title)
    if orientation == 'vertical':
        plt.grid(True, axis='y', alpha=0.3)
    else:
        plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()