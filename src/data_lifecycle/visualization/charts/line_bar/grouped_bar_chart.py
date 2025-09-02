import numpy as np
from typing import List, Union, Optional, Tuple

def plot_grouped_bar_chart(categories: List[str], group_data: List[Union[List, np.ndarray]], group_labels: List[str], title: Optional[str]=None, x_label: Optional[str]=None, y_label: Optional[str]=None, bar_width: float=0.35, colors: Optional[List[str]]=None, figsize: Tuple[int, int]=(10, 6), legend_loc: str='best') -> None:
    """
    Plot a grouped bar chart to compare multiple groups across several categories.

    This function creates a grouped bar chart where each category has multiple bars representing
    different groups. It's useful for comparing values across categories for multiple datasets.

    Args:
        categories (List[str]): Labels for the x-axis categories.
        group_data (List[Union[List, np.ndarray]]): A list where each element represents the data
            for one group across all categories. Each element should be a list or array of values
            matching the length of categories.
        group_labels (List[str]): Labels for each group to be displayed in the legend.
        title (Optional[str]): Title for the chart. Defaults to None.
        x_label (Optional[str]): Label for the x-axis. Defaults to None.
        y_label (Optional[str]): Label for the y-axis. Defaults to None.
        bar_width (float): Width of each individual bar. Defaults to 0.35.
        colors (Optional[List[str]]): Colors for each group. If None, default colors are used.
        figsize (Tuple[int, int]): Figure size as (width, height) in inches. Defaults to (10, 6).
        legend_loc (str): Location of the legend. Defaults to 'best'.

    Returns:
        None: This function displays the plot but does not return any value.

    Raises:
        ValueError: If the lengths of group_data elements don't match the number of categories,
            or if the number of group_labels doesn't match the number of groups.
    """
    import matplotlib.pyplot as plt
    if not isinstance(categories, list):
        raise ValueError('categories must be a list')
    if not isinstance(group_data, list):
        raise ValueError('group_data must be a list')
    if not isinstance(group_labels, list):
        raise ValueError('group_labels must be a list')
    n_categories = len(categories)
    n_groups = len(group_data)
    if n_groups == 0:
        raise ValueError('group_data cannot be empty')
    if n_groups != len(group_labels):
        raise ValueError('Number of group_labels must match number of groups in group_data')
    for (i, data) in enumerate(group_data):
        if len(data) != n_categories:
            raise ValueError(f'Length of group_data[{i}] ({len(data)}) does not match number of categories ({n_categories})')
    group_arrays = [np.asarray(data) for data in group_data]
    plt.figure(figsize=figsize)
    x_positions = np.arange(n_categories)
    total_width = bar_width * n_groups
    group_offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_groups)
    plot_colors = colors if colors is not None else [f'C{i}' for i in range(n_groups)]
    if len(plot_colors) < n_groups:
        plot_colors = (plot_colors * (n_groups // len(plot_colors) + 1))[:n_groups]
    for (i, (data, label, offset, color)) in enumerate(zip(group_arrays, group_labels, group_offsets, plot_colors)):
        plt.bar(x_positions + offset, data, bar_width, label=label, color=color)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.xticks(x_positions, categories)
    plt.legend(loc=legend_loc)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()