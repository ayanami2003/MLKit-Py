import numpy as np
from typing import Union, List, Optional
import matplotlib.pyplot as plt

def plot_interval_forecast(x_values: Union[np.ndarray, List[float]], y_predictions: Union[np.ndarray, List[float]], lower_bounds: Union[np.ndarray, List[float]], upper_bounds: Union[np.ndarray, List[float]], title: Optional[str]='Interval Forecast', x_label: Optional[str]='Time', y_label: Optional[str]='Values', interval_color: Optional[str]='lightblue', prediction_color: Optional[str]='blue', alpha: Optional[float]=0.3, show_grid: Optional[bool]=True, figsize: Optional[tuple]=(10, 6)) -> None:
    """
    Visualize prediction intervals for time series forecasting.

    This function plots the predicted values along with their corresponding upper and lower bounds,
    providing a visual representation of the forecast uncertainty. The plot includes customizable 
    styling for the prediction line, interval shading, labels, and figure properties.

    Args:
        x_values (Union[np.ndarray, List[float]]): The x-axis values, typically representing time steps or indices.
        y_predictions (Union[np.ndarray, List[float]]): The central prediction values for each time step.
        lower_bounds (Union[np.ndarray, List[float]]): The lower bound of the prediction interval for each time step.
        upper_bounds (Union[np.ndarray, List[float]]): The upper bound of the prediction interval for each time step.
        title (Optional[str]): The title of the plot. Defaults to "Interval Forecast".
        x_label (Optional[str]): Label for the x-axis. Defaults to "Time".
        y_label (Optional[str]): Label for the y-axis. Defaults to "Values".
        interval_color (Optional[str]): Color of the shaded area representing the prediction interval. 
                                        Defaults to "lightblue".
        prediction_color (Optional[str]): Color of the prediction line. Defaults to "blue".
        alpha (Optional[float]): Transparency level for the interval shading, between 0 and 1. Defaults to 0.3.
        show_grid (Optional[bool]): Whether to display a grid on the plot. Defaults to True.
        figsize (Optional[tuple]): Figure size as (width, height) in inches. Defaults to (10, 6).

    Returns:
        None: This function displays the plot but does not return any value.

    Raises:
        ValueError: If the lengths of the input arrays do not match.
    """
    x_vals = np.asarray(x_values)
    y_preds = np.asarray(y_predictions)
    lower = np.asarray(lower_bounds)
    upper = np.asarray(upper_bounds)
    if not x_vals.shape[0] == y_preds.shape[0] == lower.shape[0] == upper.shape[0]:
        raise ValueError('All input arrays must have the same length.')
    plt.figure(figsize=figsize)
    plt.fill_between(x_vals, lower, upper, color=interval_color, alpha=alpha, label='Prediction Interval')
    plt.plot(x_vals, y_preds, color=prediction_color, label='Predictions', linewidth=2)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.legend()
    plt.grid(show_grid, alpha=0.3)
    plt.show()