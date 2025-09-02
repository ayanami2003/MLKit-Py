from typing import Optional, Union, List
import numpy as np
from general.structures.data_batch import DataBatch

def plot_upper_confidence_band(x_values: Union[np.ndarray, List[float]], y_predictions: Union[np.ndarray, List[float]], confidence_intervals: Union[np.ndarray, List[float]], title: Optional[str]='Upper Confidence Band', x_label: Optional[str]='X Values', y_label: Optional[str]='Predictions', band_color: Optional[str]='lightblue', line_color: Optional[str]='blue', alpha: Optional[float]=0.3) -> None:
    """
    Plot the upper confidence band for predictions.

    This function visualizes the upper confidence band around predicted values,
    typically used to show uncertainty bounds in regression or forecasting tasks.
    The band represents the upper limit of the confidence interval at each point.

    Args:
        x_values (Union[np.ndarray, List[float]]): X-axis values corresponding to predictions.
        y_predictions (Union[np.ndarray, List[float]]): Predicted Y-values (central estimates).
        confidence_intervals (Union[np.ndarray, List[float]]): Upper confidence bounds 
            (same length as predictions). These represent the absolute upper limit 
            (prediction + margin) rather than just the margin itself.
        title (Optional[str]): Title of the plot. Defaults to "Upper Confidence Band".
        x_label (Optional[str]): Label for the X-axis. Defaults to "X Values".
        y_label (Optional[str]): Label for the Y-axis. Defaults to "Predictions".
        band_color (Optional[str]): Color of the confidence band area. 
            Defaults to "lightblue".
        line_color (Optional[str]): Color of the prediction line. Defaults to "blue".
        alpha (Optional[float]): Transparency level of the confidence band. 
            Should be between 0 and 1. Defaults to 0.3.

    Returns:
        None: This function produces a plot as a side effect and does not return data.

    Raises:
        ValueError: If input arrays have mismatched lengths or invalid dimensions.
    """
    import matplotlib.pyplot as plt
    x_vals = np.asarray(x_values)
    y_preds = np.asarray(y_predictions)
    conf_ints = np.asarray(confidence_intervals)
    if x_vals.ndim != 1 or y_preds.ndim != 1 or conf_ints.ndim != 1:
        raise ValueError('All input arrays must be 1-dimensional')
    if not len(x_vals) == len(y_preds) == len(conf_ints):
        raise ValueError('All input arrays must have the same length')
    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals[sorted_indices]
    y_sorted = y_preds[sorted_indices]
    ci_sorted = conf_ints[sorted_indices]
    plt.figure()
    plt.fill_between(x_sorted, y_sorted, ci_sorted, color=band_color, alpha=alpha, label='Upper Confidence Band')
    plt.plot(x_sorted, y_sorted, color=line_color, linewidth=2, label='Predictions')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()