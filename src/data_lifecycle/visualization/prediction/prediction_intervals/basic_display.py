import numpy as np
from typing import Union, Optional, List
import matplotlib.pyplot as plt

def display_prediction_intervals(x_values: Union[np.ndarray, List[float]], y_predictions: Union[np.ndarray, List[float]], lower_bounds: Union[np.ndarray, List[float]], upper_bounds: Union[np.ndarray, List[float]], title: Optional[str]='Prediction Intervals', x_label: Optional[str]='X Values', y_label: Optional[str]='Predictions', interval_color: Optional[str]='lightblue', prediction_color: Optional[str]='blue', alpha: Optional[float]=0.3) -> None:
    """
    Display prediction intervals along with predicted values in a 2D plot.

    This function creates a visualization showing the predicted values with their corresponding
    prediction intervals. The intervals are displayed as a shaded region around the prediction line,
    providing an intuitive representation of the uncertainty in predictions.

    Args:
        x_values (Union[np.ndarray, List[float]]): The x-axis values for the plot.
        y_predictions (Union[np.ndarray, List[float]]): The predicted y-values corresponding to x_values.
        lower_bounds (Union[np.ndarray, List[float]]): The lower bounds of the prediction intervals.
        upper_bounds (Union[np.ndarray, List[float]]): The upper bounds of the prediction intervals.
        title (Optional[str]): The title of the plot. Defaults to 'Prediction Intervals'.
        x_label (Optional[str]): Label for the x-axis. Defaults to 'X Values'.
        y_label (Optional[str]): Label for the y-axis. Defaults to 'Predictions'.
        interval_color (Optional[str]): Color for the prediction interval shading. Defaults to 'lightblue'.
        prediction_color (Optional[str]): Color for the prediction line. Defaults to 'blue'.
        alpha (Optional[float]): Transparency level for the interval shading. Defaults to 0.3.

    Returns:
        None: This function displays the plot but does not return any value.

    Raises:
        ValueError: If the lengths of the input arrays/lists do not match.
        TypeError: If the inputs are not of the expected types.
    """
    for (name, arr) in [('x_values', x_values), ('y_predictions', y_predictions), ('lower_bounds', lower_bounds), ('upper_bounds', upper_bounds)]:
        if not isinstance(arr, (np.ndarray, list)):
            raise TypeError(f'{name} must be a numpy array or list of floats')
        if isinstance(arr, list):
            if not all((isinstance(x, (int, float)) for x in arr)):
                raise TypeError(f'All elements in {name} must be numeric')
    x_vals = np.asarray(x_values)
    y_preds = np.asarray(y_predictions)
    lower = np.asarray(lower_bounds)
    upper = np.asarray(upper_bounds)
    if not (x_vals.ndim == 1 and y_preds.ndim == 1 and (lower.ndim == 1) and (upper.ndim == 1)):
        raise ValueError('All input arrays must be 1-dimensional')
    if not len(x_vals) == len(y_preds) == len(lower) == len(upper):
        raise ValueError('All input arrays must have the same length')
    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals[sorted_indices]
    y_sorted = y_preds[sorted_indices]
    lower_sorted = lower[sorted_indices]
    upper_sorted = upper[sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.fill_between(x_sorted, lower_sorted, upper_sorted, color=interval_color, alpha=alpha, label='Prediction Interval')
    plt.plot(x_sorted, y_sorted, color=prediction_color, linewidth=2, label='Predictions')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()