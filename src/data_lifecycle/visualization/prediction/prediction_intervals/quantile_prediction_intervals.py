from typing import Union, List, Optional
import numpy as np

def plot_quantile_prediction_intervals(x_values: Union[np.ndarray, List[float]], y_predictions: Union[np.ndarray, List[float]], lower_quantiles: Union[np.ndarray, List[float]], upper_quantiles: Union[np.ndarray, List[float]], title: Optional[str]='Quantile Prediction Intervals', x_label: Optional[str]='X Values', y_label: Optional[str]='Predictions', interval_color: Optional[str]='lightblue', prediction_color: Optional[str]='blue', alpha: Optional[float]=0.3, quantile_labels: Optional[List[str]]=None) -> None:
    """
    Plot prediction intervals based on quantiles.

    This function visualizes prediction intervals by plotting the predicted values along with
    user-provided lower and upper quantiles. The intervals are displayed as filled areas,
    while the predictions are shown as a line. This approach is particularly useful for
    quantile regression models or when prediction intervals are derived from empirical quantiles.

    Args:
        x_values (Union[np.ndarray, List[float]]): The x-axis values corresponding to the predictions.
        y_predictions (Union[np.ndarray, List[float]]): The central prediction values (e.g., median or mean).
        lower_quantiles (Union[np.ndarray, List[float]]): The lower bound values of the prediction interval
            (e.g., 5th percentile).
        upper_quantiles (Union[np.ndarray, List[float]]): The upper bound values of the prediction interval
            (e.g., 95th percentile).
        title (Optional[str]): The title of the plot. Defaults to 'Quantile Prediction Intervals'.
        x_label (Optional[str]): Label for the x-axis. Defaults to 'X Values'.
        y_label (Optional[str]): Label for the y-axis. Defaults to 'Predictions'.
        interval_color (Optional[str]): Color for the interval fill area. Defaults to 'lightblue'.
        prediction_color (Optional[str]): Color for the prediction line. Defaults to 'blue'.
        alpha (Optional[float]): Transparency level for the interval fill (0-1). Defaults to 0.3.
        quantile_labels (Optional[List[str]]): Labels for the quantiles (e.g., ['5%', '95%']). If provided,
            must match the length of lower_quantiles and upper_quantiles.

    Returns:
        None: Displays the plot but does not return any value.

    Raises:
        ValueError: If lengths of input arrays do not match or if quantile_labels length is inconsistent.
    """
    import matplotlib.pyplot as plt
    x_vals = np.asarray(x_values)
    y_preds = np.asarray(y_predictions)
    lower = np.asarray(lower_quantiles)
    upper = np.asarray(upper_quantiles)
    if not (x_vals.ndim == 1 and y_preds.ndim == 1 and (lower.ndim == 1) and (upper.ndim == 1)):
        raise ValueError('All input arrays must be 1-dimensional')
    if not len(x_vals) == len(y_preds) == len(lower) == len(upper):
        raise ValueError('All input arrays must have the same length')
    if quantile_labels is not None:
        if len(quantile_labels) != 2:
            raise ValueError('quantile_labels must contain exactly two labels [lower, upper]')
    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals[sorted_indices]
    y_sorted = y_preds[sorted_indices]
    lower_sorted = lower[sorted_indices]
    upper_sorted = upper[sorted_indices]
    plt.figure(figsize=(10, 6))
    label = None
    if quantile_labels is not None:
        label = f'{quantile_labels[0]}-{quantile_labels[1]} Interval'
    plt.fill_between(x_sorted, lower_sorted, upper_sorted, color=interval_color, alpha=alpha, label=label)
    plt.plot(x_sorted, y_sorted, color=prediction_color, linewidth=2, label='Predictions')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()