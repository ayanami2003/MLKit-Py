import numpy as np
from typing import Union, List, Optional
import matplotlib.pyplot as plt

def plot_ensemble_prediction_intervals(x_values: Union[np.ndarray, List[float]], y_predictions: Union[np.ndarray, List[float]], lower_bounds: Union[np.ndarray, List[float]], upper_bounds: Union[np.ndarray, List[float]], ensemble_members: Optional[List[str]]=None, title: Optional[str]='Ensemble Prediction Intervals', x_label: Optional[str]='X Values', y_label: Optional[str]='Predictions', interval_colors: Optional[List[str]]=None, prediction_color: Optional[str]='blue', alpha: Optional[float]=0.3) -> None:
    """
    Visualize prediction intervals from an ensemble of models.

    This function plots the predicted values along with their corresponding prediction intervals,
    allowing for visual comparison of uncertainty across multiple ensemble members. Each ensemble
    member's prediction interval is displayed with customizable colors.

    Args:
        x_values (Union[np.ndarray, List[float]]): The x-axis values for plotting.
        y_predictions (Union[np.ndarray, List[float]]): Predicted values for each ensemble member.
            Shape should be (n_samples,) or (n_samples, n_ensemble_members).
        lower_bounds (Union[np.ndarray, List[float]]): Lower bounds of prediction intervals.
            Shape should match y_predictions.
        upper_bounds (Union[np.ndarray, List[float]]): Upper bounds of prediction intervals.
            Shape should match y_predictions.
        ensemble_members (Optional[List[str]]): Names of ensemble members. If None, members will
            be labeled generically.
        title (Optional[str]): Title for the plot. Defaults to "Ensemble Prediction Intervals".
        x_label (Optional[str]): Label for x-axis. Defaults to "X Values".
        y_label (Optional[str]): Label for y-axis. Defaults to "Predictions".
        interval_colors (Optional[List[str]]): Colors for each ensemble member's interval.
            If None, default colors will be used.
        prediction_color (Optional[str]): Color for the main prediction line. Defaults to "blue".
        alpha (Optional[float]): Transparency level for interval shading. Defaults to 0.3.

    Returns:
        None: This function displays the plot but does not return any value.

    Raises:
        ValueError: If input arrays have mismatched shapes or invalid dimensions.
    """
    x_values = np.asarray(x_values)
    y_predictions = np.asarray(y_predictions)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    if y_predictions.shape != lower_bounds.shape or y_predictions.shape != upper_bounds.shape:
        raise ValueError('Shapes of y_predictions, lower_bounds, and upper_bounds must match.')
    if y_predictions.ndim == 1:
        n_samples = y_predictions.shape[0]
        n_ensemble_members = 1
        y_predictions = y_predictions.reshape(-1, 1)
        lower_bounds = lower_bounds.reshape(-1, 1)
        upper_bounds = upper_bounds.reshape(-1, 1)
    elif y_predictions.ndim == 2:
        (n_samples, n_ensemble_members) = y_predictions.shape
    else:
        raise ValueError('y_predictions must be 1D or 2D array.')
    if x_values.shape[0] != n_samples:
        raise ValueError('Length of x_values must match number of samples in y_predictions.')
    if ensemble_members is None:
        ensemble_members = [f'Model {i + 1}' for i in range(n_ensemble_members)]
    elif len(ensemble_members) != n_ensemble_members:
        raise ValueError('Length of ensemble_members must match number of ensemble members in y_predictions.')
    if interval_colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        interval_colors = [default_colors[i % len(default_colors)] for i in range(n_ensemble_members)]
    elif len(interval_colors) != n_ensemble_members:
        raise ValueError('Length of interval_colors must match number of ensemble members.')
    plt.figure(figsize=(10, 6))
    for i in range(n_ensemble_members):
        plt.fill_between(x_values, lower_bounds[:, i], upper_bounds[:, i], color=interval_colors[i], alpha=alpha, label=ensemble_members[i])
        plt.plot(x_values, y_predictions[:, i], color=interval_colors[i], linestyle='--', linewidth=1)
    if n_ensemble_members > 1:
        mean_predictions = np.mean(y_predictions, axis=1)
        plt.plot(x_values, mean_predictions, color=prediction_color, linewidth=2, label='Mean Prediction')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()