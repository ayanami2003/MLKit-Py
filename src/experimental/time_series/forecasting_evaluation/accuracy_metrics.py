from general.structures.data_batch import DataBatch
from typing import Union, Optional, Dict, Any
import numpy as np

def evaluate_forecast_accuracy(y_true: Union[np.ndarray, DataBatch], y_pred: Union[np.ndarray, DataBatch], metrics: Optional[list]=None, sample_weights: Optional[np.ndarray]=None) -> Dict[str, float]:
    """
    Evaluate the accuracy of time series forecasts using various metrics.

    This function computes a set of accuracy metrics to assess the performance of time series forecasting models.
    It supports common metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
    Mean Absolute Percentage Error (MAPE), and others. Users can specify which metrics to compute,
    or let the function compute a default set.

    Args:
        y_true (Union[np.ndarray, DataBatch]): The true/target values of the time series. If a DataBatch is provided,
                                               it must contain the actual values in its data attribute.
        y_pred (Union[np.ndarray, DataBatch]): The forecasted/predicted values of the time series. If a DataBatch is
                                               provided, it must contain the predicted values in its data attribute.
        metrics (Optional[list]): A list of metric names to compute. If None, a default set of metrics will be computed.
                                  Supported metrics include: 'mae', 'rmse', 'mape', 'smape', 'mdape', 'mse', 'r2'.
        sample_weights (Optional[np.ndarray]): Weights for each sample to compute weighted metrics. If provided,
                                              must be the same length as y_true and y_pred.

    Returns:
        Dict[str, float]: A dictionary mapping metric names to their computed values.

    Raises:
        ValueError: If y_true and y_pred have mismatched shapes or if unsupported metric names are provided.
        TypeError: If inputs are not of the expected types.
    """
    if isinstance(y_true, DataBatch):
        y_true = y_true.data
    if isinstance(y_pred, DataBatch):
        y_pred = y_pred.data
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f'y_true and y_pred must have the same shape. Got {y_true.shape} and {y_pred.shape}')
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    if sample_weights is not None:
        if not isinstance(sample_weights, np.ndarray):
            sample_weights = np.asarray(sample_weights)
        if sample_weights.shape[0] != y_true_flat.shape[0]:
            raise ValueError('sample_weights must have the same length as y_true and y_pred')
        sample_weights = sample_weights.flatten()
    else:
        sample_weights = np.ones_like(y_true_flat)
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape', 'smape', 'mdape', 'mse', 'r2']
    supported_metrics = {'mae', 'rmse', 'mape', 'smape', 'mdape', 'mse', 'r2'}
    unsupported_metrics = set(metrics) - supported_metrics
    if unsupported_metrics:
        raise ValueError(f'Unsupported metrics: {unsupported_metrics}. Supported metrics are: {supported_metrics}')
    results = {}
    residuals = y_true_flat - y_pred_flat
    abs_residuals = np.abs(residuals)

    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)
    if 'mae' in metrics:
        results['mae'] = weighted_mean(abs_residuals, sample_weights)
    if 'mse' in metrics:
        squared_residuals = residuals ** 2
        results['mse'] = weighted_mean(squared_residuals, sample_weights)
    if 'rmse' in metrics:
        if 'mse' in results:
            results['rmse'] = np.sqrt(results['mse'])
        else:
            squared_residuals = residuals ** 2
            mse = weighted_mean(squared_residuals, sample_weights)
            results['rmse'] = np.sqrt(mse)
    if 'mape' in metrics:
        nonzero_mask = y_true_flat != 0
        if np.any(nonzero_mask):
            abs_percentage_errors = np.abs(residuals[nonzero_mask] / y_true_flat[nonzero_mask])
            results['mape'] = 100 * weighted_mean(abs_percentage_errors, sample_weights[nonzero_mask])
        else:
            results['mape'] = np.inf
    if 'smape' in metrics:
        denominator = (np.abs(y_true_flat) + np.abs(y_pred_flat)) / 2
        nonzero_mask = denominator != 0
        if np.any(nonzero_mask):
            smape_values = np.abs(residuals[nonzero_mask] / denominator[nonzero_mask])
            results['smape'] = 100 * weighted_mean(smape_values, sample_weights[nonzero_mask])
        else:
            results['smape'] = np.inf
    if 'mdape' in metrics:
        nonzero_mask = y_true_flat != 0
        if np.any(nonzero_mask):
            abs_percentage_errors = np.abs(residuals[nonzero_mask] / y_true_flat[nonzero_mask])
            sorted_errors = np.sort(abs_percentage_errors)
            weights_sorted = sample_weights[nonzero_mask][np.argsort(abs_percentage_errors)]
            cumulative_weights = np.cumsum(weights_sorted)
            total_weight = np.sum(weights_sorted)
            median_idx = np.searchsorted(cumulative_weights, total_weight / 2)
            results['mdape'] = 100 * sorted_errors[median_idx] if len(sorted_errors) > 0 else 0.0
        else:
            results['mdape'] = np.inf
    if 'r2' in metrics:
        y_true_mean = weighted_mean(y_true_flat, sample_weights)
        total_ss = weighted_mean((y_true_flat - y_true_mean) ** 2, sample_weights)
        residual_ss = weighted_mean(residuals ** 2, sample_weights)
        if total_ss == 0:
            results['r2'] = 0.0 if residual_ss == 0 else -np.inf
        else:
            results['r2'] = 1 - residual_ss / total_ss
    return results