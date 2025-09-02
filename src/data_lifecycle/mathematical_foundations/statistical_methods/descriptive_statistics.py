from typing import Union, Tuple, Optional, Dict
import numpy as np
from general.structures.data_batch import DataBatch


# ...(code omitted)...


def optimize_mean(data: Union[np.ndarray, DataBatch], axis: Optional[int]=None, weights: Optional[np.ndarray]=None) -> Union[float, np.ndarray]:
    """
    Optimize or compute the weighted mean of the provided data.

    This function computes the arithmetic mean of the input data, optionally along a specified axis.
    If weights are provided, it calculates the weighted mean. This is useful in scenarios where
    certain observations contribute more significantly to the central tendency than others.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data as a NumPy array or a DataBatch object.
                                             If DataBatch, uses the `.data` attribute.
        axis (Optional[int]): Axis along which to compute the mean. If None, computes over the flattened array.
        weights (Optional[np.ndarray]): Array of weights associated with the data values.
                                        Must be broadcastable to the shape of data.

    Returns:
        Union[float, np.ndarray]: Optimized or computed mean value(s). A scalar if axis is None,
                                  otherwise an array of means along the specified axis.

    Raises:
        ValueError: If weights do not match the shape of data or if data is empty.
    """
    if isinstance(data, DataBatch):
        data = data.data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.size == 0:
        raise ValueError('Input data is empty')
    if weights is not None:
        weights = np.asarray(weights)
        try:
            np.broadcast_arrays(data, weights)
        except ValueError:
            raise ValueError('Weights array is not broadcastable to the shape of data')
        if np.any(weights < 0):
            raise ValueError('Weights must be non-negative')
        if np.all(weights == 0):
            raise ValueError('At least one weight must be positive')
        weighted_data = data * weights
        sum_weights = np.sum(weights, axis=axis, keepdims=True)
        sum_weighted_data = np.sum(weighted_data, axis=axis, keepdims=True)
        return np.squeeze(sum_weighted_data / sum_weights, axis=axis) if axis is not None else (sum_weighted_data / sum_weights).item()
    else:
        return np.mean(data, axis=axis)

def median_absolute_deviation(data: Union[np.ndarray, DataBatch], axis: Optional[int]=None, center: Optional[Union[float, np.ndarray]]=None) -> Union[float, np.ndarray]:
    """
    Compute the Median Absolute Deviation (MAD) of the input data.

    The Median Absolute Deviation is a robust measure of statistical dispersion,
    defined as the median of the absolute deviations from the dataset's median.
    It is less sensitive to outliers compared to standard deviation.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data as a NumPy array or a DataBatch object.
                                             If DataBatch, uses the `.data` attribute.
        axis (Optional[int]): Axis along which to compute the MAD. If None, computes over the flattened array.
        center (Optional[Union[float, np.ndarray]]): The center point around which to compute deviations.
                                                     If None, uses the median of the data along the specified axis.

    Returns:
        Union[float, np.ndarray]: The computed MAD value(s). A scalar if axis is None,
                                  otherwise an array of MAD values along the specified axis.

    Raises:
        ValueError: If the input data is empty or invalid.
    """
    if isinstance(data, DataBatch):
        data = data.data
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.size == 0:
        raise ValueError('Input data is empty')
    if center is None:
        center = np.median(data, axis=axis, keepdims=True)
    abs_devs = np.abs(data - center)
    return np.median(abs_devs, axis=axis)

def average_absolute_deviation(data: Union[np.ndarray, DataBatch], axis: Optional[int]=None, center: Optional[Union[float, np.ndarray]]=None) -> Union[float, np.ndarray]:
    """
    Compute the Average Absolute Deviation (AAD) of the input data.

    The Average Absolute Deviation is a measure of statistical dispersion,
    defined as the average of the absolute deviations from the dataset's mean or median.
    It provides a robust alternative to standard deviation for assessing variability.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data as a NumPy array or a DataBatch object.
                                             If DataBatch, uses the `.data` attribute.
        axis (Optional[int]): Axis along which to compute the AAD. If None, computes over the flattened array.
        center (Optional[Union[float, np.ndarray]]): The center point around which to compute deviations.
                                                     If None, uses the mean of the data along the specified axis.

    Returns:
        Union[float, np.ndarray]: The computed AAD value(s). A scalar if axis is None,
                                  otherwise an array of AAD values along the specified axis.

    Raises:
        ValueError: If the input data is empty or invalid.
    """
    if isinstance(data, DataBatch):
        data = data.data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.size == 0:
        raise ValueError('Input data is empty')
    if center is None:
        center = np.mean(data, axis=axis, keepdims=True)
    else:
        center = np.asarray(center)
    abs_deviations = np.abs(data - center)
    return np.mean(abs_deviations, axis=axis)

def coefficient_of_variation(data: Union[np.ndarray, DataBatch], axis: Optional[int]=None, ddof: int=0) -> Union[float, np.ndarray]:
    """
    Compute the Coefficient of Variation (CV) of the input data.

    The Coefficient of Variation is a standardized measure of dispersion,
    defined as the ratio of the standard deviation to the mean (expressed as a percentage).
    It is useful for comparing the variability of datasets with different units or scales.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data as a NumPy array or a DataBatch object.
                                             If DataBatch, uses the `.data` attribute.
        axis (Optional[int]): Axis along which to compute the CV. If None, computes over the flattened array.
        ddof (int): Delta degrees of freedom for standard deviation calculation. Defaults to 0.

    Returns:
        Union[float, np.ndarray]: The computed CV value(s). A scalar if axis is None,
                                  otherwise an array of CV values along the specified axis.
                                  Values are expressed as ratios (not percentages).

    Raises:
        ValueError: If the mean is zero (leading to division by zero) or if the input data is empty/invalid.
        ZeroDivisionError: Explicitly raised when the mean is zero.
    """
    if isinstance(data, DataBatch):
        data = data.data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if data.size == 0:
        raise ValueError('Input data is empty')
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=ddof)
    if np.any(mean == 0):
        raise ZeroDivisionError('Mean is zero, cannot compute coefficient of variation')
    cv = std / mean
    return cv


# ...(code omitted)...


def rolling_deviation(data: Union[np.ndarray, DataBatch], window_size: int, axis: Optional[int]=None, ddof: int=0) -> Union[np.ndarray, DataBatch]:
    """
    Compute the rolling (moving) standard deviation over a specified window.

    This function calculates the standard deviation within a sliding window across the data,
    providing insight into local variability and volatility. It is particularly useful for
    time-series analysis or any sequential data where localized patterns are of interest.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data as a NumPy array or a DataBatch object.
                                             If DataBatch, applies rolling deviation to the `.data` attribute.
        window_size (int): Size of the rolling window. Must be positive and <= data length along axis.
        axis (Optional[int]): Axis along which to compute the rolling deviation.
                              If None, flattens the array before computation.
        ddof (int): Delta degrees of freedom for standard deviation calculation. Defaults to 0.

    Returns:
        Union[np.ndarray, DataBatch]: Rolling standard deviation values.
                                      If input was DataBatch, returns a new DataBatch with updated data.
                                      Output shape depends on window size and padding strategy (not padded).

    Raises:
        ValueError: If window_size is invalid (<= 0 or > data length along axis) or if data is empty.
    """
    is_data_batch = isinstance(data, DataBatch)
    if is_data_batch:
        raw_data = data.data
    else:
        raw_data = data
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.asarray(raw_data)
    if raw_data.size == 0:
        raise ValueError('Input data is empty')
    if axis is None:
        processed_data = raw_data.flatten()
        axis = 0
    else:
        processed_data = raw_data
    if axis < 0:
        axis = processed_data.ndim + axis
    if axis >= processed_data.ndim or axis < 0:
        raise ValueError(f'Axis {axis} is out of bounds for array of dimension {processed_data.ndim}')
    data_length = processed_data.shape[axis]
    if window_size <= 0:
        raise ValueError('Window size must be positive')
    if window_size > data_length:
        raise ValueError('Window size cannot be larger than data length along the specified axis')
    processed_data = np.moveaxis(processed_data, axis, 0)
    from numpy.lib.stride_tricks import sliding_window_view
    windowed_data = sliding_window_view(processed_data, window_shape=window_size, axis=0)
    result = np.std(windowed_data, axis=-1, ddof=ddof)
    result = np.moveaxis(result, 0, axis)
    if is_data_batch:
        return DataBatch(data=result, labels=data.labels, metadata=data.metadata.copy() if data.metadata else None, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
    else:
        return result


# ...(code omitted)...


def calculate_r_squared(y_true: Union[np.ndarray, DataBatch], y_pred: Union[np.ndarray, DataBatch]) -> float:
    """
    Calculate the R-squared (coefficient of determination) for predicted values.

    R-squared measures the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s). It provides an indication
    of the goodness of fit of the model, with values closer to 1 indicating a better fit.

    Args:
        y_true (Union[np.ndarray, DataBatch]): True target values as a NumPy array or DataBatch.
                                               If DataBatch, uses the `.labels` attribute.
        y_pred (Union[np.ndarray, DataBatch]): Predicted target values as a NumPy array or DataBatch.
                                               If DataBatch, uses the `.data` attribute.

    Returns:
        float: The R-squared value, ranging from -∞ to 1.
               A value of 1 indicates perfect prediction, 0 indicates the model predicts the mean.

    Raises:
        ValueError: If y_true and y_pred have different shapes or are empty.
    """
    if isinstance(y_true, DataBatch):
        y_true = y_true.labels
    if isinstance(y_pred, DataBatch):
        y_pred = y_pred.data
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError('Input arrays cannot be empty')
    if y_true.shape != y_pred.shape:
        raise ValueError(f'Shape mismatch: y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}')
    y_true_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else float('-inf')
    r_squared = 1 - ss_res / ss_tot
    return float(r_squared)