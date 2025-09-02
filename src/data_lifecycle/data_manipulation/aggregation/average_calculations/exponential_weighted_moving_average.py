import numpy as np
from typing import Union, Optional
from general.structures.data_batch import DataBatch

def exponential_weighted_moving_average(data: Union[np.ndarray, DataBatch], alpha: float=0.3, axis: Optional[int]=None, adjust: bool=True) -> Union[np.ndarray, DataBatch]:
    """
    Calculate the exponentially weighted moving average (EWMA) of the input data.

    This function computes the EWMA along a specified axis of a numpy array or 
    DataBatch object. The EWMA assigns exponentially decreasing weights to past 
    observations, making it sensitive to recent changes in the data.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data as a numpy array or DataBatch.
                                            If DataBatch, the EWMA is computed on the
                                            `.data` attribute.
        alpha (float): Smoothing factor (0 < alpha <= 1). Higher values give more
                       weight to recent observations. Defaults to 0.3.
        axis (Optional[int]): Axis along which to compute the EWMA. If None, the
                              EWMA is computed over the flattened array. Defaults to None.
        adjust (bool): Whether to apply adjustment to the weights. If True, uses
                       weights that sum to 1 for finite windows. Defaults to True.

    Returns:
        Union[np.ndarray, DataBatch]: Exponentially weighted moving average of the input data.
                                    If input was a DataBatch, returns a DataBatch with the
                                    same metadata but updated data.

    Raises:
        ValueError: If alpha is not in the range (0, 1].
        TypeError: If data is neither a numpy array nor a DataBatch.
    """
    if not 0 < alpha <= 1:
        raise ValueError('Alpha must be in the range (0, 1]')
    if isinstance(data, DataBatch):
        input_array = np.asarray(data.data)
        is_databatch = True
    elif isinstance(data, np.ndarray):
        input_array = data
        is_databatch = False
    else:
        raise TypeError('Data must be either a numpy array or a DataBatch')
    original_shape = None
    if axis is None:
        original_shape = input_array.shape
        input_array = input_array.flatten()
        axis = -1
    if input_array.size == 1:
        result = input_array.copy()
    else:
        result = np.zeros_like(input_array, dtype=float)
        if axis == -1:
            result[0] = input_array[0]
            if adjust:
                weighted_sum = input_array[0]
                sum_weights = 1.0
                for i in range(1, len(input_array)):
                    weighted_sum = (1 - alpha) * weighted_sum + alpha * input_array[i]
                    sum_weights = (1 - alpha) * sum_weights + alpha
                    result[i] = weighted_sum / sum_weights
            else:
                for i in range(1, len(input_array)):
                    result[i] = alpha * input_array[i] + (1 - alpha) * result[i - 1]
        else:
            n = input_array.shape[axis]
            slc_first = [slice(None)] * input_array.ndim
            slc_first[axis] = 0
            result[tuple(slc_first)] = input_array[tuple(slc_first)]
            if adjust:
                for i in range(1, n):
                    slc_prev = [slice(None)] * input_array.ndim
                    slc_prev[axis] = i - 1
                    slc_curr = [slice(None)] * input_array.ndim
                    slc_curr[axis] = i
                    prev_result = result[tuple(slc_prev)]
                    curr_input = input_array[tuple(slc_curr)]
                    weighted_sum = (1 - alpha) * prev_result + alpha * curr_input
                    sum_weights = 1 - (1 - alpha) ** (i + 1)
                    result[tuple(slc_curr)] = weighted_sum / sum_weights
            else:
                for i in range(1, n):
                    slc_prev = [slice(None)] * input_array.ndim
                    slc_prev[axis] = i - 1
                    slc_curr = [slice(None)] * input_array.ndim
                    slc_curr[axis] = i
                    prev_result = result[tuple(slc_prev)]
                    curr_input = input_array[tuple(slc_curr)]
                    result[tuple(slc_curr)] = alpha * curr_input + (1 - alpha) * prev_result
    if original_shape is not None:
        result = result.reshape(original_shape)
    if is_databatch:
        return DataBatch(data=result, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
    else:
        return result