import numpy as np
from typing import Union, Optional
from general.structures.data_batch import DataBatch

def moving_window(data: Union[np.ndarray, DataBatch], window_size: int, step_size: int=1, axis: Optional[int]=None) -> Union[np.ndarray, DataBatch]:
    """
    Apply a moving window operation over the input data to create overlapping subsequences.
    
    This function generates a sequence of overlapping windows from the input data. Each window
    contains `window_size` consecutive elements, and windows are generated with a specified
    step size. The operation can be applied along any axis of multidimensional data.
    
    Args:
        data (Union[np.ndarray, DataBatch]): Input data array or batch to apply the moving window to.
        window_size (int): Size of each window (number of elements per window).
        step_size (int): Step size between consecutive windows. Defaults to 1.
        axis (Optional[int]): Axis along which to apply the moving window. If None, applies to the
                             first axis (0). For DataBatch objects, applies to the data attribute.
                             
    Returns:
        Union[np.ndarray, DataBatch]: Array or DataBatch containing the windowed data. For arrays,
                                    the result will have shape [..., num_windows, window_size] where
                                    the windowed dimension replaces the original dimension. For
                                    DataBatch objects, the data attribute will contain the windowed
                                    data while other attributes remain unchanged.
                                    
    Raises:
        ValueError: If window_size is larger than the data dimension, or if step_size is not positive.
        TypeError: If data is not a supported type.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError('window_size must be a positive integer')
    if not isinstance(step_size, int) or step_size <= 0:
        raise ValueError('step_size must be a positive integer')
    if isinstance(data, DataBatch):
        if not isinstance(data.data, np.ndarray):
            raise TypeError('DataBatch data must be a numpy array for moving window operation')
        axis = axis if axis is not None else 0
        if axis < 0:
            axis += data.data.ndim
        if axis < 0 or axis >= data.data.ndim:
            raise ValueError(f'axis {axis} is out of bounds for array of dimension {data.data.ndim}')
        if window_size > data.data.shape[axis]:
            raise ValueError(f'window_size ({window_size}) cannot be larger than data dimension ({data.data.shape[axis]}) along axis {axis}')
        result_data = _apply_sliding_window(data.data, window_size, step_size, axis)
        return DataBatch(data=result_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
    elif isinstance(data, np.ndarray):
        axis = axis if axis is not None else 0
        if axis < 0:
            axis += data.ndim
        if axis < 0 or axis >= data.ndim:
            raise ValueError(f'axis {axis} is out of bounds for array of dimension {data.ndim}')
        if window_size > data.shape[axis]:
            raise ValueError(f'window_size ({window_size}) cannot be larger than data dimension ({data.shape[axis]}) along axis {axis}')
        return _apply_sliding_window(data, window_size, step_size, axis)
    else:
        raise TypeError(f'data must be a numpy array or DataBatch, got {type(data)}')