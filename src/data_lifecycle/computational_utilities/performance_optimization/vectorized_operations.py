import numpy as np
from typing import Any, Callable, Union

def apply_vectorized_operation(data: Union[np.ndarray, list], operation: Callable[[np.ndarray], np.ndarray]) -> Union[np.ndarray, list]:
    """
    Apply a vectorized operation to the input data for performance optimization.

    This function takes an array-like data structure and applies a user-defined vectorized
    operation to it. The operation should be designed to work efficiently with NumPy arrays
    to achieve performance benefits. If the input is a list, it will be converted to a NumPy
    array for processing and then converted back to a list in the result.

    Args:
        data (Union[np.ndarray, list]): The input data to transform. Can be a NumPy array or list.
        operation (Callable[[np.ndarray], np.ndarray]): A vectorized function that takes a NumPy array
            as input and returns a transformed NumPy array.

    Returns:
        Union[np.ndarray, list]: The transformed data in the same format as the input.

    Raises:
        TypeError: If the operation is not callable or if the data is not a supported type.
        ValueError: If the operation does not return an array of the same shape as the input.
    """
    if not callable(operation):
        raise TypeError('The operation must be a callable function.')
    if isinstance(data, list):
        original_is_list = True
        try:
            np_data = np.array(data)
        except Exception as e:
            raise TypeError('Input data is a list but could not be converted to a NumPy array.') from e
    elif isinstance(data, np.ndarray):
        original_is_list = False
        np_data = data
    else:
        raise TypeError('Input data must be either a NumPy array or a list.')
    try:
        result_array = operation(np_data)
    except Exception as e:
        raise TypeError('The provided operation failed when applied to the NumPy array.') from e
    if not isinstance(result_array, np.ndarray):
        raise ValueError('The operation must return a NumPy array.')
    if result_array.shape != np_data.shape:
        raise ValueError('The operation must preserve the shape of the input data.')
    if original_is_list:
        return result_array.tolist()
    else:
        return result_array