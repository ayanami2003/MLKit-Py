from typing import List, Callable, Optional, Any

def custom_sort(data: List[Any], key_func: Optional[Callable[[Any], Any]]=None, reverse: bool=False) -> List[Any]:
    """
    Sort a list of elements using a custom key function and order specification.

    This function provides flexible sorting capabilities for lists of any data type.
    Users can specify a custom key function to determine sort order and choose between
    ascending and descending order. The function maintains stability for equal elements.

    Args:
        data (List[Any]): The list of elements to sort. Can contain any data type.
        key_func (Optional[Callable[[Any], Any]]): A function that takes an element and returns
            a value to sort by. If None, elements are sorted by their natural order.
        reverse (bool): If True, sort in descending order. If False (default), sort in ascending order.

    Returns:
        List[Any]: A new list containing the sorted elements.

    Raises:
        TypeError: If elements cannot be compared or if key_func is not callable.
    """
    if key_func is not None and (not callable(key_func)):
        raise TypeError('key_func must be callable')
    try:
        if key_func is None:
            return sorted(data, reverse=reverse)
        else:
            return sorted(data, key=key_func, reverse=reverse)
    except TypeError as e:
        raise TypeError(f'Unable to sort data: {str(e)}')