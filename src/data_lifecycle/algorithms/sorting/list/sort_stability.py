from typing import List, Any, Callable, Optional

def stable_sort(data: List[Any], key_func: Optional[Callable[[Any], Any]]=None, reverse: bool=False) -> List[Any]:
    """
    Sort a list while preserving the relative order of elements that compare as equal (stable sort).
    
    This function performs a stable sort on the input list. Elements that compare as equal according to
    the key function will maintain their original relative ordering in the sorted output. The sort can
    be customized with a key function and direction.
    
    Args:
        data (List[Any]): The list of elements to sort. The list contents can be any comparable type.
        key_func (Optional[Callable[[Any], Any]]): A callable that extracts a comparison key from each element.
                                                  If None, elements are compared directly.
        reverse (bool): If True, sort in descending order. If False (default), sort in ascending order.
        
    Returns:
        List[Any]: A new list containing all elements from the input, sorted in stable order.
                  The original list is not modified.
                  
    Examples:
        >>> stable_sort([3, 1, 2, 1])
        [1, 1, 2, 3]
        
        >>> stable_sort([('A', 2), ('B', 1), ('A', 1)], key_func=lambda x: x[0])
        [('A', 2), ('A', 1), ('B', 1)]
        
        >>> stable_sort([3, 1, 2, 1], reverse=True)
        [3, 2, 1, 1]
    """
    if not isinstance(data, list):
        raise TypeError('Input data must be a list')
    if key_func is not None and (not callable(key_func)):
        raise TypeError('key_func must be callable or None')
    try:
        return sorted(data, key=key_func, reverse=reverse)
    except TypeError as e:
        raise TypeError(f'Unable to sort data: {str(e)}')