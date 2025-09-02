from typing import List, TypeVar, Callable, Optional
T = TypeVar('T')

def binary_search(arr: List[T], target: T, key: Optional[Callable[[T], T]]=None, left: int=0, right: Optional[int]=None) -> int:
    """
    Perform binary search on a sorted array to find the index of a target element.

    This function implements the classic binary search algorithm to efficiently locate
    a target value in a sorted array. It supports custom key functions for comparing
    elements and allows searching within a subarray.

    Args:
        arr (List[T]): The sorted list to search in. Must be sorted in ascending order
            according to the key function (if provided) or natural ordering.
        target (T): The element to search for.
        key (Optional[Callable[[T], T]]): Optional function to extract a comparison key
            from each element. Defaults to None (identity function).
        left (int): Starting index of the subarray to search. Defaults to 0.
        right (Optional[int]): Ending index of the subarray to search. Defaults to
            len(arr) - 1 if not specified.

    Returns:
        int: The index of the target element if found, otherwise -1.

    Raises:
        ValueError: If the array is empty or the specified subarray indices are invalid.
    """
    if not arr:
        raise ValueError('Array cannot be empty')
    if right is None:
        right = len(arr) - 1
    if left < 0 or right >= len(arr) or left > right:
        raise ValueError('Invalid subarray indices')
    if key is None:
        key = lambda x: x
    target_key = key(target)
    while left <= right:
        mid = left + (right - left) // 2
        mid_key = key(arr[mid])
        if mid_key == target_key:
            return mid
        elif mid_key < target_key:
            left = mid + 1
        else:
            right = mid - 1
    return -1