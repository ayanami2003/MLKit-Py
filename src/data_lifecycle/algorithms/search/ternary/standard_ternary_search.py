from typing import Callable

def ternary_search(func: Callable[[float], float], left: float, right: float, tol: float=1e-09, max_iter: int=1000, find_min: bool=True) -> float:
    """
    Perform a ternary search to find the minimum or maximum of a unimodal function within a given interval.

    Ternary search is a divide-and-conquer optimization technique that works on unimodal functions.
    It divides the search space into three parts and eliminates one-third of the search space at each iteration,
    converging towards the extremum (minimum or maximum) of the function.

    Args:
        func (Callable[[float], float]): The unimodal function to optimize. Must accept a single float argument
                                         and return a float value.
        left (float): The left boundary of the search interval.
        right (float): The right boundary of the search interval.
        tol (float, optional): The tolerance for convergence. The search stops when the interval size is less than
                               or equal to this value. Defaults to 1e-9.
        max_iter (int, optional): The maximum number of iterations allowed. Defaults to 1000.
        find_min (bool, optional): If True, finds the minimum of the function. If False, finds the maximum.
                                   Defaults to True.

    Returns:
        float: The x-coordinate of the extremum (minimum or maximum) of the function within the specified interval.

    Raises:
        ValueError: If left >= right, or if tol <= 0, or if max_iter <= 0.
        RuntimeError: If the search does not converge within the maximum number of iterations.
    """
    if left >= right:
        raise ValueError('Left boundary must be less than right boundary')
    if tol <= 0:
        raise ValueError('Tolerance must be positive')
    if max_iter <= 0:
        raise ValueError('Maximum iterations must be positive')
    for i in range(max_iter):
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        f1 = func(mid1)
        f2 = func(mid2)
        if find_min:
            if f1 < f2:
                right = mid2
            else:
                left = mid1
        elif f1 > f2:
            right = mid2
        else:
            left = mid1
        if right - left <= tol:
            return (left + right) / 2
    raise RuntimeError(f'Ternary search did not converge within {max_iter} iterations')