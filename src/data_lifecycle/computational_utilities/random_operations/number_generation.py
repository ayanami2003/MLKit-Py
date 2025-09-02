import numpy as np
from typing import Union, Optional

def generate_poisson_numbers(rate: Union[float, np.ndarray], size: Optional[int]=None, random_state: Optional[int]=None) -> Union[int, np.ndarray]:
    """
    Generate random numbers from a Poisson distribution with a given rate parameter.

    This function generates random samples from a Poisson distribution, which models the number of events
    occurring within a fixed interval of time or space when these events happen with a known constant mean rate
    and independently of the time since the last event.

    Args:
        rate (Union[float, np.ndarray]): The rate parameter (lambda) of the Poisson distribution.
                                         Must be non-negative. Can be a scalar or array-like.
        size (Optional[int]): The number of samples to generate. If None, a single value is returned
                              if rate is scalar. If rate is an array, size is ignored and an array of
                              the same shape as rate is returned.
        random_state (Optional[int]): Seed for the random number generator for reproducibility.

    Returns:
        Union[int, np.ndarray]: A single integer if size is None and rate is scalar, otherwise
                                an array of integers with the specified size or matching the shape of rate.

    Raises:
        ValueError: If rate contains negative values.
    """
    rate_array = np.asarray(rate)
    if np.any(rate_array < 0):
        raise ValueError('Rate parameter must be non-negative')
    rng = np.random.default_rng(random_state)
    if rate_array.ndim > 0 or (isinstance(rate, (list, np.ndarray)) and np.asarray(rate).size > 1):
        return rng.poisson(rate_array)
    if size is not None:
        return rng.poisson(rate_array, size=size)
    return int(rng.poisson(rate_array))