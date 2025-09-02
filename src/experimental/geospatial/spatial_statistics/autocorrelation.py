import numpy as np
from typing import Union, Optional
from general.structures.data_batch import DataBatch

def compute_morans_i_statistic(values: Union[np.ndarray, DataBatch], coordinates: Union[np.ndarray, DataBatch], weights: Optional[Union[np.ndarray, str]]='inverse_distance', standardize: bool=True, permutations: int=999, random_state: Optional[int]=None) -> dict:
    """
    Compute Moran's I statistic for spatial autocorrelation analysis.
    
    Moran's I is a measure of spatial autocorrelation that evaluates whether nearby locations
    have similar values. Values close to +1 indicate strong positive spatial autocorrelation,
    values close to -1 indicate strong negative spatial autocorrelation, and values around 0
    suggest randomness.
    
    Args:
        values (Union[np.ndarray, DataBatch]): Array of numeric values for which to compute spatial autocorrelation.
            If DataBatch, uses the primary data attribute.
        coordinates (Union[np.ndarray, DataBatch]): Array of shape (n, 2) containing spatial coordinates [x, y] or [lat, lon]
            for each observation. If DataBatch, uses the primary data attribute.
        weights (Optional[Union[np.ndarray, str]]): Spatial weights matrix or method to construct one.
            If str, supports "inverse_distance" or "binary". If ndarray, must be of shape (n, n).
            Defaults to "inverse_distance".
        standardize (bool): Whether to standardize the values by subtracting the mean.
            Defaults to True.
        permutations (int): Number of permutations for significance testing.
            Defaults to 999.
        random_state (Optional[int]): Random seed for reproducibility of permutations.
            Defaults to None.
            
    Returns:
        dict: Dictionary containing:
            - "statistic" (float): The computed Moran's I value.
            - "expected" (float): Expected value under null hypothesis.
            - "p_value" (float): P-value from permutation test.
            - "z_score" (float): Standardized test statistic.
            
    Raises:
        ValueError: If input arrays have mismatched dimensions or invalid weight specifications.
        TypeError: If inputs are not of supported types.
    """
    if isinstance(values, DataBatch):
        values = values.data
    if isinstance(coordinates, DataBatch):
        coordinates = coordinates.data
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim != 2 or values.shape[1] != 1:
        raise ValueError('Values must be a 1D array or a 2D array with a single column')
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError('Coordinates must be a 2D array with shape (n, 2)')
    if values.shape[0] != coordinates.shape[0]:
        raise ValueError('Number of values must match number of coordinates')
    n = values.shape[0]
    if n < 2:
        raise ValueError('At least two observations are required')
    values = values.flatten()
    if isinstance(weights, str):
        if weights == 'inverse_distance':
            diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(distances, np.inf)
            weights_matrix = 1.0 / distances
            np.fill_diagonal(weights_matrix, 0.0)
        elif weights == 'binary':
            diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            weights_matrix = np.zeros((n, n))
            for i in range(n):
                nearest_idx = np.argmin(distances[i] + np.where(np.arange(n) == i, np.inf, 0))
                weights_matrix[i, nearest_idx] = 1
                weights_matrix[nearest_idx, i] = 1
        else:
            raise ValueError("Unsupported weights method. Choose 'inverse_distance' or 'binary'")
    elif isinstance(weights, np.ndarray):
        if weights.shape != (n, n):
            raise ValueError('Weights matrix must have shape (n, n) where n is the number of observations')
        weights_matrix = weights
    else:
        raise TypeError("Weights must be either a string ('inverse_distance', 'binary') or a numpy array")
    if standardize:
        values = values - np.mean(values)
    numerator = np.sum(weights_matrix * np.outer(values, values))
    denominator = np.sum(values ** 2)
    W_sum = np.sum(weights_matrix)
    if denominator == 0 or W_sum == 0:
        morans_i = 0.0
    else:
        morans_i = n / W_sum * (numerator / denominator)
    expected = -1.0 / (n - 1)
    rng = np.random.default_rng(random_state)
    permuted_stats = np.zeros(permutations)
    for i in range(permutations):
        shuffled_values = rng.permutation(values)
        perm_num = np.sum(weights_matrix * np.outer(shuffled_values, shuffled_values))
        if denominator != 0 and W_sum != 0:
            permuted_stats[i] = n / W_sum * (perm_num / denominator)
        else:
            permuted_stats[i] = 0.0
    extreme_count = np.sum(np.abs(permuted_stats) >= np.abs(morans_i))
    p_value = (extreme_count + 1) / (permutations + 1)
    if permutations > 1:
        perm_mean = np.mean(permuted_stats)
        perm_std = np.std(permuted_stats)
        if perm_std > 0:
            z_score = (morans_i - perm_mean) / perm_std
        else:
            z_score = 0.0
    else:
        z_score = 0.0
    return {'statistic': float(morans_i), 'expected': float(expected), 'p_value': float(p_value), 'z_score': float(z_score)}