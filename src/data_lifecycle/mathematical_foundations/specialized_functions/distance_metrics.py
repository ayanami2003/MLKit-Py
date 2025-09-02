import numpy as np
from typing import Union

def manhattan_distance(a: Union[np.ndarray, list], b: Union[np.ndarray, list]) -> float:
    """
    Compute the Manhattan distance (L1 norm) between two vectors.

    Manhattan distance is the sum of the absolute differences of their Cartesian coordinates.
    It is commonly used in grid-based pathfinding and compressed sensing.

    Args:
        a (Union[np.ndarray, list]): First input vector.
        b (Union[np.ndarray, list]): Second input vector.

    Returns:
        float: Manhattan distance between the two vectors.

    Raises:
        ValueError: If the input vectors have different shapes.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('Input vectors must have the same shape')
    return np.sum(np.abs(a - b))

def cosine_similarity(a: Union[np.ndarray, list], b: Union[np.ndarray, list]) -> float:
    """
    Compute the cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two non-zero vectors,
    providing a value between -1 and 1 where 1 indicates identical direction,
    0 indicates orthogonality, and -1 indicates opposite directions.

    Args:
        a (Union[np.ndarray, list]): First input vector.
        b (Union[np.ndarray, list]): Second input vector.

    Returns:
        float: Cosine similarity between the two vectors.

    Raises:
        ValueError: If the input vectors have different shapes or are zero vectors.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('Input vectors must have the same shape')
    if np.all(a == 0) or np.all(b == 0):
        raise ValueError('Input vectors must not be zero vectors')
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError('Input vectors must not be zero vectors')
    return float(dot_product / (norm_a * norm_b))

def mahalanobis_distance(x: Union[np.ndarray, list], mean: Union[np.ndarray, list], cov_inv: np.ndarray) -> float:
    """
    Compute the Mahalanobis distance between a point and a distribution.

    The Mahalanobis distance measures the distance between a point and a distribution,
    taking into account the covariance structure of the data. It is unitless and
    scale-invariant, making it useful for detecting outliers in multivariate data.

    Args:
        x (Union[np.ndarray, list]): Input vector (point) of shape (n_features,).
        mean (Union[np.ndarray, list]): Mean vector of the distribution of shape (n_features,).
        cov_inv (np.ndarray): Inverse of the covariance matrix of the distribution of shape (n_features, n_features).

    Returns:
        float: Mahalanobis distance between the point and the distribution.

    Raises:
        ValueError: If dimensions of inputs do not align properly.
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Input vector 'x' must be 1-dimensional, got {x.ndim} dimensions")
    if mean.ndim != 1:
        raise ValueError(f"Mean vector 'mean' must be 1-dimensional, got {mean.ndim} dimensions")
    if x.shape[0] != mean.shape[0]:
        raise ValueError(f"Dimensions of 'x' ({x.shape[0]}) and 'mean' ({mean.shape[0]}) must match")
    if cov_inv.ndim != 2:
        raise ValueError(f"Covariance inverse matrix 'cov_inv' must be 2-dimensional, got {cov_inv.ndim} dimensions")
    if cov_inv.shape[0] != cov_inv.shape[1]:
        raise ValueError(f"Covariance inverse matrix 'cov_inv' must be square, got shape {cov_inv.shape}")
    if cov_inv.shape[0] != x.shape[0]:
        raise ValueError(f"Dimensions of 'cov_inv' ({cov_inv.shape[0]}) must match dimensions of 'x' ({x.shape[0]})")
    diff = x - mean
    dist_squared = np.einsum('i,ij,j', diff, cov_inv, diff)
    return float(np.sqrt(dist_squared))