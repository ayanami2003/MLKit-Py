import numpy as np
from typing import Union, Optional
from general.structures.data_batch import DataBatch

def compute_population_stability_index(baseline_data: Union[np.ndarray, DataBatch], current_data: Union[np.ndarray, DataBatch], bins: int=10, epsilon: float=1e-10, feature_names: Optional[list]=None) -> dict:
    """
    Compute the Population Stability Index (PSI) between baseline and current data distributions.

    The Population Stability Index measures the shift in distribution of features over time,
    commonly used to detect data drift in machine learning pipelines. PSI values indicate:
    - < 0.1: No significant change
    - 0.1 - 0.2: Moderate change
    - > 0.2: Significant change

    This function computes PSI for each feature separately and returns a dictionary with
    feature names as keys and their respective PSI values.

    Args:
        baseline_data (Union[np.ndarray, DataBatch]): Reference data representing the baseline distribution.
            If DataBatch, uses the 'data' attribute.
        current_data (Union[np.ndarray, DataBatch]): Current data to compare against baseline.
            If DataBatch, uses the 'data' attribute.
        bins (int): Number of bins to use for histogram discretization. Defaults to 10.
        epsilon (float): Small value to avoid division by zero. Defaults to 1e-10.
        feature_names (Optional[list]): List of feature names for labeling results.
            If None and DataBatch provided with feature_names, those will be used.

    Returns:
        dict: Dictionary mapping feature names to their PSI values.

    Raises:
        ValueError: If baseline_data and current_data have different numbers of features.
        ValueError: If bins <= 0.
    """
    if bins <= 0:
        raise ValueError('bins must be positive')
    if isinstance(baseline_data, DataBatch):
        baseline_array = np.asarray(baseline_data.data)
        if feature_names is None:
            feature_names = getattr(baseline_data, 'feature_names', None)
    else:
        baseline_array = np.asarray(baseline_data)
    if isinstance(current_data, DataBatch):
        current_array = np.asarray(current_data.data)
        if feature_names is None:
            feature_names = getattr(current_data, 'feature_names', None)
    else:
        current_array = np.asarray(current_data)
    if baseline_array.ndim == 1:
        baseline_array = baseline_array.reshape(-1, 1)
    if current_array.ndim == 1:
        current_array = current_array.reshape(-1, 1)
    if baseline_array.ndim != 2 or current_array.ndim != 2:
        raise ValueError('Input data must be 1D or 2D arrays')
    n_baseline_features = baseline_array.shape[1]
    n_current_features = current_array.shape[1]
    if n_baseline_features != n_current_features:
        raise ValueError(f'Baseline data has {n_baseline_features} features, but current data has {n_current_features} features')
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_baseline_features)]
    elif len(feature_names) != n_baseline_features:
        raise ValueError(f'Length of feature_names ({len(feature_names)}) does not match number of features ({n_baseline_features})')
    psi_results = {}
    for (i, feature_name) in enumerate(feature_names):
        baseline_feature = baseline_array[:, i]
        current_feature = current_array[:, i]
        baseline_feature = baseline_feature[~np.isnan(baseline_feature)]
        current_feature = current_feature[~np.isnan(current_feature)]
        if len(baseline_feature) == 0 or len(current_feature) == 0:
            raise ValueError(f"No valid data points found for feature '{feature_name}' after removing NaNs")
        combined_data = np.concatenate([baseline_feature, current_feature])
        bin_edges = np.histogram_bin_edges(combined_data, bins=bins)
        (baseline_hist, _) = np.histogram(baseline_feature, bins=bin_edges)
        (current_hist, _) = np.histogram(current_feature, bins=bin_edges)
        baseline_pct = baseline_hist / len(baseline_feature)
        current_pct = current_hist / len(current_feature)
        baseline_pct = np.where(baseline_pct == 0, epsilon, baseline_pct)
        current_pct = np.where(current_pct == 0, epsilon, current_pct)
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        psi_results[feature_name] = psi
    return psi_results