from typing import Union, Optional, Dict, Any
from general.structures.data_batch import DataBatch
import numpy as np
from typing import Union, Optional, Dict, Any, List
from collections import deque
import math


# ...(code omitted)...


def detect_concept_drift_with_adwin(reference_data: Union[np.ndarray, DataBatch], current_data: Union[np.ndarray, DataBatch], delta: float=0.01, bucket_size: int=5, feature_names: Optional[List[str]]=None) -> Dict[str, Any]:
    """
    Detect concept drift using the ADWIN (Adaptive Windowing) algorithm.

    This function applies the ADWIN algorithm to detect changes in the underlying data distribution
    between a reference dataset and current data stream. ADWIN maintains a sliding window of recent
    data and dynamically adjusts its size to detect changes while providing statistical guarantees.

    ADWIN is particularly effective for detecting gradual and sudden concept drift in streaming data
    scenarios where the data distribution may change over time.

    Args:
        reference_data (Union[np.ndarray, DataBatch]): Baseline/reference data used to establish
            the initial distribution. Can be a numpy array of shape (n_samples, n_features) or
            a DataBatch object containing the reference data.
        current_data (Union[np.ndarray, DataBatch]): Current data stream to monitor for drift.
            Should have the same feature structure as reference_data.
        feature_names (Optional[list]): List of feature names corresponding to columns in the data.
            If None, features will be indexed numerically.
        delta (float): Confidence level for detecting changes. Smaller values make detection more
            conservative (less sensitive). Must be between 0 and 1. Default is 0.01.
        bucket_size (int): Size of buckets used in the ADWIN algorithm. Controls the granularity
            of change detection. Larger values may miss short-term changes but reduce false positives.
            Default is 5.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'drift_detected' (bool): Whether concept drift was detected
            - 'change_points' (list): Indices where drift was detected
            - 'p_values' (list): P-values for each feature at change points
            - 'window_sizes' (list): Sizes of windows at each time step
            - 'feature_contributions' (dict): Contribution of each feature to detected drift

    Raises:
        ValueError: If reference_data and current_data have incompatible shapes or feature counts.
        ValueError: If delta is not between 0 and 1.
    """
    if not 0 < delta < 1:
        raise ValueError('delta must be between 0 and 1')
    if isinstance(reference_data, DataBatch):
        ref_array = reference_data.data
        if feature_names is None and reference_data.feature_names is not None:
            feature_names = reference_data.feature_names
    else:
        ref_array = reference_data
    if isinstance(current_data, DataBatch):
        curr_array = current_data.data
        if feature_names is None and current_data.feature_names is not None:
            feature_names = current_data.feature_names
    else:
        curr_array = current_data
    if not isinstance(ref_array, np.ndarray):
        ref_array = np.asarray(ref_array)
    if not isinstance(curr_array, np.ndarray):
        curr_array = np.asarray(curr_array)
    if ref_array.size == 0 or curr_array.size == 0:
        return {'drift_detected': False, 'change_points': [], 'p_values': [], 'window_sizes': [], 'feature_contributions': {}}
    if ref_array.ndim == 1:
        ref_array = ref_array.reshape(-1, 1)
    if curr_array.ndim == 1:
        curr_array = curr_array.reshape(-1, 1)
    if ref_array.shape[1] != curr_array.shape[1]:
        raise ValueError('Reference and current data must have the same number of features')
    n_features = ref_array.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError('Length of feature_names must match number of features')
    combined_data = np.vstack([ref_array, curr_array])
    detectors = [ADWINDetector(delta=delta, bucket_size=bucket_size) for _ in range(n_features)]
    change_points = []
    p_values = []
    window_sizes = []
    feature_changes = np.zeros(n_features)
    for t in range(len(combined_data)):
        changes_detected_at_t = []
        p_vals_at_t = []
        for f in range(n_features):
            detectors[f].add_element(combined_data[t, f])
            if detectors[f].detected_change():
                changes_detected_at_t.append(f)
                p_vals_at_t.append(delta)
        if changes_detected_at_t:
            change_points.append(t)
            p_values.append(p_vals_at_t)
            for f in changes_detected_at_t:
                feature_changes[f] += 1
        if t > 0:
            window_sizes.append(detectors[0].get_window_size())
    total_changes = np.sum(feature_changes)
    if total_changes > 0:
        feature_contributions = {feature_names[i]: feature_changes[i] / total_changes for i in range(n_features)}
    else:
        feature_contributions = {name: 0.0 for name in feature_names}
    return {'drift_detected': len(change_points) > 0, 'change_points': change_points, 'p_values': p_values, 'window_sizes': window_sizes, 'feature_contributions': feature_contributions}