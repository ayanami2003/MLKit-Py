import numpy as np
from general.structures.data_batch import DataBatch
from typing import Union, Optional, Tuple
from scipy.stats import ks_2samp
from typing import Union, Optional

def perform_kolmogorov_smirnov_test_for_drift(reference_data: Union[np.ndarray, DataBatch], current_data: Union[np.ndarray, DataBatch], feature_names: Optional[list]=None, alpha: float=0.05) -> dict:
    """
    Perform the Kolmogorov-Smirnov test to detect distribution drift between two datasets.

    This function compares the empirical cumulative distribution functions of reference and current data
    for specified features using the two-sample Kolmogorov-Smirnov test. It determines whether a statistically
    significant distribution shift has occurred based on the provided significance level.

    Args:
        reference_data (Union[np.ndarray, DataBatch]): The reference/baseline dataset representing the expected distribution.
            If DataBatch, uses the `.data` attribute. Should be 1D (single feature) or 2D (multiple features).
        current_data (Union[np.ndarray, DataBatch]): The current dataset to compare against the reference.
            If DataBatch, uses the `.data` attribute. Must have the same number of features as reference_data.
        feature_names (Optional[list]): Optional list of feature names corresponding to columns in the data.
            If None and DataBatch is provided, will attempt to use DataBatch.feature_names.
        alpha (float): Significance level for the test (default: 0.05). Features with p-values less than alpha
            are considered to have significant distribution drift.

    Returns:
        dict: A dictionary containing:
            - 'statistics': dict mapping feature names to KS test statistics (D values)
            - 'p_values': dict mapping feature names to p-values
            - 'drift_detected': dict mapping feature names to boolean drift indicators
            - 'significant_drift_features': list of feature names where drift was detected
            - 'alpha': significance level used

    Raises:
        ValueError: If data dimensions don't match or if inputs are invalid.
        TypeError: If inputs are not numpy arrays or DataBatch objects.
    """
    if isinstance(reference_data, DataBatch):
        ref_array = reference_data.data
        if feature_names is None:
            feature_names = getattr(reference_data, 'feature_names', None)
    elif isinstance(reference_data, np.ndarray):
        ref_array = reference_data
    else:
        raise TypeError('reference_data must be either a numpy array or a DataBatch object.')
    if isinstance(current_data, DataBatch):
        curr_array = current_data.data
        if feature_names is None:
            feature_names = getattr(current_data, 'feature_names', None)
    elif isinstance(current_data, np.ndarray):
        curr_array = current_data
    else:
        raise TypeError('current_data must be either a numpy array or a DataBatch object.')
    if ref_array.ndim == 1:
        ref_array = ref_array.reshape(-1, 1)
    if curr_array.ndim == 1:
        curr_array = curr_array.reshape(-1, 1)
    if ref_array.ndim != 2 or curr_array.ndim != 2:
        raise ValueError('Input data must be 1D or 2D arrays.')
    n_ref_features = ref_array.shape[1]
    n_curr_features = curr_array.shape[1]
    if n_ref_features != n_curr_features:
        raise ValueError(f'Number of features in reference ({n_ref_features}) and current ({n_curr_features}) data must match.')
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_ref_features)]
    elif len(feature_names) != n_ref_features:
        raise ValueError(f'Length of feature_names ({len(feature_names)}) does not match number of features ({n_ref_features}).')
    statistics = {}
    p_values = {}
    drift_detected = {}
    significant_drift_features = []
    for (i, feature_name) in enumerate(feature_names):
        ref_feature_data = ref_array[:, i]
        curr_feature_data = curr_array[:, i]
        ref_feature_data = ref_feature_data[~np.isnan(ref_feature_data)]
        curr_feature_data = curr_feature_data[~np.isnan(curr_feature_data)]
        if len(ref_feature_data) == 0 or len(curr_feature_data) == 0:
            raise ValueError(f"No valid data points found for feature '{feature_name}' after removing NaNs.")
        (ks_statistic, p_value) = ks_2samp(ref_feature_data, curr_feature_data)
        statistics[feature_name] = ks_statistic
        p_values[feature_name] = p_value
        is_drift = p_value < alpha
        drift_detected[feature_name] = is_drift
        if is_drift:
            significant_drift_features.append(feature_name)
    return {'statistics': statistics, 'p_values': p_values, 'drift_detected': drift_detected, 'significant_drift_features': significant_drift_features, 'alpha': alpha}