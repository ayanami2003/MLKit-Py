from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import Union, Optional, Dict, Any
import numpy as np

def compute_summary_statistics(data: Union[DataBatch, FeatureSet], include_quantiles: bool=True, quantile_levels: Optional[list]=None, custom_functions: Optional[Dict[str, callable]]=None) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics for numerical data.

    This function calculates common statistical measures such as mean, median, standard deviation,
    variance, min, max, and optionally quantiles for each numerical feature in the input data.
    It supports both DataBatch and FeatureSet input formats and allows customization of computed
    statistics through additional functions.

    Args:
        data (Union[DataBatch, FeatureSet]): Input data containing numerical features.
        include_quantiles (bool): Whether to include quantile calculations. Defaults to True.
        quantile_levels (Optional[list]): Specific quantile levels to compute if include_quantiles is True.
                                          Defaults to [0.25, 0.5, 0.75] if None.
        custom_functions (Optional[Dict[str, callable]]): Dictionary mapping names to custom
                                                          statistical functions to apply to each feature.

    Returns:
        Dict[str, Any]: A dictionary containing computed statistics for each feature, including:
                        - count: Number of non-null observations
                        - mean: Average value
                        - std: Standard deviation
                        - min: Minimum value
                        - max: Maximum value
                        - quantiles (optional): Specified quantile values
                        - custom metrics (optional): Results from custom functions

    Raises:
        ValueError: If the input data format is unsupported or contains non-numerical features
                    that cannot be processed.
        TypeError: If custom_functions contains non-callable values.
    """
    if not isinstance(data, (DataBatch, FeatureSet)):
        raise ValueError('Input data must be either DataBatch or FeatureSet')
    if isinstance(data, DataBatch):
        raw_data = data.data
        feature_names = data.feature_names
    else:
        raw_data = data.features
        feature_names = data.feature_names
    if not isinstance(raw_data, np.ndarray):
        try:
            array_data = np.array(raw_data, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f'Failed to convert data to numerical array: {e}')
    elif not np.issubdtype(raw_data.dtype, np.number):
        try:
            array_data = raw_data.astype(float)
        except (ValueError, TypeError) as e:
            raise ValueError(f'Data contains non-numerical values that cannot be converted: {e}')
    else:
        array_data = raw_data
    if array_data.ndim == 1:
        array_data = array_data.reshape(-1, 1)
    elif array_data.ndim != 2:
        raise ValueError('Data must be 1D or 2D')
    if custom_functions:
        for (name, func) in custom_functions.items():
            if not callable(func):
                raise TypeError(f"Custom function '{name}' is not callable")
    if include_quantiles and quantile_levels is None:
        quantile_levels = [0.25, 0.5, 0.75]
    results = {}
    n_features = array_data.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError("Number of feature names doesn't match number of features in data")
    for (i, feature_name) in enumerate(feature_names):
        feature_data = array_data[:, i]
        valid_data = feature_data[~np.isnan(feature_data)]
        feature_stats = {'count': len(valid_data), 'mean': np.mean(valid_data) if len(valid_data) > 0 else np.nan, 'std': np.std(valid_data, ddof=1) if len(valid_data) > 1 else np.nan, 'min': np.min(valid_data) if len(valid_data) > 0 else np.nan, 'max': np.max(valid_data) if len(valid_data) > 0 else np.nan}
        if include_quantiles and len(valid_data) > 0:
            quantiles = np.quantile(valid_data, quantile_levels)
            for (j, q) in enumerate(quantile_levels):
                feature_stats[str(q)] = quantiles[j]
        if custom_functions and len(valid_data) > 0:
            for (func_name, func) in custom_functions.items():
                try:
                    feature_stats[func_name] = func(valid_data)
                except Exception as e:
                    feature_stats[func_name] = np.nan
        results[feature_name] = feature_stats
    return results