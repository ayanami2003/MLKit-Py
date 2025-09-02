from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import Union, Dict, Any, Optional
import numpy as np
import inspect

def summarize_data(data: Union[DataBatch, FeatureSet], include_details: bool=True, custom_metrics: Optional[Dict[str, callable]]=None) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the input data, providing key statistics and metadata.

    This function computes essential descriptive statistics for datasets encapsulated in either
    DataBatch or FeatureSet structures. The output includes measures such as count, mean, standard
    deviation, min/max values, and optionally detailed quantiles and custom metric evaluations.

    Args:
        data (Union[DataBatch, FeatureSet]): The input data to summarize. Can be either a DataBatch
                                             containing raw data or a FeatureSet with processed features.
        include_details (bool): If True, includes additional details like quantiles and data types.
                                Defaults to True.
        custom_metrics (Optional[Dict[str, callable]]): A dictionary mapping metric names to functions
                                                        that compute custom summary statistics. Each
                                                        function should accept a 1D array-like input.

    Returns:
        Dict[str, Any]: A dictionary containing the computed summary statistics. Keys represent
                        feature names (if applicable) or general metrics, and values are the
                        corresponding statistical measures or metadata.

    Raises:
        ValueError: If the input data structure is unsupported or malformed.
        TypeError: If custom metric functions do not conform to expected signatures.
    """
    if not isinstance(data, (DataBatch, FeatureSet)):
        raise ValueError('Input data must be either DataBatch or FeatureSet')
    if isinstance(data, DataBatch):
        raw_data = data.data
        feature_names = data.feature_names
        feature_types = None
    else:
        raw_data = data.features
        feature_names = data.feature_names
        feature_types = data.feature_types
    if not isinstance(raw_data, np.ndarray):
        try:
            array_data = np.asarray(raw_data, dtype=float)
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
    if custom_metrics:
        for (name, func) in custom_metrics.items():
            if not callable(func):
                raise TypeError(f"Custom metric '{name}' is not callable")
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if len(params) != 1:
                raise TypeError(f"Custom metric '{name}' must accept exactly one argument (array-like)")
            try:
                test_input = np.array([1, 2, 3])
                result = func(test_input)
                if not np.isscalar(result) and (not (isinstance(result, (np.ndarray, list)) and len(np.atleast_1d(result)) == 1)):
                    raise TypeError(f"Custom metric '{name}' must return a scalar value")
            except Exception as e:
                raise TypeError(f"Custom metric '{name}' failed validity check: {e}")
    results = {}
    n_features = array_data.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError("Number of feature names doesn't match number of features in data")
    for (i, feature_name) in enumerate(feature_names):
        feature_data = array_data[:, i]
        valid_data = feature_data[~np.isnan(feature_data)]
        feature_stats = {'count': len(valid_data), 'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else np.nan, 'std': float(np.std(valid_data, ddof=1)) if len(valid_data) > 1 else np.nan, 'min': float(np.min(valid_data)) if len(valid_data) > 0 else np.nan, 'max': float(np.max(valid_data)) if len(valid_data) > 0 else np.nan}
        if include_details:
            if len(valid_data) > 0:
                quantiles = np.quantile(valid_data, [0.25, 0.5, 0.75])
                feature_stats.update({'0.25': float(quantiles[0]), '0.5': float(quantiles[1]), '0.75': float(quantiles[2])})
            if isinstance(data, FeatureSet):
                if feature_types is not None and i < len(feature_types):
                    feature_stats['dtype'] = feature_types[i]
                else:
                    feature_stats['dtype'] = str(data.features.dtype)
            else:
                feature_stats['dtype'] = str(array_data.dtype)
        if custom_metrics and len(valid_data) > 0:
            for (metric_name, metric_func) in custom_metrics.items():
                try:
                    result = metric_func(valid_data)
                    if isinstance(result, (np.ndarray, list)):
                        result = np.atleast_1d(result).item()
                    feature_stats[metric_name] = float(result) if not np.isnan(result) and (not np.isinf(result)) else result
                except Exception:
                    feature_stats[metric_name] = np.nan
        results[feature_name] = feature_stats
    return results