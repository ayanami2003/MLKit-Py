from typing import Union, Dict, Any, Optional
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Union, Dict, Any, Optional, List
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

def validate_synthetic_data_quality(real_data: Union[np.ndarray, DataBatch, FeatureSet], synthetic_data: Union[np.ndarray, DataBatch, FeatureSet], metrics: Optional[list]=None, threshold: float=0.1) -> Dict[str, Any]:
    """
    Validate the quality of synthetic data by comparing it with real data across multiple statistical metrics.

    This function computes various statistical measures to assess how well the synthetic data preserves
    the statistical properties of the real data. It supports multiple comparison metrics and allows
    setting a quality threshold for validation decisions.

    Args:
        real_data (Union[np.ndarray, DataBatch, FeatureSet]): The original real data used as reference.
        synthetic_data (Union[np.ndarray, DataBatch, FeatureSet]): The synthetic data generated to be validated.
        metrics (Optional[list]): List of specific metrics to compute. If None, uses a default set of metrics.
                                Supported metrics include 'ks_test', 'correlation_diff', 'mean_diff',
                                'std_diff', 'pairwise_dist', and 'coverage'.
        threshold (float): Quality threshold for validation. Metrics exceeding this threshold will be flagged
                           as failing quality checks. Defaults to 0.1.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'overall_quality' (bool): Whether the synthetic data passes all quality checks.
            - 'metric_scores' (Dict[str, float]): Individual metric scores.
            - 'failed_metrics' (list): List of metrics that failed the quality threshold.
            - 'details' (Dict[str, Any]): Additional details for each metric computed.

    Raises:
        ValueError: If real_data and synthetic_data have incompatible shapes or types.
        ValueError: If an unsupported metric is specified in the metrics list.
    """

    def _extract_array(data):
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, DataBatch):
            return data.data
        elif isinstance(data, FeatureSet):
            return data.features
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')

    def _compute_ks_test(real_array, synthetic_array):
        scores = []
        details = {}
        for i in range(real_array.shape[1]):
            try:
                (statistic, p_value) = ks_2samp(real_array[:, i], synthetic_array[:, i])
                scores.append(statistic)
                details[f'feature_{i}'] = {'statistic': statistic, 'p_value': p_value}
            except Exception as e:
                scores.append(1.0)
                details[f'feature_{i}'] = {'error': str(e)}
        return (np.mean(scores), details)

    def _compute_correlation_diff(real_array, synthetic_array):
        real_corr = np.corrcoef(real_array.T)
        synth_corr = np.corrcoef(synthetic_array.T)
        diff = np.abs(real_corr - synth_corr)
        return (np.mean(diff), {'correlation_difference_matrix': diff.tolist()})

    def _compute_mean_diff(real_array, synthetic_array):
        real_mean = np.mean(real_array, axis=0)
        synth_mean = np.mean(synthetic_array, axis=0)
        diff = np.abs(real_mean - synth_mean)
        return (np.mean(diff), {'mean_differences': diff.tolist(), 'real_means': real_mean.tolist(), 'synth_means': synth_mean.tolist()})

    def _compute_std_diff(real_array, synthetic_array):
        real_std = np.std(real_array, axis=0)
        synth_std = np.std(synthetic_array, axis=0)
        diff = np.abs(real_std - synth_std)
        return (np.mean(diff), {'std_differences': diff.tolist(), 'real_stds': real_std.tolist(), 'synth_stds': synth_std.tolist()})

    def _compute_pairwise_dist(real_array, synthetic_array):
        max_samples = min(1000, real_array.shape[0], synthetic_array.shape[0])
        real_sample_idx = np.random.choice(real_array.shape[0], min(max_samples, real_array.shape[0]), replace=False)
        synth_sample_idx = np.random.choice(synthetic_array.shape[0], min(max_samples, synthetic_array.shape[0]), replace=False)
        real_sample = real_array[real_sample_idx]
        synth_sample = synthetic_array[synth_sample_idx]
        distances_real = cdist(real_sample[:min(100, len(real_sample))], real_sample[:min(100, len(real_sample))])
        distances_synth = cdist(synth_sample[:min(100, len(synth_sample))], synth_sample[:min(100, len(synth_sample))])
        try:
            (statistic, p_value) = ks_2samp(distances_real.flatten(), distances_synth.flatten())
            return (statistic, {'distance_ks_statistic': statistic, 'distance_p_value': p_value})
        except Exception as e:
            return (1.0, {'error': str(e)})

    def _compute_coverage(real_array, synthetic_array):
        coverage_scores = []
        details = {}
        for i in range(real_array.shape[1]):
            real_min = np.min(real_array[:, i])
            real_max = np.max(real_array[:, i])
            if real_min == real_max:
                in_range = synthetic_array[:, i] == real_min
                coverage = np.mean(in_range)
            else:
                in_range = (synthetic_array[:, i] >= real_min) & (synthetic_array[:, i] <= real_max)
                coverage = np.mean(in_range)
            coverage_scores.append(coverage)
            details[f'feature_{i}'] = {'coverage': coverage, 'real_min': float(real_min), 'real_max': float(real_max), 'synth_min': float(np.min(synthetic_array[:, i])), 'synth_max': float(np.max(synthetic_array[:, i]))}
        avg_coverage = np.mean(coverage_scores)
        details['average_coverage'] = float(avg_coverage)
        return (avg_coverage, details)
    real_array = _extract_array(real_data)
    synthetic_array = _extract_array(synthetic_data)
    if not isinstance(real_array, np.ndarray):
        real_array = np.array(real_array)
    if not isinstance(synthetic_array, np.ndarray):
        synthetic_array = np.array(synthetic_array)
    if real_array.size == 0 or synthetic_array.size == 0:
        raise ValueError('Input data cannot be empty')
    if real_array.ndim == 1:
        real_array = real_array.reshape(-1, 1)
    if synthetic_array.ndim == 1:
        synthetic_array = synthetic_array.reshape(-1, 1)
    if real_array.ndim != 2 or synthetic_array.ndim != 2:
        raise ValueError('Input data must be 1 or 2-dimensional')
    if real_array.shape[1] != synthetic_array.shape[1]:
        raise ValueError('Real and synthetic data must have the same number of features')
    if metrics is None:
        metrics = ['ks_test', 'correlation_diff', 'mean_diff', 'std_diff', 'pairwise_dist', 'coverage']
    supported_metrics = {'ks_test', 'correlation_diff', 'mean_diff', 'std_diff', 'pairwise_dist', 'coverage'}
    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(f'Unsupported metric: {metric}. Supported metrics: {supported_metrics}')
    metric_scores = {}
    details = {}
    if 'ks_test' in metrics:
        (metric_scores['ks_test'], details['ks_test']) = _compute_ks_test(real_array, synthetic_array)
    if 'correlation_diff' in metrics:
        (metric_scores['correlation_diff'], details['correlation_diff']) = _compute_correlation_diff(real_array, synthetic_array)
    if 'mean_diff' in metrics:
        (metric_scores['mean_diff'], details['mean_diff']) = _compute_mean_diff(real_array, synthetic_array)
    if 'std_diff' in metrics:
        (metric_scores['std_diff'], details['std_diff']) = _compute_std_diff(real_array, synthetic_array)
    if 'pairwise_dist' in metrics:
        (metric_scores['pairwise_dist'], details['pairwise_dist']) = _compute_pairwise_dist(real_array, synthetic_array)
    if 'coverage' in metrics:
        (metric_scores['coverage'], details['coverage']) = _compute_coverage(real_array, synthetic_array)
    failed_metrics = [metric for (metric, score) in metric_scores.items() if score > threshold]
    overall_quality = len(failed_metrics) == 0
    return {'overall_quality': overall_quality, 'metric_scores': metric_scores, 'failed_metrics': failed_metrics, 'details': details}