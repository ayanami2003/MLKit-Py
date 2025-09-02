from general.structures.data_batch import DataBatch
from typing import Union, Dict, Any, Optional, List
import numpy as np
from scipy import stats


# ...(code omitted)...


def perform_stationarity_tests(data: Union[np.ndarray, DataBatch], test_types: Optional[List[str]]=None, regression: str='c', lags: Optional[int]=None) -> Dict[str, Any]:
    """
    Perform multiple stationarity tests on a time series.

    This function provides a unified interface to run various stationarity tests
    on a time series, including KPSS, Phillips-Perron, and other common tests.
    It returns a comprehensive report of all test results for easy comparison.

    Args:
        data (Union[np.ndarray, DataBatch]): Input time series data. If DataBatch is provided,
            it should contain a single time series in the data attribute.
        test_types (Optional[List[str]]): List of test types to perform. Options include
            'kpss', 'phillips_perron', 'adf' (Augmented Dickey-Fuller), etc.
            If None, defaults to ['kpss', 'phillips_perron'].
        regression (str): The deterministic trend specification. 'c' for constant (default),
            'ct' for constant and trend.
        lags (Optional[int]): Number of lags to use in the Newey-West estimator for relevant tests.
            If None, lags are automatically determined for each test.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'test_results': Dictionary mapping test names to their individual results
            - 'summary': Overall stationarity assessment based on all tests
            - 'recommendations': Suggestions for further analysis based on test outcomes

    Raises:
        ValueError: If the input data is invalid or test_types contains unsupported tests
    """
    if isinstance(data, DataBatch):
        ts_data = np.asarray(data.data)
    elif isinstance(data, np.ndarray):
        ts_data = data
    else:
        raise ValueError('Input data must be a numpy array or DataBatch')
    if ts_data.ndim > 1:
        if ts_data.shape[1] == 1:
            ts_data = ts_data.flatten()
        elif ts_data.shape[0] == 1:
            ts_data = ts_data.flatten()
        else:
            raise ValueError('Only univariate time series are supported')
    if len(ts_data) < 2:
        raise ValueError('Time series must have at least two observations')
    if test_types is None:
        test_types = ['kpss', 'phillips_perron']
    valid_tests = {'kpss', 'phillips_perron', 'adf'}
    for t in test_types:
        if t not in valid_tests:
            raise ValueError(f"Unsupported test type '{t}'. Valid options: {valid_tests}")
    if regression not in ('c', 'ct'):
        raise ValueError("regression must be either 'c' (constant) or 'ct' (constant and trend)")
    if lags is not None and (not isinstance(lags, int) or lags < 0):
        raise ValueError('lags must be a non-negative integer or None')
    test_results = {}
    if 'kpss' in test_types:
        try:
            (stat, p_value, lags_used, critical_values) = kpss_test(ts_data, regression=regression, lags=lags)
            decision = 'stationary' if p_value > 0.05 else 'non-stationary'
            test_results['kpss'] = {'statistic': stat, 'p_value': p_value, 'lags': lags_used, 'critical_values': critical_values, 'decision': decision}
        except Exception as e:
            test_results['kpss'] = {'error': str(e)}
    if 'phillips_perron' in test_types:
        try:
            (stat, p_value, critical_values) = phillips_perron_test(ts_data, regression=regression)
            decision = 'stationary' if p_value < 0.05 else 'non-stationary'
            test_results['phillips_perron'] = {'statistic': stat, 'p_value': p_value, 'critical_values': critical_values, 'decision': decision}
        except Exception as e:
            test_results['phillips_perron'] = {'error': str(e)}
    if 'adf' in test_types:
        try:
            (stat, p_value, used_lag, critical_values) = augmented_dickey_fuller(ts_data, regression=regression, max_lags=lags)
            decision = 'stationary' if p_value < 0.05 else 'non-stationary'
            test_results['adf'] = {'statistic': stat, 'p_value': p_value, 'used_lag': used_lag, 'critical_values': critical_values, 'decision': decision}
        except Exception as e:
            test_results['adf'] = {'error': str(e)}
    decisions = [res.get('decision') for res in test_results.values() if isinstance(res, dict) and 'decision' in res]
    if not decisions:
        summary = 'no_decision'
        recommendation = 'Unable to make a determination due to test failures.'
    else:
        stationary_count = decisions.count('stationary')
        non_stationary_count = decisions.count('non-stationary')
        if stationary_count > non_stationary_count:
            summary = 'likely_stationary'
        elif non_stationary_count > stationary_count:
            summary = 'likely_non_stationary'
        else:
            summary = 'inconclusive'
        recommendations = []
        if summary == 'likely_non_stationary' or non_stationary_count > 0:
            recommendations.append('Consider differencing the time series to achieve stationarity.')
        if regression == 'c':
            recommendations.append('If a trend is visually apparent, consider using regression="ct".')
        if lags is None:
            recommendations.append('Automatic lag selection was used. Consider specifying lags if domain knowledge suggests a particular value.')
        recommendation = ' '.join(recommendations) if recommendations else 'The time series appears to be stationary.'
    return {'test_results': test_results, 'summary': summary, 'recommendations': recommendation}