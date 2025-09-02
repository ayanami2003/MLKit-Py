from typing import Union, Optional, Dict, Any, List
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from scipy.stats import chi2

def check_autocorrelation_in_residuals(residuals: Union[np.ndarray, DataBatch, FeatureSet], lags: Optional[int]=None, alpha: float=0.05) -> Dict[str, Any]:
    """
    Check for autocorrelation in model residuals using statistical tests.

    This function performs autocorrelation analysis on model residuals to detect
    temporal dependencies that might violate model assumptions. It computes
    autocorrelation coefficients and conducts statistical tests to determine
    if significant autocorrelation exists at specified lags.

    Args:
        residuals (Union[np.ndarray, DataBatch, FeatureSet]): Model residuals to analyze.
            Can be provided as raw array, DataBatch, or FeatureSet format.
        lags (Optional[int]): Number of lags to test for autocorrelation.
            If None, defaults to 10% of sample size or 20, whichever is smaller.
        alpha (float): Significance level for statistical tests. Defaults to 0.05.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'autocorrelations': Array of autocorrelation coefficients
            - 'lags': Array of lag values tested
            - 'significant_lags': List of lags with significant autocorrelation
            - 'test_statistic': Test statistic value
            - 'p_values': P-values for each lag
            - 'is_autocorrelated': Boolean indicating presence of significant autocorrelation

    Raises:
        ValueError: If residuals array is empty or has insufficient samples.
        TypeError: If residuals are not in supported format.
    """
    if isinstance(residuals, DataBatch):
        res_array = np.array(residuals.data).flatten()
    elif isinstance(residuals, FeatureSet):
        res_array = residuals.features.flatten()
    elif isinstance(residuals, np.ndarray):
        res_array = residuals.flatten()
    else:
        raise TypeError('Residuals must be np.ndarray, DataBatch, or FeatureSet.')
    if res_array.size == 0:
        raise ValueError('Residuals array is empty.')
    n = len(res_array)
    if n < 2:
        raise ValueError('Need at least 2 observations to compute autocorrelation.')
    if lags is None:
        lags = min(max(1, n // 10), 20)
    max_lags = min(lags, n - 1)
    if max_lags != lags and lags is not None:
        lags = max_lags
    if lags <= 0:
        raise ValueError('Number of lags must be positive.')
    autocorrs = np.zeros(lags)
    for k in range(1, lags + 1):
        if k < len(res_array):
            autocorrs[k - 1] = np.corrcoef(res_array[:-k], res_array[k:])[0, 1]
        else:
            autocorrs[k - 1] = 0
    lb_stats = np.zeros(lags)
    p_vals = np.zeros(lags)
    for k in range(1, lags + 1):
        rho_squared_terms = autocorrs[:k] ** 2 / (n - np.arange(1, k + 1))
        lb_stat = n * (n + 2) * np.sum(rho_squared_terms)
        lb_stats[k - 1] = lb_stat
        p_vals[k - 1] = 1 - chi2.cdf(lb_stat, df=k)
    significant_lags = [i + 1 for i in range(lags) if p_vals[i] < alpha]
    is_autocorrelated = len(significant_lags) > 0
    return {'autocorrelations': autocorrs, 'lags': np.arange(1, lags + 1), 'significant_lags': significant_lags, 'test_statistic': lb_stats[-1] if lags > 0 else 0.0, 'p_values': p_vals, 'is_autocorrelated': is_autocorrelated}


# ...(code omitted)...


def multicollinearity_diagnosis_with_vif(X: Union[np.ndarray, FeatureSet, DataBatch], feature_names: Optional[List[str]]=None, vif_threshold: float=5.0) -> Dict[str, Any]:
    """
    Diagnose multicollinearity among features using Variance Inflation Factor (VIF).

    This function computes the Variance Inflation Factor for each feature to
    quantify multicollinearity. VIF measures how much the variance of estimated
    regression coefficients is inflated due to collinearity. Higher VIF values
    indicate more severe multicollinearity.

    Args:
        X (Union[np.ndarray, FeatureSet, DataBatch]): Input feature matrix.
        feature_names (Optional[List[str]]): Names of features. If None, attempts
            to extract from X if it's a FeatureSet or DataBatch.
        vif_threshold (float): VIF threshold above which features are considered
            problematic. Common values are 5.0 or 10.0. Defaults to 5.0.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'vif_values': Dictionary mapping feature names to VIF values
            - 'problematic_features': List of features with VIF > threshold
            - 'max_vif': Maximum VIF value found
            - 'mean_vif': Average VIF across all features
            - 'vif_summary': Summary statistics of VIF values

    Raises:
        ValueError: If X has insufficient samples or contains constant features.
        TypeError: If X is not in supported format.
        RuntimeError: If VIF calculation fails due to numerical issues.
    """
    if isinstance(X, np.ndarray):
        X_array = X
    elif isinstance(X, FeatureSet):
        X_array = X.features
        if feature_names is None:
            feature_names = X.feature_names
    elif isinstance(X, DataBatch):
        X_array = X.data
        if feature_names is None:
            feature_names = X.feature_names
    else:
        raise TypeError('X must be a numpy array, FeatureSet, or DataBatch.')
    if X_array.ndim != 2:
        raise ValueError('X must be a 2D array.')
    (n_samples, n_features) = X_array.shape
    if n_samples <= n_features:
        raise ValueError(f'Insufficient samples ({n_samples}) for {n_features} features. VIF requires more samples than features.')
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError('Length of feature_names must match number of features in X.')
    constant_features = []
    for (i, name) in enumerate(feature_names):
        if np.var(X_array[:, i]) == 0:
            constant_features.append(name)
    if constant_features:
        raise ValueError(f'Constant features detected: {constant_features}. These must be removed before VIF calculation.')
    vif_values = {}
    eps = 1e-10
    for (i, name) in enumerate(feature_names):
        y_reg = X_array[:, i]
        X_reg = np.delete(X_array, i, axis=1)
        X_reg_with_const = np.column_stack([np.ones(X_reg.shape[0]), X_reg])
        try:
            XtX = X_reg_with_const.T @ X_reg_with_const
            XtX_reg = XtX + np.eye(XtX.shape[0]) * eps
            XtY = X_reg_with_const.T @ y_reg
            try:
                beta = np.linalg.solve(XtX_reg, XtY)
                y_pred = X_reg_with_const @ beta
                ss_res = np.sum((y_reg - y_pred) ** 2)
                ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
                if ss_tot < eps:
                    r_squared = 0.0
                else:
                    r_squared = 1.0 - ss_res / ss_tot
                if r_squared >= 1.0 - eps:
                    vif = np.inf
                else:
                    vif = 1.0 / (1.0 - r_squared)
                vif_values[name] = vif
            except np.linalg.LinAlgError:
                try:
                    beta = np.linalg.pinv(XtX_reg) @ XtY
                    y_pred = X_reg_with_const @ beta
                    ss_res = np.sum((y_reg - y_pred) ** 2)
                    ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
                    if ss_tot < eps:
                        r_squared = 0.0
                    else:
                        r_squared = 1.0 - ss_res / ss_tot
                    if r_squared >= 1.0 - eps:
                        vif = np.inf
                    else:
                        vif = 1.0 / (1.0 - r_squared)
                    vif_values[name] = vif
                except Exception:
                    raise RuntimeError(f'Numerical error in VIF calculation for feature {name}')
        except Exception:
            raise RuntimeError(f'Failed to calculate VIF for feature {name}')
    problematic_features = [name for (name, vif) in vif_values.items() if vif > vif_threshold]
    vif_array = np.array(list(vif_values.values()))
    finite_vif = vif_array[np.isfinite(vif_array)]
    vif_summary = {'count': len(vif_array), 'finite_count': len(finite_vif), 'infinite_count': len(vif_array) - len(finite_vif), 'min': float(np.min(finite_vif)) if len(finite_vif) > 0 else 0.0, 'max': float(np.max(finite_vif)) if len(finite_vif) > 0 else 0.0, 'mean': float(np.mean(finite_vif)) if len(finite_vif) > 0 else 0.0, 'median': float(np.median(finite_vif)) if len(finite_vif) > 0 else 0.0, 'std': float(np.std(finite_vif, ddof=1)) if len(finite_vif) > 1 else 0.0}
    return {'vif_values': vif_values, 'problematic_features': problematic_features, 'max_vif': float(np.max(vif_array)) if len(vif_array) > 0 else 0.0, 'mean_vif': float(np.mean(finite_vif)) if len(finite_vif) > 0 else 0.0, 'vif_summary': vif_summary}

def assess_normality_of_residuals(residuals: Union[np.ndarray, DataBatch, FeatureSet], alpha: float=0.05, test_types: Optional[List[str]]=None) -> Dict[str, Any]:
    """
    Assess the normality of model residuals using statistical tests.

    This function evaluates whether model residuals follow a normal distribution,
    which is a common assumption in many statistical models. It applies multiple
    normality tests and provides visualization-ready statistics.

    Args:
        residuals (Union[np.ndarray, DataBatch, FeatureSet]): Model residuals to test.
        alpha (float): Significance level for statistical tests. Defaults to 0.05.
        test_types (Optional[List[str]]): List of normality tests to perform.
            Options include 'shapiro-wilk', 'anderson-darling', 'kolmogorov-smirnov'.
            If None, performs all available tests.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'test_results': Dictionary of test names and their results
            - 'is_normal': Boolean indicating if residuals are normally distributed
            - 'p_values': P-values from each test
            - 'test_statistics': Test statistic values
            - 'confidence': Confidence level in normality assessment
            - 'recommendations': Suggestions based on test outcomes

    Raises:
        ValueError: If residuals array is empty or too small for testing.
        TypeError: If residuals are not in supported format.
    """
    if isinstance(residuals, np.ndarray):
        residuals_data = residuals.flatten()
    elif isinstance(residuals, DataBatch):
        residuals_data = np.asarray(residuals.data).flatten()
    elif isinstance(residuals, FeatureSet):
        residuals_data = residuals.features.flatten()
    else:
        raise TypeError('Residuals must be np.ndarray, DataBatch, or FeatureSet')
    if residuals_data.size == 0:
        test_results = {}
        p_values = {}
        test_statistics = {}
        available_tests = ['shapiro-wilk', 'anderson-darling', 'kolmogorov-smirnov']
        tests_to_run = test_types if test_types is not None else available_tests
        for test in tests_to_run:
            if test == 'shapiro-wilk':
                test_results[test] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': 'Residuals array is empty'}
                p_values[test] = None
                test_statistics[test] = None
            elif test == 'anderson-darling':
                test_results[test] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': 'Residuals array is empty'}
                p_values[test] = None
                test_statistics[test] = None
            elif test == 'kolmogorov-smirnov':
                test_results[test] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': 'Residuals array is empty'}
                p_values[test] = None
                test_statistics[test] = None
        return {'test_results': test_results, 'is_normal': bool(False), 'p_values': p_values, 'test_statistics': test_statistics, 'confidence': float(0.0), 'recommendations': ['Residuals array is empty. Cannot perform normality tests.']}
    residuals_data = residuals_data[~np.isnan(residuals_data)]
    if residuals_data.size == 0:
        test_results = {}
        p_values = {}
        test_statistics = {}
        available_tests = ['shapiro-wilk', 'anderson-darling', 'kolmogorov-smirnov']
        tests_to_run = test_types if test_types is not None else available_tests
        for test in tests_to_run:
            if test == 'shapiro-wilk':
                test_results[test] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': 'Residuals array contains only NaN values'}
                p_values[test] = None
                test_statistics[test] = None
            elif test == 'anderson-darling':
                test_results[test] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': 'Residuals array contains only NaN values'}
                p_values[test] = None
                test_statistics[test] = None
            elif test == 'kolmogorov-smirnov':
                test_results[test] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': 'Residuals array contains only NaN values'}
                p_values[test] = None
                test_statistics[test] = None
        return {'test_results': test_results, 'is_normal': bool(False), 'p_values': p_values, 'test_statistics': test_statistics, 'confidence': float(0.0), 'recommendations': ['Residuals array contains only NaN values. Cannot perform normality tests.']}
    available_tests = ['shapiro-wilk', 'anderson-darling', 'kolmogorov-smirnov']
    if test_types is None:
        test_types = available_tests
    else:
        for test in test_types:
            if test not in available_tests:
                raise ValueError(f'Unsupported test type: {test}. Available tests: {available_tests}')
    test_results = {}
    p_values = {}
    test_statistics = {}
    if 'shapiro-wilk' in test_types:
        try:
            if 3 <= len(residuals_data) <= 5000:
                from scipy.stats import shapiro
                (stat, p_val) = shapiro(residuals_data)
                is_normal = bool(p_val > alpha)
                test_results['shapiro-wilk'] = {'statistic': float(stat), 'p_value': float(p_val), 'is_normal': bool(is_normal), 'success': True}
                p_values['shapiro-wilk'] = float(p_val)
                test_statistics['shapiro-wilk'] = float(stat)
            else:
                test_results['shapiro-wilk'] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': f'Sample size {len(residuals_data)} not in valid range [3, 5000]'}
                p_values['shapiro-wilk'] = None
                test_statistics['shapiro-wilk'] = None
        except Exception as e:
            test_results['shapiro-wilk'] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': str(e)}
            p_values['shapiro-wilk'] = None
            test_statistics['shapiro-wilk'] = None
    if 'anderson-darling' in test_types:
        try:
            from scipy.stats import anderson
            result = anderson(residuals_data, dist='norm')
            critical_values = result.critical_values
            significance_levels = result.significance_level
            idx = np.argmin(np.abs(significance_levels - alpha * 100))
            critical_value = critical_values[idx]
            is_normal = bool(result.statistic < critical_value)
            test_results['anderson-darling'] = {'statistic': float(result.statistic), 'critical_value': float(critical_value), 'significance_level': float(significance_levels[idx]), 'p_value': None, 'is_normal': bool(is_normal), 'success': True}
            p_values['anderson-darling'] = None
            test_statistics['anderson-darling'] = float(result.statistic)
        except Exception as e:
            test_results['anderson-darling'] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': str(e)}
            p_values['anderson-darling'] = None
            test_statistics['anderson-darling'] = None
    if 'kolmogorov-smirnov' in test_types:
        try:
            from scipy.stats import kstest, norm
            (mu, sigma) = (np.mean(residuals_data), np.std(residuals_data, ddof=1))
            if sigma > 0:
                (stat, p_val) = kstest(residuals_data, lambda x: norm.cdf(x, mu, sigma))
                is_normal = bool(p_val > alpha)
                test_results['kolmogorov-smirnov'] = {'statistic': float(stat), 'p_value': float(p_val), 'is_normal': bool(is_normal), 'success': True}
                p_values['kolmogorov-smirnov'] = float(p_val)
                test_statistics['kolmogorov-smirnov'] = float(stat)
            else:
                test_results['kolmogorov-smirnov'] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': 'Standard deviation is zero'}
                p_values['kolmogorov-smirnov'] = None
                test_statistics['kolmogorov-smirnov'] = None
        except Exception as e:
            test_results['kolmogorov-smirnov'] = {'statistic': None, 'p_value': None, 'is_normal': None, 'success': False, 'reason': str(e)}
            p_values['kolmogorov-smirnov'] = None
            test_statistics['kolmogorov-smirnov'] = None
    successful_tests = [test_name for (test_name, result) in test_results.items() if result.get('success', False)]
    if successful_tests:
        normal_tests = sum((1 for test_name in successful_tests if test_results[test_name].get('is_normal', False)))
        is_normal_overall = bool(normal_tests >= len(successful_tests) / 2)
        confidence = float(normal_tests / len(successful_tests)) if successful_tests else float(0.0)
    else:
        is_normal_overall = bool(False)
        confidence = float(0.0)
    recommendations = []
    if not successful_tests:
        recommendations.append('No tests could be successfully performed.')
    else:
        if is_normal_overall:
            recommendations.append('Residuals appear to be normally distributed.')
        else:
            recommendations.append('Residuals do not appear to be normally distributed.')
        failed_tests = [test_name for (test_name, result) in test_results.items() if not result.get('success', True)]
        if failed_tests:
            recommendations.append(f"Tests that failed: {', '.join(failed_tests)}")
        if len(residuals_data) < 10:
            recommendations.append('Sample size is small, which may affect test reliability.')
        elif len(residuals_data) > 5000:
            recommendations.append('Sample size is large, consider visual inspection as well.')
    return {'test_results': test_results, 'is_normal': bool(is_normal_overall), 'p_values': p_values, 'test_statistics': test_statistics, 'confidence': float(confidence), 'recommendations': recommendations}

def probability_plot_analysis(data: Union[np.ndarray, DataBatch, FeatureSet], distribution: str='normal', confidence_level: float=0.95) -> Dict[str, Any]:
    """
    Perform probability plot analysis to assess distributional fit.

    This function generates probability plots (Q-Q plots) to visually and
    statistically assess how well data fits a specified theoretical distribution.
    It's particularly useful for residual analysis and distributional assumption checking.

    Args:
        data (Union[np.ndarray, DataBatch, FeatureSet]): Data to analyze.
        distribution (str): Theoretical distribution to compare against.
            Supported options: 'normal', 'exponential', 'uniform', 'lognormal'.
            Defaults to 'normal'.
        confidence_level (float): Confidence level for uncertainty bands.
            Defaults to 0.95 (95%).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'quantiles': Theoretical quantiles of specified distribution
            - 'ordered_data': Empirical quantiles from input data
            - 'correlation_coefficient': Correlation between theoretical and empirical quantiles
            - 'confidence_intervals': Upper and lower bounds for uncertainty bands
            - 'distribution_fit': Goodness of fit measure
            - 'outliers': Indices of potential outliers
            - 'plot_data': Data formatted for visualization

    Raises:
        ValueError: If data is empty or distribution not supported.
        TypeError: If data is not in supported format.
    """
    if isinstance(data, np.ndarray):
        vals = data.flatten()
    elif isinstance(data, DataBatch):
        vals = data.data.values.flatten()
    elif isinstance(data, FeatureSet):
        vals = data.X.values.flatten()
    else:
        raise TypeError('Data must be np.ndarray, DataBatch, or FeatureSet')
    if vals.size == 0:
        raise ValueError('Input data is empty')
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        raise ValueError('All data values are NaN')
    dist_map = {'normal': ('norm', lambda x: (x - np.mean(x)) / np.std(x, ddof=1)), 'exponential': ('expon', lambda x: x / np.mean(x)), 'uniform': ('uniform', lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))), 'lognormal': ('lognorm', lambda x: x[x > 0])}
    if distribution not in dist_map:
        raise ValueError(f"Distribution '{distribution}' not supported. Choose from {list(dist_map.keys())}")
    (dist_name, transform_func) = dist_map[distribution]
    if distribution == 'lognormal':
        vals = vals[vals > 0]
        if vals.size == 0:
            raise ValueError('No positive values found for lognormal distribution')
        transformed_vals = np.log(vals)
        transformed_vals = (transformed_vals - np.mean(transformed_vals)) / np.std(transformed_vals, ddof=1)
    else:
        transformed_vals = transform_func(vals)
    sorted_vals = np.sort(transformed_vals)
    n = len(sorted_vals)
    plotting_positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    from scipy.stats import norm, expon, uniform, lognorm
    if distribution == 'normal':
        theoretical_quantiles = norm.ppf(plotting_positions)
    elif distribution == 'exponential':
        theoretical_quantiles = expon.ppf(plotting_positions)
    elif distribution == 'uniform':
        theoretical_quantiles = uniform.ppf(plotting_positions)
    elif distribution == 'lognormal':
        theoretical_quantiles = norm.ppf(plotting_positions)
    correlation = np.corrcoef(theoretical_quantiles, sorted_vals)[0, 1]
    alpha = 1 - confidence_level
    z_alpha = norm.ppf(1 - alpha / 2)
    se = np.sqrt(plotting_positions * (1 - plotting_positions) / n)
    upper_bound = theoretical_quantiles + z_alpha * se
    lower_bound = theoretical_quantiles - z_alpha * se
    mid_idx = slice(int(n * 0.25), int(n * 0.75))
    (slope, intercept) = np.polyfit(theoretical_quantiles[mid_idx], sorted_vals[mid_idx], 1)
    fitted_values = slope * theoretical_quantiles + intercept
    residuals = sorted_vals - fitted_values
    std_residuals = np.std(residuals)
    outlier_indices = np.where(np.abs(residuals) > 2 * std_residuals)[0].tolist()
    plot_data = {'theoretical': theoretical_quantiles.tolist(), 'empirical': sorted_vals.tolist(), 'upper_confidence': upper_bound.tolist(), 'lower_confidence': lower_bound.tolist(), 'fitted_line': fitted_values.tolist()}
    return {'quantiles': theoretical_quantiles.tolist(), 'ordered_data': sorted_vals.tolist(), 'correlation_coefficient': float(correlation), 'confidence_intervals': {'upper': upper_bound.tolist(), 'lower': lower_bound.tolist()}, 'distribution_fit': float(correlation), 'outliers': outlier_indices, 'plot_data': plot_data}


# ...(code omitted)...


def model_specification_testing(model: BaseModel, X: Union[np.ndarray, DataBatch, FeatureSet], y: Union[np.ndarray, List], test_types: Optional[List[str]]=None, alpha: float=0.05) -> Dict[str, Any]:
    """
    Perform model specification tests to validate model assumptions.

    This function conducts various statistical tests to verify that the
    model satisfies its underlying assumptions. These tests help ensure
    the validity of model inferences and predictions.

    Args:
        model (BaseModel): Fitted model implementing the BaseModel interface.
        X (Union[np.ndarray, DataBatch, FeatureSet]): Input features.
        y (Union[np.ndarray, List]): Target values.
        test_types (Optional[List[str]]): Specific tests to perform.
            Options include 'ramsey-reset', 'link-test', 'heteroskedasticity'.
            If None, performs all relevant tests based on model type.
        alpha (float): Significance level for statistical tests. Defaults to 0.05.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'test_results': Detailed results for each test performed
            - 'specification_valid': Boolean indicating if model passes all tests
            - 'failed_tests': List of tests that indicated specification issues
            - 'recommendations': Suggestions for model improvement
            - 'confidence': Overall confidence in model specification

    Raises:
        ValueError: If model is not fitted or data dimensions mismatch.
        TypeError: If inputs are not in supported formats.
        RuntimeError: If required residuals or predictions are not available.
    """
    if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
        raise ValueError('Model must be fitted and implement a predict method')
    if isinstance(X, np.ndarray):
        X_array = X
    elif isinstance(X, DataBatch):
        X_array = np.array(X.data)
    elif isinstance(X, FeatureSet):
        X_array = X.features
    else:
        raise TypeError('X must be np.ndarray, DataBatch, or FeatureSet')
    if isinstance(y, np.ndarray):
        y_array = y
    elif isinstance(y, list):
        y_array = np.array(y)
    else:
        raise TypeError('y must be np.ndarray or List')
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError('Number of samples in X and y must match')
    try:
        y_pred = model.predict(X)
        if isinstance(y_pred, (list, tuple)):
            y_pred = np.array(y_pred)
    except Exception as e:
        raise RuntimeError(f'Failed to get predictions from model: {str(e)}')
    residuals = y_array - y_pred.flatten()
    available_tests = ['ramsey-reset', 'link-test', 'heteroskedasticity']
    if test_types is None:
        tests_to_run = available_tests
    else:
        for test in test_types:
            if test not in available_tests:
                raise ValueError(f'Unsupported test type: {test}. Available tests: {available_tests}')
        tests_to_run = test_types
    test_results = {}
    failed_tests = []
    recommendations = []
    passed_tests = 0
    total_tests = len(tests_to_run)
    if 'ramsey-reset' in tests_to_run:
        try:
            y_pred_sq = y_pred ** 2
            y_pred_cube = y_pred ** 3
            n = len(y_array)
            k = X_array.shape[1] if X_array.ndim > 1 else 1
            if X_array.ndim == 1:
                X_aux = np.column_stack([np.ones(n), X_array.reshape(-1, 1), y_pred_sq, y_pred_cube])
            else:
                X_aux = np.column_stack([np.ones(n), X_array, y_pred_sq, y_pred_cube])
            try:
                XtX = X_aux.T @ X_aux
                XtY = X_aux.T @ residuals
                beta = np.linalg.solve(XtX, XtY)
                residuals_aux = residuals - X_aux @ beta
                ss_res = np.sum(residuals_aux ** 2)
                ss_tot = np.sum((residuals - np.mean(residuals)) ** 2)
                if ss_tot > 0:
                    r_squared_aux = 1 - ss_res / ss_tot
                else:
                    r_squared_aux = 0
                if X_array.ndim == 1:
                    X_rest = np.column_stack([np.ones(n), X_array.reshape(-1, 1)])
                else:
                    X_rest = np.column_stack([np.ones(n), X_array])
                XtX_rest = X_rest.T @ X_rest
                XtY_rest = X_rest.T @ residuals
                beta_rest = np.linalg.solve(XtX_rest, XtY_rest)
                residuals_rest = residuals - X_rest @ beta_rest
                ss_res_rest = np.sum(residuals_rest ** 2)
                df_full = X_aux.shape[1]
                df_rest = X_rest.shape[1]
                df_diff = df_full - df_rest
                if df_diff > 0 and ss_res > 0:
                    f_stat = (ss_res_rest - ss_res) / df_diff / (ss_res / (n - df_full))
                    from scipy.stats import f
                    p_value = 1 - f.cdf(f_stat, df_diff, n - df_full)
                    is_valid = p_value > alpha
                    test_results['ramsey-reset'] = {'f_statistic': float(f_stat), 'p_value': float(p_value), 'r_squared_aux': float(r_squared_aux), 'is_valid': bool(is_valid), 'degrees_of_freedom': (int(df_diff), int(n - df_full))}
                    if not is_valid:
                        failed_tests.append('ramsey-reset')
                        recommendations.append('Ramsey RESET test indicates potential functional form misspecification. Consider adding polynomial terms or transforming variables.')
                    else:
                        passed_tests += 1
                else:
                    test_results['ramsey-reset'] = {'f_statistic': None, 'p_value': None, 'r_squared_aux': float(r_squared_aux), 'is_valid': None, 'error': 'Could not compute F-statistic'}
            except np.linalg.LinAlgError:
                test_results['ramsey-reset'] = {'f_statistic': None, 'p_value': None, 'r_squared_aux': None, 'is_valid': None, 'error': 'Matrix singularity in auxiliary regression'}
        except Exception as e:
            test_results['ramsey-reset'] = {'f_statistic': None, 'p_value': None, 'r_squared_aux': None, 'is_valid': None, 'error': str(e)}
    if 'link-test' in tests_to_run:
        try:
            y_pred_sq = y_pred ** 2
            X_link = np.column_stack([np.ones(len(y_array)), y_pred, y_pred_sq])
            try:
                XtX = X_link.T @ X_link
                XtY = X_link.T @ y_array
                beta = np.linalg.solve(XtX, XtY)
                y_fitted = X_link @ beta
                residuals_link = y_array - y_fitted
                ss_res = np.sum(residuals_link ** 2)
                sigma_sq = ss_res / (len(y_array) - 3)
                var_beta = sigma_sq * np.linalg.inv(XtX)
                se_beta2 = np.sqrt(var_beta[2, 2])
                if se_beta2 > 0:
                    t_stat = beta[2] / se_beta2
                    from scipy.stats import t as t_dist
                    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), len(y_array) - 3))
                    is_valid = p_value > alpha
                    test_results['link-test'] = {'t_statistic': float(t_stat), 'p_value': float(p_value), 'coefficient_yhat_sq': float(beta[2]), 'se_yhat_sq': float(se_beta2), 'is_valid': bool(is_valid)}
                    if not is_valid:
                        failed_tests.append('link-test')
                        recommendations.append('Link test suggests non-linearity in the relationship. Consider polynomial terms or non-linear transformations.')
                    else:
                        passed_tests += 1
                else:
                    test_results['link-test'] = {'t_statistic': None, 'p_value': None, 'coefficient_yhat_sq': float(beta[2]), 'se_yhat_sq': 0.0, 'is_valid': None, 'error': 'Zero standard error'}
            except np.linalg.LinAlgError:
                test_results['link-test'] = {'t_statistic': None, 'p_value': None, 'coefficient_yhat_sq': None, 'se_yhat_sq': None, 'is_valid': None, 'error': 'Matrix singularity in link test regression'}
        except Exception as e:
            test_results['link-test'] = {'t_statistic': None, 'p_value': None, 'coefficient_yhat_sq': None, 'se_yhat_sq': None, 'is_valid': None, 'error': str(e)}
    if 'heteroskedasticity' in tests_to_run:
        try:
            residuals_sq = residuals ** 2
            if X_array.ndim == 1:
                X_het = np.column_stack([np.ones(len(residuals)), X_array.reshape(-1, 1)])
            else:
                X_het = np.column_stack([np.ones(len(residuals)), X_array])
            try:
                XtX = X_het.T @ X_het
                XtY = X_het.T @ residuals_sq
                beta = np.linalg.solve(XtX, XtY)
                fitted_vals = X_het @ beta
                residuals_het = residuals_sq - fitted_vals
                ss_res = np.sum(residuals_het ** 2)
                ss_tot = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
                if ss_tot > 0:
                    r_squared = 1 - ss_res / ss_tot
                else:
                    r_squared = 0
                n = len(residuals_sq)
                k = X_het.shape[1]
                lm_stat = n * r_squared
                from scipy.stats import chi2
                p_value = 1 - chi2.cdf(lm_stat, k - 1)
                is_valid = p_value > alpha
                test_results['heteroskedasticity'] = {'lm_statistic': float(lm_stat), 'p_value': float(p_value), 'r_squared': float(r_squared), 'is_valid': bool(is_valid), 'degrees_of_freedom': int(k - 1)}
                if not is_valid:
                    failed_tests.append('heteroskedasticity')
                    recommendations.append('Evidence of heteroskedasticity detected. Consider using robust standard errors or weighted least squares.')
                else:
                    passed_tests += 1
            except np.linalg.LinAlgError:
                test_results['heteroskedasticity'] = {'lm_statistic': None, 'p_value': None, 'r_squared': None, 'is_valid': None, 'error': 'Matrix singularity in heteroskedasticity test'}
        except Exception as e:
            test_results['heteroskedasticity'] = {'lm_statistic': None, 'p_value': None, 'r_squared': None, 'is_valid': None, 'error': str(e)}
    if total_tests > 0:
        confidence = passed_tests / total_tests
        specification_valid = len(failed_tests) == 0
    else:
        confidence = 1.0
        specification_valid = True
        recommendations.append('No tests were performed.')
    if len(failed_tests) > 0:
        recommendations.append('Model specification issues detected. Consider revising the model structure.')
    elif total_tests > 0:
        recommendations.append('Model appears to satisfy specification tests.')
    return {'test_results': test_results, 'specification_valid': bool(specification_valid), 'failed_tests': failed_tests, 'passed_tests': [test for test in tests_to_run if test not in failed_tests], 'recommendations': recommendations, 'confidence': float(confidence)}