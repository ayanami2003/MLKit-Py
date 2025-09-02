from typing import Union, Dict, Any, List
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Union, Dict, Any
from scipy.stats import ks_2samp

def kolmogorov_smirnov_test(data1: Union[np.ndarray, FeatureSet], data2: Union[np.ndarray, FeatureSet], alternative: str='two-sided') -> Dict[str, Any]:
    """
    Perform the Kolmogorov-Smirnov test for goodness of fit between two samples.
    
    This test compares the empirical distribution functions of two samples to determine
    if they come from the same distribution. It is a non-parametric test that works
    with continuous distributions.
    
    The test is suitable for comparing:
    - Two independent samples (two-sample test)
    - A sample against a reference distribution (one-sample test when data2 is theoretical)
    
    Args:
        data1 (Union[np.ndarray, FeatureSet]): First sample data. If FeatureSet, uses .features attribute.
        data2 (Union[np.ndarray, FeatureSet]): Second sample data. If FeatureSet, uses .features attribute.
                                                 Can also represent a theoretical distribution.
        alternative (str): Defines the alternative hypothesis. Options:
                          'two-sided' (default): distributions are different
                          'less': CDF of data1 is less than CDF of data2
                          'greater': CDF of data1 is greater than CDF of data2
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'statistic' (float): KS test statistic D
            - 'p_value' (float): p-value for the test
            - 'significant' (bool): Whether the result is statistically significant (p < 0.05)
            - 'alternative' (str): The alternative hypothesis used
    
    Raises:
        ValueError: If data dimensions are incompatible or alternative parameter is invalid.
    """
    valid_alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in valid_alternatives:
        raise ValueError(f"Invalid alternative '{alternative}'. Must be one of {valid_alternatives}")
    if isinstance(data1, FeatureSet):
        data1 = data1.features
    if isinstance(data2, FeatureSet):
        data2 = data2.features
    data1 = np.asarray(data1).flatten()
    data2 = np.asarray(data2).flatten()
    if len(data1) == 0 or len(data2) == 0:
        raise ValueError('Both data arrays must be non-empty')
    scipy_result = ks_2samp(data1, data2, alternative=alternative)
    significant = scipy_result.pvalue < 0.05
    return {'statistic': float(scipy_result.statistic), 'p_value': float(scipy_result.pvalue), 'significant': bool(significant), 'alternative': alternative}

def shapiro_wilk_test(data: Union[np.ndarray, FeatureSet]) -> Dict[str, Any]:
    """
    Perform the Shapiro-Wilk test for normality.
    
    This test evaluates the null hypothesis that the data comes from a normally distributed population.
    It is particularly effective for small sample sizes (n < 50) and is considered one of the most
    powerful normality tests for this range.
    
    Args:
        data (Union[np.ndarray, FeatureSet]): Sample data to test for normality.
                                            If FeatureSet, uses .features attribute.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'statistic' (float): Shapiro-Wilk test statistic W
            - 'p_value' (float): p-value for the test
            - 'significant' (bool): Whether the result is statistically significant (p < 0.05)
            - 'n_samples' (int): Number of observations used in the test
            - 'normal' (bool): Indicates if data appears to be normally distributed (not significant)
    
    Raises:
        ValueError: If sample size is outside the valid range (3-5000) or data is not 1-dimensional.
    """
    from scipy.stats import shapiro
    if isinstance(data, FeatureSet):
        raw_data = data.features
    else:
        raw_data = data
    arr = np.asarray(raw_data).flatten()
    n = len(arr)
    if n < 3 or n > 5000:
        raise ValueError(f'Sample size must be between 3 and 5000, got {n}')
    (statistic, p_value) = shapiro(arr)
    significant = p_value < 0.05
    normal = not significant
    return {'statistic': float(statistic), 'p_value': float(p_value), 'significant': bool(significant), 'n_samples': n, 'normal': bool(normal)}

def anderson_darling_test(data: Union[np.ndarray, FeatureSet], distribution: str='norm') -> Dict[str, Any]:
    """
    Perform the Anderson-Darling test for goodness of fit.
    
    This test evaluates whether a sample comes from a specified distribution. It is particularly
    sensitive to the tails of the distribution compared to other goodness-of-fit tests like
    Kolmogorov-Smirnov. The test returns critical values for different significance levels.
    
    Args:
        data (Union[np.ndarray, FeatureSet]): Sample data to test.
                                           If FeatureSet, uses .features attribute.
        distribution (str): Distribution to test against. Supported options:
                           'norm' (normal distribution, default)
                           'expon' (exponential)
                           'logistic'
                           'gumbel' (Gumbel type I)
                           'gumbel_l' (left-skewed Gumbel)
                           'gumbel_r' (right-skewed Gumbel)
                           'extreme1' (alias for gumbel_r)
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'statistic' (float): Anderson-Darling test statistic A²
            - 'critical_values' (List[float]): Critical values for different significance levels
            - 'significance_level' (List[float]): Corresponding significance levels (e.g., 1%, 5%, 10%)
            - 'p_value' (float): Approximate p-value (if available for the distribution)
            - 'distribution' (str): Tested distribution
            - 'n_samples' (int): Number of observations used in the test
            - 'fits' (bool): Indicates if data fits the distribution (statistic < critical value at 5%)
    
    Raises:
        ValueError: If distribution is not supported or data is not 1-dimensional.
    """
    if isinstance(data, FeatureSet):
        data_array = data.features
    else:
        data_array = data
    if data_array.ndim != 1:
        raise ValueError('Data must be 1-dimensional.')
    supported_dists = {'norm': 'norm', 'expon': 'expon', 'logistic': 'logistic', 'gumbel': 'gumbel_r', 'gumbel_r': 'gumbel_r', 'gumbel_l': 'gumbel_l', 'extreme1': 'gumbel_r'}
    if distribution not in supported_dists:
        raise ValueError(f'Unsupported distribution: {distribution}. Supported distributions: {list(supported_dists.keys())}')
    from scipy.stats import anderson
    dist_name = supported_dists[distribution]
    try:
        result = anderson(data_array, dist_name)
    except Exception as e:
        raise ValueError(f'Error performing Anderson-Darling test: {str(e)}')
    try:
        idx_5_percent = list(result.significance_level).index(5.0)
    except ValueError:
        sig_levels = result.significance_level
        idx_5_percent = len([level for level in sig_levels if level <= 5.0]) - 1
        if idx_5_percent < 0:
            idx_5_percent = 0
    fits = result.statistic < result.critical_values[idx_5_percent]
    p_val = None
    if distribution == 'norm' and len(result.critical_values) > 0:
        A_squared = result.statistic
        if A_squared < 0.2:
            p_val = 1 - np.exp(-13.436 + 101.14 * A_squared - 223.73 * A_squared ** 2)
        elif A_squared < 0.34:
            p_val = 1 - np.exp(-8.318 + 42.796 * A_squared - 59.938 * A_squared ** 2)
        elif A_squared < 0.6:
            p_val = np.exp(0.9177 - 4.279 * A_squared - 1.38 * A_squared ** 2)
        elif A_squared < 10:
            p_val = np.exp(1.2937 - 5.709 * A_squared + 0.0186 * A_squared ** 2)
        else:
            p_val = 0.0
    return {'statistic': float(result.statistic), 'critical_values': [float(cv) for cv in result.critical_values], 'significance_level': [float(sl) for sl in result.significance_level], 'p_value': float(p_val) if p_val is not None else None, 'distribution': distribution, 'n_samples': int(len(data_array)), 'fits': bool(fits)}


# ...(code omitted)...


def sign_test(data1: Union[np.ndarray, FeatureSet], data2: Union[np.ndarray, FeatureSet], alternative: str='two-sided') -> Dict[str, Any]:
    """
    Perform the sign test for paired samples.
    
    This non-parametric test evaluates whether there is a median difference between
    paired observations. It is based on the signs of the differences between paired
    observations and does not assume any specific distribution for the data.
    
    Args:
        data1 (Union[np.ndarray, FeatureSet]): First sample of paired observations.
        data2 (Union[np.ndarray, FeatureSet]): Second sample of paired observations.
        alternative (str): Defines the alternative hypothesis. Options:
                          'two-sided' (default): median difference is not zero
                          'less': median of data1 is less than median of data2
                          'greater': median of data1 is greater than median of data2
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'statistic' (int): Number of positive differences (or test statistic)
            - 'p_value' (float): p-value for the test
            - 'significant' (bool): Whether the result is statistically significant (p < 0.05)
            - 'alternative' (str): The alternative hypothesis used
            - 'n_samples' (int): Number of non-zero differences used in the test
            - 'median_difference' (float): Median of the differences between pairs
    
    Raises:
        ValueError: If data dimensions don't match or alternative parameter is invalid.
    """
    valid_alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in valid_alternatives:
        raise ValueError(f"Invalid alternative '{alternative}'. Must be one of {valid_alternatives}")
    if isinstance(data1, FeatureSet):
        data1 = data1.features
    if isinstance(data2, FeatureSet):
        data2 = data2.features
    data1 = np.asarray(data1).flatten()
    data2 = np.asarray(data2).flatten()
    if len(data1) != len(data2):
        raise ValueError('Paired samples must have the same length')
    if len(data1) == 0:
        raise ValueError('Data arrays must be non-empty')
    differences = data1 - data2
    non_zero_diffs = differences[differences != 0]
    n = len(non_zero_diffs)
    if n == 0:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False, 'alternative': alternative, 'n_samples': 0, 'median_difference': float(np.median(differences))}
    n_pos = np.sum(non_zero_diffs > 0)
    n_neg = n - n_pos
    if alternative == 'two-sided':
        statistic = min(n_pos, n_neg)
        from scipy.stats import binom
        p_value = 2 * binom.cdf(statistic, n, 0.5)
        if p_value > 1.0:
            p_value = 1.0
    elif alternative == 'greater':
        statistic = n_pos
        from scipy.stats import binom
        p_value = binom.sf(statistic - 1, n, 0.5)
    else:
        statistic = n_neg
        from scipy.stats import binom
        p_value = binom.cdf(statistic, n, 0.5)
    median_difference = float(np.median(differences))
    significant = p_value < 0.05
    return {'statistic': int(statistic), 'p_val': p_value}