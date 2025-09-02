from general.structures.data_batch import DataBatch
from typing import Union, Dict, Any, Optional, List
import numpy as np
import warnings
from general.base_classes.validator_base import BaseValidator
from scipy import stats


# ...(code omitted)...


class IQROutlierDetector(BaseValidator):
    """
    Detects outliers in data using the Interquartile Range (IQR) method.
    
    This validator identifies outliers by calculating the first quartile (Q1),
    third quartile (Q3), and interquartile range (IQR = Q3 - Q1) for each feature.
    Data points outside the range [Q1 - k*IQR, Q3 + k*IQR] are considered outliers,
    where k is a multiplier (typically 1.5).
    
    Attributes:
        name (str): Name of the validator instance.
        iqr_multiplier (float): Multiplier for IQR to define outlier bounds.
        outlier_flags (Dict[str, List[bool]]): Boolean flags indicating outliers for each feature.
    """

    def __init__(self, name: Optional[str]=None, iqr_multiplier: float=1.5):
        """
        Initialize the IQROutlierDetector.
        
        Args:
            name (Optional[str]): Name for the validator instance.
            iqr_multiplier (float): Multiplier for IQR to define outlier bounds (default: 1.5).
        """
        super().__init__(name)
        self.iqr_multiplier = iqr_multiplier
        self.outlier_flags: Dict[str, List[bool]] = {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Detect outliers in the input data using the IQR method.
        
        For each feature in the data batch, calculates IQR-based outlier bounds
        and identifies data points that fall outside these bounds.
        
        Args:
            data (DataBatch): Input data to analyze for outliers.
            **kwargs: Additional parameters for validation.
            
        Returns:
            bool: True if no outliers are detected, False otherwise.
        """
        self.reset_validation_state()
        self.outlier_flags = {}
        if not isinstance(data.data, np.ndarray):
            try:
                array_data = np.array(data.data)
            except Exception as e:
                self.add_error(f'Failed to convert data to numpy array: {e}')
                return False
        else:
            array_data = data.data
        if array_data.ndim == 1:
            array_data = array_data.reshape(-1, 1)
        (n_rows, n_cols) = array_data.shape
        if data.feature_names:
            feature_names = data.feature_names
        else:
            feature_names = [f'col_{i}' for i in range(n_cols)]
        has_outliers = False
        for i in range(n_cols):
            feature_name = feature_names[i]
            feature_data = array_data[:, i]
            if not np.issubdtype(feature_data.dtype, np.number):
                self.outlier_flags[feature_name] = [False] * n_rows
                continue
            clean_data = feature_data[~np.isnan(feature_data)]
            if len(clean_data) == 0:
                self.outlier_flags[feature_name] = [False] * n_rows
                continue
            q1 = np.percentile(clean_data, 25)
            q3 = np.percentile(clean_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            feature_outlier_flags = []
            outlier_count = 0
            for j in range(n_rows):
                value = feature_data[j]
                if np.isnan(value):
                    is_outlier = False
                else:
                    is_outlier = value < lower_bound or value > upper_bound
                feature_outlier_flags.append(is_outlier)
                if is_outlier:
                    outlier_count += 1
                    has_outliers = True
            self.outlier_flags[feature_name] = feature_outlier_flags
            if outlier_count > 0:
                warnings.warn(f"Feature '{feature_name}' contains {outlier_count} outliers", UserWarning)
        return not has_outliers

    def get_outlier_indices(self) -> Dict[str, List[int]]:
        """
        Get the indices of outliers for each feature.
        
        Returns:
            Dict[str, List[int]]: Dictionary mapping feature names to lists of outlier indices.
        """
        if not hasattr(self, 'outlier_flags') or not self.outlier_flags:
            return {}
        outlier_indices = {}
        for (feature_name, flags) in self.outlier_flags.items():
            indices = [i for (i, is_outlier) in enumerate(flags) if is_outlier]
            outlier_indices[feature_name] = indices
        return outlier_indices

    def get_outlier_flags(self) -> Dict[str, List[bool]]:
        """
        Get the outlier flags for each feature.
        
        Returns:
            Dict[str, List[bool]]: Dictionary mapping feature names to lists of outlier flags.
        """
        return getattr(self, 'outlier_flags', {})

def shapiro_wilk_test(data: Union[DataBatch, list], feature_names: Optional[List[str]]=None, significance_level: float=0.05) -> Dict[str, Dict[str, Any]]:
    """
    Perform the Shapiro-Wilk test for normality on input data.
    
    The Shapiro-Wilk test evaluates the null hypothesis that a sample comes from
    a normally distributed population. It is particularly effective for small
    sample sizes (n < 50) but can be applied to larger datasets as well.
    
    This test is commonly used in data validation workflows to verify assumptions
    about data distribution before applying parametric statistical methods.
    
    Args:
        data (Union[DataBatch, list]): Input data to test for normality. If DataBatch,
            tests are performed on specified features. If list, treats as single feature.
        feature_names (Optional[list]): Names of features to test if data is DataBatch.
            If None, tests all numeric features.
        significance_level (float): Significance level for the test (default: 0.05).
            
    Returns:
        Dict[str, Any]: Dictionary containing test results for each feature:
            - 'statistic': W test statistic for each feature
            - 'p_value': p-value for each feature
            - 'is_normal': Boolean indicating if feature passes normality test
            - 'critical_value': Critical value at the specified significance level
            - 'sample_size': Number of observations for each feature
            
    Raises:
        ValueError: If data is empty or contains non-numeric values.
        TypeError: If data is not a supported type.
    """
    from scipy import stats
    results = {}
    if not 0 < significance_level < 1:
        raise ValueError('Significance level must be between 0 and 1')
    if isinstance(data, DataBatch):
        if isinstance(data.data, np.ndarray):
            data_array = data.data
        else:
            data_array = np.array(data.data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        if feature_names is not None:
            if data.feature_names is None:
                raise ValueError('feature_names specified but DataBatch has no feature names')
            try:
                feature_indices = [data.feature_names.index(name) for name in feature_names]
            except ValueError as e:
                raise ValueError(f'Specified feature name not found in DataBatch: {e}')
        else:
            if data.feature_names is not None:
                feature_names_local = data.feature_names
            else:
                feature_names_local = [f'feature_{i}' for i in range(data_array.shape[1])]
            numeric_features = []
            feature_indices = []
            for (i, name) in enumerate(feature_names_local):
                try:
                    col_data = data_array[:, i]
                    _ = np.array(col_data, dtype=float)
                    numeric_features.append(name)
                    feature_indices.append(i)
                except (ValueError, TypeError):
                    continue
            if feature_names is None:
                feature_names = numeric_features if numeric_features else feature_names_local[:data_array.shape[1]]
                if not feature_indices:
                    feature_indices = list(range(min(len(feature_names_local), data_array.shape[1])))
            else:
                feature_indices = list(range(data_array.shape[1]))
        for (i, feature_idx) in enumerate(feature_indices):
            feature_name = feature_names[i] if i < len(feature_names) else f'feature_{feature_idx}'
            feature_data = data_array[:, feature_idx]
            feature_data = feature_data[~np.isnan(feature_data)]
            n = len(feature_data)
            if n < 3:
                raise ValueError(f"Shapiro-Wilk test requires at least 3 samples, got {n} for feature '{feature_name}'")
            if n > 5000:
                raise ValueError(f"Shapiro-Wilk test is not reliable for sample sizes > 5000, got {n} for feature '{feature_name}'")
            try:
                (statistic, p_value) = stats.shapiro(feature_data)
                is_normal = p_value > significance_level
                critical_value = significance_level
                results[feature_name] = {'statistic': float(statistic), 'p_value': float(p_value), 'is_normal': bool(is_normal), 'critical_value': float(critical_value), 'sample_size': int(n)}
            except Exception as e:
                raise ValueError(f"Error performing Shapiro-Wilk test on feature '{feature_name}': {e}")
    elif isinstance(data, list):
        try:
            data_array = np.array(data, dtype=float)
        except (ValueError, TypeError):
            raise ValueError('Input list must contain numeric data')
        data_array = data_array[~np.isnan(data_array)]
        n = len(data_array)
        if n < 3:
            raise ValueError(f'Shapiro-Wilk test requires at least 3 samples, got {n}')
        if n > 5000:
            raise ValueError(f'Shapiro-Wilk test is not reliable for sample sizes > 5000, got {n}')
        try:
            (statistic, p_value) = stats.shapiro(data_array)
            is_normal = p_value > significance_level
            critical_value = significance_level
            results['feature_0'] = {'statistic': float(statistic), 'p_value': float(p_value), 'is_normal': bool(is_normal), 'critical_value': float(critical_value), 'sample_size': int(n)}
        except Exception as e:
            raise ValueError(f'Error performing Shapiro-Wilk test: {e}')
    else:
        raise TypeError(f'Unsupported input type: {type(data)}. Expected DataBatch or list.')
    return results

def anderson_darling_test(data: Union[DataBatch, list], feature_names: Optional[list]=None, distribution: str='norm') -> Dict[str, Any]:
    """
    Perform the Anderson-Darling test for distributional goodness of fit.
    
    The Anderson-Darling test evaluates whether a sample comes from a specified 
    distribution. It is particularly sensitive to the tails of the distribution,
    making it more powerful than the Kolmogorov-Smirnov test for this purpose.
    
    This test supports multiple distributions including normal, exponential, 
    logistic, and Gumbel distributions. For normal distributions, it provides
    critical values for common significance levels.
    
    Args:
        data (Union[DataBatch, list]): Input data to test for distributional fit. 
            If DataBatch, tests are performed on specified features. If list, treats as single feature.
        feature_names (Optional[list]): Names of features to test if data is DataBatch.
            If None, tests all numeric features.
        distribution (str): Name of the distribution to test against. Supported values:
            'norm' (normal), 'expon' (exponential), 'logistic', 'gumbel' (default: 'norm').
            
    Returns:
        Dict[str, Any]: Dictionary containing test results for each feature:
            - 'statistic': Anderson-Darling test statistic for each feature
            - 'critical_values': Critical values for the test
            - 'significance_levels': Corresponding significance levels for critical values
            - 'p_value': Estimated p-value (if available for the distribution)
            - 'is_dist_fit': Boolean indicating if feature fits the specified distribution
            - 'distribution': Name of the tested distribution
            - 'sample_size': Number of observations for each feature
            
    Raises:
        ValueError: If data is empty, contains non-numeric values, or distribution is unsupported.
        TypeError: If data is not a supported type.
    """
    supported_distributions = ['norm', 'expon', 'logistic', 'gumbel']
    if distribution not in supported_distributions:
        raise ValueError(f"Unsupported distribution '{distribution}'. Supported distributions: {supported_distributions}")
    features = {}
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError('Input data is empty')
        try:
            feature_data = np.array(data, dtype=float)
        except (ValueError, TypeError):
            raise ValueError('Data contains non-numeric values')
        valid_data = feature_data[~np.isnan(feature_data)]
        if len(valid_data) == 0:
            raise ValueError('Input data contains no valid numeric values')
        features['feature_0'] = valid_data
        feature_names_list = ['feature_0']
    elif isinstance(data, DataBatch):
        if isinstance(data.data, np.ndarray):
            array_data = data.data
        else:
            array_data = np.array(data.data)
        if array_data.size == 0:
            raise ValueError('Input data is empty')
        if array_data.ndim == 1:
            array_data = array_data.reshape(-1, 1)
        elif array_data.ndim != 2:
            raise ValueError('Data must be 1D or 2D')
        if data.feature_names is not None:
            all_feature_names = data.feature_names
        else:
            all_feature_names = [f'feature_{i}' for i in range(array_data.shape[1])]
        if feature_names is None:
            feature_names_list = []
            for (i, name) in enumerate(all_feature_names):
                try:
                    col_data = array_data[:, i]
                    col_data_float = np.array(col_data, dtype=float)
                    valid_data = col_data_float[~np.isnan(col_data_float)]
                    if len(valid_data) > 0:
                        feature_names_list.append(name)
                except (ValueError, TypeError):
                    continue
            if not feature_names_list:
                feature_names_list = all_feature_names[:array_data.shape[1]]
        else:
            invalid_features = set(feature_names) - set(all_feature_names)
            if invalid_features:
                raise ValueError(f'Specified feature names not found in data: {invalid_features}')
            feature_names_list = feature_names
        for name in feature_names_list:
            try:
                idx = all_feature_names.index(name)
                feature_data = array_data[:, idx]
                feature_data_float = np.array(feature_data, dtype=float)
                valid_data = feature_data_float[~np.isnan(feature_data_float)]
                if len(valid_data) == 0:
                    raise ValueError(f"Feature '{name}' contains no valid numeric data")
                features[name] = valid_data
            except (ValueError, TypeError) as e:
                raise ValueError(f"Feature '{name}' contains non-numeric data: {e}")
    else:
        raise TypeError('Data must be either a DataBatch or a list')
    results = {}
    for (feature_name, valid_data) in features.items():
        n = len(valid_data)
        if n == 0:
            raise ValueError(f"Feature '{feature_name}' contains no valid data")
        if distribution == 'gumbel':
            test_result = stats.anderson(valid_data, dist='gumbel')
        else:
            test_result = stats.anderson(valid_data, dist=distribution)
        p_value = None
        is_dist_fit = False
        if distribution == 'norm':
            A_squared = float(test_result.statistic)
            if A_squared < 0.2:
                p_value = 1 - np.exp(-13.436 + 101.14 * A_squared - 223.73 * A_squared ** 2)
            elif A_squared < 0.34:
                p_value = 1 - np.exp(-8.318 + 42.796 * A_squared - 59.938 * A_squared ** 2)
            elif A_squared < 0.6:
                p_value = np.exp(0.9177 - 4.279 * A_squared - 1.38 * A_squared ** 2)
            elif A_squared < 10:
                p_value = np.exp(1.2937 - 5.709 * A_squared + 0.0186 * A_squared ** 2)
            else:
                p_value = 0
            p_value = float(max(0, min(1, p_value)))
            is_dist_fit = bool(p_value > 0.05)
        else:
            critical_values = test_result.critical_values
            significance_levels = test_result.significance_level
            for (i, crit_val) in enumerate(critical_values):
                if test_result.statistic < crit_val:
                    is_dist_fit = True
                    break
            is_dist_fit = bool(is_dist_fit)
        results[feature_name] = {'statistic': float(test_result.statistic), 'critical_values': [float(cv) for cv in test_result.critical_values], 'significance_levels': [float(sl) for sl in test_result.significance_level], 'p_value': p_value, 'is_dist_fit': bool(is_dist_fit), 'distribution': distribution, 'sample_size': int(n)}
    return results

def kolmogorov_smirnov_test(data: Union[DataBatch, list], feature_names: Optional[List[str]]=None, reference_distribution: Union[str, callable]='norm', reference_params: tuple=(), significance_level: float=0.05) -> Dict[str, Any]:
    """
    Perform the Kolmogorov-Smirnov test for distributional goodness of fit.
    
    The Kolmogorov-Smirnov test evaluates the null hypothesis that a sample comes
    from a specified distribution. It compares the empirical distribution function
    of the sample with the cumulative distribution function of the reference
    distribution or compares two empirical distributions.
    
    This test is less sensitive to tail differences compared to Anderson-Darling
    but is more general and can be applied to any continuous distribution.
    
    Args:
        data (Union[DataBatch, list]): Input data to test for distributional fit.
            If DataBatch, tests are performed on specified features. If list, treats as single feature.
        feature_names (Optional[list]): Names of features to test if data is DataBatch.
            If None, tests all numeric features.
        reference_distribution (Union[str, callable]): Reference distribution to test against.
            Can be a string naming a distribution ('norm', 'expon', 'uniform', etc.) or
            a callable that computes the CDF.
        reference_params (Optional[tuple]): Parameters for the reference distribution.
            For example, (mean, std) for a normal distribution.
            
    Returns:
        Dict[str, Any]: Dictionary containing test results for each feature:
            - 'statistic': Kolmogorov-Smirnov test statistic (D) for each feature
            - 'p_value': p-value for each feature
            - 'is_dist_fit': Boolean indicating if feature fits the specified distribution
            - 'reference_distribution': Name or description of the reference distribution
            - 'sample_size': Number of observations for each feature
            - 'critical_value': Critical value at 5% significance level
            
    Raises:
        ValueError: If data is empty, contains non-numeric values, or distribution parameters are invalid.
        TypeError: If data is not a supported type.
    """
    from scipy import stats
    import numpy as np
    if not 0 < significance_level < 1:
        raise ValueError('Significance level must be between 0 and 1')
    results = {}
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError('Input data is empty')
        try:
            data_array = np.array(data, dtype=float)
        except (ValueError, TypeError):
            raise ValueError('Input list must contain numeric data')
        data_array = data_array[~np.isnan(data_array)]
        if len(data_array) == 0:
            raise ValueError('Input data contains no valid numeric values')
        if len(data_array) < 2:
            raise ValueError('Kolmogorov-Smirnov test requires at least 2 samples')
        try:
            if callable(reference_distribution):

                def vectorized_cdf(x):
                    return np.vectorize(lambda xi: reference_distribution(xi, *reference_params))(x)
                (statistic, p_value) = stats.kstest(data_array, vectorized_cdf)
            else:
                (statistic, p_value) = stats.kstest(data_array, reference_distribution, args=reference_params)
        except Exception as e:
            raise ValueError(f'Error performing Kolmogorov-Smirnov test: {e}')
        n = len(data_array)
        critical_value = 1.36 / np.sqrt(n)
        is_dist_fit = p_value > significance_level
        results['feature_0'] = {'statistic': float(statistic), 'p_value': float(p_value), 'is_dist_fit': bool(is_dist_fit), 'reference_distribution': reference_distribution if isinstance(reference_distribution, str) else 'custom_cdf', 'sample_size': int(n), 'critical_value': float(critical_value)}
    elif isinstance(data, DataBatch):
        if isinstance(data.data, np.ndarray):
            data_array = data.data
        else:
            data_array = np.array(data.data)
        if data_array.size == 0:
            raise ValueError('Input data is empty')
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        elif data_array.ndim != 2:
            raise ValueError('Data must be 1D or 2D')
        if data.feature_names is not None:
            all_feature_names = data.feature_names
        else:
            all_feature_names = [f'feature_{i}' for i in range(data_array.shape[1])]
        if feature_names is None:
            selected_features = []
            selected_indices = []
            for (i, name) in enumerate(all_feature_names):
                try:
                    col_data = data_array[:, i]
                    col_data_float = np.array(col_data, dtype=float)
                    valid_data = col_data_float[~np.isnan(col_data_float)]
                    if len(valid_data) >= 2:
                        selected_features.append(name)
                        selected_indices.append(i)
                except (ValueError, TypeError):
                    continue
            if not selected_features:
                selected_features = all_feature_names[:data_array.shape[1]]
                selected_indices = list(range(min(len(all_feature_names), data_array.shape[1])))
        else:
            invalid_features = set(feature_names) - set(all_feature_names)
            if invalid_features:
                raise ValueError(f'Specified feature names not found in data: {invalid_features}')
            selected_features = feature_names
            selected_indices = [all_feature_names.index(name) for name in feature_names]
        for (name, idx) in zip(selected_features, selected_indices):
            try:
                feature_data = data_array[:, idx]
                feature_data_float = np.array(feature_data, dtype=float)
                valid_data = feature_data_float[~np.isnan(feature_data_float)]
                if len(valid_data) == 0:
                    raise ValueError(f"Feature '{name}' contains no valid numeric data")
                if len(valid_data) < 2:
                    raise ValueError(f"Feature '{name}' requires at least 2 samples for KS test, got {len(valid_data)}")
                if callable(reference_distribution):

                    def vectorized_cdf(x):
                        return np.vectorize(lambda xi: reference_distribution(xi, *reference_params))(x)
                    (statistic, p_value) = stats.kstest(valid_data, vectorized_cdf)
                else:
                    (statistic, p_value) = stats.kstest(valid_data, reference_distribution, args=reference_params)
                n = len(valid_data)
                critical_value = 1.36 / np.sqrt(n)
                is_dist_fit = p_value > significance_level
                results[name] = {'statistic': float(statistic), 'p_value': float(p_value), 'is_dist_fit': bool(is_dist_fit), 'reference_distribution': reference_distribution if isinstance(reference_distribution, str) else 'custom_cdf', 'sample_size': int(n), 'critical_value': float(critical_value)}
            except Exception as e:
                raise ValueError(f"Error performing Kolmogorov-Smirnov test on feature '{name}': {e}")
    else:
        raise TypeError(f'Unsupported input type: {type(data)}. Expected DataBatch or list.')
    return results

def autocorrelation_test(data: Union[DataBatch, list], feature_names: Optional[list]=None, max_lags: int=10, significance_level: float=0.05) -> Dict[str, Any]:
    """
    Perform autocorrelation tests on time series data to detect serial dependence.
    
    This function computes autocorrelation coefficients for specified lags and
    performs statistical tests to determine if autocorrelations are significantly
    different from zero. It helps identify patterns, cycles, and dependencies
    in time series data that violate independence assumptions.
    
    The test is particularly useful for validating assumptions in time series
    analysis and detecting non-randomness in data sequences.
    
    Args:
        data (Union[DataBatch, list]): Time series data to test for autocorrelation.
            If DataBatch, tests are performed on specified features. If list, treats as single series.
        feature_names (Optional[list]): Names of features to test if data is DataBatch.
            If None, tests all numeric features.
        max_lags (int): Maximum number of lags to compute autocorrelations for (default: 10).
        significance_level (float): Significance level for testing autocorrelations (default: 0.05).
            
    Returns:
        Dict[str, Any]: Dictionary containing autocorrelation test results for each feature:
            - 'autocorrelations': Autocorrelation coefficients for each lag
            - 'partial_autocorrelations': Partial autocorrelation coefficients for each lag
            - 'significant_lags': List of lags with significant autocorrelations
            - 'q_statistic': Ljung-Box Q-statistic for overall autocorrelation
            - 'q_p_value': P-value for the Q-statistic
            - 'has_autocorrelation': Boolean indicating presence of significant autocorrelation
            - 'sample_size': Number of observations
            - 'lags': Array of lag values tested
            
    Raises:
        ValueError: If data is empty, contains non-numeric values, or max_lags is invalid.
        TypeError: If data is not a supported type.
    """
    if not isinstance(max_lags, int) or max_lags < 0:
        raise ValueError('max_lags must be a non-negative integer')
    if not 0 < significance_level < 1:
        raise ValueError('significance_level must be between 0 and 1')
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError('Input data cannot be empty')
        try:
            series_data = np.array(data, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f'Failed to convert input data to numerical array: {e}')
        if len(series_data) < 2:
            raise ValueError('Sample size must be at least 2 for autocorrelation test')
        adjusted_max_lags = min(max_lags, len(series_data) - 1)
        n = len(series_data)
        lags = list(range(adjusted_max_lags + 1))
        autocorrelations = []
        for lag in lags:
            if lag == 0:
                autocorrelations.append(1.0)
            else:
                x = series_data[:-lag]
                y = series_data[lag:]
                if len(x) > 1 and np.std(x) > 0 and (np.std(y) > 0):
                    corr = np.corrcoef(x, y)[0, 1]
                    autocorrelations.append(corr if not np.isnan(corr) else 0.0)
                else:
                    autocorrelations.append(0.0)
        autocorrelations = np.array(autocorrelations)
        partial_autocorrelations = _compute_partial_autocorrelations(autocorrelations.tolist(), adjusted_max_lags)
        partial_autocorrelations = np.array(partial_autocorrelations)
        significant_lags = []
        for i in range(1, len(autocorrelations)):
            if i <= adjusted_max_lags:
                threshold = 1.96 / np.sqrt(n)
                if abs(autocorrelations[i]) > threshold:
                    significant_lags.append(i)
        q_statistic = 0
        for k in range(1, adjusted_max_lags + 1):
            if k < len(autocorrelations):
                rho_k = autocorrelations[k]
                q_statistic += rho_k ** 2 / (n - k)
        q_statistic *= n * (n + 2)
        q_p_value = 1 - stats.chi2.cdf(q_statistic, df=adjusted_max_lags) if adjusted_max_lags > 0 else 1.0
        has_autocorrelation = q_p_value < significance_level
        return {'autocorrelations': autocorrelations, 'partial_autocorrelations': partial_autocorrelations, 'significant_lags': significant_lags, 'q_statistic': float(q_statistic), 'q_p_value': float(q_p_value), 'has_autocorrelation': bool(has_autocorrelation), 'sample_size': n, 'lags': np.array(lags)}
    elif isinstance(data, DataBatch):
        raw_data = data.data
        all_feature_names = data.feature_names
        if not isinstance(raw_data, np.ndarray):
            try:
                array_data = np.array(raw_data, dtype=float)
            except (ValueError, TypeError) as e:
                raise ValueError(f'Failed to convert DataBatch data to numerical array: {e}')
        elif not np.issubdtype(raw_data.dtype, np.number):
            try:
                array_data = raw_data.astype(float)
            except (ValueError, TypeError) as e:
                raise ValueError(f'DataBatch contains non-numerical values that cannot be converted: {e}')
        else:
            array_data = raw_data
        if array_data.ndim == 1:
            array_data = array_data.reshape(-1, 1)
        elif array_data.ndim != 2:
            raise ValueError('DataBatch data must be 1D or 2D')
        if feature_names is None:
            if all_feature_names is None:
                feature_names_selected = [f'feature_{i}' for i in range(array_data.shape[1])]
            else:
                feature_names_selected = all_feature_names
            selected_features = []
            selected_indices = []
            for (i, name) in enumerate(feature_names_selected):
                try:
                    col_data = array_data[:, i]
                    col_data_float = np.array(col_data, dtype=float)
                    valid_data = col_data_float[~np.isnan(col_data_float)]
                    if len(valid_data) >= 2:
                        selected_features.append(name)
                        selected_indices.append(i)
                except (ValueError, TypeError):
                    continue
            if not selected_features:
                selected_features = feature_names_selected[:array_data.shape[1]]
                selected_indices = list(range(min(len(feature_names_selected), array_data.shape[1])))
        else:
            if all_feature_names is None:
                all_feature_names = [f'feature_{i}' for i in range(array_data.shape[1])]
            invalid_features = set(feature_names) - set(all_feature_names)
            if invalid_features:
                raise ValueError(f'Specified feature names not found: {invalid_features}')
            selected_features = feature_names
            selected_indices = [all_feature_names.index(name) for name in feature_names]
        results = {}
        for (name, idx) in zip(selected_features, selected_indices):
            try:
                feature_data = array_data[:, idx]
                feature_data_float = np.array(feature_data, dtype=float)
                valid_data = feature_data_float[~np.isnan(feature_data_float)]
                if len(valid_data) == 0:
                    raise ValueError(f"Feature '{name}' contains no valid numeric data")
                if len(valid_data) < 2:
                    raise ValueError(f"Feature '{name}' requires at least 2 samples for autocorrelation test, got {len(valid_data)}")
                adjusted_max_lags = min(max_lags, len(valid_data) - 1)
                n = len(valid_data)
                lags = list(range(adjusted_max_lags + 1))
                autocorrelations = []
                for lag in lags:
                    if lag == 0:
                        autocorrelations.append(1.0)
                    else:
                        x = valid_data[:-lag]
                        y = valid_data[lag:]
                        if len(x) > 1 and np.std(x) > 0 and (np.std(y) > 0):
                            corr = np.corrcoef(x, y)[0, 1]
                            autocorrelations.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            autocorrelations.append(0.0)
                autocorrelations = np.array(autocorrelations)
                partial_autocorrelations = _compute_partial_autocorrelations(autocorrelations.tolist(), adjusted_max_lags)
                partial_autocorrelations = np.array(partial_autocorrelations)
                significant_lags = []
                for i in range(1, len(autocorrelations)):
                    if i <= adjusted_max_lags:
                        threshold = 1.96 / np.sqrt(n)
                        if abs(autocorrelations[i]) > threshold:
                            significant_lags.append(i)
                q_statistic = 0
                for k in range(1, adjusted_max_lags + 1):
                    if k < len(autocorrelations):
                        rho_k = autocorrelations[k]
                        q_statistic += rho_k ** 2 / (n - k)
                q_statistic *= n * (n + 2)
                q_p_value = 1 - stats.chi2.cdf(q_statistic, df=adjusted_max_lags) if adjusted_max_lags > 0 else 1.0
                has_autocorrelation = q_p_value < significance_level
                results[name] = {'autocorrelations': autocorrelations, 'partial_autocorrelations': partial_autocorrelations, 'significant_lags': significant_lags, 'q_statistic': float(q_statistic), 'q_p_value': float(q_p_value), 'has_autocorrelation': bool(has_autocorrelation), 'sample_size': n, 'lags': np.array(lags)}
            except Exception as e:
                raise ValueError(f"Error performing autocorrelation test on feature '{name}': {e}")
        return results
    else:
        raise TypeError(f'Unsupported input type: {type(data)}. Expected DataBatch or list.')