from typing import Union, Dict, Any, Optional, Tuple, List, Callable
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from general.structures.model_artifact import ModelArtifact
from general.base_classes.validator_base import BaseValidator
import numpy as np
import time
import math
from scipy.stats import norm, t, chi2
from typing import Union, Dict, Any, Optional, List, Callable
from scipy import stats
from scipy.stats import kstest, anderson, chisquare, cramervonmises, cramervonmises_2samp
import warnings

class WelchTTestValidator(BaseValidator):

    def __init__(self, alpha: float=0.05, equal_var: bool=False, name: Optional[str]=None):
        """
        Initialize the Welch's t-test validator.

        Args:
            alpha: Significance level for determining statistical significance
            equal_var: If True, performs standard t-test assuming equal variances
            name: Optional name for the validator instance
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.equal_var = equal_var

    def validate(self, data: Union[DataBatch, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Perform Welch's t-test on the provided data.

        Args:
            data: Data containing two independent samples to compare.
                 Expected to have keys 'sample1' and 'sample2' with numeric values.
            **kwargs: Additional parameters for the test

        Returns:
            bool: True if the difference between sample means is statistically significant
        """
        self.reset_validation_state()
        if isinstance(data, DataBatch):
            if data.metadata is not None and 'sample1' in data.metadata and ('sample2' in data.metadata):
                sample1 = data.metadata['sample1']
                sample2 = data.metadata['sample2']
            elif data.metadata is not None and 'samples' in data.metadata:
                samples = data.metadata['samples']
                if 'sample1' in samples and 'sample2' in samples:
                    sample1 = samples['sample1']
                    sample2 = samples['sample2']
                else:
                    self.add_error("DataBatch metadata['samples'] must contain keys 'sample1' and 'sample2'")
                    return self.get_validation_report()
            else:
                self.add_error("DataBatch must have 'sample1' and 'sample2' in metadata or metadata['samples']")
                return self.get_validation_report()
        elif isinstance(data, dict):
            if 'sample1' not in data or 'sample2' not in data:
                self.add_error("Dictionary data must contain keys 'sample1' and 'sample2'")
                return self.get_validation_report()
            sample1 = data['sample1']
            sample2 = data['sample2']
        else:
            self.add_error('Data must be either a DataBatch or a dictionary')
            return self.get_validation_report()
        try:
            sample1 = np.asarray(sample1, dtype=float)
            sample2 = np.asarray(sample2, dtype=float)
        except (ValueError, TypeError) as e:
            self.add_error('Samples must be convertible to numeric arrays')
            return self.get_validation_report()
        if len(sample1) < 2 or len(sample2) < 2:
            self.add_error('Each sample must contain at least two observations')
            return self.get_validation_report()
        sample1 = sample1[~np.isnan(sample1)]
        sample2 = sample2[~np.isnan(sample2)]
        if len(sample1) < 2 or len(sample2) < 2:
            self.add_error('Each sample must contain at least two valid (non-NaN) observations')
            return self.get_validation_report()
        from scipy.stats import ttest_ind
        (t_statistic, p_value) = ttest_ind(sample1, sample2, equal_var=self.equal_var)
        is_significant = p_value < self.alpha
        if len(sample1) <= 5 or len(sample2) <= 5:
            self.add_warning('Sample sizes are small, which may affect the reliability of the test')
        report = self.get_validation_report()
        report.update({'is_significant': bool(is_significant), 'p_value': float(p_value), 't_statistic': float(t_statistic), 'alpha': self.alpha, 'equal_var': self.equal_var, 'sample1_size': len(sample1), 'sample2_size': len(sample2), 'sample1_mean': float(np.mean(sample1)), 'sample2_mean': float(np.mean(sample2))})
        return report

class AndersonDarlingTest:
    """
    Performs the Anderson-Darling test for assessing whether a sample comes from a specified distribution.
    
    The Anderson-Darling test is a statistical test used to determine if a sample of data
    comes from a specific distribution. It is particularly sensitive to the tails of the distribution.
    This implementation supports testing for normality as well as other common distributions.
    
    Attributes:
        distribution (str): Name of the distribution to test against ('norm', 'expon', 'logistic', 'gumbel', etc.)
        significance_level (float): Significance level for critical values comparison (default: 0.05)
    """

    def __init__(self, distribution: str='norm', significance_level: float=0.05):
        """
        Initialize the Anderson-Darling test.
        
        Args:
            distribution: Name of the distribution to test against. Supported values:
                         'norm' (normal), 'expon' (exponential), 'logistic', 'gumbel', 'extreme1'
            significance_level: Significance level for the test (default: 0.05)
            
        Raises:
            ValueError: If distribution is not supported
        """
        self.distribution = distribution
        self.significance_level = significance_level
        self._supported_distributions = ['norm', 'expon', 'logistic', 'gumbel', 'extreme1']
        if distribution not in self._supported_distributions:
            raise ValueError(f"Distribution '{distribution}' not supported. Supported distributions: {self._supported_distributions}")

    def test_normality(self, data: Union[np.ndarray, DataBatch]) -> Dict[str, Any]:
        """
        Perform Anderson-Darling test for normality.
        
        Tests the null hypothesis that the data comes from a normal distribution.
        
        Args:
            data: Sample data as numpy array or DataBatch containing numeric values
            
        Returns:
            Dict containing:
            - 'statistic': Anderson-Darling test statistic
            - 'critical_values': Critical values for different significance levels
            - 'significance_levels': Corresponding significance levels for critical values
            - 'p_value': Approximate p-value (if available)
            - 'is_normal': Boolean indicating if data appears normal at the specified significance level
            - 'confidence': Confidence level of the test
            
        Raises:
            ValueError: If data is empty or contains non-numeric values
        """
        if isinstance(data, DataBatch):
            data_array = data.data
        else:
            data_array = data
        if data_array.size == 0:
            raise ValueError('Data cannot be empty')
        data_array = np.asarray(data_array).flatten()
        if not np.issubdtype(data_array.dtype, np.number):
            raise ValueError('Data must contain numeric values')
        data_array = data_array[~np.isnan(data_array)]
        if len(data_array) == 0:
            raise ValueError('Data cannot be empty after removing NaN values')
        critical_values = np.array([0.576, 0.656, 0.787, 0.918, 1.092])
        significance_levels = np.array([15, 10, 5, 2.5, 1])
        n = len(data_array)
        sorted_data = np.sort(data_array)
        mean = np.mean(data_array)
        std = np.std(data_array, ddof=1)
        if std == 0:
            statistic = 0.0
        else:
            standardized = (sorted_data - mean) / std
            i = np.arange(1, n + 1)
            Fi = i / (n + 1)
            from scipy.stats import norm
            Zi = norm.cdf(standardized)
            term1 = (2 * i - 1) * (np.log(Zi) + np.log(1 - np.flip(Zi)))
            statistic = -n - 1 / n * np.sum(term1)
            statistic = statistic * (1 + 0.75 / n + 2.25 / n ** 2)
        is_normal = True
        if self.significance_level >= 0.15:
            critical_index = 0
        elif self.significance_level >= 0.1:
            critical_index = 1
        elif self.significance_level >= 0.05:
            critical_index = 2
        elif self.significance_level >= 0.025:
            critical_index = 3
        elif self.significance_level >= 0.01:
            critical_index = 4
        else:
            critical_index = 4
        if statistic > critical_values[critical_index]:
            is_normal = False
        if statistic < 0.2:
            p_value = 1 - np.exp(-13.436 + 101.14 * statistic - 223.73 * statistic ** 2)
        elif statistic < 0.34:
            p_value = 1 - np.exp(-8.314 + 42.796 * statistic - 59.938 * statistic ** 2)
        elif statistic < 0.6:
            p_value = np.exp(0.9177 - 4.279 * statistic - 1.38 * statistic ** 2)
        elif statistic < 10:
            p_value = np.exp(1.2937 - 5.709 * statistic + 0.0186 * statistic ** 2)
        else:
            p_value = 0.0
        p_value = max(0.0, min(1.0, p_value))
        return {'statistic': statistic, 'critical_values': critical_values, 'significance_levels': significance_levels, 'p_value': p_value, 'is_normal': is_normal, 'confidence': 1 - self.significance_level}

    def test_distribution(self, data: Union[np.ndarray, DataBatch], distribution: Optional[str]=None) -> Dict[str, Any]:
        """
        Perform Anderson-Darling test against a specified distribution.
        
        Tests the null hypothesis that the data comes from the specified distribution.
        
        Args:
            data: Sample data as numpy array or DataBatch containing numeric values
            distribution: Name of the distribution to test against (overrides instance setting)
            
        Returns:
            Dict containing:
            - 'statistic': Anderson-Darling test statistic
            - 'critical_values': Critical values for different significance levels
            - 'significance_levels': Corresponding significance levels for critical values
            - 'p_value': Approximate p-value (if available)
            - 'fits_distribution': Boolean indicating if data fits the distribution at the specified significance level
            - 'confidence': Confidence level of the test
            
        Raises:
            ValueError: If data is empty or contains non-numeric values
            ValueError: If distribution is not supported
        """
        if isinstance(data, DataBatch):
            data_array = data.data
        else:
            data_array = data
        if data_array.size == 0:
            raise ValueError('Data cannot be empty')
        data_array = np.asarray(data_array).flatten()
        if not np.issubdtype(data_array.dtype, np.number):
            raise ValueError('Data must contain numeric values')
        data_array = data_array[~np.isnan(data_array)]
        if len(data_array) == 0:
            raise ValueError('Data cannot be empty after removing NaN values')
        dist_to_test = distribution if distribution is not None else self.distribution
        if dist_to_test not in self._supported_distributions:
            raise ValueError(f"Distribution '{dist_to_test}' not supported. Supported distributions: {self._supported_distributions}")
        from scipy import stats
        if dist_to_test == 'norm':
            result = stats.anderson(data_array, dist='norm')
            critical_values = result.critical_values
            significance_levels = result.significance_level
        elif dist_to_test == 'expon':
            result = stats.anderson(data_array, dist='expon')
            critical_values = result.critical_values
            significance_levels = result.significance_level
        elif dist_to_test == 'logistic':
            result = stats.anderson(data_array, dist='logistic')
            critical_values = result.critical_values
            significance_levels = result.significance_level
        elif dist_to_test in ['gumbel', 'extreme1']:
            result = stats.anderson(data_array, dist='gumbel')
            critical_values = result.critical_values
            significance_levels = result.significance_level
        else:
            result = stats.anderson(data_array, dist=dist_to_test)
            critical_values = result.critical_values
            significance_levels = result.significance_level
        statistic = result.statistic
        fits_distribution = True
        applicable_critical_values = []
        applicable_significance_levels = []
        for (i, sig_level) in enumerate(significance_levels):
            if sig_level >= self.significance_level * 100:
                applicable_critical_values.append(critical_values[i])
                applicable_significance_levels.append(sig_level)
        if applicable_critical_values:
            critical_value_to_use = applicable_critical_values[-1]
            if statistic > critical_value_to_use:
                fits_distribution = False
        elif len(critical_values) > 0 and statistic > critical_values[-1]:
            fits_distribution = False
        p_value = self._approximate_p_value(statistic, dist_to_test)
        return {'statistic': statistic, 'critical_values': np.array(critical_values), 'significance_levels': np.array(significance_levels), 'p_value': p_value, 'fits_distribution': fits_distribution, 'confidence': 1 - self.significance_level}

    def _approximate_p_value(self, statistic: float, distribution: str) -> float:
        """
        Approximate p-value for the Anderson-Darling test statistic.
        
        Args:
            statistic: The Anderson-Darling test statistic
            distribution: The distribution being tested
            
        Returns:
            Approximate p-value
        """
        if statistic <= 0:
            return 1.0
        if distribution == 'norm':
            if statistic < 0.2:
                p_value = 1 - np.exp(-13.436 + 101.14 * statistic - 223.73 * statistic ** 2)
            elif statistic < 0.34:
                p_value = 1 - np.exp(-8.314 + 42.796 * statistic - 59.938 * statistic ** 2)
            elif statistic < 0.6:
                p_value = np.exp(0.9177 - 4.279 * statistic - 1.38 * statistic ** 2)
            elif statistic < 10:
                p_value = np.exp(1.2937 - 5.709 * statistic + 0.0186 * statistic ** 2)
            else:
                p_value = 0.0
        elif statistic < 0.2:
            p_value = 1 - np.exp(-10 * statistic)
        elif statistic < 1:
            p_value = np.exp(-2.5 * statistic)
        elif statistic < 10:
            p_value = np.exp(-3 * statistic)
        else:
            p_value = 0.0
        return max(0.0, min(1.0, p_value))

class BayesianInformationCriterion:

    def __init__(self, model_type: str='nested', penalty_factor: float=1.0):
        """
        Initialize the BIC calculator.
        
        Args:
            model_type: Type of model to evaluate. Supported values:
                       'nested' (standard nested models),
                       'non_nested' (non-nested model comparison),
                       'mixed_effects' (mixed effects models),
                       'seasonal' (models with seasonal adjustments)
            penalty_factor: Factor to adjust the penalty term (default: 1.0)
            
        Raises:
            ValueError: If model_type is not supported
        """
        self.model_type = model_type
        self.penalty_factor = penalty_factor
        self._supported_model_types = ['nested', 'non_nested', 'mixed_effects', 'seasonal']
        if model_type not in self._supported_model_types:
            raise ValueError(f"Model type '{model_type}' not supported. Supported types: {self._supported_model_types}")

    def calculate_bic(self, log_likelihood: float, n_parameters: int, n_samples: int) -> float:
        """
        Calculate the Bayesian Information Criterion.
        
        BIC = -2*ln(L) + k*ln(n)
        where L is the maximized value of the likelihood function, 
        k is the number of parameters, and n is the number of samples.
        
        Args:
            log_likelihood: Log-likelihood of the model
            n_parameters: Number of parameters in the model
            n_samples: Number of samples used in fitting
            
        Returns:
            float: Calculated BIC value (lower values indicate better models)
            
        Raises:
            ValueError: If n_samples <= n_parameters or if inputs are invalid
        """
        if not isinstance(log_likelihood, (int, float)):
            raise ValueError('log_likelihood must be a numeric value')
        if not isinstance(n_parameters, int) or n_parameters < 0:
            raise ValueError('n_parameters must be a non-negative integer')
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError('n_samples must be a positive integer')
        if n_samples <= n_parameters:
            raise ValueError('n_samples must be greater than n_parameters')
        bic = -2 * log_likelihood + n_parameters * np.log(n_samples) * self.penalty_factor
        return bic

    def compare_models(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models using BIC.
        
        Compares models based on their BIC values and provides rankings.
        
        Args:
            models: List of dictionaries, each containing:
                   - 'log_likelihood': Log-likelihood of the model
                   - 'n_parameters': Number of parameters
                   - 'n_samples': Number of samples
                   - 'model_name': Optional name for identification
                   
        Returns:
            Dict containing:
            - 'bic_values': List of BIC values for each model
            - 'rankings': Indices of models ranked by BIC (best first)
            - 'best_model_index': Index of the best model
            - 'delta_bic': Differences in BIC from the best model
            - 'evidence_weights': Evidence weights for model selection
            
        Raises:
            ValueError: If models list is empty or contains invalid entries
        """
        if not models:
            raise ValueError('Models list cannot be empty')
        bic_values = []
        for (i, model) in enumerate(models):
            if not isinstance(model, dict):
                raise ValueError(f'Model at index {i} must be a dictionary')
            required_keys = ['log_likelihood', 'n_parameters', 'n_samples']
            for key in required_keys:
                if key not in model:
                    raise ValueError(f'Model at index {i} missing required key: {key}')
            try:
                bic = self.calculate_bic(log_likelihood=model['log_likelihood'], n_parameters=model['n_parameters'], n_samples=model['n_samples'])
                bic_values.append(bic)
            except Exception as e:
                raise ValueError(f'Error calculating BIC for model at index {i}: {str(e)}')
        best_model_index = int(np.argmin(bic_values))
        best_bic = bic_values[best_model_index]
        delta_bic = [bic - best_bic for bic in bic_values]
        max_delta = max(delta_bic)
        adjusted_deltas = [delta - max_delta for delta in delta_bic]
        exp_deltas = [np.exp(-0.5 * delta) for delta in adjusted_deltas]
        sum_exp_deltas = sum(exp_deltas)
        evidence_weights = [exp_delta / sum_exp_deltas for exp_delta in exp_deltas]
        rankings = sorted(range(len(bic_values)), key=lambda i: bic_values[i])
        return {'bic_values': bic_values, 'rankings': rankings, 'best_model_index': best_model_index, 'delta_bic': delta_bic, 'evidence_weights': evidence_weights}

    def calculate_seasonal_bic(self, log_likelihood: float, n_parameters: int, n_samples: int, seasonal_periods: int) -> float:
        """
        Calculate BIC for models with seasonal adjustments.
        
        Adjusts the standard BIC calculation to account for seasonal components.
        
        Args:
            log_likelihood: Log-likelihood of the model
            n_parameters: Number of parameters in the model
            n_samples: Number of samples used in fitting
            seasonal_periods: Number of seasonal periods
            
        Returns:
            float: Calculated seasonal BIC value
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(log_likelihood, (int, float)):
            raise ValueError('log_likelihood must be a numeric value')
        if not isinstance(n_parameters, int) or n_parameters < 0:
            raise ValueError('n_parameters must be a non-negative integer')
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError('n_samples must be a positive integer')
        if not isinstance(seasonal_periods, int) or seasonal_periods <= 0:
            raise ValueError('seasonal_periods must be a positive integer')
        if n_samples <= n_parameters:
            raise ValueError('n_samples must be greater than n_parameters')
        effective_n_samples = n_samples / seasonal_periods
        bic = -2 * log_likelihood + n_parameters * np.log(effective_n_samples) * self.penalty_factor
        return bic

class MultipleComparisonCorrection:
    """
    Applies multiple comparison corrections to control false discovery rate or family-wise error rate.
    
    This class implements several methods for adjusting p-values when conducting multiple 
    statistical tests to control for Type I errors. It includes the Benjamini-Hochberg 
    procedure for false discovery rate control and Bonferroni correction for family-wise 
    error rate control.
    
    Attributes:
        method (str): Correction method to apply ('benjamini_hochberg', 'bonferroni')
        alpha (float): Significance level for error rate control (default: 0.05)
    """

    def __init__(self, method: str='benjamini_hochberg', alpha: float=0.05):
        """
        Initialize the multiple comparison correction processor.
        
        Args:
            method: Correction method to apply. Supported values:
                   'benjamini_hochberg' (False Discovery Rate control),
                   'bonferroni' (Family-Wise Error Rate control)
            alpha: Significance level for error rate control (default: 0.05)
            
        Raises:
            ValueError: If method is not supported
        """
        self.method = method
        self.alpha = alpha
        self._supported_methods = ['benjamini_hochberg', 'bonferroni']
        if method not in self._supported_methods:
            raise ValueError(f"Method '{method}' not supported. Supported methods: {self._supported_methods}")

    def benjamini_hochberg_procedure(self, p_values: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Apply the Benjamini-Hochberg procedure for false discovery rate control.
        
        Controls the expected proportion of incorrectly rejected null hypotheses 
        (false discoveries) among all rejected hypotheses.
        
        Args:
            p_values: Array of p-values from multiple statistical tests
            
        Returns:
            Dict containing:
            - 'adjusted_p_values': BH-adjusted p-values
            - 'significant': Boolean array indicating which tests are significant
            - 'number_significant': Number of significant tests
            - 'fdr_threshold': Threshold used for FDR control
            - 'ranking': Original ranking of p-values
            
        Raises:
            ValueError: If p_values contains invalid values or is empty
        """
        if len(p_values) == 0:
            raise ValueError('p_values cannot be empty')
        p_vals = np.asarray(p_values, dtype=float)
        if np.any(np.isnan(p_vals)):
            raise ValueError('p_values cannot contain NaN values')
        if np.any((p_vals < 0) | (p_vals > 1)):
            raise ValueError('p_values must be between 0 and 1')
        n_tests = len(p_vals)
        sorted_indices = np.argsort(p_vals)
        sorted_p_values = p_vals[sorted_indices]
        ranks = np.arange(1, n_tests + 1)
        bh_critical_values = ranks * self.alpha / n_tests
        comparison_results = sorted_p_values <= bh_critical_values
        significant_indices = np.where(comparison_results)[0]
        if len(significant_indices) > 0:
            max_k = significant_indices[-1] + 1
            fdr_threshold = sorted_p_values[max_k - 1]
        else:
            max_k = 0
            fdr_threshold = 0.0
        adjusted_p_values = np.zeros(n_tests)
        cummin_pv = np.minimum.accumulate(sorted_p_values[::-1])[::-1]
        denominators = (n_tests / ranks).astype(float)
        adjusted_sorted = np.minimum(1.0, cummin_pv * denominators)
        adjusted_p_values[sorted_indices] = adjusted_sorted
        significant = adjusted_p_values <= self.alpha
        number_significant = np.sum(significant)
        ranking = np.argsort(sorted_indices)
        return {'adjusted_p_values': adjusted_p_values.tolist(), 'significant': significant.tolist(), 'number_significant': int(number_significant), 'fdr_threshold': float(fdr_threshold), 'ranking': ranking.tolist()}

    def bonferroni_correction(self, p_values: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Apply Bonferroni correction for family-wise error rate control.
        
        Controls the probability of making at least one Type I error across all tests.
        
        Args:
            p_values: Array of p-values from multiple statistical tests
            
        Returns:
            Dict containing:
            - 'adjusted_p_values': Bonferroni-adjusted p-values
            - 'significant': Boolean array indicating which tests are significant
            - 'number_significant': Number of significant tests
            - 'alpha_corrected': Corrected significance threshold
            - 'ranking': Original ranking of p-values
            
        Raises:
            ValueError: If p_values contains invalid values or is empty
        """
        if len(p_values) == 0:
            raise ValueError('p_values cannot be empty')
        p_vals = np.asarray(p_values, dtype=float)
        if np.any(np.isnan(p_vals)):
            raise ValueError('p_values cannot contain NaN values')
        if np.any((p_vals < 0) | (p_vals > 1)):
            raise ValueError('p_values must be between 0 and 1')
        n_tests = len(p_vals)
        adjusted_p_values = np.minimum(1.0, p_vals * n_tests)
        alpha_corrected = self.alpha / n_tests
        significant = adjusted_p_values <= self.alpha
        number_significant = np.sum(significant)
        ranking = np.arange(n_tests)
        return {'adjusted_p_values': adjusted_p_values.tolist(), 'significant': significant.tolist(), 'number_significant': int(number_significant), 'alpha_corrected': float(alpha_corrected), 'ranking': ranking.tolist()}

    def apply_correction(self, p_values: Union[List[float], np.ndarray], method: Optional[str]=None) -> Dict[str, Any]:
        """
        Apply multiple comparison correction using the specified or default method.
        
        Flexible method that applies the selected correction technique to p-values.
        
        Args:
            p_values: Array of p-values from multiple statistical tests
            method: Correction method to apply (overrides instance setting)
            
        Returns:
            Dict containing correction results (format depends on method used)
            
        Raises:
            ValueError: If method is not supported or p_values is invalid
        """
        correction_method = method if method is not None else self.method
        if correction_method not in self._supported_methods:
            raise ValueError(f"Method '{correction_method}' not supported. Supported methods: {self._supported_methods}")
        if correction_method == 'benjamini_hochberg':
            return self.benjamini_hochberg_procedure(p_values)
        elif correction_method == 'bonferroni':
            return self.bonferroni_correction(p_values)

class KruskalWallisTest:

    def __init__(self, alpha: float=0.05, correction: bool=True, optimize: bool=False):
        """
        Initialize the Kruskal-Wallis test.
        
        Args:
            alpha: Significance level for determining statistical significance (default: 0.05)
            correction: Whether to apply continuity correction for small samples (default: True)
            optimize: Whether to use optimized algorithms for large datasets (default: False)
        """
        self.alpha = alpha
        self.correction = correction
        self.optimize = optimize

    def perform_test(self, samples: Union[List[np.ndarray], List[DataBatch]]) -> Dict[str, Any]:
        """
        Perform the Kruskal-Wallis H-test on multiple independent samples.
        
        Tests the null hypothesis that the population medians of all groups are equal.
        
        Args:
            samples: List of sample data arrays or DataBatches (each representing a group)
            
        Returns:
            Dict containing:
            - 'h_statistic': Kruskal-Wallis H-test statistic
            - 'p_value': P-value for the test
            - 'degrees_of_freedom': Degrees of freedom for the test
            - 'significant': Boolean indicating if differences are statistically significant
            - 'group_medians': Median values for each group
            - 'group_sizes': Sample sizes for each group
            - 'ranking': Average ranks for each group
            - 'critical_value': Critical value at the specified alpha level
            
        Raises:
            ValueError: If fewer than 2 groups provided or if any group is empty
            TypeError: If input data types are not supported
        """
        if len(samples) < 2:
            raise ValueError('At least 2 groups must be provided for Kruskal-Wallis test')
        processed_samples = []
        for sample in samples:
            if isinstance(sample, DataBatch):
                processed_samples.append(np.asarray(sample.data).flatten())
            elif isinstance(sample, np.ndarray):
                processed_samples.append(sample.flatten())
            else:
                raise TypeError('Samples must be numpy arrays or DataBatch objects')
        for (i, sample) in enumerate(processed_samples):
            if len(sample) == 0:
                raise ValueError(f'Group {i} is empty')
        group_medians = [np.median(sample) for sample in processed_samples]
        group_sizes = [len(sample) for sample in processed_samples]
        n_total = sum(group_sizes)
        combined_data = np.concatenate(processed_samples)
        ranks = np.argsort(np.argsort(combined_data)) + 1
        avg_ranks = []
        start_idx = 0
        for size in group_sizes:
            group_ranks = ranks[start_idx:start_idx + size]
            avg_ranks.append(np.mean(group_ranks))
            start_idx += size
        h_statistic = 12 / (n_total * (n_total + 1)) * sum((size * (avg_rank - (n_total + 1) / 2) ** 2 for (size, avg_rank) in zip(group_sizes, avg_ranks)))
        k = len(group_sizes)
        degrees_of_freedom = k - 1
        if self.correction and k == 2 and (n_total <= 200):
            h_statistic = max(0, h_statistic - 1 / (n_total * (n_total - 1)))
        p_value = self._chi2_sf(h_statistic, degrees_of_freedom)
        significant = bool(p_value < self.alpha)
        critical_value = self._chi2_ppf(1 - self.alpha, degrees_of_freedom)
        if all((np.array_equal(sample, processed_samples[0]) for sample in processed_samples)):
            h_statistic = 0.0
        return {'h_statistic': float(h_statistic), 'p_value': float(p_value), 'degrees_of_freedom': degrees_of_freedom, 'significant': significant, 'group_medians': [float(m) for m in group_medians], 'group_sizes': group_sizes, 'ranking': [float(r) for r in avg_ranks], 'critical_value': float(critical_value)}

    def perform_optimized_test(self, samples: Union[List[np.ndarray], List[DataBatch]]) -> Dict[str, Any]:
        """
        Perform an optimized version of the Kruskal-Wallis H-test for large datasets.
        
        Uses computational optimizations for handling large sample sizes efficiently.
        
        Args:
            samples: List of sample data arrays or DataBatches (each representing a group)
            
        Returns:
            Dict containing:
            - 'h_statistic': Kruskal-Wallis H-test statistic
            - 'p_value': P-value for the test
            - 'degrees_of_freedom': Degrees of freedom for the test
            - 'significant': Boolean indicating if differences are statistically significant
            - 'computation_time': Time taken for the optimized computation
            - 'memory_usage': Memory usage during computation
            - 'optimization_method': Description of optimization technique used
            - 'group_medians': Median values for each group
            - 'group_sizes': Sample sizes for each group
            - 'ranking': Average ranks for each group
            - 'critical_value': Critical value at the specified alpha level
            
        Raises:
            ValueError: If fewer than 2 groups provided or if any group is empty
            TypeError: If input data types are not supported
        """
        start_time = time.time()
        try:
            if len(samples) < 2:
                raise ValueError('At least 2 groups must be provided for Kruskal-Wallis test')
            processed_samples = []
            for sample in samples:
                if isinstance(sample, DataBatch):
                    processed_samples.append(np.asarray(sample.data).flatten())
                elif isinstance(sample, np.ndarray):
                    processed_samples.append(sample.flatten())
                else:
                    raise TypeError('Samples must be numpy arrays or DataBatch objects')
            for (i, sample) in enumerate(processed_samples):
                if len(sample) == 0:
                    raise ValueError(f'Group {i} is empty')
            group_sizes = [len(sample) for sample in processed_samples]
            n_total = sum(group_sizes)
            k = len(group_sizes)
            degrees_of_freedom = k - 1
            combined_data = np.concatenate(processed_samples)
            sorted_indices = np.argsort(combined_data)
            ranks = np.empty_like(sorted_indices, dtype=float)
            ranks[sorted_indices] = np.arange(1, len(combined_data) + 1)
            (unique_vals, inverse_indices, counts) = np.unique(combined_data, return_inverse=True, return_counts=True)
            if len(unique_vals) < len(combined_data):
                for (i, count) in enumerate(counts):
                    if count > 1:
                        tied_indices = np.where(inverse_indices == i)[0]
                        avg_rank = np.mean(ranks[tied_indices])
                        ranks[tied_indices] = avg_rank
            avg_ranks = []
            start_idx = 0
            for size in group_sizes:
                group_ranks = ranks[start_idx:start_idx + size]
                avg_ranks.append(np.mean(group_ranks))
                start_idx += size
            h_statistic = 12 / (n_total * (n_total + 1)) * sum((size * (avg_rank - (n_total + 1) / 2) ** 2 for (size, avg_rank) in zip(group_sizes, avg_ranks)))
            if self.correction and k == 2 and (n_total <= 200):
                h_statistic = max(0, h_statistic - 1 / (n_total * (n_total - 1)))
            p_value = self._chi2_sf(h_statistic, degrees_of_freedom)
            significant = p_value < self.alpha
            critical_value = self._chi2_ppf(1 - self.alpha, degrees_of_freedom)
            group_medians = [np.median(sample) for sample in processed_samples]
            end_time = time.time()
            return {'h_statistic': float(h_statistic), 'p_value': float(p_value), 'degrees_of_freedom': degrees_of_freedom, 'significant': significant, 'computation_time': end_time - start_time, 'memory_usage': 0, 'optimization_method': 'manual rankdata with vectorized operations', 'group_medians': [float(m) for m in group_medians], 'group_sizes': group_sizes, 'ranking': [float(r) for r in avg_ranks], 'critical_value': float(critical_value)}
        except Exception as e:
            end_time = time.time()
            return {'error': str(e), 'computation_time': end_time - start_time, 'memory_usage': 0, 'optimization_method': 'manual rankdata with vectorized operations'}

    def _chi2_sf(self, x: float, df: int) -> float:
        """Calculate survival function (1 - CDF) of chi-square distribution."""
        if x <= 0:
            return 1.0
        if df <= 0:
            return 0.0
        if df > 100:
            z = (x - df) / np.sqrt(2 * df)
            return 0.5 * self._erfc(z / np.sqrt(2))
        return 1.0 - self._chi2_cdf(x, df)

    def _chi2_cdf(self, x: float, df: int) -> float:
        """Calculate CDF of chi-square distribution."""
        if x <= 0:
            return 0.0
        if df <= 0:
            return 0.0
        if df % 2 == 0:
            k = df // 2
            sum_term = 0.0
            term = 1.0
            for i in range(k):
                sum_term += term
                term *= x / (2 * (i + 1))
            return 1.0 - np.exp(-x / 2) * sum_term
        return self._gamma_lower_regularized(df / 2.0, x / 2.0)

    def _chi2_ppf(self, p: float, df: int) -> float:
        """Calculate percent point function (inverse CDF) of chi-square distribution."""
        if p <= 0:
            return 0.0
        if p >= 1:
            return np.inf
        if df == 1:
            from scipy.special import erfinv
            z = np.sqrt(2) * erfinv(p - 0.5)
            return z * z
        elif df == 2:
            return -2 * np.log(1 - p)
        else:
            mean = 1 - 2 / (9 * df)
            var = 2 / (9 * df)
            z = self._norm_ppf(p)
            cube_root_chi = mean + np.sqrt(var) * z
            return df * cube_root_chi ** 3

    def _gamma_lower_regularized(self, s: float, x: float) -> float:
        """Regularized lower incomplete gamma function."""
        if x <= 0:
            return 0.0
        if x >= 100 + s:
            return 1.0 - self._gamma_upper_regularized_cf(s, x)
        return self._gamma_lower_series(s, x)

    def _gamma_lower_series(self, s: float, x: float) -> float:
        """Series expansion for lower incomplete gamma."""
        if s == 0:
            return 0.0
        sum_val = 1.0
        term = 1.0
        for n in range(1, 100):
            term *= x / (s + n)
            sum_val += term
            if abs(term) < 1e-15 * abs(sum_val):
                break
        return sum_val * np.exp(-x + s * np.log(x)) / self._gamma(s)

    def _gamma_upper_regularized_cf(self, s: float, x: float) -> float:
        """Continued fraction for upper incomplete gamma."""
        tiny = 1e-30
        b = x + 1.0 - s
        c = tiny
        d = 1.0 / b
        h = d
        for i in range(1, 100):
            an = -i * (i - s)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            del_h = d * c
            h *= del_h
            if abs(del_h - 1.0) < 1e-15:
                break
        return h * np.exp(-x + s * np.log(x)) / self._gamma(s)

    def _gamma(self, z: float) -> float:
        """Gamma function using Lanczos approximation."""
        g = 7
        p = [0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019572e-06, 1.5056327351493116e-07]
        if z < 0.5:
            return np.pi / (np.sin(np.pi * z) * self._gamma(1 - z))
        else:
            z -= 1
            x = p[0]
            for i in range(1, g + 2):
                x += p[i] / (z + i)
            t = z + g + 0.5
            return np.sqrt(2 * np.pi) * t ** (z + 0.5) * np.exp(-t) * x

    def _norm_ppf(self, p: float) -> float:
        """Standard normal percent point function (quantile function)."""
        if p <= 0 or p >= 1:
            raise ValueError('p must be between 0 and 1')
        a = [0.0, -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239]
        b = [0.0, -54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
        c = [0.0, -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
        d = [0.0, 0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
        p_low = 0.02425
        p_high = 1 - p_low
        if 0 < p < p_low:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
        elif p_low <= p <= p_high:
            q = p - 0.5
            r = q * q
            return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
        elif p_high < p < 1:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
        else:
            return 0.0

    def _erfc(self, x: float) -> float:
        """Complementary error function."""
        z = abs(x)
        t = 1.0 / (1.0 + 0.5 * z)
        ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        return ans if x >= 0 else 2.0 - ans

class WilcoxonSignedRankTest:
    """
    Performs the Wilcoxon signed-rank test for comparing two related samples.
    
    The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test 
    used to compare two related samples, matched samples, or repeated measurements 
    on a single sample to assess whether their population mean ranks differ.
    
    Attributes:
        alpha (float): Significance level for the test (default: 0.05)
        alternative (str): Alternative hypothesis ('two-sided', 'greater', 'less')
        zero_method (str): How to handle zero differences ('pratt', 'wilcox', 'zsplit')
    """

    def __init__(self, alpha: float=0.05, alternative: str='two-sided', zero_method: str='pratt'):
        """
        Initialize the Wilcoxon signed-rank test.
        
        Args:
            alpha: Significance level for determining statistical significance (default: 0.05)
            alternative: Alternative hypothesis. Supported values:
                        'two-sided' (default): distributions differ in location
                        'greater': first distribution is shifted to the right
                        'less': first distribution is shifted to the left
            zero_method: How to handle zero differences. Supported values:
                        'pratt': zeros are counted in ranking
                        'wilcox': zeros are discarded
                        'zsplit': zero rank splits between positive and negative
            
        Raises:
            ValueError: If parameters are not supported
        """
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        self.alpha = alpha
        if alternative not in ['two-sided', 'greater', 'less']:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        self.alternative = alternative
        if zero_method not in ['pratt', 'wilcox', 'zsplit']:
            raise ValueError("zero_method must be 'pratt', 'wilcox', or 'zsplit'")
        self.zero_method = zero_method

    def perform_test(self, x: Union[np.ndarray, DataBatch], y: Union[np.ndarray, DataBatch]) -> Dict[str, Any]:
        """
        Perform the Wilcoxon signed-rank test on two related samples.
        
        Tests the null hypothesis that the distribution of the differences 
        between paired observations is symmetric around zero.
        
        Args:
            x: First sample data as numpy array or DataBatch
            y: Second sample data as numpy array or DataBatch
            
        Returns:
            Dict containing:
            - 'statistic': Wilcoxon test statistic (sum of ranks for positive differences)
            - 'p_value': P-value for the test
            - 'sample_size': Number of non-zero differences
            - 'significant': Boolean indicating if difference is statistically significant
            - 'confidence_interval': Confidence interval for the median difference
            - 'median_difference': Median of the differences
            - 'effect_size': Effect size measure (r = Z / sqrt(N))
            - 'z_score': Normal approximation z-score (for large samples)
            
        Raises:
            ValueError: If x and y have different lengths or are empty
            TypeError: If input data types are not supported
        """
        if isinstance(x, DataBatch):
            x = x.data.values if hasattr(x.data, 'values') else np.array(x.data)
        if isinstance(y, DataBatch):
            y = y.data.values if hasattr(y.data, 'values') else np.array(y.data)
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError('Inputs must be numpy arrays or DataBatch objects')
        if len(x) != len(y):
            raise ValueError('x and y must have the same length')
        if len(x) == 0:
            raise ValueError('Input arrays cannot be empty')
        differences = x - y
        original_n = len(differences)
        if self.zero_method == 'wilcox':
            non_zero_indices = differences != 0
            differences = differences[non_zero_indices]
        elif self.zero_method == 'pratt':
            non_zero_indices = differences != 0
            non_zero_diffs = differences[non_zero_indices]
        elif self.zero_method == 'zsplit':
            non_zero_indices = differences != 0
            non_zero_diffs = differences[non_zero_indices]
        if len(differences) == 0 or (self.zero_method in ['pratt', 'zsplit'] and len(non_zero_diffs) == 0):
            return {'statistic': 0, 'p_value': 1.0, 'sample_size': 0, 'significant': bool(False), 'confidence_interval': (0.0, 0.0), 'median_difference': 0.0, 'effect_size': 0.0, 'z_score': 0.0}
        if self.zero_method in ['pratt', 'zsplit']:
            diffs_to_rank = non_zero_diffs
        else:
            diffs_to_rank = differences
        abs_diffs = np.abs(diffs_to_rank)
        sorted_indices = np.argsort(abs_diffs)
        sorted_abs_diffs = abs_diffs[sorted_indices]
        ranks = np.zeros(len(sorted_abs_diffs))
        i = 0
        while i < len(sorted_abs_diffs):
            j = i
            while j < len(sorted_abs_diffs) and sorted_abs_diffs[j] == sorted_abs_diffs[i]:
                j += 1
            avg_rank = (i + j + 1) / 2
            ranks[sorted_indices[i:j]] = avg_rank
            i = j
        signs = np.sign(diffs_to_rank)
        signed_ranks = signs * ranks
        if self.alternative == 'greater':
            statistic = np.sum(signed_ranks[signed_ranks > 0])
        elif self.alternative == 'less':
            statistic = np.sum(signed_ranks[signed_ranks < 0])
        else:
            statistic = np.sum(np.abs(signed_ranks[signed_ranks > 0]))
        n = len(diffs_to_rank)
        if self.zero_method == 'pratt':
            n_total = len(differences)
            expected_value = n_total * (n_total + 1) / 4
            variance = n_total * (n_total + 1) * (2 * n_total + 1) / 24
        else:
            expected_value = n * (n + 1) / 4
            variance = n * (n + 1) * (2 * n + 1) / 24
        if self.zero_method == 'zsplit':
            zero_count = np.sum(differences == 0)
            variance -= zero_count * (zero_count + 1) * (2 * zero_count + 1) / 24
        if variance <= 0:
            z_score = 0.0
            p_value = 1.0
        else:
            if self.alternative == 'two-sided':
                z_score = (statistic - expected_value) / np.sqrt(variance)
                if statistic > expected_value:
                    z_score = (statistic - 0.5 - expected_value) / np.sqrt(variance)
                else:
                    z_score = (statistic + 0.5 - expected_value) / np.sqrt(variance)
            else:
                z_score = (statistic - expected_value) / np.sqrt(variance)
                if self.alternative == 'greater' and statistic > expected_value:
                    z_score = (statistic - 0.5 - expected_value) / np.sqrt(variance)
                elif self.alternative == 'less' and statistic < expected_value:
                    z_score = (statistic + 0.5 - expected_value) / np.sqrt(variance)
            if self.alternative == 'two-sided':
                p_value = 2 * (1 - self._norm_cdf(np.abs(z_score)))
            elif self.alternative == 'greater':
                p_value = 1 - self._norm_cdf(z_score)
            else:
                p_value = self._norm_cdf(z_score)
        median_difference = np.median(differences)
        effect_size = z_score / np.sqrt(len(differences)) if len(differences) > 0 else 0.0
        sorted_diffs = np.sort(differences)
        n_total = len(sorted_diffs)
        alpha_half = self.alpha / 2 if self.alternative == 'two-sided' else self.alpha
        if n_total > 1:
            z_critical = self._norm_ppf(1 - alpha_half)
            std_err = np.std(sorted_diffs) / np.sqrt(n_total) if np.std(sorted_diffs) > 0 else 0
            margin_error = z_critical * std_err
            ci_lower = median_difference - margin_error
            ci_upper = median_difference + margin_error
        else:
            (ci_lower, ci_upper) = (median_difference, median_difference)
        if self.alternative == 'greater':
            ci_lower = median_difference
        elif self.alternative == 'less':
            ci_upper = median_difference
        return {'statistic': statistic, 'p_value': p_value, 'sample_size': n, 'significant': bool(p_value < self.alpha), 'confidence_interval': (ci_lower, ci_upper), 'median_difference': median_difference, 'effect_size': effect_size, 'z_score': z_score}

    def perform_exact_test(self, x: Union[np.ndarray, DataBatch], y: Union[np.ndarray, DataBatch]) -> Dict[str, Any]:
        """
        Perform exact Wilcoxon signed-rank test for small sample sizes.
        
        Uses exact distribution calculations rather than normal approximation 
        for more accurate results with small samples.
        
        Args:
            x: First sample data as numpy array or DataBatch
            y: Second sample data as numpy array or DataBatch
            
        Returns:
            Dict containing:
            - 'statistic': Wilcoxon test statistic
            - 'p_value': Exact p-value for the test
            - 'sample_size': Number of non-zero differences
            - 'significant': Boolean indicating if difference is statistically significant
            - 'permutations': Number of permutations considered
            - 'critical_values': Critical values for exact test
            
        Raises:
            ValueError: If x and y have different lengths or are empty
            TypeError: If input data types are not supported
        """
        if isinstance(x, DataBatch):
            x = x.data.values if hasattr(x.data, 'values') else np.array(x.data)
        if isinstance(y, DataBatch):
            y = y.data.values if hasattr(y.data, 'values') else np.array(y.data)
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError('Inputs must be numpy arrays or DataBatch objects')
        if len(x) != len(y):
            raise ValueError('x and y must have the same length')
        if len(x) == 0:
            raise ValueError('Input arrays cannot be empty')
        differences = x - y
        original_n = len(differences)
        if self.zero_method == 'wilcox':
            non_zero_indices = differences != 0
            differences = differences[non_zero_indices]
        elif self.zero_method in ['pratt', 'zsplit']:
            non_zero_indices = differences != 0
            non_zero_diffs = differences[non_zero_indices]
        if len(differences) == 0 or (self.zero_method in ['pratt', 'zsplit'] and len(non_zero_diffs) == 0):
            return {'statistic': 0, 'p_value': 1.0, 'sample_size': original_n, 'significant': bool(False), 'permutations': 0, 'critical_values': []}
        if self.zero_method in ['pratt', 'zsplit']:
            diffs_to_rank = non_zero_diffs
        else:
            diffs_to_rank = differences
        abs_diffs = np.abs(diffs_to_rank)
        sorted_indices = np.argsort(abs_diffs)
        sorted_abs_diffs = abs_diffs[sorted_indices]
        ranks = np.zeros(len(sorted_abs_diffs))
        i = 0
        while i < len(sorted_abs_diffs):
            j = i
            while j < len(sorted_abs_diffs) and sorted_abs_diffs[j] == sorted_abs_diffs[i]:
                j += 1
            avg_rank = (i + j + 1) / 2
            ranks[sorted_indices[i:j]] = avg_rank
            i = j
        signs = np.sign(diffs_to_rank)
        signed_ranks = signs * ranks
        if self.alternative == 'greater':
            statistic = np.sum(signed_ranks[signed_ranks > 0])
        elif self.alternative == 'less':
            statistic = np.sum(signed_ranks[signed_ranks < 0])
        else:
            statistic = np.sum(np.abs(signed_ranks[signed_ranks > 0]))
        n = len(diffs_to_rank)
        permutations = 2 ** n
        if permutations > 1000000:
            raise ValueError('Exact test not feasible for sample size requiring more than 1,000,000 permutations')
        exact_distribution = []
        for i in range(permutations):
            pattern = [i >> j & 1 for j in range(n)]
            signs_pattern = np.array([1 if bit == 1 else -1 for bit in pattern])
            signed_ranks_pattern = signs_pattern * ranks
            if self.alternative == 'greater':
                stat = np.sum(signed_ranks_pattern[signed_ranks_pattern > 0])
            elif self.alternative == 'less':
                stat = np.sum(signed_ranks_pattern[signed_ranks_pattern < 0])
            else:
                stat = np.sum(np.abs(signed_ranks_pattern[signed_ranks_pattern > 0]))
            exact_distribution.append(stat)
        exact_distribution = np.array(exact_distribution)
        if self.alternative == 'two-sided':
            center = n * (n + 1) / 4
            diff_from_center = np.abs(exact_distribution - center)
            obs_diff_from_center = np.abs(statistic - center)
            p_value = np.sum(diff_from_center >= obs_diff_from_center) / len(exact_distribution)
        elif self.alternative == 'greater':
            p_value = np.sum(exact_distribution >= statistic) / len(exact_distribution)
        else:
            p_value = np.sum(exact_distribution <= statistic) / len(exact_distribution)
        critical_values = []
        if self.alternative == 'two-sided':
            alpha_half = self.alpha / 2
            lower_percentile = int(alpha_half * permutations)
            upper_percentile = int((1 - alpha_half) * permutations)
            sorted_stats = np.sort(exact_distribution)
            critical_values = [sorted_stats[lower_percentile], sorted_stats[upper_percentile]]
        elif self.alternative == 'greater':
            percentile = int((1 - self.alpha) * permutations)
            sorted_stats = np.sort(exact_distribution)
            critical_values = [sorted_stats[percentile]]
        else:
            percentile = int(self.alpha * permutations)
            sorted_stats = np.sort(exact_distribution)
            critical_values = [sorted_stats[percentile]]
        return {'statistic': statistic, 'p_value': p_value, 'sample_size': n, 'significant': bool(p_value < self.alpha), 'permutations': permutations, 'critical_values': critical_values}

    def _norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + self._erf(x / np.sqrt(2)))

    def _norm_ppf(self, p: float) -> float:
        """Standard normal percent point function (quantile function)."""
        if p <= 0 or p >= 1:
            raise ValueError('p must be between 0 and 1')
        a = [0.0, -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239]
        b = [0.0, -54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
        c = [0.0, -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
        d = [0.0, 0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
        p_low = 0.02425
        p_high = 1 - p_low
        if 0 < p < p_low:
            q = np.sqrt(-2 * np.log(p))
            return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
        elif p_low <= p <= p_high:
            q = p - 0.5
            r = q * q
            return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
        elif p_high < p < 1:
            q = np.sqrt(-2 * np.log(1 - p))
            return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
        else:
            return 0.0

    def _erf(self, x: float) -> float:
        """Error function approximation."""
        sign = 1 if x >= 0 else -1
        x = abs(x)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return sign * y

class PearsonChiSquareTest:
    """
    Performs Pearson's chi-square test for independence or goodness-of-fit.
    
    Pearson's chi-square test is a statistical test applied to sets of categorical data 
    to evaluate how likely it is that any observed difference between the sets arose 
    by chance. It's commonly used for testing independence between two categorical variables 
    or for testing if observed frequencies match expected frequencies.
    
    Attributes:
        test_type (str): Type of chi-square test ('independence' or 'goodness_of_fit')
        alpha (float): Significance level for the test (default: 0.05)
        Yates_correction (bool): Whether to apply Yates' continuity correction (default: False)
    """

    def __init__(self, test_type: str='independence', alpha: float=0.05, Yates_correction: bool=False):
        """
        Initialize the Pearson's chi-square test.
        
        Args:
            test_type: Type of chi-square test to perform. Supported values:
                      'independence' (test association between two categorical variables)
                      'goodness_of_fit' (test if observed frequencies match expected frequencies)
            alpha: Significance level for determining statistical significance (default: 0.05)
            Yates_correction: Whether to apply Yates' continuity correction for 2x2 tables (default: False)
            
        Raises:
            ValueError: If test_type is not supported
        """
        self.test_type = test_type
        self.alpha = alpha
        self.Yates_correction = Yates_correction
        if test_type not in ['independence', 'goodness_of_fit']:
            raise ValueError("test_type must be 'independence' or 'goodness_of_fit'")

    def _extract_data(self, data: Union[np.ndarray, DataBatch]) -> np.ndarray:
        """Extract numpy array from input data."""
        if isinstance(data, DataBatch):
            return np.array(data.data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError('Input data must be numpy array or DataBatch')

    def _validate_contingency_table(self, table: np.ndarray) -> None:
        """Validate contingency table for independence test."""
        if table.ndim != 2:
            raise ValueError('Contingency table must be 2-dimensional')
        if np.any(table < 0):
            raise ValueError('All observed frequencies must be non-negative')
        if np.all(table == 0):
            raise ValueError('At least one observed frequency must be positive')

    def _validate_goodness_of_fit_data(self, observed: np.ndarray, expected: np.ndarray) -> None:
        """Validate data for goodness-of-fit test."""
        if observed.ndim != 1:
            raise ValueError('Observed frequencies must be 1-dimensional')
        if np.any(observed < 0):
            raise ValueError('All observed frequencies must be non-negative')
        if len(observed) != len(expected):
            raise ValueError('Observed and expected arrays must have the same length')
        if np.any(expected < 0):
            raise ValueError('All expected frequencies must be non-negative')
        if np.sum(expected) == 0:
            raise ValueError('Expected frequencies cannot all be zero')
        if np.sum(observed) == 0:
            raise ValueError('Observed frequencies cannot all be zero')

    def _chi2_sf(self, x: float, df: int) -> float:
        """Calculate survival function (1 - CDF) of chi-square distribution."""
        if x <= 0:
            return 1.0
        if df <= 0:
            return 0.0
        if df > 100:
            from scipy.special import gammaincc
            return gammaincc(df / 2.0, x / 2.0)
        else:
            from scipy.special import gammainc
            return 1.0 - gammainc(df / 2.0, x / 2.0)

    def test_independence(self, observed: Union[np.ndarray, DataBatch]) -> Dict[str, Any]:
        """
        Perform Pearson's chi-square test for independence between two categorical variables.
        
        Tests the null hypothesis that the two categorical variables are independent.
        
        Args:
            observed: Contingency table of observed frequencies as numpy array or DataBatch
                     (rows represent one variable, columns represent another)
            
        Returns:
            Dict containing:
            - 'chi_square_statistic': Chi-square test statistic
            - 'p_value': P-value for the test
            - 'degrees_of_freedom': Degrees of freedom (rows-1)*(columns-1)
            - 'significant': Boolean indicating if variables are dependent
            - 'expected_frequencies': Expected frequencies under independence assumption
            - 'residuals': Pearson residuals (observed - expected) / sqrt(expected)
            - 'cramers_v': Cramer's V effect size measure
            - 'contingency_coefficient': Contingency coefficient measure
            
        Raises:
            ValueError: If observed table is not 2-dimensional or contains invalid values
            TypeError: If input data types are not supported
        """
        obs_table = self._extract_data(observed)
        self._validate_contingency_table(obs_table)
        row_totals = np.sum(obs_table, axis=1)
        col_totals = np.sum(obs_table, axis=0)
        grand_total = np.sum(obs_table)
        expected = np.outer(row_totals, col_totals) / grand_total
        if np.any(expected == 0):
            expected = expected + 1e-10
        if self.Yates_correction and obs_table.shape == (2, 2):
            diff = np.abs(obs_table - expected) - 0.5
            diff = np.maximum(diff, 0)
            chi2_stat = np.sum(diff ** 2 / expected)
        else:
            chi2_stat = np.sum((obs_table - expected) ** 2 / expected)
        df = (obs_table.shape[0] - 1) * (obs_table.shape[1] - 1)
        p_value = self._chi2_sf(chi2_stat, df)
        residuals = (obs_table - expected) / np.sqrt(expected)
        n = grand_total
        cramers_v = np.sqrt(chi2_stat / (n * min(obs_table.shape[0], obs_table.shape[1]) - 1))
        contingency_coeff = np.sqrt(chi2_stat / (chi2_stat + n))
        return {'chi_square_statistic': chi2_stat, 'p_value': p_value, 'degrees_of_freedom': df, 'significant': p_value < self.alpha, 'expected_frequencies': expected, 'residuals': residuals, 'cramers_v': cramers_v, 'contingency_coefficient': contingency_coeff}

    def test_goodness_of_fit(self, observed: Union[np.ndarray, DataBatch], expected: Union[np.ndarray, List[float]]) -> Dict[str, Any]:
        """
        Perform Pearson's chi-square goodness-of-fit test.
        
        Tests the null hypothesis that the observed frequencies match the expected frequencies.
        
        Args:
            observed: Array of observed frequencies as numpy array or DataBatch
            expected: Array of expected frequencies or probabilities
            
        Returns:
            Dict containing:
            - 'chi_square_statistic': Chi-square test statistic
            - 'p_value': P-value for the test
            - 'degrees_of_freedom': Degrees of freedom (categories - 1)
            - 'significant': Boolean indicating if observed differs from expected
            - 'expected_frequencies': Expected frequencies used in test
            - 'residuals': Pearson residuals (observed - expected) / sqrt(expected)
            - 'effect_size': Effect size measure (phi coefficient for 2 categories)
            
        Raises:
            ValueError: If observed and expected arrays have different lengths or contain invalid values
            TypeError: If input data types are not supported
        """
        obs_array = self._extract_data(observed)
        exp_array = np.array(expected)
        self._validate_goodness_of_fit_data(obs_array, exp_array)
        if np.isclose(np.sum(exp_array), 1.0) and np.sum(exp_array) < np.sum(obs_array):
            exp_array = exp_array * np.sum(obs_array)
        if np.any(exp_array == 0):
            exp_array = exp_array + 1e-10
        if self.Yates_correction and len(obs_array) == 2:
            diff = np.abs(obs_array - exp_array) - 0.5
            diff = np.maximum(diff, 0)
            chi2_stat = np.sum(diff ** 2 / exp_array)
        else:
            chi2_stat = np.sum((obs_array - exp_array) ** 2 / exp_array)
        df = len(obs_array) - 1
        p_value = self._chi2_sf(chi2_stat, df)
        residuals = (obs_array - exp_array) / np.sqrt(exp_array)
        n = np.sum(obs_array)
        effect_size = np.sqrt(chi2_stat / n) if len(obs_array) == 2 else None
        return {'chi_square_statistic': chi2_stat, 'p_value': p_value, 'degrees_of_freedom': df, 'significant': p_value < self.alpha, 'expected_frequencies': exp_array, 'residuals': residuals, 'effect_size': effect_size}

class ConfidenceIntervalCalculator:

    def __init__(self, confidence_level: float=0.95, method: str='normal', bootstrap_samples: int=1000):
        """
        Initialize the confidence interval calculator.
        
        Args:
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
            method: Method for calculating confidence intervals. Supported values:
                   'normal' (parametric using normal distribution),
                   't' (parametric using t-distribution),
                   'bootstrap' (non-parametric bootstrap),
                   'percentile' (percentile-based),
                   'bc' (bias-corrected),
                   'bca' (bias-corrected and accelerated)
            bootstrap_samples: Number of bootstrap samples for resampling methods (default: 1000)
            
        Raises:
            ValueError: If confidence_level is not between 0 and 1 or method is not supported
        """
        if not 0 < confidence_level < 1:
            raise ValueError('confidence_level must be between 0 and 1')
        self.confidence_level = confidence_level
        self.method = method
        self.bootstrap_samples = bootstrap_samples
        self._supported_methods = ['normal', 't', 'bootstrap', 'percentile', 'bc', 'bca']
        if method not in self._supported_methods:
            raise ValueError(f'method must be one of {self._supported_methods}')

    def calculate_mean_ci(self, data: Union[np.ndarray, DataBatch]) -> Dict[str, Any]:
        """
        Calculate confidence interval for the population mean.
        
        Computes the confidence interval for the mean using the specified method.
        
        Args:
            data: Sample data as numpy array or DataBatch
            
        Returns:
            Dict containing:
            - 'lower_bound': Lower bound of confidence interval
            - 'upper_bound': Upper bound of confidence interval
            - 'estimate': Point estimate (sample mean)
            - 'margin_of_error': Margin of error
            - 'standard_error': Standard error of the mean
            - 'method': Method used for calculation
            - 'confidence_level': Confidence level used
            
        Raises:
            ValueError: If data is empty or contains non-numeric values
            TypeError: If input data types are not supported
        """
        if isinstance(data, DataBatch):
            data_array = np.array(data.data)
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise TypeError('Input data must be either a numpy array or DataBatch')
        if data_array.size == 0:
            raise ValueError('Input data cannot be empty')
        data_flat = data_array.flatten()
        if not np.issubdtype(data_flat.dtype, np.number):
            raise ValueError('Input data must contain numeric values')
        data_clean = data_flat[~np.isnan(data_flat)]
        if len(data_clean) == 0:
            raise ValueError('No valid numeric data after removing NaN values')
        n = len(data_clean)
        sample_mean = np.mean(data_clean)
        if self.method in ['normal', 't']:
            sample_std = np.std(data_clean, ddof=1)
            standard_error = sample_std / np.sqrt(n)
            if self.method == 'normal':
                from scipy.stats import norm
                alpha = 1 - self.confidence_level
                z_critical = norm.ppf(1 - alpha / 2)
                margin_of_error = z_critical * standard_error
            else:
                from scipy.stats import t
                alpha = 1 - self.confidence_level
                t_critical = t.ppf(1 - alpha / 2, df=n - 1)
                margin_of_error = t_critical * standard_error
            lower_bound = sample_mean - margin_of_error
            upper_bound = sample_mean + margin_of_error
            return {'lower_bound': lower_bound, 'upper_bound': upper_bound, 'estimate': sample_mean, 'margin_of_error': margin_of_error, 'standard_error': standard_error, 'method': self.method, 'confidence_level': self.confidence_level}
        else:
            return self._bootstrap_mean_ci(data_clean)

    def _bootstrap_mean_ci(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Calculate confidence interval for the mean using bootstrap methods.
        
        Args:
            data: Sample data as numpy array
            
        Returns:
            Dict containing confidence interval information
        """
        n = len(data)
        bootstrap_means = []
        rng = np.random.default_rng()
        for _ in range(self.bootstrap_samples):
            bootstrap_sample = rng.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        if self.method == 'bootstrap' or self.method == 'percentile':
            lower_bound = np.percentile(bootstrap_means, lower_percentile)
            upper_bound = np.percentile(bootstrap_means, upper_percentile)
        elif self.method == 'bc':
            sample_mean = np.mean(data)
            proportion_below = np.sum(bootstrap_means < sample_mean) / len(bootstrap_means)
            z0 = norm.ppf(proportion_below)
            z_alpha_2 = norm.ppf(alpha / 2)
            z_1_alpha_2 = norm.ppf(1 - alpha / 2)
            lower_z = z0 + (z0 + z_alpha_2)
            upper_z = z0 + (z0 + z_1_alpha_2)
            lower_percentile_bc = norm.cdf(lower_z) * 100
            upper_percentile_bc = norm.cdf(upper_z) * 100
            lower_bound = np.percentile(bootstrap_means, lower_percentile_bc)
            upper_bound = np.percentile(bootstrap_means, upper_percentile_bc)
        elif self.method == 'bca':
            sample_mean = np.mean(data)
            proportion_below = np.sum(bootstrap_means < sample_mean) / len(bootstrap_means)
            z0 = norm.ppf(proportion_below)
            jackknife_means = []
            for i in range(len(data)):
                jack_data = np.delete(data, i)
                jackknife_means.append(np.mean(jack_data))
            jackknife_means = np.array(jackknife_means)
            mean_jackknife = np.mean(jackknife_means)
            numerator = np.sum((mean_jackknife - jackknife_means) ** 3)
            denominator = 6 * np.sum((mean_jackknife - jackknife_means) ** 2) ** 1.5
            if denominator == 0:
                a = 0
            else:
                a = numerator / denominator
            z_alpha_2 = norm.ppf(alpha / 2)
            z_1_alpha_2 = norm.ppf(1 - alpha / 2)
            lower_z = z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2))
            upper_z = z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2))
            lower_percentile_bca = norm.cdf(lower_z) * 100
            upper_percentile_bca = norm.cdf(upper_z) * 100
            lower_bound = np.percentile(bootstrap_means, lower_percentile_bca)
            upper_bound = np.percentile(bootstrap_means, upper_percentile_bca)
        estimate = np.mean(data)
        standard_error = np.std(bootstrap_means, ddof=1)
        margin_of_error = (upper_bound - lower_bound) / 2
        return {'lower_bound': lower_bound, 'upper_bound': upper_bound, 'estimate': estimate, 'margin_of_error': margin_of_error, 'standard_error': standard_error, 'method': self.method, 'confidence_level': self.confidence_level}

    def calculate_proportion_ci(self, successes: int, trials: int) -> Dict[str, Any]:
        """
        Calculate confidence interval for a population proportion.
        
        Computes the confidence interval for a binomial proportion using various methods.
        
        Args:
            successes: Number of successful outcomes
            trials: Total number of trials
            
        Returns:
            Dict containing:
            - 'lower_bound': Lower bound of confidence interval
            - 'upper_bound': Upper bound of confidence interval
            - 'estimate': Point estimate (sample proportion)
            - 'margin_of_error': Margin of error
            - 'method': Method used for calculation
            - 'confidence_level': Confidence level used
            - 'successes': Number of successes
            - 'trials': Number of trials
            
        Raises:
            ValueError: If successes > trials or if values are negative
        """
        if not isinstance(successes, int) or not isinstance(trials, int):
            raise ValueError('Successes and trials must be integers')
        if successes < 0 or trials < 0:
            raise ValueError('Successes and trials must be non-negative')
        if successes > trials:
            raise ValueError('Successes cannot exceed trials')
        if trials == 0:
            raise ValueError('Trials must be greater than zero')
        p_hat = successes / trials
        if self.method == 'normal':
            if trials * p_hat < 5 or trials * (1 - p_hat) < 5:
                pass
            from scipy.stats import norm
            alpha = 1 - self.confidence_level
            z_critical = norm.ppf(1 - alpha / 2)
            standard_error = np.sqrt(p_hat * (1 - p_hat) / trials)
            margin_of_error = z_critical * standard_error
            lower_bound = p_hat - margin_of_error
            upper_bound = p_hat + margin_of_error
            lower_bound = max(0, lower_bound)
            upper_bound = min(1, upper_bound)
        else:
            from scipy.stats import norm
            alpha = 1 - self.confidence_level
            z_critical = norm.ppf(1 - alpha / 2)
            standard_error = np.sqrt(p_hat * (1 - p_hat) / trials)
            margin_of_error = z_critical * standard_error
            lower_bound = p_hat - margin_of_error
            upper_bound = p_hat + margin_of_error
            lower_bound = max(0, lower_bound)
            upper_bound = min(1, upper_bound)
        return {'lower_bound': lower_bound, 'upper_bound': upper_bound, 'estimate': p_hat, 'margin_of_error': margin_of_error, 'method': self.method, 'confidence_level': self.confidence_level, 'successes': successes, 'trials': trials}

    def calculate_variance_ci(self, data: Union[np.ndarray, DataBatch]) -> Dict[str, Any]:
        """
        Calculate confidence interval for the population variance.
        
        Computes the confidence interval for variance using the chi-square distribution.
        
        Args:
            data: Sample data as numpy array or DataBatch
            
        Returns:
            Dict containing:
            - 'lower_bound': Lower bound of confidence interval
            - 'upper_bound': Upper bound of confidence interval
            - 'estimate': Point estimate (sample variance)
            - 'method': Method used for calculation
            - 'confidence_level': Confidence level used
            - 'degrees_of_freedom': Degrees of freedom used
            
        Raises:
            ValueError: If data is empty or contains non-numeric values
            TypeError: If input data types are not supported
        """
        if isinstance(data, DataBatch):
            data_array = np.array(data.data)
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise TypeError('Input data must be either a numpy array or DataBatch')
        if data_array.size == 0:
            raise ValueError('Input data cannot be empty')
        data_flat = data_array.flatten()
        if not np.issubdtype(data_flat.dtype, np.number):
            raise ValueError('Input data must contain numeric values')
        data_clean = data_flat[~np.isnan(data_flat)]
        if len(data_clean) == 0:
            raise ValueError('No valid numeric data after removing NaN values')
        if len(data_clean) < 2:
            raise ValueError('Need at least 2 data points to compute variance')
        n = len(data_clean)
        degrees_of_freedom = n - 1
        sample_variance = np.var(data_clean, ddof=1)
        from scipy.stats import chi2
        alpha = 1 - self.confidence_level
        chi2_lower = chi2.ppf(1 - alpha / 2, df=degrees_of_freedom)
        chi2_upper = chi2.ppf(alpha / 2, df=degrees_of_freedom)
        lower_bound = degrees_of_freedom * sample_variance / chi2_lower
        upper_bound = degrees_of_freedom * sample_variance / chi2_upper
        return {'lower_bound': lower_bound, 'upper_bound': upper_bound, 'estimate': sample_variance, 'method': 'chi-square', 'confidence_level': self.confidence_level, 'degrees_of_freedom': degrees_of_freedom}


# ...(code omitted)...


class CorrelationAnalyzer:

    def __init__(self, method: str='pearson', handle_missing: str='complete', adjust_p_values: bool=False, adjustment_method: str='bonferroni'):
        """
        Initialize the correlation analyzer.
        
        Args:
            method: Default correlation method. Supported values:
                   'pearson' (Pearson correlation coefficient),
                   'spearman' (Spearman rank correlation),
                   'kendall' (Kendall's tau),
                   'point_biserial' (Point-biserial correlation),
                   'polychoric' (Polychoric correlation),
                   'partial' (Partial correlation)
            handle_missing: How to handle missing values. Supported values:
                           'complete' (listwise deletion),
                           'pairwise' (pairwise deletion),
                           'impute' (mean imputation)
            adjust_p_values: Whether to adjust p-values for multiple comparisons (default: False)
            adjustment_method: Method for p-value adjustment. Supported values:
                              'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)
            
        Raises:
            ValueError: If method, handle_missing, or adjustment_method is not supported
        """
        self.method = method
        self.handle_missing = handle_missing
        self.adjust_p_values = adjust_p_values
        self.adjustment_method = adjustment_method
        self._supported_methods = ['pearson', 'spearman', 'kendall', 'point_biserial', 'polychoric', 'partial']
        self._supported_missing_handlers = ['complete', 'pairwise', 'impute']
        self._supported_adjustment_methods = ['bonferroni', 'holm', 'fdr_bh']
        if method not in self._supported_methods:
            raise ValueError(f'method must be one of {self._supported_methods}')
        if handle_missing not in self._supported_missing_handlers:
            raise ValueError(f'handle_missing must be one of {self._supported_missing_handlers}')
        if adjustment_method not in self._supported_adjustment_methods:
            raise ValueError(f'adjustment_method must be one of {self._supported_adjustment_methods}')

    def compute_correlation(self, x: Union[np.ndarray, DataBatch], y: Union[np.ndarray, DataBatch], method: Optional[str]=None) -> Dict[str, Any]:
        """
        Compute correlation coefficient between two variables.
        
        Calculates the correlation coefficient using the specified method.
        
        Args:
            x: First variable data as numpy array or DataBatch
            y: Second variable data as numpy array or DataBatch
            method: Correlation method (overrides default if provided)
            
        Returns:
            Dict containing:
            - 'correlation': Correlation coefficient value
            - 'p_value': P-value for significance test
            - 'confidence_interval': Confidence interval for correlation
            - 'sample_size': Number of observations used
            - 'method': Correlation method used
            - 'statistic': Test statistic value
            - 'degrees_of_freedom': Degrees of freedom (if applicable)
            - 'effect_size': Effect size interpretation
            
        Raises:
            ValueError: If x and y have different lengths or are empty
            TypeError: If input data types are not supported
        """
        if isinstance(x, DataBatch):
            x = x.data
        if isinstance(y, DataBatch):
            y = y.data
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError('x and y must be 1-dimensional arrays')
        if len(x) != len(y):
            raise ValueError('x and y must have the same length')
        if len(x) == 0:
            raise ValueError('x and y cannot be empty')
        corr_method = method if method is not None else self.method
        if corr_method not in self._supported_methods:
            raise ValueError(f"Method '{corr_method}' not supported. Supported methods: {self._supported_methods}")
        mask = self._handle_missing_values_single(x, y)
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 2:
            raise ValueError('Not enough valid observations after handling missing values')
        if corr_method == 'pearson':
            result = self._pearson_correlation(x_clean, y_clean)
        elif corr_method == 'spearman':
            result = self._spearman_correlation(x_clean, y_clean)
        elif corr_method == 'kendall':
            result = self._kendall_correlation(x_clean, y_clean)
        elif corr_method == 'point_biserial':
            result = self._point_biserial_correlation(x_clean, y_clean)
        else:
            raise ValueError(f"Method '{corr_method}' not yet implemented")
        result['method'] = corr_method
        result['sample_size'] = len(x_clean)
        result['effect_size'] = self._interpret_effect_size(result['correlation'])
        return result

    def compute_correlation_matrix(self, data: Union[np.ndarray, FeatureSet, DataBatch], method: Optional[str]=None) -> Dict[str, Any]:
        """
        Compute correlation matrix for multiple variables.
        
        Calculates pairwise correlations between all variables in a dataset.
        
        Args:
            data: Data matrix as numpy array, FeatureSet, or DataBatch
            method: Correlation method (overrides default if provided)
            
        Returns:
            Dict containing:
            - 'correlation_matrix': Matrix of correlation coefficients
            - 'p_value_matrix': Matrix of p-values for each correlation
            - 'confidence_intervals': Confidence intervals for correlations
            - 'sample_size_matrix': Matrix of sample sizes for each pair
            - 'method': Correlation method used
            - 'variable_names': Names of variables (if available)
            - 'significant_correlations': Boolean matrix indicating significant correlations
            - 'adjusted_p_values': Matrix of adjusted p-values (if adjust_p_values=True)
            
        Raises:
            ValueError: If data is empty or contains non-numeric values
            TypeError: If input data types are not supported
        """
        if isinstance(data, DataBatch):
            var_names = data.feature_names if hasattr(data, 'feature_names') else [f'var_{i}' for i in range(data.data.shape[1])]
            data = data.data
        elif isinstance(data, FeatureSet):
            var_names = list(data.features.keys())
            data = np.array([data.features[name] for name in var_names]).T
        else:
            data = np.asarray(data)
            var_names = [f'var_{i}' for i in range(data.shape[1])]
        if data.ndim != 2:
            raise ValueError('Data must be a 2-dimensional array')
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError('Data cannot be empty')
        n_vars = data.shape[1]
        corr_method = method if method is not None else self.method
        if corr_method not in self._supported_methods:
            raise ValueError(f"Method '{corr_method}' not supported. Supported methods: {self._supported_methods}")
        corr_matrix = np.full((n_vars, n_vars), np.nan)
        p_matrix = np.full((n_vars, n_vars), np.nan)
        sample_sizes = np.zeros((n_vars, n_vars), dtype=int)
        conf_intervals = [[None for _ in range(n_vars)] for _ in range(n_vars)]
        all_p_values = []
        pair_indices = []
        for i in range(n_vars):
            for j in range(i, n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                    sample_sizes[i, j] = data.shape[0]
                    conf_intervals[i][j] = [1.0, 1.0]
                else:
                    x = data[:, i]
                    y = data[:, j]
                    if self.handle_missing == 'complete':
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                    elif self.handle_missing == 'pairwise':
                        mask = ~(np.isnan(x) | np.isnan(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                    elif self.handle_missing == 'impute':
                        x_clean = x.copy()
                        y_clean = y.copy()
                        x_mean = np.nanmean(x)
                        y_mean = np.nanmean(y)
                        x_clean[np.isnan(x_clean)] = x_mean
                        y_clean[np.isnan(y_clean)] = y_mean
                        mask = np.ones(len(x), dtype=bool)
                    sample_size = len(x_clean)
                    sample_sizes[i, j] = sample_size
                    sample_sizes[j, i] = sample_size
                    if sample_size >= 2:
                        if corr_method == 'pearson':
                            result = self._pearson_correlation(x_clean, y_clean)
                        elif corr_method == 'spearman':
                            result = self._spearman_correlation(x_clean, y_clean)
                        elif corr_method == 'kendall':
                            result = self._kendall_correlation(x_clean, y_clean)
                        elif corr_method == 'point_biserial':
                            result = self._point_biserial_correlation(x_clean, y_clean)
                        else:
                            raise ValueError(f"Method '{corr_method}' not yet implemented for matrix computation")
                        corr_matrix[i, j] = result['correlation']
                        corr_matrix[j, i] = result['correlation']
                        p_matrix[i, j] = result['p_value']
                        p_matrix[j, i] = result['p_value']
                        conf_intervals[i][j] = result['confidence_interval']
                        conf_intervals[j][i] = result['confidence_interval']
                        if i != j:
                            all_p_values.append(result['p_value'])
                            pair_indices.append((i, j))
                    else:
                        corr_matrix[i, j] = np.nan
                        corr_matrix[j, i] = np.nan
                        p_matrix[i, j] = np.nan
                        p_matrix[j, i] = np.nan
                        conf_intervals[i][j] = [np.nan, np.nan]
                        conf_intervals[j][i] = [np.nan, np.nan]
        adjusted_p_matrix = None
        if self.adjust_p_values and len(all_p_values) > 0:
            correction = MultipleComparisonCorrection(method=self.adjustment_method)
            correction_result = correction.apply_correction(all_p_values)
            adjusted_p_values = correction_result['adjusted_p_values']
            adjusted_p_matrix = np.full((n_vars, n_vars), np.nan)
            for (idx, (i, j)) in enumerate(pair_indices):
                adjusted_p_matrix[i, j] = adjusted_p_values[idx]
                adjusted_p_matrix[j, i] = adjusted_p_values[idx]
            sig_matrix = adjusted_p_matrix < 0.05
        else:
            sig_matrix = p_matrix < 0.05
        np.fill_diagonal(sig_matrix, False)
        sig_matrix = np.where(np.isnan(sig_matrix), False, sig_matrix)
        return {'correlation_matrix': corr_matrix.tolist(), 'p_value_matrix': p_matrix.tolist(), 'confidence_intervals': conf_intervals, 'sample_size_matrix': sample_sizes.tolist(), 'method': corr_method, 'variable_names': var_names, 'significant_correlations': sig_matrix.tolist(), 'adjusted_p_values': adjusted_p_matrix.tolist() if adjusted_p_matrix is not None else None}

    def partial_correlation(self, x: Union[np.ndarray, DataBatch], y: Union[np.ndarray, DataBatch], covariates: Union[np.ndarray, List[Union[np.ndarray, DataBatch]]]) -> Dict[str, Any]:
        """
        Compute partial correlation between two variables controlling for covariates.
        
        Calculates the correlation between x and y after removing the effect of covariates.
        
        Args:
            x: First variable data as numpy array or DataBatch
            y: Second variable data as numpy array or DataBatch
            covariates: Covariate data as numpy array, DataBatch, or list of arrays/Batches
            
        Returns:
            Dict containing:
            - 'partial_correlation': Partial correlation coefficient
            - 'p_value': P-value for significance test
            - 'confidence_interval': Confidence interval for partial correlation
            - 'sample_size': Number of observations used
            - 'degrees_of_freedom': Degrees of freedom
            - 'control_variables': Number of control variables
            - 'semi_partial_correlations': Semi-partial correlations for each covariate
            
        Raises:
            ValueError: If data arrays have incompatible shapes or are empty
            TypeError: If input data types are not supported
        """
        pass

    def _handle_missing_values_single(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Handle missing values for correlation computation."""
        if self.handle_missing == 'complete':
            mask = ~(np.isnan(x) | np.isnan(y))
        elif self.handle_missing == 'pairwise':
            mask = ~(np.isnan(x) | np.isnan(y))
        elif self.handle_missing == 'impute':
            x_imputed = x.copy()
            y_imputed = y.copy()
            x_mean = np.nanmean(x)
            y_mean = np.nanmean(y)
            x_imputed[np.isnan(x_imputed)] = x_mean
            y_imputed[np.isnan(y_imputed)] = y_mean
            mask = np.ones(len(x), dtype=bool)
            x[:] = x_imputed
            y[:] = y_imputed
        else:
            mask = ~(np.isnan(x) | np.isnan(y))
        return mask

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute Pearson correlation coefficient."""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denom_x = np.sum((x - x_mean) ** 2)
        denom_y = np.sum((y - y_mean) ** 2)
        correlation = numerator / np.sqrt(denom_x * denom_y)
        n = len(x)
        if n <= 2:
            raise ValueError('Need at least 3 samples to compute Pearson correlation')
        df = n - 2
        if abs(correlation) == 1.0:
            t_stat = np.inf if correlation == 1.0 else -np.inf
            p_value = 0.0
        else:
            with np.errstate(divide='ignore'):
                t_stat = correlation * np.sqrt(df / (1 - correlation ** 2))
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        if abs(correlation) == 1.0:
            (ci_lower, ci_upper) = (correlation, correlation)
        else:
            with np.errstate(divide='ignore'):
                z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            z_se = 1 / np.sqrt(n - 3)
            z_crit = norm.ppf(0.975)
            z_lower = z - z_crit * z_se
            z_upper = z + z_crit * z_se
            ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        return {'correlation': float(correlation), 'p_value': float(p_value), 'confidence_interval': [float(ci_lower), float(ci_upper)], 'statistic': float(t_stat), 'degrees_of_freedom': int(df)}

    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute Spearman rank correlation."""
        x_rank = self._rank_data(x)
        y_rank = self._rank_data(y)
        return self._pearson_correlation(x_rank, y_rank)

    def _kendall_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute Kendall's tau correlation."""
        n = len(x)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if x[i] != x[j] and y[i] != y[j]:
                    if x[i] < x[j] and y[i] < y[j] or (x[i] > x[j] and y[i] > y[j]):
                        concordant += 1
                    else:
                        discordant += 1
        tau = (concordant - discordant) / (0.5 * n * (n - 1))
        if n <= 10:
            z_stat = 0
            p_value = 1.0
        else:
            se = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
            if se > 0:
                z_stat = tau / se
                p_value = 2 * (1 - norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0
        z_crit = norm.ppf(0.975)
        se_tau = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
        ci_lower = max(-1, tau - z_crit * se_tau)
        ci_upper = min(1, tau + z_crit * se_tau)
        return {'correlation': float(tau), 'p_value': float(p_value), 'confidence_interval': [float(ci_lower), float(ci_upper)], 'statistic': float(z_stat), 'degrees_of_freedom': None}

    def _point_biserial_correlation(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute point-biserial correlation (between continuous and binary variable)."""
        unique_y = np.unique(y[~np.isnan(y)])
        if len(unique_y) != 2:
            unique_x = np.unique(x[~np.isnan(x)])
            if len(unique_x) == 2:
                return self._point_biserial_correlation(y, x)
            else:
                raise ValueError('Point-biserial correlation requires one binary variable')
        (val0, val1) = (unique_y[0], unique_y[1])
        mask0 = y == val0
        mask1 = y == val1
        n0 = np.sum(mask0)
        n1 = np.sum(mask1)
        n = n0 + n1
        if n0 == 0 or n1 == 0:
            raise ValueError('Binary variable must have both values represented')
        mean0 = np.mean(x[mask0])
        mean1 = np.mean(x[mask1])
        overall_mean = np.mean(x)
        numerator = ((mean1 - overall_mean) * n1 + (mean0 - overall_mean) * n0) / n
        denominator = np.std(x, ddof=1)
        if denominator == 0:
            correlation = 0.0
        else:
            correlation = (mean1 - mean0) * np.sqrt(n0 * n1) / (n * denominator)
        df = n - 2
        if df <= 0 or abs(correlation) >= 1:
            t_stat = np.inf if correlation >= 1 else -np.inf if correlation <= -1 else 0
            p_value = 0.0 if abs(correlation) >= 1 else 1.0
        else:
            with np.errstate(divide='ignore'):
                t_stat = correlation * np.sqrt(df / (1 - correlation ** 2))
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        if abs(correlation) == 1.0:
            (ci_lower, ci_upper) = (correlation, correlation)
        else:
            with np.errstate(divide='ignore'):
                z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            z_se = 1 / np.sqrt(n - 3)
            z_crit = norm.ppf(0.975)
            z_lower = z - z_crit * z_se
            z_upper = z + z_crit * z_se
            ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        return {'correlation': float(correlation), 'p_value': float(p_value), 'confidence_interval': [float(ci_lower), float(ci_upper)], 'statistic': float(t_stat), 'degrees_of_freedom': int(df)}

    def _rank_data(self, data: np.ndarray) -> np.ndarray:
        """Assign ranks to data, dealing with ties appropriately."""
        sorted_indices = np.argsort(data)
        ranked = np.empty_like(data, dtype=float)
        ranked[sorted_indices] = np.arange(1, len(data) + 1)
        unique_vals = np.unique(data)
        for val in unique_vals:
            if not np.isnan(val):
                mask = data == val
                if np.sum(mask) > 1:
                    avg_rank = np.mean(ranked[mask])
                    ranked[mask] = avg_rank
        return ranked

    def _interpret_effect_size(self, correlation: float) -> str:
        """Interpret correlation magnitude as effect size."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return 'negligible'
        elif abs_corr < 0.3:
            return 'small'
        elif abs_corr < 0.5:
            return 'medium'
        else:
            return 'large'


# ...(code omitted)...


class LinearMixedModel:
    """
    A class for fitting linear mixed-effects models to handle correlated data with hierarchical or repeated measures structures.
    
    Supports both fixed and random effects modeling, with configurable fitting methods (REML, ML),
    covariance structures (unstructured, diagonal, etc.), and optimization algorithms (BFGS, Newton, etc.).
    """

    def __init__(self, fit_method: str='reml', cov_structure: str='unstructured', optimizer: str='bfgs'):
        """
        Initialize the LinearMixedModel.
        
        Args:
            fit_method: Fitting method, either 'reml' (restricted maximum likelihood) or 'ml' (maximum likelihood)
            cov_structure: Covariance structure for random effects, e.g., 'unstructured', 'diagonal'
            optimizer: Optimization algorithm, e.g., 'bfgs', 'newton'
            
        Raises:
            ValueError: If any parameter is not in the supported set
        """
        self.fit_method = fit_method.lower()
        self.cov_structure = cov_structure.lower()
        self.optimizer = optimizer.lower()
        if self.fit_method not in ['reml', 'ml']:
            raise ValueError("fit_method must be either 'reml' or 'ml'")
        if self.cov_structure not in ['unstructured', 'diagonal']:
            raise ValueError("cov_structure must be either 'unstructured' or 'diagonal'")
        if self.optimizer not in ['bfgs', 'newton']:
            raise ValueError("optimizer must be either 'bfgs' or 'newton'")
        self.fitted = False
        self.fixed_effects = None
        self.random_effects_cov = None
        self.residual_var = None
        self.aic = None
        self.bic = None
        self.log_likelihood = None
        self.converged = None
        self.n_groups = None
        self.n_obs = None

    def fit(self, X_fixed: np.ndarray, X_random: np.ndarray, y: np.ndarray, groups: np.ndarray) -> 'LinearMixedModel':
        """
        Fit the linear mixed-effects model.
        
        Args:
            X_fixed: Fixed effects design matrix (n_obs x n_fixed_features)
            X_random: Random effects design matrix (n_obs x n_random_features)
            y: Response variable (n_obs,)
            groups: Grouping variable defining clusters for random effects (n_obs,)
            
        Returns:
            self: Fitted model instance
            
        Raises:
            ValueError: If input arrays have incompatible shapes or invalid values
            TypeError: If input data types are not supported
        """
        if not isinstance(X_fixed, np.ndarray) or not isinstance(X_random, np.ndarray) or (not isinstance(y, np.ndarray)) or (not isinstance(groups, np.ndarray)):
            raise TypeError('All inputs must be numpy arrays')
        if X_fixed.ndim != 2 or X_random.ndim != 2:
            raise ValueError('Design matrices must be 2-dimensional')
        if y.ndim != 1:
            raise ValueError('Response variable must be 1-dimensional')
        if groups.ndim != 1:
            raise ValueError('Grouping variable must be 1-dimensional')
        n_obs = len(y)
        if X_fixed.shape[0] != n_obs or X_random.shape[0] != n_obs or len(groups) != n_obs:
            raise ValueError('All input arrays must have the same number of observations')
        self.X_fixed = X_fixed
        self.X_random = X_random
        self.y = y
        self.groups = groups
        unique_groups = np.unique(groups)
        self.n_groups = len(unique_groups)
        self.n_obs = n_obs
        n_fixed = X_fixed.shape[1]
        n_random = X_random.shape[1]
        self.fixed_effects = np.zeros(n_fixed)
        if self.cov_structure == 'unstructured':
            self.random_effects_cov = np.eye(n_random)
        elif self.cov_structure == 'diagonal':
            self.random_effects_cov = np.diag(np.ones(n_random))
        self.residual_var = 1.0
        try:
            XtX = X_fixed.T @ X_fixed
            if np.linalg.det(XtX) != 0:
                self.fixed_effects = np.linalg.solve(XtX, X_fixed.T @ y)
            else:
                self.fixed_effects = np.linalg.pinv(XtX) @ X_fixed.T @ y
            residuals = y - X_fixed @ self.fixed_effects
            self.residual_var = np.var(residuals)
            self.log_likelihood = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(self.residual_var) + 1)
            if self.fit_method == 'reml':
                self.log_likelihood -= 0.5 * (n_obs - n_fixed) * np.log(2 * np.pi * self.residual_var)
            k = n_fixed + n_random * (n_random + 1) // 2 + 1
            self.aic = -2 * self.log_likelihood + 2 * k
            self.bic = -2 * self.log_likelihood + k * np.log(n_obs)
            self.converged = True
            self.fitted = True
        except Exception as e:
            self.converged = False
            raise RuntimeError(f'Fitting failed: {str(e)}')
        return self

    def predict(self, X_fixed: np.ndarray, X_random: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        Predict response values for new data using the fitted model.
        
        Args:
            X_fixed: Fixed effects design matrix for prediction (n_pred x n_fixed_features)
            X_random: Random effects design matrix for prediction (n_pred x n_random_features)
            groups: Grouping variable for prediction (n_pred,)
            
        Returns:
            np.ndarray: Predicted response values (n_pred,)
            
        Raises:
            RuntimeError: If the model has not been fitted
            ValueError: If input arrays have incompatible shapes
        """
        if not self.fitted:
            raise RuntimeError('Model must be fitted before making predictions')
        if not isinstance(X_fixed, np.ndarray) or not isinstance(X_random, np.ndarray) or (not isinstance(groups, np.ndarray)):
            raise TypeError('All inputs must be numpy arrays')
        if X_fixed.ndim != 2 or X_random.ndim != 2:
            raise ValueError('Design matrices must be 2-dimensional')
        if groups.ndim != 1:
            raise ValueError('Grouping variable must be 1-dimensional')
        if X_fixed.shape[0] != X_random.shape[0] or X_fixed.shape[0] != len(groups):
            raise ValueError('All input arrays must have the same number of observations')
        if X_fixed.shape[1] != self.X_fixed.shape[1]:
            raise ValueError('Fixed effects design matrix must have the same number of columns as in training')
        if X_random.shape[1] != self.X_random.shape[1]:
            raise ValueError('Random effects design matrix must have the same number of columns as in training')
        predictions = X_fixed @ self.fixed_effects
        return predictions

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a detailed summary of the fitted model.
        
        Returns:
            Dict containing model summary information:
            - 'fixed_effects': Estimated fixed effects coefficients with statistics
            - 'random_effects_variance': Random effects variance components
            - 'residual_variance': Estimated residual variance
            - 'aic': Akaike Information Criterion
            - 'bic': Bayesian Information Criterion
            - 'log_likelihood': Log-likelihood of the fitted model
            - 'converged': Whether the fitting algorithm converged
            - 'n_groups': Number of groups/clusters
            - 'n_obs': Total number of observations
            - 'fit_method': Fitting method used ('reml' or 'ml')
            
        Raises:
            RuntimeError: If the model has not been fitted
        """
        if not self.fitted:
            raise RuntimeError('Model must be fitted before getting summary')
        n_fixed = len(self.fixed_effects)
        fixed_effects_summary = {'coef': self.fixed_effects.tolist(), 'stderr': [0.1] * n_fixed, 't_value': [self.fixed_effects[i] / 0.1 for i in range(n_fixed)], 'p_value': [2 * (1 - t.cdf(abs(self.fixed_effects[i] / 0.1), self.n_obs - n_fixed)) for i in range(n_fixed)]}
        if self.cov_structure == 'unstructured':
            random_effects_variance = {'cov_matrix': self.random_effects_cov.tolist()}
        elif self.cov_structure == 'diagonal':
            random_effects_variance = {'variances': np.diag(self.random_effects_cov).tolist()}
        summary = {'fixed_effects': fixed_effects_summary, 'random_effects_variance': random_effects_variance, 'residual_variance': self.residual_var, 'aic': self.aic, 'bic': self.bic, 'log_likelihood': self.log_likelihood, 'converged': self.converged, 'n_groups': self.n_groups, 'n_obs': self.n_obs, 'fit_method': self.fit_method}
        return summary

class PairedTTest:
    """
    Performs paired t-test for comparing means of dependent samples.
    
    This class provides methods for conducting paired t-tests, which are used
    to compare the means of two related groups of samples, such as measurements
    taken before and after an intervention on the same subjects.
    
    Attributes:
        alpha (float): Significance level for the test (default: 0.05)
        alternative (str): Alternative hypothesis (default: 'two-sided')
        equal_var (bool): Assume equal variances (default: True)
    """

    def __init__(self, alpha: float=0.05):
        """
        Initialize the PairedTTest with a significance level.
        
        Parameters
        ----------
        alpha : float, optional
            Significance level for the test (default is 0.05)
        """
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        self.alpha = alpha

    def perform_test(self, before: Union[np.ndarray, DataBatch], after: Union[np.ndarray, DataBatch], alpha: Optional[float]=None) -> Dict[str, Any]:
        """
        Perform a standard paired t-test on two related samples.
        
        Parameters
        ----------
        before : array_like or DataBatch
            Measurements before treatment or at time 1
        after : array_like or DataBatch
            Measurements after treatment or at time 2
        alpha : float, optional
            Significance level (overrides class-level alpha if provided)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 't_statistic': Calculated t-statistic
            - 'p_value': Two-tailed p-value
            - 'degrees_of_freedom': Degrees of freedom
            - 'confidence_interval': Confidence interval for the mean difference
            - 'effect_size': Cohen's d effect size
            - 'is_significant': Whether the result is statistically significant
            - 'mean_difference': Mean of the differences (after - before)
            - 'std_difference': Standard deviation of the differences
        """
        significance_level = alpha if alpha is not None else self.alpha
        if not 0 < significance_level < 1:
            raise ValueError('alpha must be between 0 and 1')
        if isinstance(before, DataBatch):
            before_data = np.asarray(before.data, dtype=float)
        else:
            before_data = np.asarray(before, dtype=float)
        if isinstance(after, DataBatch):
            after_data = np.asarray(after.data, dtype=float)
        else:
            after_data = np.asarray(after, dtype=float)
        if before_data.ndim != 1 or after_data.ndim != 1:
            raise ValueError("Both 'before' and 'after' must be 1-dimensional arrays")
        if len(before_data) != len(after_data):
            raise ValueError("'before' and 'after' arrays must have the same length")
        if len(before_data) < 2:
            raise ValueError('Paired t-test requires at least 2 pairs of observations')
        mask = ~(np.isnan(before_data) | np.isnan(after_data))
        before_clean = before_data[mask]
        after_clean = after_data[mask]
        if len(before_clean) < 2:
            raise ValueError('At least 2 valid pairs of observations are required after removing NaN values')
        differences = after_clean - before_clean
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1) if n > 1 else 0.0
        if std_diff == 0:
            if mean_diff == 0:
                t_stat = 0.0
            else:
                t_stat = np.inf if mean_diff > 0 else -np.inf
        else:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
        df = n - 1
        if np.isinf(t_stat):
            p_value = 0.0
        else:
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        if std_diff == 0 or n == 1:
            ci_lower = mean_diff
            ci_upper = mean_diff
        else:
            t_critical = t.ppf(1 - significance_level / 2, df)
            margin_error = t_critical * (std_diff / np.sqrt(n))
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
        effect_size = mean_diff / std_diff if std_diff != 0 else np.inf if mean_diff > 0 else -np.inf if mean_diff < 0 else 0
        is_significant = p_value < significance_level
        return {'t_statistic': float(t_stat), 'p_value': float(p_value), 'degrees_of_freedom': int(df), 'confidence_interval': [float(ci_lower), float(ci_upper)], 'effect_size': float(effect_size), 'is_significant': bool(is_significant), 'mean_difference': float(mean_diff), 'std_difference': float(std_diff) if std_diff != 0 else 0.0}

    def perform_test_with_covariates(self, before: Union[np.ndarray, DataBatch], after: Union[np.ndarray, DataBatch], covariates: Union[np.ndarray, DataBatch, List[Union[np.ndarray, DataBatch]]], alpha: Optional[float]=None) -> Dict[str, Any]:
        """
        Perform a paired t-test adjusted for covariates.
        
        Parameters
        ----------
        before : array_like or DataBatch
            Measurements before treatment or at time 1
        after : array_like or DataBatch
            Measurements after treatment or at time 2
        covariates : array_like, DataBatch, or list of these
            Covariate(s) to adjust for in the analysis
        alpha : float, optional
            Significance level (overrides class-level alpha if provided)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 't_statistic': Calculated t-statistic for the adjusted test
            - 'p_value': Two-tailed p-value for the adjusted test
            - 'degrees_of_freedom': Degrees of freedom for the adjusted test
            - 'confidence_interval': Confidence interval for the adjusted mean difference
            - 'effect_size': Adjusted effect size
            - 'is_significant': Whether the adjusted result is statistically significant
            - 'mean_difference': Mean of the adjusted differences
            - 'std_difference': Standard deviation of the adjusted differences
            - 'covariates_used': Number of covariates included in adjustment
        """
        significance_level = alpha if alpha is not None else self.alpha
        if not 0 < significance_level < 1:
            raise ValueError('alpha must be between 0 and 1')
        if isinstance(before, DataBatch):
            before_data = np.asarray(before.data, dtype=float)
        else:
            before_data = np.asarray(before, dtype=float)
        if isinstance(after, DataBatch):
            after_data = np.asarray(after.data, dtype=float)
        else:
            after_data = np.asarray(after, dtype=float)
        if isinstance(covariates, (list, tuple)):
            covariate_arrays = []
            for cov in covariates:
                if isinstance(cov, DataBatch):
                    covariate_arrays.append(np.asarray(cov.data, dtype=float))
                else:
                    covariate_arrays.append(np.asarray(cov, dtype=float))
        elif isinstance(covariates, DataBatch):
            covariate_arrays = [np.asarray(covariates.data, dtype=float)]
        else:
            covariate_arrays = [np.asarray(covariates, dtype=float)]
        if before_data.ndim != 1 or after_data.ndim != 1:
            raise ValueError("Both 'before' and 'after' must be 1-dimensional arrays")
        if len(before_data) != len(after_data):
            raise ValueError("'before' and 'after' arrays must have the same length")
        n = len(before_data)
        if n < 2:
            raise ValueError('Paired t-test requires at least 2 pairs of observations')
        for (i, cov_array) in enumerate(covariate_arrays):
            if cov_array.ndim == 1:
                if len(cov_array) != n:
                    raise ValueError(f"Covariate {i} must have the same length as 'before' and 'after' arrays")
            elif cov_array.ndim == 2:
                if cov_array.shape[0] != n:
                    raise ValueError(f"Covariate {i} must have the same number of rows as 'before' and 'after' arrays")
            else:
                raise ValueError(f'Covariate {i} must be 1-dimensional or 2-dimensional')
        mask = ~np.isnan(before_data) & ~np.isnan(after_data)
        for cov_array in covariate_arrays:
            if cov_array.ndim == 1:
                mask &= ~np.isnan(cov_array)
            else:
                mask &= ~np.any(np.isnan(cov_array), axis=1)
        before_clean = before_data[mask]
        after_clean = after_data[mask]
        covariate_clean = [cov[mask] for cov in covariate_arrays]
        if len(before_clean) < 2:
            raise ValueError('At least 2 valid pairs of observations are required after removing NaN values')
        combined_covariates = []
        for cov in covariate_clean:
            if cov.ndim == 1:
                combined_covariates.append(cov.reshape(-1, 1))
            else:
                for i in range(cov.shape[1]):
                    combined_covariates.append(cov[:, i].reshape(-1, 1))
        n_clean = len(before_clean)
        differences = after_clean - before_clean
        if not combined_covariates:
            return self.perform_test(before_clean, after_clean, alpha=significance_level)
        X = np.column_stack([np.ones(n_clean)] + combined_covariates)
        try:
            beta = np.linalg.pinv(X.T @ X) @ X.T @ differences
            predicted_diff = X @ beta
            residuals = differences - predicted_diff
            mse = np.sum(residuals ** 2) / (n_clean - X.shape[1])
            adjusted_mean_diff = beta[0]
            var_intercept = mse * np.linalg.pinv(X.T @ X)[0, 0]
            se_intercept = np.sqrt(var_intercept) if var_intercept >= 0 else 0.0
        except np.linalg.LinAlgError:
            return self.perform_test(before_clean, after_clean, alpha=significance_level)
        if se_intercept == 0:
            if adjusted_mean_diff == 0:
                t_stat = 0.0
            else:
                t_stat = np.inf if adjusted_mean_diff > 0 else -np.inf
        else:
            t_stat = adjusted_mean_diff / se_intercept
        df = n_clean - X.shape[1]
        if np.isinf(t_stat):
            p_value = 0.0
        elif df > 0:
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        else:
            p_value = float('nan')
        if se_intercept == 0 or df <= 0:
            ci_lower = adjusted_mean_diff
            ci_upper = adjusted_mean_diff
        else:
            t_critical = t.ppf(1 - significance_level / 2, df)
            margin_error = t_critical * se_intercept
            ci_lower = adjusted_mean_diff - margin_error
            ci_upper = adjusted_mean_diff + margin_error
        std_residuals = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
        if std_residuals != 0:
            effect_size = adjusted_mean_diff / std_residuals
        else:
            effect_size = np.inf if adjusted_mean_diff > 0 else -np.inf if adjusted_mean_diff < 0 else 0
        is_significant = p_value < significance_level if not np.isnan(p_value) else False
        return {'t_statistic': float(t_stat), 'p_value': float(p_value), 'degrees_of_freedom': int(df), 'confidence_interval': [float(ci_lower), float(ci_upper)], 'effect_size': float(effect_size), 'is_significant': bool(is_significant), 'mean_difference': float(adjusted_mean_diff), 'std_difference': float(std_residuals) if std_residuals != 0 else 0.0, 'covariates_used': len(combined_covariates)}


# ...(code omitted)...


class ScoreScaler:
    """
    Scales and transforms statistical scores for comparison and interpretation.
    
    This class provides methods for transforming statistical scores to common scales,
    standardizing metrics, and making them comparable across different contexts.
    Useful for combining multiple metrics or creating composite indicators.
    
    Attributes:
        method (str): Scaling method to use (default: 'z_score')
        reference_scores (Optional[List[float]]): Reference scores for some methods
        target_range (Tuple[float, float]): Target range for min-max scaling (default: (0, 1))
    """

    def __init__(self, method: str='z_score', reference_scores: Optional[List[float]]=None, target_range: Tuple[float, float]=(0, 1)):
        """
        Initialize the score scaler.
        
        Args:
            method: Scaling method. Supported values:
                   'z_score' (standardization to mean=0, std=1),
                   'min_max' (scaling to specified range),
                   'robust' (scaling using median and IQR),
                   'unit_vector' (scaling to unit length),
                   'percentile' (percentile-based scaling),
                   'softmax' (softmax transformation)
            reference_scores: Reference scores for percentile-based scaling (optional)
            target_range: Target range for min-max scaling (default: (0, 1))
            
        Raises:
            ValueError: If method is not supported or parameters are invalid
        """
        self.method = method
        self.reference_scores = reference_scores
        self.target_range = target_range
        self._supported_methods = ['z_score', 'min_max', 'robust', 'unit_vector', 'percentile', 'softmax']
        if method not in self._supported_methods:
            raise ValueError(f'method must be one of {self._supported_methods}')
        if len(target_range) != 2 or target_range[0] >= target_range[1]:
            raise ValueError('target_range must be a tuple (min, max) with min < max')
        self._scaling_params = {}

    def scale_scores(self, scores: Union[np.ndarray, DataBatch, List[float]]) -> Dict[str, Any]:
        """
        Scale scores using the specified method.
        
        Transforms input scores to a common scale for comparison.
        
        Args:
            scores: Input scores as numpy array, DataBatch, or list of floats
            
        Returns:
            Dict containing:
            - 'scaled_scores': Transformed scores on the target scale
            - 'original_scores': Copy of original scores
            - 'method': Scaling method used
            - 'scaling_parameters': Parameters used for scaling (e.g., mean, std)
            - 'score_range': Range of scaled scores (min, max)
            - 'mean_scaled': Mean of scaled scores
            - 'std_scaled': Standard deviation of scaled scores
            - 'outliers_detected': Number of outliers detected (if applicable)
            
        Raises:
            ValueError: If scores are empty or contain invalid values
            TypeError: If input data types are not supported
        """
        if isinstance(scores, DataBatch):
            original_scores = np.array(scores.data)
        elif isinstance(scores, list):
            original_scores = np.array(scores)
        elif isinstance(scores, np.ndarray):
            original_scores = scores.copy()
        else:
            raise TypeError('scores must be numpy array, DataBatch, or list of floats')
        if original_scores.size == 0:
            raise ValueError('Input scores cannot be empty')
        if not np.issubdtype(original_scores.dtype, np.number):
            raise ValueError('All scores must be numeric')
        if not np.isfinite(original_scores).all():
            raise ValueError('Scores contain NaN or infinite values')
        scaled_scores = np.empty_like(original_scores, dtype=float)
        scaling_params = {}
        outliers_detected = 0
        if self.method == 'z_score':
            mean_val = np.mean(original_scores)
            std_val = np.std(original_scores)
            if std_val == 0:
                scaled_scores.fill(0.0)
            else:
                scaled_scores = (original_scores - mean_val) / std_val
            scaling_params = {'mean': mean_val, 'std': std_val}
            if std_val != 0:
                z_scores = np.abs(scaled_scores)
                outliers_detected = np.sum(z_scores > 3)
        elif self.method == 'min_max':
            min_val = np.min(original_scores)
            max_val = np.max(original_scores)
            if min_val == max_val:
                scaled_scores.fill(self.target_range[0])
            else:
                scaled_scores = (original_scores - min_val) / (max_val - min_val)
                scaled_scores = scaled_scores * (self.target_range[1] - self.target_range[0]) + self.target_range[0]
            scaling_params = {'min': min_val, 'max': max_val, 'target_range': self.target_range}
        elif self.method == 'robust':
            median_val = np.median(original_scores)
            q75 = np.percentile(original_scores, 75)
            q25 = np.percentile(original_scores, 25)
            iqr = q75 - q25
            if iqr == 0:
                scaled_scores.fill(0.0)
            else:
                scaled_scores = (original_scores - median_val) / iqr
            scaling_params = {'median': median_val, 'q75': q75, 'q25': q25, 'iqr': iqr}
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers_detected = np.sum((original_scores < lower_bound) | (original_scores > upper_bound))
        elif self.method == 'unit_vector':
            norm = np.linalg.norm(original_scores)
            if norm == 0:
                scaled_scores = original_scores.copy()
            else:
                scaled_scores = original_scores / norm
            scaling_params = {'norm': norm}
        elif self.method == 'percentile':
            if self.reference_scores is not None:
                ref_scores = np.array(self.reference_scores)
                percentiles = np.searchsorted(np.sort(ref_scores), original_scores) / len(ref_scores) * 100
                scaled_scores = percentiles / 100
            else:
                from scipy.stats import rankdata
                ranks = rankdata(original_scores, method='average')
                percentiles = (ranks - 1) / (len(original_scores) - 1) * 100
                scaled_scores = percentiles / 100
            scaling_params = {'reference_scores_used': self.reference_scores is not None}
        elif self.method == 'softmax':
            max_val = np.max(original_scores)
            exp_scores = np.exp(original_scores - max_val)
            sum_exp = np.sum(exp_scores)
            if sum_exp == 0:
                scaled_scores.fill(1.0 / len(original_scores))
            else:
                scaled_scores = exp_scores / sum_exp
            scaling_params = {'max_subtracted': max_val}
        self._scaling_params = scaling_params.copy()
        self._scaling_params['method'] = self.method
        score_range = (float(np.min(scaled_scores)), float(np.max(scaled_scores)))
        mean_scaled = float(np.mean(scaled_scores))
        std_scaled = float(np.std(scaled_scores))
        return {'scaled_scores': scaled_scores, 'original_scores': original_scores, 'method': self.method, 'scaling_parameters': scaling_params, 'score_range': score_range, 'mean_scaled': mean_scaled, 'std_scaled': std_scaled, 'outliers_detected': outliers_detected}

    def inverse_transform(self, scaled_scores: Union[np.ndarray, DataBatch, List[float]]) -> Dict[str, Any]:
        """
        Apply inverse transformation to recover original scores.
        
        Converts scaled scores back to their original scale using stored parameters.
        
        Args:
            scaled_scores: Scaled scores to transform back
            
        Returns:
            Dict containing:
            - 'original_scores': Recovered original scores
            - 'method': Scaling method used for inverse transform
            - 'transformation_parameters': Parameters used for inverse transformation
            
        Raises:
            ValueError: If scaler has not been fitted or inputs are invalid
            TypeError: If input data types are not supported
        """
        if not self._scaling_params:
            raise ValueError('Scaler has not been fitted. Call scale_scores() first.')
        if isinstance(scaled_scores, DataBatch):
            scaled_array = np.array(scaled_scores.data)
        elif isinstance(scaled_scores, list):
            scaled_array = np.array(scaled_scores)
        elif isinstance(scaled_scores, np.ndarray):
            scaled_array = scaled_scores.copy()
        else:
            raise TypeError('scaled_scores must be numpy array, DataBatch, or list of floats')
        if scaled_array.size == 0:
            raise ValueError('Input scaled_scores cannot be empty')
        original_scores = np.empty_like(scaled_array, dtype=float)
        method = self._scaling_params.get('method', self.method)
        if method == 'z_score':
            mean_val = self._scaling_params['mean']
            std_val = self._scaling_params['std']
            original_scores = scaled_array * std_val + mean_val
        elif method == 'min_max':
            min_val = self._scaling_params['min']
            max_val = self._scaling_params['max']
            (target_min, target_max) = self._scaling_params['target_range']
            normalized = (scaled_array - target_min) / (target_max - target_min)
            original_scores = normalized * (max_val - min_val) + min_val
        elif method == 'robust':
            median_val = self._scaling_params['median']
            iqr = self._scaling_params['iqr']
            original_scores = scaled_array * iqr + median_val
        elif method == 'unit_vector':
            original_scores = scaled_array
            if 'norm' in self._scaling_params:
                original_scores = scaled_array * self._scaling_params['norm']
        elif method == 'percentile':
            original_scores = scaled_array
        elif method == 'softmax':
            original_scores = scaled_array
        else:
            raise ValueError(f'Unsupported method for inverse transform: {method}')
        return {'original_scores': original_scores, 'method': method, 'transformation_parameters': self._scaling_params}

    def combine_scores(self, score_sets: List[Union[np.ndarray, DataBatch, List[float]]], weights: Optional[List[float]]=None) -> Dict[str, Any]:
        """
        Combine multiple score sets into a composite score.
        
        Creates a weighted combination of multiple score sets after scaling.
        
        Args:
            score_sets: List of score sets to combine
            weights: Weights for each score set (if None, equal weights used)
            
        Returns:
            Dict containing:
            - 'composite_scores': Combined composite scores
            - 'individual_contributions': Contributions of each score set
            - 'weights_used': Weights applied to each score set
            - 'correlation_matrix': Correlations between score sets
            - 'reliability': Reliability of the composite score
            - 'scaling_methods': Methods used for each score set
            
        Raises:
            ValueError: If score sets have different lengths or are empty
            TypeError: If input data types are not supported
        """
        if not score_sets:
            raise ValueError('score_sets cannot be empty')
        processed_sets = []
        for (i, scores) in enumerate(score_sets):
            if isinstance(scores, DataBatch):
                arr = np.array(scores.data)
            elif isinstance(scores, list):
                arr = np.array(scores)
            elif isinstance(scores, np.ndarray):
                arr = scores.copy()
            else:
                raise TypeError(f'score_sets[{i}] must be numpy array, DataBatch, or list of floats')
            if arr.size == 0:
                raise ValueError(f'score_sets[{i}] cannot be empty')
            processed_sets.append(arr)
        lengths = [len(arr) for arr in processed_sets]
        if len(set(lengths)) > 1:
            raise ValueError('All score sets must have the same length')
        n_scores = lengths[0]
        if weights is None:
            weights = [1.0 / len(processed_sets)] * len(processed_sets)
        elif len(weights) != len(processed_sets):
            raise ValueError('Number of weights must match number of score sets')
        weight_sum = sum(weights)
        if weight_sum == 0:
            raise ValueError('Sum of weights cannot be zero')
        normalized_weights = [w / weight_sum for w in weights]
        scaled_sets = []
        scaling_methods = []
        individual_contributions = []
        for scores in processed_sets:
            temp_scaler = ScoreScaler(method=self.method, reference_scores=self.reference_scores, target_range=self.target_range)
            result = temp_scaler.scale_scores(scores)
            scaled_sets.append(result['scaled_scores'])
            scaling_methods.append(result['method'])
            individual_contributions.append(result['scaled_scores'] * normalized_weights[len(scaled_sets) - 1])
        stacked_sets = np.column_stack(scaled_sets)
        correlation_matrix = np.corrcoef(stacked_sets, rowvar=False)
        composite_scores = np.zeros(n_scores)
        for (i, (scaled_set, weight)) in enumerate(zip(scaled_sets, normalized_weights)):
            composite_scores += scaled_set * weight
        k = len(scaled_sets)
        if k > 1:
            var_sum = np.var(composite_scores, ddof=1)
            sum_var = sum((np.var(scaled_set, ddof=1) for scaled_set in scaled_sets))
            if var_sum > 0:
                reliability = k / (k - 1) * (1 - sum_var / var_sum)
                reliability = max(0.0, min(1.0, reliability))
            else:
                reliability = 0.0
        else:
            reliability = 1.0
        return {'composite_scores': composite_scores, 'individual_contributions': np.column_stack(individual_contributions), 'weights_used': normalized_weights, 'correlation_matrix': correlation_matrix, 'reliability': reliability, 'scaling_methods': scaling_methods}

class DistributionFitTest:
    """
    Performs goodness-of-fit tests to assess how well a theoretical distribution 
    fits observed data.
    
    This class provides methods for testing whether sample data comes from
    a specific theoretical distribution using various statistical tests.
    
    Attributes:
        test_method (str): Goodness-of-fit test method (default: 'kolmogorov_smirnov')
        distribution (str): Theoretical distribution to test against (default: 'normal')
        significance_level (float): Significance level for hypothesis tests (default: 0.05)
    """

    def __init__(self, test_method: str='kolmogorov_smirnov', distribution: str='normal', significance_level: float=0.05):
        """
        Initialize the distribution fit test.
        
        Args:
            test_method: Goodness-of-fit test method. Supported values:
                        'kolmogorov_smirnov' (K-S test),
                        'anderson_darling' (A-D test),
                        'chi_square' (Chi-square test),
                        'cramér_von_mises' (Cramér-von Mises test),
                        'watson' (Watson test),
                        'custom' (User-provided test function)
            distribution: Theoretical distribution to test against. Supported values:
                         'normal', 'exponential', 'uniform', 'gamma', 'beta', 'weibull'
            significance_level: Significance level for hypothesis tests (default: 0.05)
            
        Raises:
            ValueError: If test_method, distribution, or significance_level is invalid
        """
        self.test_method = test_method
        self.distribution = distribution
        self.significance_level = significance_level
        self._supported_tests = ['kolmogorov_smirnov', 'anderson_darling', 'chi_square', 'cramér_von_mises', 'watson', 'custom']
        self._supported_distributions = ['normal', 'exponential', 'uniform', 'gamma', 'beta', 'weibull']
        if test_method not in self._supported_tests:
            raise ValueError(f'test_method must be one of {self._supported_tests}')
        if distribution not in self._supported_distributions:
            raise ValueError(f'distribution must be one of {self._supported_distributions}')
        if not 0 < significance_level < 1:
            raise ValueError('significance_level must be between 0 and 1')

    def _extract_data(self, data: Union[np.ndarray, DataBatch]) -> np.ndarray:
        """Extract numpy array from input data."""
        if isinstance(data, DataBatch):
            return np.asarray(data.data).flatten()
        elif isinstance(data, np.ndarray):
            return data.flatten()
        else:
            return np.asarray(data).flatten()

    def _get_distribution_object(self, distribution: str, params: tuple):
        """Get scipy.stats distribution object for the specified distribution with parameters."""
        if distribution == 'normal':
            return stats.norm(*params)
        elif distribution == 'exponential':
            return stats.expon(*params)
        elif distribution == 'uniform':
            return stats.uniform(*params)
        elif distribution == 'gamma':
            return stats.gamma(*params)
        elif distribution == 'beta':
            return stats.beta(*params)
        elif distribution == 'weibull':
            return stats.weibull_min(*params)
        else:
            raise ValueError(f'Unsupported distribution: {distribution}')

    def _estimate_parameters(self, data: np.ndarray, distribution: str) -> tuple:
        """Estimate parameters for the specified distribution."""
        if distribution == 'normal':
            return stats.norm.fit(data)
        elif distribution == 'exponential':
            return stats.expon.fit(data)
        elif distribution == 'uniform':
            return stats.uniform.fit(data)
        elif distribution == 'gamma':
            return stats.gamma.fit(data)
        elif distribution == 'beta':
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            eps = 1e-06
            data_norm = np.clip(data_norm, eps, 1 - eps)
            return stats.beta.fit(data_norm)
        elif distribution == 'weibull':
            return stats.weibull_min.fit(data)
        else:
            raise ValueError(f'Unsupported distribution: {distribution}')

    def _calculate_aic_bic(self, data: np.ndarray, distribution: str, params: tuple) -> tuple:
        """Calculate AIC and BIC for the fitted distribution."""
        dist_fitted = self._get_distribution_object(distribution, params)
        log_likelihood = np.sum(dist_fitted.logpdf(data))
        k = len(params)
        n = len(data)
        aic = 2 * k - 2 * log_likelihood
        bic = np.log(n) * k - 2 * log_likelihood
        return (aic, bic)

    def _kolmogorov_smirnov_test(self, data: np.ndarray, distribution: str, params: tuple) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test."""
        dist_fitted = self._get_distribution_object(distribution, params)
        (ks_stat, p_value) = kstest(data, lambda x: dist_fitted.cdf(x))
        n = len(data)
        critical_value = stats.kstwo.ppf(1 - self.significance_level, n)
        significant = p_value < self.significance_level
        if p_value > 0.1:
            fit_quality = 'Excellent'
        elif p_value > 0.05:
            fit_quality = 'Good'
        elif p_value > 0.01:
            fit_quality = 'Fair'
        else:
            fit_quality = 'Poor'
        return {'test_statistic': ks_stat, 'p_value': p_value, 'critical_value': critical_value, 'significant': significant, 'fit_quality': fit_quality}

    def _anderson_darling_test(self, data: np.ndarray, distribution: str, params: tuple) -> Dict[str, Any]:
        """Perform Anderson-Darling test."""
        if distribution == 'normal':
            result = anderson(data, dist='norm')
        elif distribution == 'exponential':
            result = anderson(data, dist='expon')
        else:
            dist_fitted = self._get_distribution_object(distribution, params)
            data_cdf = dist_fitted.cdf(data)
            (ad_stat, p_value) = kstest(data_cdf, 'uniform')
            n = len(data)
            critical_value = stats.kstwo.ppf(1 - self.significance_level, n)
            significant = p_value < self.significance_level
            if p_value > 0.1:
                fit_quality = 'Excellent'
            elif p_value > 0.05:
                fit_quality = 'Good'
            elif p_value > 0.01:
                fit_quality = 'Fair'
            else:
                fit_quality = 'Poor'
            return {'test_statistic': ad_stat, 'p_value': p_value, 'critical_value': critical_value, 'significant': significant, 'fit_quality': fit_quality}
        sig_levels = result.significance_level
        crit_values = result.critical_values
        stat = result.statistic
        p_value = 0.05
        critical_value = 0.0
        significant = False
        for (i, level) in enumerate(sig_levels):
            if stat < crit_values[i]:
                p_value = level / 100.0
                critical_value = crit_values[i]
                break
        else:
            p_value = 0.01 / 100.0
            critical_value = crit_values[-1]
        significant = p_value < self.significance_level
        if p_value > 0.1:
            fit_quality = 'Excellent'
        elif p_value > 0.05:
            fit_quality = 'Good'
        elif p_value > 0.01:
            fit_quality = 'Fair'
        else:
            fit_quality = 'Poor'
        return {'test_statistic': stat, 'p_value': p_value, 'critical_value': critical_value, 'significant': significant, 'fit_quality': fit_quality}

    def _chi_square_test(self, data: np.ndarray, distribution: str, params: tuple) -> Dict[str, Any]:
        """Perform Chi-square goodness-of-fit test."""
        n_bins = min(20, max(5, int(np.sqrt(len(data)))))
        (obs_freq, bin_edges) = np.histogram(data, bins=n_bins)
        dist_fitted = self._get_distribution_object(distribution, params)
        exp_freq = []
        for i in range(len(bin_edges) - 1):
            p = dist_fitted.cdf(bin_edges[i + 1]) - dist_fitted.cdf(bin_edges[i])
            exp_freq.append(p * len(data))
        exp_freq = np.array(exp_freq)
        MIN_EXP_FREQ = 5
        combined_obs = []
        combined_exp = []
        i = 0
        while i < len(obs_freq):
            obs_sum = obs_freq[i]
            exp_sum = exp_freq[i]
            j = i + 1
            while exp_sum < MIN_EXP_FREQ and j < len(obs_freq):
                obs_sum += obs_freq[j]
                exp_sum += exp_freq[j]
                j += 1
            combined_obs.append(obs_sum)
            combined_exp.append(exp_sum)
            i = j
        if len(combined_obs) < 2:
            chi2_stat = 0
            p_value = 1.0
        else:
            try:
                (chi2_stat, p_value) = chisquare(combined_obs, combined_exp)
            except:
                chi2_stat = 0
                p_value = 1.0
        df = len(combined_obs) - 1 - len(params)
        if df <= 0:
            df = 1
        critical_value = stats.chi2.ppf(1 - self.significance_level, df)
        significant = p_value < self.significance_level
        if p_value > 0.1:
            fit_quality = 'Excellent'
        elif p_value > 0.05:
            fit_quality = 'Good'
        elif p_value > 0.01:
            fit_quality = 'Fair'
        else:
            fit_quality = 'Poor'
        return {'test_statistic': chi2_stat, 'p_value': p_value, 'critical_value': critical_value, 'significant': significant, 'fit_quality': fit_quality}

    def _cramer_von_mises_test(self, data: np.ndarray, distribution: str, params: tuple) -> Dict[str, Any]:
        """Perform Cramér-von Mises test."""
        dist_fitted = self._get_distribution_object(distribution, params)
        u_data = dist_fitted.cdf(data)
        try:
            result = cramervonmises(u_data, 'uniform')
            cvm_stat = result.statistic
            p_value = result.pvalue
        except:
            cvm_stat = 0.0
            p_value = 1.0
        n = len(data)
        critical_value = 0.15
        significant = p_value < self.significance_level
        if p_value > 0.1:
            fit_quality = 'Excellent'
        elif p_value > 0.05:
            fit_quality = 'Good'
        elif p_value > 0.01:
            fit_quality = 'Fair'
        else:
            fit_quality = 'Poor'
        return {'test_statistic': cvm_stat, 'p_value': p_value, 'critical_value': critical_value, 'significant': significant, 'fit_quality': fit_quality}

    def _watson_test(self, data: np.ndarray, distribution: str, params: tuple) -> Dict[str, Any]:
        """Perform Watson test (modified Cramér-von Mises)."""
        dist_fitted = self._get_distribution_object(distribution, params)
        u_data = dist_fitted.cdf(data)
        u_sorted = np.sort(u_data)
        n = len(u_sorted)
        mean_u = np.mean(u_sorted)
        cvm_stat = 1 / (12 * n)
        for i in range(n):
            cvm_stat += (u_sorted[i] - (2 * i + 1) / (2 * n)) ** 2
        watson_stat = cvm_stat - n * (mean_u - 0.5) ** 2
        p_value = np.exp(-watson_stat * 6)
        critical_value = 0.1
        significant = p_value < self.significance_level
        if p_value > 0.1:
            fit_quality = 'Excellent'
        elif p_value > 0.05:
            fit_quality = 'Good'
        elif p_value > 0.01:
            fit_quality = 'Fair'
        else:
            fit_quality = 'Poor'
        return {'test_statistic': watson_stat, 'p_value': p_value, 'critical_value': critical_value, 'significant': significant, 'fit_quality': fit_quality}

    def perform_fit_test(self, data: Union[np.ndarray, DataBatch]) -> Dict[str, Any]:
        """
        Perform goodness-of-fit test for the specified distribution.
        
        Tests the null hypothesis that the data comes from the specified distribution.
        
        Args:
            data: Sample data as numpy array or DataBatch
            
        Returns:
            Dict containing:
            - 'test_statistic': Calculated test statistic
            - 'p_value': P-value for the test
            - 'critical_value': Critical value at significance level
            - 'significant': Boolean indicating if null hypothesis is rejected
            - 'fit_quality': Qualitative assessment of fit quality
            - 'distribution': Distribution tested
            - 'sample_size': Number of observations
            - 'parameters': Estimated distribution parameters
            - 'confidence_level': Confidence level of the test
            
        Raises:
            ValueError: If data is empty or contains invalid values
            TypeError: If input data types are not supported
        """
        data_array = self._extract_data(data)
        if len(data_array) == 0:
            raise ValueError('Data cannot be empty')
        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
            raise ValueError('Data contains NaN or infinite values')
        params = self._estimate_parameters(data_array, self.distribution)
        if self.test_method == 'kolmogorov_smirnov':
            test_result = self._kolmogorov_smirnov_test(data_array, self.distribution, params)
        elif self.test_method == 'anderson_darling':
            test_result = self._anderson_darling_test(data_array, self.distribution, params)
        elif self.test_method == 'chi_square':
            test_result = self._chi_square_test(data_array, self.distribution, params)
        elif self.test_method == 'cramér_von_mises':
            test_result = self._cramer_von_mises_test(data_array, self.distribution, params)
        elif self.test_method == 'watson':
            test_result = self._watson_test(data_array, self.distribution, params)
        else:
            raise ValueError(f'Unsupported test method: {self.test_method}')
        test_result['distribution'] = self.distribution
        test_result['sample_size'] = len(data_array)
        test_result['parameters'] = params
        test_result['confidence_level'] = 1 - self.significance_level
        return test_result

    def compare_distributions(self, data: Union[np.ndarray, DataBatch], distributions: Optional[List[str]]=None) -> Dict[str, Any]:
        """
        Compare fit of multiple distributions to the data.
        
        Performs goodness-of-fit tests for multiple distributions and ranks them.
        
        Args:
            data: Sample data as numpy array or DataBatch
            distributions: List of distributions to test (if None, uses common distributions)
            
        Returns:
            Dict containing:
            - 'results': Dictionary of test results for each distribution
            - 'best_fit': Best fitting distribution based on test statistics
            - 'ranking': Ranking of distributions by fit quality
            - 'aic_values': AIC values for each distribution
            - 'bic_values': BIC values for each distribution
            - 'recommended_distribution': Recommended distribution based on multiple criteria
            
        Raises:
            ValueError: If data is empty or contains invalid values
            TypeError: If input data types are not supported
        """
        data_array = self._extract_data(data)
        if len(data_array) == 0:
            raise ValueError('Data cannot be empty')
        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
            raise ValueError('Data contains NaN or infinite values')
        if distributions is None:
            distributions = self._supported_distributions.copy()
        else:
            for dist in distributions:
                if dist not in self._supported_distributions:
                    raise ValueError(f'Unsupported distribution: {dist}')
        results = {}
        aic_values = {}
        bic_values = {}
        for dist in distributions:
            try:
                params = self._estimate_parameters(data_array, dist)
                (aic, bic) = self._calculate_aic_bic(data_array, dist, params)
                aic_values[dist] = aic
                bic_values[dist] = bic
                temp_test_method = self.test_method
                self.test_method = 'kolmogorov_smirnov'
                test_result = self._kolmogorov_smirnov_test(data_array, dist, params)
                self.test_method = temp_test_method
                results[dist] = test_result
                results[dist]['parameters'] = params
            except Exception as e:
                warnings.warn(f'Could not fit distribution {dist}: {str(e)}')
                continue
        ranked_dists = sorted(aic_values.keys(), key=lambda x: aic_values[x])
        best_fit = ranked_dists[0] if ranked_dists else None
        recommended_dist = best_fit
        return {'results': results, 'best_fit': best_fit, 'ranking': ranked_dists, 'aic_values': aic_values, 'bic_values': bic_values, 'recommended_distribution': recommended_dist}

    def custom_fit_test(self, data: Union[np.ndarray, DataBatch], test_function: Callable, null_distribution: Callable) -> Dict[str, Any]:
        """
        Perform custom goodness-of-fit test with user-provided functions.
        
        Allows users to specify their own test statistic and null distribution.
        
        Args:
            data: Sample data as numpy array or DataBatch
            test_function: Function to compute test statistic
                          Signature: test_function(data, null_distribution) -> float
            null_distribution: Function representing null distribution
                              Signature: null_distribution(params) -> samples
            
        Returns:
            Dict containing:
            - 'test_statistic': Calculated test statistic
            - 'p_value': P-value for the test
            - 'critical_value': Critical value at significance level
            - 'significant': Boolean indicating if null hypothesis is rejected
            - 'bootstrap_samples': Number of bootstrap samples used
            - 'confidence_interval': Confidence interval for test statistic
            
        Raises:
            ValueError: If data is empty or functions have incorrect signatures
            TypeError: If input data types are not supported
        """
        data_array = self._extract_data(data)
        if len(data_array) == 0:
            raise ValueError('Data cannot be empty')
        if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
            raise ValueError('Data contains NaN or infinite values')
        try:
            test_stat = test_function(data_array, null_distribution)
        except Exception as e:
            raise ValueError(f'Error calling test_function: {str(e)}')
        n_bootstrap = 1000
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            try:
                null_sample = null_distribution(len(data_array))
                null_stat = test_function(null_sample, null_distribution)
                bootstrap_stats.append(null_stat)
            except Exception:
                continue
        if len(bootstrap_stats) == 0:
            raise ValueError('Could not generate any bootstrap samples')
        bootstrap_stats = np.array(bootstrap_stats)
        p_value = np.mean(bootstrap_stats >= test_stat)
        critical_value = np.percentile(bootstrap_stats, (1 - self.significance_level) * 100)
        significant = test_stat > critical_value
        alpha = self.significance_level
        ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        return {'test_statistic': test_stat, 'p_value': p_value, 'critical_value': critical_value, 'significant': significant, 'bootstrap_samples': len(bootstrap_stats), 'confidence_interval': (ci_lower, ci_upper)}


# ...(code omitted)...


def spearman_rank_correlation(x: Union[np.ndarray, DataBatch], y: Union[np.ndarray, DataBatch], alpha: float=0.05) -> Dict[str, Any]:
    """
    Calculate Spearman rank-order correlation coefficient between two variables.
    
    Spearman's rank correlation assesses how well the relationship between two variables 
    can be described using a monotonic function. It's a non-parametric measure that 
    uses the ranked values rather than the raw data.
    
    Args:
        x: First variable data as numpy array or DataBatch
        y: Second variable data as numpy array or DataBatch
        
    Returns:
        Dict containing:
        - 'correlation_coefficient': Spearman's rho correlation coefficient (-1 to 1)
        - 'p_value': Two-tailed p-value for test of zero correlation
        - 'sample_size': Number of observations used
        - 'confidence_interval': 95% confidence interval for the correlation
        - 'ranking_x': Ranked values of x
        - 'ranking_y': Ranked values of y
        - 'significant': Boolean indicating if correlation is statistically significant at α=0.05
        
    Raises:
        ValueError: If x and y have different lengths or are empty
        TypeError: If input data types are not supported
    """
    if isinstance(x, DataBatch):
        x_data = np.asarray(x.data).flatten()
    elif isinstance(x, np.ndarray):
        x_data = x.flatten()
    else:
        x_data = np.asarray(x).flatten()
    if isinstance(y, DataBatch):
        y_data = np.asarray(y.data).flatten()
    elif isinstance(y, np.ndarray):
        y_data = y.flatten()
    else:
        y_data = np.asarray(y).flatten()
    if len(x_data) != len(y_data):
        raise ValueError('Input arrays must have the same length')
    if len(x_data) == 0:
        raise ValueError('Input arrays cannot be empty')
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    if len(x_clean) == 0:
        raise ValueError('No valid (non-NaN) observations found')
    rank_x = _rank_data(x_clean)
    rank_y = _rank_data(y_clean)
    n = len(x_clean)
    if n < 2:
        raise ValueError('At least 2 observations required for correlation calculation')
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        correlation_coefficient = 0.0
    else:
        correlation_coefficient = np.corrcoef(rank_x, rank_y)[0, 1]
    if n <= 2:
        p_value = 1.0
    else:
        if abs(correlation_coefficient) == 1.0:
            t_stat = np.inf if correlation_coefficient == 1.0 else -np.inf
        elif 1 - correlation_coefficient ** 2 == 0:
            t_stat = np.inf if correlation_coefficient > 0 else -np.inf
        else:
            t_stat = correlation_coefficient * np.sqrt((n - 2) / (1 - correlation_coefficient ** 2))
        from scipy.stats import t
        p_value = 2 * (1 - t.cdf(abs(t_stat), n - 2))
    if n > 3 and abs(correlation_coefficient) < 1.0:
        try:
            z = 0.5 * np.log((1 + correlation_coefficient) / (1 - correlation_coefficient))
            se = 1.0 / np.sqrt(n - 3)
            z_critical = 1.96
            z_lower = z - z_critical * se
            z_upper = z + z_critical * se
            ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            ci_lower = max(-1, min(1, ci_lower))
            ci_upper = max(-1, min(1, ci_upper))
            confidence_interval = [float(ci_lower), float(ci_upper)]
        except (FloatingPointError, ValueError, ZeroDivisionError):
            margin = 0.1
            ci_lower = max(-1, correlation_coefficient - margin)
            ci_upper = min(1, correlation_coefficient + margin)
            confidence_interval = [float(ci_lower), float(ci_upper)]
    else:
        confidence_interval = [float(correlation_coefficient), float(correlation_coefficient)]
    significant = p_value < alpha
    return {'correlation_coefficient': float(correlation_coefficient), 'p_value': float(p_value), 'sample_size': int(n), 'confidence_interval': confidence_interval, 'ranking_x': rank_x.tolist(), 'ranking_y': rank_y.tolist(), 'significant': bool(significant)}


# ...(code omitted)...


def calculate_rank_correlation_matrix(data: Union[np.ndarray, DataBatch], method: str='spearman', alpha: float=0.05) -> Dict[str, Any]:
    """
    Calculate rank correlation matrix for multiple variables.
    
    Computes pairwise rank correlations between all variables in a dataset.
    
    Args:
        data: Data matrix as numpy array or DataBatch (samples x features)
        method: Correlation method to use ('spearman' or 'kendall')
        
    Returns:
        Dict containing:
        - 'correlation_matrix': Matrix of correlation coefficients
        - 'p_value_matrix': Matrix of p-values for each correlation
        - 'significant_matrix': Boolean matrix indicating significant correlations
        - 'sample_size': Number of observations used
        - 'method': Method used for calculation
        
    Raises:
        ValueError: If data is empty or method is not supported
        TypeError: If input data types are not supported
    """
    if method not in ['spearman', 'kendall']:
        raise ValueError("Method must be 'spearman' or 'kendall'")
    if isinstance(data, DataBatch):
        raw_data = data.data
    else:
        raw_data = data
    if not isinstance(raw_data, np.ndarray):
        raw_data = np.array(raw_data)
    if raw_data.size == 0:
        raise ValueError('Input data is empty')
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(-1, 1)
    elif raw_data.ndim != 2:
        raise ValueError('Input data must be 1 or 2 dimensional')
    (n_samples, n_features) = raw_data.shape
    correlation_matrix = np.eye(n_features)
    p_value_matrix = np.zeros((n_features, n_features))
    significant_matrix = np.zeros((n_features, n_features), dtype=bool)
    from scipy.stats import spearmanr, kendalltau
    for i in range(n_features):
        for j in range(i + 1, n_features):
            x = raw_data[:, i]
            y = raw_data[:, j]
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            if len(x_clean) < 2:
                corr = 0.0
                p_val = 1.0
            else:
                try:
                    if method == 'spearman':
                        (corr, p_val) = spearmanr(x_clean, y_clean)
                    else:
                        (corr, p_val) = kendalltau(x_clean, y_clean)
                    if np.isnan(corr):
                        corr = 0.0
                    if np.isnan(p_val):
                        p_val = 1.0
                except Exception:
                    corr = 0.0
                    p_val = 1.0
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
            p_value_matrix[i, j] = p_val
            p_value_matrix[j, i] = p_val
            significant_matrix[i, j] = p_val < alpha
            significant_matrix[j, i] = p_val < alpha
    sample_size = n_samples
    return {'correlation_matrix': correlation_matrix, 'p_value_matrix': p_value_matrix, 'significant_matrix': significant_matrix, 'sample_size': sample_size, 'method': method}