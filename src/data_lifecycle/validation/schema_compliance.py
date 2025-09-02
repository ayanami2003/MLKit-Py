from typing import Dict, Any, Optional, List, Union, Callable
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
import numpy as np


# ...(code omitted)...


class ColumnPresenceValidator(BaseValidator):
    """
    Validates that required columns are present in a data batch.
    
    This validator checks whether all specified columns exist in the data batch.
    It can operate in strict mode (no extra columns allowed) or permissive mode
    (only checks for required columns).
    
    Attributes:
        required_columns (List[str]): Columns that must be present
        strict_mode (bool): If True, fails when extra columns are present
        allowed_columns (Optional[List[str]]): Whitelist of permitted columns in strict mode
        
    Methods:
        validate: Performs column presence validation on a data batch
        set_required_columns: Updates the list of required columns
    """

    def __init__(self, required_columns: List[str], strict_mode: bool=False, allowed_columns: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the ColumnPresenceValidator with required columns.
        
        Args:
            required_columns: List of column names that must be present
            strict_mode: If True, only allows specified columns
            allowed_columns: Optional whitelist of column names when in strict mode
            name: Optional name for the validator
        """
        super().__init__(name)
        self.required_columns = required_columns
        self.strict_mode = strict_mode
        self.allowed_columns = allowed_columns

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate that required columns are present in the data batch.
        
        Args:
            data: DataBatch to validate
            **kwargs: Additional validation parameters
            
        Returns:
            bool: True if all required columns are present (and no extra columns in strict mode), False otherwise
        """
        if data.feature_names is not None:
            data_columns = set(data.feature_names)
        else:
            self.add_error('Data batch does not contain feature names')
            return False
        required_columns_set = set(self.required_columns)
        missing_columns = required_columns_set - data_columns
        if missing_columns:
            self.add_error(f'Missing required columns: {list(missing_columns)}')
            return False
        if self.strict_mode:
            allowed_columns_set = required_columns_set.copy()
            if self.allowed_columns is not None:
                allowed_columns_set.update(self.allowed_columns)
            unexpected_columns = data_columns - allowed_columns_set
            if unexpected_columns:
                self.add_error(f'Unexpected columns found in strict mode: {list(unexpected_columns)}')
                return False
        return True

    def set_required_columns(self, columns: List[str]) -> None:
        """
        Update the list of required columns.
        
        Args:
            columns: New list of required column names
        """
        if columns is None:
            raise ValueError('Required columns cannot be None')
        self.required_columns = list(columns)


# ...(code omitted)...


class DistributionalValidator(BaseValidator):
    """
    Validates data against expected statistical distributions and properties.
    
    This validator performs statistical tests and checks to ensure data follows
    expected distributions, has appropriate moments (mean, variance), and meets
    other distributional expectations. It supports both univariate and multivariate
    distributional validations.
    
    Attributes:
        distribution_rules (List[Dict[str, Any]]): Rules defining expected distributions
        test_configurations (Dict[str, Dict[str, Any]]): Configuration for statistical tests
        alpha_threshold (float): Significance level for statistical tests
        
    Methods:
        validate: Performs distributional validation on a data batch
        add_distribution_rule: Adds a distribution expectation rule
        configure_statistical_test: Configures parameters for a statistical test
    """

    def __init__(self, distribution_rules=None, test_configurations=None, alpha_threshold=0.05, name=None):
        """
        Initialize the DistributionalValidator with distribution rules and test configurations.
        
        Args:
            distribution_rules: List of rules defining expected data distributions
            test_configurations: Configuration parameters for statistical tests
            alpha_threshold: Significance level for statistical tests (default: 0.05)
            name: Optional name for the validator
        """
        super().__init__(name)
        self.distribution_rules = distribution_rules or []
        self.test_configurations = test_configurations or {}
        self.alpha_threshold = alpha_threshold

    def validate(self, data, **kwargs) -> bool:
        """
        Validate a data batch against distributional expectations.
        
        Args:
            data: DataBatch to validate
            **kwargs: Additional validation parameters
            
        Returns:
            bool: True if distributional validation passes, False otherwise
        """
        self.reset_validation_state()
        if not hasattr(data, 'data') or data.data is None:
            self.add_error('Invalid data batch: missing data attribute')
            return False
        for rule in self.distribution_rules:
            try:
                column = rule.get('column')
                distribution = rule.get('distribution', 'normal')
                test_type = rule.get('test', 'ks')
                if column is not None:
                    if hasattr(data, 'features') and column in data.features.columns:
                        test_data = data.features[column].values
                    elif isinstance(data.data, dict) and column in data.data:
                        test_data = data.data[column]
                    else:
                        self.add_error(f"Column '{column}' not found in data")
                        continue
                else:
                    test_data = data.data.flatten() if hasattr(data.data, 'flatten') else data.data
                import numpy as np
                test_data = np.asarray(test_data).flatten()
                test_data = test_data[~np.isnan(test_data)]
                if len(test_data) < 2:
                    self.add_error(f"Insufficient data for column '{column}' (need at least 2 values)")
                    continue
                test_passed = self._apply_statistical_test(test_data, distribution, test_type, rule)
                if not test_passed:
                    column_name = column if column is not None else 'data'
                    self.add_error(f"Distribution test failed for '{column_name}' (distribution: {distribution}, test: {test_type})")
            except Exception as e:
                column_name = rule.get('column', 'unknown')
                self.add_error(f"Error validating distribution for column '{column_name}': {str(e)}")
                continue
        return len(self.validation_errors) == 0

    def add_distribution_rule(self, rule) -> None:
        """
        Add a distribution expectation rule.
        
        Args:
            rule: Rule defining expected distribution (e.g., normal, uniform, etc.)
        """
        if not isinstance(rule, dict):
            raise TypeError('Rule must be a dictionary')
        required_keys = ['distribution']
        for key in required_keys:
            if key not in rule:
                raise ValueError(f'Rule missing required key: {key}')
        supported_distributions = ['normal', 'uniform', 'exponential', 'poisson']
        if rule['distribution'] not in supported_distributions:
            raise ValueError(f"Unsupported distribution: {rule['distribution']}. Supported distributions: {supported_distributions}")
        supported_tests = ['ks', 'anderson', 'shapiro']
        test_type = rule.get('test', 'ks')
        if test_type not in supported_tests:
            raise ValueError(f'Unsupported test type: {test_type}. Supported tests: {supported_tests}')
        self.distribution_rules.append(rule)

    def configure_statistical_test(self, test_name: str, parameters) -> None:
        """
        Configure parameters for a statistical test.
        
        Args:
            test_name: Name of the statistical test to configure
            parameters: Configuration parameters for the test
        """
        if not isinstance(parameters, dict):
            raise TypeError('Parameters must be a dictionary')
        supported_tests = ['ks', 'anderson', 'shapiro', 'chi2']
        if test_name not in supported_tests:
            raise ValueError(f'Unsupported test: {test_name}. Supported tests: {supported_tests}')
        self.test_configurations[test_name] = parameters

    def _apply_statistical_test(self, data, distribution: str, test_type: str, rule: dict) -> bool:
        """
        Apply a statistical test to check if data follows the expected distribution.
        
        Args:
            data: Data to test
            distribution: Expected distribution name
            test_type: Type of statistical test to apply
            rule: The distribution rule being applied
            
        Returns:
            bool: True if test passes, False otherwise
        """
        import numpy as np
        from scipy import stats
        test_config = self.test_configurations.get(test_type, {})
        significance_level = test_config.get('alpha', self.alpha_threshold)
        try:
            if test_type == 'ks':
                return self._kolmogorov_smirnov_test(data, distribution, significance_level)
            elif test_type == 'anderson':
                return self._anderson_darling_test(data, distribution, significance_level)
            elif test_type == 'shapiro' and distribution == 'normal':
                return self._shapiro_wilk_test(data, significance_level)
            else:
                raise ValueError(f'Unsupported test/distribution combination: {test_type}/{distribution}')
        except Exception as e:
            raise RuntimeError(f'Failed to apply {test_type} test for {distribution} distribution: {str(e)}')

    def _kolmogorov_smirnov_test(self, data, distribution: str, significance_level: float) -> bool:
        """Perform Kolmogorov-Smirnov test."""
        from scipy import stats
        import numpy as np
        if distribution == 'normal':
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            (_, p_value) = stats.kstest(data, 'norm', args=(mean, std))
        elif distribution == 'uniform':
            a = np.min(data)
            b = np.max(data)
            (_, p_value) = stats.kstest(data, 'uniform', args=(a, b - a))
        elif distribution == 'exponential':
            (loc, scale) = stats.expon.fit(data, floc=0)
            (_, p_value) = stats.kstest(data, 'expon', args=(loc, scale))
        else:
            raise ValueError(f'KS test not supported for distribution: {distribution}')
        return p_value > significance_level

    def _anderson_darling_test(self, data, distribution: str, significance_level: float) -> bool:
        """Perform Anderson-Darling test."""
        from scipy import stats
        import numpy as np
        if distribution == 'normal':
            result = stats.anderson(data, dist='norm')
            sig_level_percent = significance_level * 100
            idx = np.argmin(np.abs(result.significance_level - sig_level_percent))
            return result.statistic < result.critical_values[idx]
        else:
            raise ValueError(f'Anderson-Darling test not supported for distribution: {distribution}')

    def _shapiro_wilk_test(self, data, significance_level: float) -> bool:
        """Perform Shapiro-Wilk test for normality."""
        from scipy import stats
        import numpy as np
        if len(data) < 3 or len(data) > 5000:
            raise ValueError('Shapiro-Wilk test requires 3 to 5000 samples')
        (_, p_value) = stats.shapiro(data)
        return p_value > significance_level