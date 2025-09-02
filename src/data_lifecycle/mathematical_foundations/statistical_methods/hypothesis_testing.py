from typing import Any, Dict, Union, Tuple, List
from abc import ABC, abstractmethod
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
import numpy as np
from scipy.stats import kruskal

class StatisticalTest(ABC):

    def __init__(self, name: str=None, significance_level: float=0.05):
        """
        Initialize the statistical test.

        Args:
            name (str): Optional name for the test instance.
            significance_level (float): Default significance level for hypothesis testing (default: 0.05).
        """
        self.name = name or self.__class__.__name__
        if not 0 < significance_level < 1:
            raise ValueError('Significance level must be between 0 and 1')
        self.significance_level = significance_level
        self.last_result = None

    @abstractmethod
    def perform_test(self, data: Union[DataBatch, dict, list], **kwargs) -> Dict[str, Any]:
        """
        Perform the statistical test on the provided data.

        Args:
            data (Union[DataBatch, dict, list]): Input data for the test. Format depends on the specific test.
            **kwargs: Additional parameters for the test (e.g., alternative hypotheses, group labels).

        Returns:
            Dict[str, Any]: A dictionary containing test results, including:
                - 'statistic': The computed test statistic
                - 'p_value': The p-value for the test
                - 'significant': Boolean indicating if result is statistically significant
                - 'interpretation': Text interpretation of the result
                - Other test-specific metrics

        Raises:
            ValueError: If the input data is invalid or does not meet test assumptions.
        """
        pass

    def validate_assumptions(self, data: Union[DataBatch, dict, list]) -> bool:
        """
        Validate assumptions required for the statistical test.

        Args:
            data (Union[DataBatch, dict, list]): Input data to validate.

        Returns:
            bool: True if all assumptions are satisfied, False otherwise.
        """
        return True

    def run_with_validation(self, data: Union[DataBatch, dict, list], validator: BaseValidator=None, **kwargs) -> Dict[str, Any]:
        """
        Run the test with optional data validation.

        Args:
            data (Union[DataBatch, dict, list]): Input data for the test.
            validator (BaseValidator): Optional validator to check data quality before testing.
            **kwargs: Additional parameters for the test.

        Returns:
            Dict[str, Any]: Test results, potentially with validation warnings or errors.

        Raises:
            ValueError: If data validation fails and prevents test execution.
        """
        if not self.validate_assumptions(data):
            raise ValueError('Test assumptions are not satisfied')
        if validator is not None:
            is_valid = validator.validate(data)
            if not is_valid:
                validation_report = validator.get_validation_report()
                raise ValueError(f'Data validation failed: {validation_report}')
        return self.perform_test(data, **kwargs)


# ...(code omitted)...


def kruskal_wallis_test(data: Union[DataBatch, List[list]], significance_level: float=0.05) -> dict:
    """
    Perform the Kruskal-Wallis H test for comparing distributions across multiple independent groups.

    This non-parametric statistical test evaluates whether samples originate from the same distribution.
    It is an extension of the Mann-Whitney U test and is particularly useful when the assumptions of
    one-way ANOVA are not met. The test works with ordinal or continuous data that does not meet 
    normality assumptions.

    Args:
        data (Union[DataBatch, List[list]]): A collection of groups/samples to compare. 
            If DataBatch is provided, assumes each column represents a group. 
            If List of lists, each inner list represents a group of observations.
        significance_level (float): The alpha level for determining statistical significance (default: 0.05).

    Returns:
        dict: A dictionary containing:
            - 'statistic' (float): The computed H-statistic
            - 'p_value' (float): The p-value for the test
            - 'significant' (bool): Whether the result is statistically significant at the given level
            - 'degrees_of_freedom' (int): Degrees of freedom for the test
            - 'interpretation' (str): Interpretation of the result

    Raises:
        ValueError: If fewer than two groups are provided or if any group is empty.
    """
    if not 0 < significance_level < 1:
        raise ValueError('Significance level must be between 0 and 1')
    groups = []
    if isinstance(data, DataBatch):
        array_data = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data
        if array_data.ndim == 0:
            raise ValueError('DataBatch must contain at least 1-dimensional data')
        elif array_data.ndim == 1:
            raise ValueError('DataBatch must be 2-dimensional to represent multiple groups as columns')
        elif array_data.ndim == 2:
            groups = [array_data[:, i] for i in range(array_data.shape[1])]
        else:
            raise ValueError('DataBatch data must be at most 2-dimensional')
    elif isinstance(data, list):
        if len(data) < 2:
            raise ValueError('At least two groups must be provided')
        for (i, group) in enumerate(data):
            if isinstance(group, (list, np.ndarray)):
                group_data = np.asarray(group)
                if group_data.size == 0:
                    raise ValueError(f'Group {i} is empty')
                groups.append(group_data)
            else:
                raise ValueError(f'Group {i} must be a list or array-like')
    else:
        raise ValueError('Data must be a DataBatch or a list of groups')
    if len(groups) < 2:
        raise ValueError('At least two groups must be provided')
    for (i, group) in enumerate(groups):
        if len(group) == 0:
            raise ValueError(f'Group {i} is empty')
    try:
        (statistic, p_value) = kruskal(*groups)
    except ValueError as e:
        if 'All numbers are identical' in str(e):
            return {'statistic': 0.0, 'p_value': 1.0, 'significant': False, 'degrees_of_freedom': len(groups) - 1, 'interpretation': 'Fail to reject null hypothesis - all groups have identical values'}
        else:
            raise ValueError(f'Error performing Kruskal-Wallis test: {str(e)}')
    significant = bool(p_value < significance_level)
    if significant:
        interpretation = 'Reject null hypothesis - significant differences found between groups'
    else:
        interpretation = 'Fail to reject null hypothesis - no significant differences found between groups'
    return {'statistic': float(statistic), 'p_value': float(p_value), 'significant': significant, 'degrees_of_freedom': len(groups) - 1, 'interpretation': interpretation}