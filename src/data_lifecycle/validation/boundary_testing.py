from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
import numpy as np
from typing import Any, Optional, Dict, List, Union


class RobustnessTester(BaseValidator):

    def __init__(self, name: Optional[str]=None, tolerance: float=0.1, test_conditions: Optional[Dict[str, Any]]=None):
        """
        Initialize the RobustnessTester.
        
        Args:
            name (Optional[str]): Name for the validator instance.
            tolerance (float): Acceptable deviation threshold for robustness tests.
            test_conditions (Optional[Dict[str, Any]]): Custom test scenarios to apply.
        """
        super().__init__(name)
        self.tolerance = tolerance
        self.test_conditions = test_conditions or {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Perform robustness testing on the input data.
        
        Applies various stress tests to the data to determine if the system handles
        extreme conditions gracefully without failing or producing unreliable outputs.
        
        Args:
            data (DataBatch): Input data batch to test for robustness.
            **kwargs: Additional parameters for validation.
            
        Returns:
            bool: True if the data passes all robustness tests, False otherwise.
        """
        self.reset_validation_state()
        if isinstance(data.data, list):
            try:
                np_data = np.array(data.data, dtype=float)
            except (ValueError, TypeError):
                self.add_error('Cannot convert data to numeric array for robustness testing')
                return False
        else:
            np_data = np.asarray(data.data, dtype=float)
        if np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
        elif np_data.ndim > 2:
            self.add_error('Only 1D and 2D data arrays are supported for robustness testing')
            return False
        failure_rate = 0.0
        if self.test_conditions.get('test_extreme_values', False):
            extreme_failure_rate = self._test_extreme_values(np_data)
            failure_rate = max(failure_rate, extreme_failure_rate)
        if self.test_conditions.get('test_missing_data', False):
            missing_failure_rate = self._test_missing_data_resilience(np_data)
            failure_rate = max(failure_rate, missing_failure_rate)
        if self.test_conditions.get('test_unexpected_inputs', False):
            unexpected_failure_rate = self._test_unexpected_inputs(data)
            failure_rate = max(failure_rate, unexpected_failure_rate)
        if self.test_conditions.get('test_data_types', False):
            dtype_failure_rate = self._test_data_type_variations(np_data)
            failure_rate = max(failure_rate, dtype_failure_rate)
        return failure_rate <= self.tolerance

    def configure_tests(self, conditions: Dict[str, Any]) -> None:
        """
        Configure specific robustness test scenarios.
        
        Args:
            conditions (Dict[str, Any]): Dictionary defining test conditions.
        """
        if not isinstance(conditions, dict):
            raise TypeError('Conditions must be a dictionary')
        self.test_conditions.update(conditions)

    def _test_extreme_values(self, data: np.ndarray) -> float:
        """
        Test how the system handles extreme values.
        
        Args:
            data (np.ndarray): Input data as numpy array.
            
        Returns:
            float: Failure rate for this test (0.0 to 1.0).
        """
        failure_count = 0
        total_tests = 0
        threshold = self.test_conditions.get('extreme_value_threshold', 3.0)
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            if np.all(np.isnan(col_data)):
                continue
            col_mean = np.nanmean(col_data)
            col_std = np.nanstd(col_data)
            if col_std == 0:
                continue
            z_scores = np.abs((col_data - col_mean) / col_std)
            extreme_mask = z_scores > threshold
            extreme_count = np.sum(extreme_mask & ~np.isnan(col_data))
            total_values = np.sum(~np.isnan(col_data))
            if total_values > 0:
                extreme_ratio = extreme_count / total_values
                if extreme_ratio > self.tolerance:
                    failure_count += 1
                total_tests += 1
        return failure_count / total_tests if total_tests > 0 else 0.0

    def _test_missing_data_resilience(self, data: np.ndarray) -> float:
        """
        Test how the system handles missing data.
        
        Args:
            data (np.ndarray): Input data as numpy array.
            
        Returns:
            float: Failure rate for this test (0.0 to 1.0).
        """
        missing_ratio = self.test_conditions.get('missing_data_ratio', 0.2)
        failure_count = 0
        total_tests = 0
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            if np.all(np.isnan(col_data)):
                continue
            actual_missing_ratio = np.sum(np.isnan(col_data)) / len(col_data)
            if actual_missing_ratio > missing_ratio:
                failure_count += 1
            total_tests += 1
        return failure_count / total_tests if total_tests > 0 else 0.0

    def _test_unexpected_inputs(self, data: DataBatch) -> float:
        """
        Test how the system handles unexpected input formats.
        
        Args:
            data (DataBatch): Input data batch.
            
        Returns:
            float: Failure rate for this test (0.0 to 1.0).
        """
        failure_count = 0
        total_tests = 3
        try:
            shape = data.get_shape()
            if len(shape) > 2:
                failure_count += 1
        except Exception:
            failure_count += 1
        if data.labels is not None:
            try:
                data_length = len(data.data) if hasattr(data.data, '__len__') else 1
                labels_length = len(data.labels) if hasattr(data.labels, '__len__') else 1
                if data_length != labels_length:
                    failure_count += 1
            except Exception:
                failure_count += 1
        elif self.test_conditions.get('expect_labels', False):
            failure_count += 1
        if self.test_conditions.get('require_metadata', False):
            if not data.metadata or len(data.metadata) == 0:
                failure_count += 1
        return failure_count / total_tests

    def _test_data_type_variations(self, data: np.ndarray) -> float:
        """
        Test how the system handles different data types.
        
        Args:
            data (np.ndarray): Input data as numpy array.
            
        Returns:
            float: Failure rate for this test (0.0 to 1.0).
        """
        failure_count = 0
        total_tests = 0
        dtype_variations = self.test_conditions.get('data_type_variations', [np.int32, np.int64, np.float32, np.float64])
        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]
            if np.all(np.isnan(col_data)):
                continue
            for dtype in dtype_variations:
                total_tests += 1
                try:
                    converted = col_data.astype(dtype)
                except (ValueError, OverflowError, TypeError):
                    failure_count += 1
        return failure_count / total_tests if total_tests > 0 else 0.0