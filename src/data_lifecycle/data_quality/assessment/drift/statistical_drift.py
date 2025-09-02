from typing import Optional, Dict, Any
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
import numpy as np
from typing import Optional
from scipy import stats

class MaximumMeanDiscrepancyDriftDetector(BaseValidator):

    def __init__(self, name: Optional[str]=None, threshold: float=0.05, kernel_type: str='rbf', gamma: float=1.0):
        """
        Initialize the MMD drift detector.
        
        Args:
            name: Optional name for the validator
            threshold: Significance threshold for drift detection (default: 0.05)
            kernel_type: Kernel function to use ('rbf' or 'linear')
            gamma: Kernel coefficient for RBF kernel
        """
        super().__init__(name)
        self.threshold = threshold
        self.kernel_type = kernel_type
        self.gamma = gamma
        self._reference_data = None

    def fit(self, data: DataBatch, **kwargs) -> 'MaximumMeanDiscrepancyDriftDetector':
        """
        Fit the detector to reference data to establish baseline distribution.
        
        Args:
            data: Reference data batch to use as baseline distribution
            **kwargs: Additional fitting parameters
            
        Returns:
            Self instance for method chaining
        """
        if not isinstance(data, DataBatch):
            raise TypeError('Data must be a DataBatch instance')
        if isinstance(data.data, list):
            ref_data = np.array(data.data)
        else:
            ref_data = np.asarray(data.data)
        if ref_data.ndim == 1:
            ref_data = ref_data.reshape(-1, 1)
        elif ref_data.ndim == 0:
            ref_data = ref_data.reshape(1, 1)
        self._reference_data = ref_data
        return self

    @property
    def reference_data(self):
        """Get the reference data."""
        return self._reference_data

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Check if new data shows significant drift from reference distribution.
        
        Args:
            data: Test data batch to check for drift
            **kwargs: Additional validation parameters
            
        Returns:
            True if significant drift detected, False otherwise
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call 'fit' first.")
        if not isinstance(data, DataBatch):
            raise TypeError('Data must be a DataBatch instance')
        mmd_value = self.compute_mmd(DataBatch(self._reference_data), data)
        return bool(mmd_value > self.threshold)

    def _compute_kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between two datasets.
        
        Args:
            x: First dataset (n_samples_x, n_features)
            y: Second dataset (n_samples_y, n_features)
            
        Returns:
            Kernel matrix (n_samples_x, n_samples_y)
        """
        if self.kernel_type == 'rbf':
            x_norm = np.sum(x ** 2, axis=1, keepdims=True)
            y_norm = np.sum(y ** 2, axis=1, keepdims=True)
            distances_sq = x_norm + y_norm.T - 2 * np.dot(x, y.T)
            return np.exp(-self.gamma * distances_sq)
        elif self.kernel_type == 'linear':
            return np.dot(x, y.T)
        else:
            raise ValueError(f'Unsupported kernel type: {self.kernel_type}')

    def compute_mmd(self, data1: DataBatch, data2: DataBatch) -> float:
        """
        Compute Maximum Mean Discrepancy between two data batches.
        
        Args:
            data1: First data batch
            data2: Second data batch
            
        Returns:
            MMD statistic value
        """
        if isinstance(data1.data, list):
            x = np.array(data1.data)
        else:
            x = np.asarray(data1.data)
        if isinstance(data2.data, list):
            y = np.array(data2.data)
        else:
            y = np.asarray(data2.data)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 0:
            x = x.reshape(1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 0:
            y = y.reshape(1, 1)
        m = x.shape[0]
        n = y.shape[0]
        k_xx = self._compute_kernel(x, x)
        k_yy = self._compute_kernel(y, y)
        k_xy = self._compute_kernel(x, y)
        term1 = np.sum(k_xx) - np.trace(k_xx)
        term1 = term1 / (m * (m - 1)) if m > 1 else 0
        term2 = np.sum(k_yy) - np.trace(k_yy)
        term2 = term2 / (n * (n - 1)) if n > 1 else 0
        term3 = np.sum(k_xy)
        term3 = 2 * term3 / (m * n) if m > 0 and n > 0 else 0
        mmd_squared = term1 + term2 - term3
        return max(0, mmd_squared)