from general.structures.data_batch import DataBatch
import numpy as np
import pandas as pd
from typing import Union, Optional, List
from general.base_classes.validator_base import BaseValidator
from scipy import stats


# ...(code omitted)...


class QuantileDriftValidator(BaseValidator):

    def __init__(self, name: Optional[str]=None, quantiles: Optional[List[float]]=None, threshold: float=0.1):
        """
        Initialize the Quantile Drift Validator.
        
        Parameters
        ----------
        name : Optional[str]
            Name for the validator instance
        quantiles : Optional[List[float]]
            Quantiles to compare (default: [0.25, 0.5, 0.75])
        threshold : float
            Maximum allowed difference in quantiles for drift detection
        """
        super().__init__(name)
        self.quantiles = quantiles or [0.25, 0.5, 0.75]
        self.threshold = threshold
        self.reference_quantiles: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, data: DataBatch, **kwargs) -> 'QuantileDriftValidator':
        """
        Fit the validator by computing reference quantiles.
        
        Calculates quantiles for each feature in the reference data
        and stores them for future comparisons.
        
        Parameters
        ----------
        data : DataBatch
            Reference data to compute quantiles from
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        QuantileDriftValidator
            Self instance for method chaining
        """
        if isinstance(data.data, pd.DataFrame):
            data_array = data.data.values
            self._feature_names = data.feature_names or list(data.data.columns)
        else:
            data_array = np.asarray(data.data)
            self._feature_names = data.feature_names
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        self.reference_quantiles = np.zeros((len(self.quantiles), data_array.shape[1]))
        for (i, q) in enumerate(self.quantiles):
            self.reference_quantiles[i] = np.quantile(data_array, q, axis=0)
        return self

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Validate if test data shows significant drift from reference data.
        
        Compares quantiles of test data against reference quantiles and
        determines if differences exceed the threshold.
        
        Parameters
        ----------
        data : DataBatch
            Test data to check for drift
        **kwargs : dict
            Additional validation parameters
            
        Returns
        -------
        bool
            True if no significant drift detected, False otherwise
            
        Raises
        ------
        ValueError
            If reference quantiles have not been fitted
        """
        if self.reference_quantiles is None:
            raise ValueError('QuantileDriftValidator must be fitted before calling validate. Missing reference quantiles.')
        differences = self.calculate_quantile_differences(data)
        for feature_diffs in differences.values():
            for diff in feature_diffs.values():
                if diff > self.threshold:
                    return bool(False)
        return bool(True)

    def calculate_quantile_differences(self, data: DataBatch) -> dict:
        """
        Calculate differences between reference and test data quantiles.
        
        Computes absolute differences between reference quantiles and
        quantiles computed from test data for each feature.
        
        Parameters
        ----------
        data : DataBatch
            Test data to compare against reference
        
        Returns
        -------
        dict
            Dictionary mapping feature names to quantile differences
        """
        if self.reference_quantiles is None:
            raise ValueError('Reference quantiles not available. Call fit() first.')
        if isinstance(data.data, pd.DataFrame):
            data_array = data.data.values
        else:
            data_array = np.asarray(data.data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        if data_array.size == 0:
            result = {}
            feature_names = self._feature_names or [f'feature_{i}' for i in range(self.reference_quantiles.shape[1])]
            for (feat_idx, feature_name) in enumerate(feature_names):
                result[feature_name] = {q: 0.0 for q in self.quantiles}
            return result
        test_quantiles = np.zeros((len(self.quantiles), data_array.shape[1]))
        for (i, q) in enumerate(self.quantiles):
            test_quantiles[i] = np.quantile(data_array, q, axis=0)
        differences = {}
        feature_names = self._feature_names or [f'feature_{i}' for i in range(self.reference_quantiles.shape[1])]
        for (feat_idx, feature_name) in enumerate(feature_names):
            differences[feature_name] = {}
            for (q_idx, q) in enumerate(self.quantiles):
                diff = abs(test_quantiles[q_idx, feat_idx] - self.reference_quantiles[q_idx, feat_idx])
                differences[feature_name][q] = float(diff)
        return differences