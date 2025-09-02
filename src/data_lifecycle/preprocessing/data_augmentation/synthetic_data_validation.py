from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from general.base_classes.validator_base import BaseValidator
from general.structures.feature_set import FeatureSet

class SyntheticDataValidator(BaseValidator):

    def __init__(self, reference_data: FeatureSet, distribution_test: str='ks', correlation_tolerance: float=0.1, uniqueness_threshold: float=0.9, name: Optional[str]=None):
        super().__init__(name)
        self.reference_data = reference_data
        self.distribution_test = distribution_test
        self.correlation_tolerance = correlation_tolerance
        self.uniqueness_threshold = uniqueness_threshold

    def validate(self, data: FeatureSet, **kwargs) -> bool:
        """
        Validate synthetic data against the reference dataset based on multiple quality criteria.

        Performs comprehensive validation checks including distribution similarity, correlation structure,
        value range adherence, and sample uniqueness. Returns True if all validations pass within
        specified thresholds, False otherwise.

        Args:
            data (FeatureSet): The synthetic data to validate.
            **kwargs: Additional validation parameters (if any).

        Returns:
            bool: True if synthetic data passes all validation checks, False otherwise.

        Raises:
            ValueError: If the synthetic data has incompatible dimensions or types compared to reference.
        """
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet instance.')
        ref_features = self.reference_data.features
        syn_features = data.features
        if ref_features.shape[1] != syn_features.shape[1]:
            raise ValueError('Synthetic data must have the same number of features as reference data.')
        if self.reference_data.feature_names is not None and data.feature_names is not None and (self.reference_data.feature_names != data.feature_names):
            raise ValueError('Synthetic data must have the same feature names as reference data.')
        metrics = self.compute_validation_metrics(data)
        distribution_valid = all((p_value > 0.05 for p_value in metrics['distribution_p_values'].values()))
        correlation_valid = metrics['correlation_diff'] <= self.correlation_tolerance
        uniqueness_valid = metrics['uniqueness_ratio'] >= self.uniqueness_threshold
        boundary_valid = metrics['boundary_violations'] == 0
        if ref_features.shape[0] < 2 or syn_features.shape[0] < 2:
            correlation_valid = True
        return distribution_valid and correlation_valid and uniqueness_valid and boundary_valid

    def compute_validation_metrics(self, data: FeatureSet) -> Dict[str, Any]:
        """
        Compute detailed validation metrics for synthetic data without performing pass/fail judgment.

        Calculates quantitative metrics that describe how closely the synthetic data matches the reference,
        including statistical distances, correlation differences, and diversity measures. These metrics
        can be used for diagnostic purposes or to tune data generation processes.

        Args:
            data (FeatureSet): The synthetic data to analyze.

        Returns:
            Dict[str, Any]: Dictionary containing validation metrics such as:
                - distribution_distances: Distance metrics per feature
                - correlation_diff: Difference in correlation structure
                - uniqueness_ratio: Ratio of unique samples
                - boundary_violations: Count of out-of-range values
        """
        ref_features = self.reference_data.features
        syn_features = data.features
        (n_samples, n_features) = syn_features.shape
        if data.feature_names is not None:
            feature_names = data.feature_names
        elif self.reference_data.feature_names is not None:
            feature_names = self.reference_data.feature_names
        else:
            feature_names = [f'f{i + 1}' for i in range(n_features)]
        distribution_distances = {}
        distribution_p_values = {}
        for i in range(n_features):
            ref_col = ref_features[:, i]
            syn_col = syn_features[:, i]
            feature_name = feature_names[i] if i < len(feature_names) else f'f{i + 1}'
            if self.distribution_test == 'ks':
                try:
                    (statistic, p_value) = stats.ks_2samp(ref_col, syn_col)
                    distribution_distances[feature_name] = float(statistic)
                    distribution_p_values[feature_name] = float(p_value)
                except:
                    distribution_distances[feature_name] = float('inf')
                    distribution_p_values[feature_name] = 0.0
            else:
                try:
                    (statistic, p_value) = stats.ks_2samp(ref_col, syn_col)
                    distribution_distances[feature_name] = float(statistic)
                    distribution_p_values[feature_name] = float(p_value)
                except:
                    distribution_distances[feature_name] = float('inf')
                    distribution_p_values[feature_name] = 0.0
        if n_features > 1:
            try:
                if ref_features.shape[0] < 2 or syn_features.shape[0] < 2:
                    correlation_diff = 1.0
                else:
                    ref_corr = np.corrcoef(ref_features, rowvar=False)
                    syn_corr = np.corrcoef(syn_features, rowvar=False)
                    ref_corr = np.nan_to_num(ref_corr, nan=0.0)
                    syn_corr = np.nan_to_num(syn_corr, nan=0.0)
                    corr_diff_matrix = np.abs(ref_corr - syn_corr)
                    triu_indices = np.triu_indices_from(corr_diff_matrix, k=1)
                    if len(triu_indices[0]) > 0:
                        corr_diff_values = corr_diff_matrix[triu_indices]
                        corr_diff_values = corr_diff_values[np.isfinite(corr_diff_values)]
                        if len(corr_diff_values) > 0:
                            correlation_diff = float(np.mean(corr_diff_values))
                        else:
                            correlation_diff = 1.0
                    else:
                        correlation_diff = 0.0
            except Exception:
                correlation_diff = 1.0
        else:
            correlation_diff = 0.0
        if n_samples > 0:
            try:
                syn_structured = np.core.records.fromarrays(syn_features.T)
                unique_count = len(np.unique(syn_structured))
                uniqueness_ratio = unique_count / n_samples
            except Exception:
                unique_count = len(np.unique(syn_features, axis=0))
                uniqueness_ratio = unique_count / n_samples
        else:
            uniqueness_ratio = 0.0
        boundary_violations = 0
        for i in range(n_features):
            ref_col = ref_features[:, i]
            syn_col = syn_features[:, i]
            (ref_min, ref_max) = (np.min(ref_col), np.max(ref_col))
            violations = np.sum((syn_col < ref_min) | (syn_col > ref_max))
            boundary_violations += int(violations)
        return {'distribution_distances': distribution_distances, 'distribution_p_values': distribution_p_values, 'correlation_diff': correlation_diff, 'uniqueness_ratio': uniqueness_ratio, 'boundary_violations': boundary_violations}