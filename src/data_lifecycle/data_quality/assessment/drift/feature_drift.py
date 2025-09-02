from typing import Dict, List, Any, Optional
from general.structures.data_batch import DataBatch
from general.base_classes.validator_base import BaseValidator
import numpy as np
from collections import deque


# ...(code omitted)...


class FeatureDriftMonitor(BaseValidator):

    def __init__(self, name: Optional[str]=None, alert_threshold: float=0.1, history_size: int=10):
        """
        Initialize the FeatureDriftMonitor.
        
        Args:
            name (Optional[str]): Name for the monitor instance.
            alert_threshold (float): Threshold for triggering drift alerts.
            history_size (int): Number of historical batches to retain.
        """
        super().__init__(name)
        self.alert_threshold = alert_threshold
        self.history_size = history_size
        self._baseline_stats: Optional[Dict[str, Any]] = None
        self._drift_history: deque = deque(maxlen=history_size)
        self._feature_names: Optional[List[str]] = None
        self._baseline_data: Optional[np.ndarray] = None

    def fit(self, data: DataBatch, **kwargs) -> 'FeatureDriftMonitor':
        """
        Initialize the monitor with a baseline dataset.
        
        Args:
            data (DataBatch): Baseline dataset for initial reference.
            **kwargs: Additional fitting parameters.
            
        Returns:
            FeatureDriftMonitor: Initialized monitor instance.
        """
        if self._baseline_stats is not None:
            raise ValueError('Monitor is already fitted. Create a new instance or reset state.')
        if isinstance(data.data, list):
            X = np.array(data.data)
        else:
            X = data.data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._feature_names = data.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self._baseline_stats = {'means': np.mean(X, axis=0), 'stds': np.std(X, axis=0), 'n_samples': X.shape[0]}
        self._baseline_data = X.copy()
        return self

    def _compute_psi(self, baseline_probs: np.ndarray, test_probs: np.ndarray, epsilon: float=1e-08) -> float:
        """
        Compute Population Stability Index between two probability distributions.
        
        Args:
            baseline_probs: Baseline probability distribution
            test_probs: Test probability distribution
            epsilon: Small value to avoid division by zero
            
        Returns:
            float: PSI value
        """
        baseline_probs = np.clip(baseline_probs, epsilon, 1)
        test_probs = np.clip(test_probs, epsilon, 1)
        psi = np.sum((test_probs - baseline_probs) * np.log(test_probs / baseline_probs))
        return psi

    def _compute_feature_drift(self, baseline_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """
        Compute drift metrics for each feature.
        
        Args:
            baseline_data: Baseline data array
            test_data: Test data array
            
        Returns:
            Dict mapping feature names to drift scores
        """
        if self._feature_names is None:
            raise ValueError('Feature names not initialized. Call fit() first.')
        drift_scores = {}
        for (i, feature_name) in enumerate(self._feature_names):
            baseline_feature = baseline_data[:, i]
            test_feature = test_data[:, i]
            combined_data = np.concatenate([baseline_feature, test_feature])
            n_bins = min(10, max(3, int(np.sqrt(len(combined_data)))))
            bins = np.linspace(combined_data.min(), combined_data.max(), n_bins + 1)
            (baseline_hist, _) = np.histogram(baseline_feature, bins=bins)
            (test_hist, _) = np.histogram(test_feature, bins=bins)
            baseline_probs = baseline_hist / (baseline_hist.sum() + 1e-10)
            test_probs = test_hist / (test_hist.sum() + 1e-10)
            psi = self._compute_psi(baseline_probs, test_probs)
            drift_scores[feature_name] = psi
        return drift_scores

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Check for feature-wise drift in the provided data compared to the baseline.
        
        Args:
            data (DataBatch): Dataset to check for drift.
            **kwargs: Additional validation parameters.
            
        Returns:
            bool: True if no feature shows significant drift (all below alert_threshold), False otherwise.
        """
        if self._baseline_data is None:
            raise ValueError('Monitor not fitted. Call fit() with baseline data first.')
        if isinstance(data.data, list):
            test_data = np.array(data.data)
        else:
            test_data = data.data
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        if test_data.shape[1] != self._baseline_data.shape[1]:
            raise ValueError(f'Feature dimension mismatch: expected {self._baseline_data.shape[1]}, got {test_data.shape[1]}')
        drift_scores = self._compute_feature_drift(self._baseline_data, test_data)
        for score in drift_scores.values():
            if score > self.alert_threshold:
                return False
        return True

    def update_and_monitor(self, data: DataBatch) -> Dict[str, Any]:
        """
        Update drift history with new data and return detailed monitoring results.
        
        Args:
            data (DataBatch): New data batch to monitor.
            
        Returns:
            Dict[str, Any]: Detailed monitoring results including drift scores and alerts.
        """
        if self._baseline_data is None:
            raise ValueError('Monitor not fitted. Call fit() with baseline data first.')
        if isinstance(data.data, list):
            test_data = np.array(data.data)
        else:
            test_data = data.data
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        if test_data.shape[1] != self._baseline_data.shape[1]:
            raise ValueError(f'Feature dimension mismatch: expected {self._baseline_data.shape[1]}, got {test_data.shape[1]}')
        drift_scores = self._compute_feature_drift(self._baseline_data, test_data)
        drifted_features = [feature for (feature, score) in drift_scores.items() if score > self.alert_threshold]
        drift_report = {'batch_id': data.batch_id, 'timestamp': data.metadata.get('timestamp') if data.metadata else None, 'drift_scores': drift_scores, 'drifted_features': drifted_features, 'alert_triggered': len(drifted_features) > 0, 'max_drift_score': max(drift_scores.values()) if drift_scores else 0.0}
        self._drift_history.append(drift_report)
        return drift_report

    def get_drift_history(self) -> List[Dict[str, Any]]:
        """
        Get the chronological history of all computed drift reports.
        
        Returns:
            List[Dict[str, Any]]: List of drift reports in chronological order.
        """
        return list(self._drift_history)

    def detect_drift_per_feature(self, data: DataBatch) -> Dict[str, Any]:
        """
        Compute drift metrics for each feature individually.
        
        Args:
            data (DataBatch): Dataset to analyze for feature-wise drift.
            
        Returns:
            Dict[str, Any]: Dictionary mapping feature names to drift metrics.
        """
        if self._baseline_data is None:
            raise ValueError('Monitor not fitted. Call fit() with baseline data first.')
        if isinstance(data.data, list):
            test_data = np.array(data.data)
        else:
            test_data = data.data
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        if test_data.shape[1] != self._baseline_data.shape[1]:
            raise ValueError(f'Feature dimension mismatch: expected {self._baseline_data.shape[1]}, got {test_data.shape[1]}')
        drift_scores = self._compute_feature_drift(self._baseline_data, test_data)
        drifted_features = {feature: score for (feature, score) in drift_scores.items() if score > self.alert_threshold}
        return {'drift_scores': drift_scores, 'drifted_features': drifted_features, 'any_drift_detected': len(drifted_features) > 0}