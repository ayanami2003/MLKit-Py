from general.base_classes.model_base import BaseModel
from typing import Optional, Dict, Any
from general.structures.feature_set import FeatureSet
import numpy as np

class CoxProportionalHazardsModel(BaseModel):

    def __init__(self, name: Optional[str]=None, alpha: float=0.05):
        """
        Initialize the Cox Proportional Hazards Model.

        Args:
            name (Optional[str]): Name of the model instance.
            alpha (float): Regularization parameter for fitting.
        """
        super().__init__(name)
        self.alpha = alpha
        self.coefficients_: Optional[np.ndarray] = None
        self.baseline_hazard_: Optional[Dict[float, float]] = None
        self.feature_names: Optional[list] = None
        self.is_fitted = False

    def fit(self, X: FeatureSet, event_times: np.ndarray, event_indicators: np.ndarray, **kwargs) -> 'CoxProportionalHazardsModel':
        """
        Fit the Cox proportional hazards model to the training data.

        Args:
            X (FeatureSet): Training features with shape (n_samples, n_features).
            event_times (np.ndarray): Array of event/censoring times with shape (n_samples,).
            event_indicators (np.ndarray): Boolean array indicating whether event occurred (True) or was censored (False).

        Returns:
            CoxProportionalHazardsModel: Fitted model instance.

        Raises:
            ValueError: If input dimensions don't match or data is invalid.
        """
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_features = X.features
        (n_samples, n_features) = X_features.shape
        if event_times.shape[0] != n_samples:
            raise ValueError(f'event_times length ({event_times.shape[0]}) must match number of samples ({n_samples})')
        if event_indicators.shape[0] != n_samples:
            raise ValueError(f'event_indicators length ({event_indicators.shape[0]}) must match number of samples ({n_samples})')
        if not np.all(np.isfinite(event_times)) or np.any(event_times < 0):
            raise ValueError('event_times must be non-negative and finite')
        if event_indicators.dtype != bool:
            raise TypeError('event_indicators must be boolean array')
        if not np.any(event_indicators):
            self.coefficients_ = np.zeros(n_features)
            self.baseline_hazard_ = {}
            self.feature_names = X.feature_names if X.feature_names is not None else [f'feature_{i}' for i in range(n_features)]
            self.is_fitted = True
            return self
        self.feature_names = X.feature_names if X.feature_names is not None else [f'feature_{i}' for i in range(n_features)]
        sort_indices = np.argsort(event_times)
        X_sorted = X_features[sort_indices]
        event_times_sorted = event_times[sort_indices]
        event_indicators_sorted = event_indicators[sort_indices]
        self.coefficients_ = np.zeros(n_features)
        max_iter = 100
        tol = 1e-06
        for iteration in range(max_iter):
            risk_scores = X_sorted @ self.coefficients_
            risk_scores = np.clip(risk_scores, -500, 500)
            exp_risk_scores = np.exp(risk_scores)
            gradient = np.zeros(n_features)
            hessian = np.zeros((n_features, n_features))
            event_mask = event_indicators_sorted
            if not np.any(event_mask):
                break
            (unique_times, time_counts) = np.unique(event_times_sorted[event_mask], return_counts=True)
            for (t, count) in zip(unique_times, time_counts):
                events_at_time_mask = (event_times_sorted == t) & event_indicators_sorted
                events_at_time = np.where(events_at_time_mask)[0]
                risk_set_mask = event_times_sorted >= t
                if not np.any(risk_set_mask):
                    continue
                risk_set_scores = exp_risk_scores[risk_set_mask]
                sum_risk_scores = np.sum(risk_set_scores)
                if sum_risk_scores <= 0:
                    continue
                for event_idx in events_at_time:
                    risk_set_indices = np.where(risk_set_mask)[0]
                    weighted_X_sum = risk_set_scores @ X_sorted[risk_set_indices] / sum_risk_scores
                    gradient += X_sorted[event_idx] - weighted_X_sum
                risk_set_indices = np.where(risk_set_mask)[0]
                weighted_X = risk_set_scores[:, np.newaxis] * X_sorted[risk_set_indices]
                mean_X = np.sum(weighted_X, axis=0) / sum_risk_scores
                weighted_X2 = np.zeros((n_features, n_features))
                for i in range(len(risk_set_scores)):
                    x_vec = X_sorted[risk_set_indices][i]
                    weighted_X2 += risk_set_scores[i] * np.outer(x_vec, x_vec)
                mean_X2 = weighted_X2 / sum_risk_scores
                hessian -= count * (mean_X2 - np.outer(mean_X, mean_X))
            hessian -= self.alpha * np.eye(n_features)
            try:
                delta = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(hessian) @ gradient
            self.coefficients_ -= delta
            if np.linalg.norm(delta) < tol:
                break
        self.baseline_hazard_ = self._compute_baseline_hazard(X_sorted, event_times_sorted, event_indicators_sorted)
        self.is_fitted = True
        return self

    def _compute_baseline_hazard(self, X: np.ndarray, event_times: np.ndarray, event_indicators: np.ndarray) -> Dict[float, float]:
        """
        Compute baseline hazard using Breslow estimator.

        Args:
            X: Feature matrix sorted by ascending event times
            event_times: Event/censoring times sorted by ascending order
            event_indicators: Boolean array indicating events

        Returns:
            Dict mapping time points to baseline hazard values
        """
        if self.coefficients_ is None:
            return {}
        (n_samples, n_features) = X.shape
        risk_scores = X @ self.coefficients_
        risk_scores = np.clip(risk_scores, -500, 500)
        exp_risk_scores = np.exp(risk_scores)
        baseline_hazard = {}
        unique_events = np.unique(event_times[event_indicators])
        for t in unique_events:
            event_mask = (event_times == t) & event_indicators
            num_events = np.sum(event_mask)
            risk_set_mask = event_times >= t
            risk_set_scores = exp_risk_scores[risk_set_mask]
            risk_set_sum = np.sum(risk_set_scores)
            if risk_set_sum <= 0:
                baseline_hazard[t] = 0.0
            else:
                baseline_hazard[t] = num_events / risk_set_sum
        return baseline_hazard

    def predict_survival_function(self, X: FeatureSet, times: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict survival function for given samples at specified time points.

        Args:
            X (FeatureSet): Samples to predict with shape (n_samples, n_features).
            times (np.ndarray): Time points at which to compute survival probabilities.

        Returns:
            np.ndarray: Survival probabilities with shape (n_samples, n_times).

        Raises:
            ValueError: If model is not fitted or input dimensions are incorrect.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before making predictions')
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_features = X.features
        (n_samples, n_features) = X_features.shape
        if self.coefficients_ is None or len(self.coefficients_) != n_features:
            raise ValueError('Model coefficients dimension mismatch')
        if self.baseline_hazard_ is None:
            raise ValueError('Baseline hazard not computed')
        risk_scores = X_features @ self.coefficients_
        risk_scores = np.clip(risk_scores, -500, 500)
        risk_ratios = np.exp(risk_scores)
        sorted_times = np.sort(times)
        sorted_unique_event_times = np.array(sorted(self.baseline_hazard_.keys()))
        n_times = len(sorted_times)
        cumulative_hazards = np.zeros((n_samples, n_times))
        baseline_cumulative_hazard = np.zeros(len(sorted_times))
        current_baseline_hazard = 0.0
        j = 0
        for t_event in sorted_unique_event_times:
            current_baseline_hazard += self.baseline_hazard_.get(t_event, 0.0)
            while j < n_times and sorted_times[j] <= t_event:
                baseline_cumulative_hazard[j] = current_baseline_hazard
                j += 1
        while j < n_times:
            baseline_cumulative_hazard[j] = current_baseline_hazard
            j += 1
        for i in range(n_samples):
            cumulative_hazards[i, :] = risk_ratios[i] * baseline_cumulative_hazard
        survival_probs = np.exp(-cumulative_hazards)
        survival_probs = np.clip(survival_probs, 0.0, 1.0)
        return survival_probs

    def predict_hazard_ratios(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict hazard ratios for given samples relative to baseline.

        Args:
            X (FeatureSet): Samples to predict with shape (n_samples, n_features).

        Returns:
            np.ndarray: Hazard ratios with shape (n_samples,).

        Raises:
            ValueError: If model is not fitted.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before making predictions')
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_features = X.features
        (n_samples, n_features) = X_features.shape
        if self.coefficients_ is None or len(self.coefficients_) != n_features:
            raise ValueError('Model coefficients dimension mismatch')
        risk_scores = X_features @ self.coefficients_
        risk_scores = np.clip(risk_scores, -500, 500)
        hazard_ratios = np.exp(risk_scores)
        hazard_ratios = np.maximum(hazard_ratios, np.finfo(float).eps)
        return hazard_ratios

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict hazard ratios for compatibility with BaseModel interface.

        Args:
            X (FeatureSet): Samples to predict with shape (n_samples, n_features).

        Returns:
            np.ndarray: Hazard ratios with shape (n_samples,).
        """
        return self.predict_hazard_ratios(X)

    def score(self, X: FeatureSet, event_times: np.ndarray, event_indicators: np.ndarray, **kwargs) -> float:
        """
        Compute the concordance index (c-index) for the model.

        Args:
            X (FeatureSet): Test features with shape (n_samples, n_features).
            event_times (np.ndarray): Event/censoring times with shape (n_samples,).
            event_indicators (np.ndarray): Event indicators with shape (n_samples,).

        Returns:
            float: Concordance index value between 0 and 1.

        Raises:
            ValueError: If model is not fitted or input dimensions are incorrect.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before scoring')
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_features = X.features
        (n_samples, n_features) = X_features.shape
        if event_times.shape[0] != n_samples:
            raise ValueError(f'event_times length ({event_times.shape[0]}) must match number of samples ({n_samples})')
        if event_indicators.shape[0] != n_samples:
            raise ValueError(f'event_indicators length ({event_indicators.shape[0]}) must match number of samples ({n_samples})')
        if self.coefficients_ is None or len(self.coefficients_) != n_features:
            raise ValueError('Model coefficients dimension mismatch')
        risk_scores = X_features @ self.coefficients_
        concordant_pairs = 0
        usable_pairs = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if event_indicators[i] or event_indicators[j]:
                    if event_indicators[i] and event_indicators[j]:
                        usable_pairs += 1
                        if (risk_scores[i] > risk_scores[j]) == (event_times[i] < event_times[j]):
                            concordant_pairs += 1
                        elif risk_scores[i] == risk_scores[j]:
                            concordant_pairs += 0.5
                    elif event_indicators[i] and (not event_indicators[j]):
                        if event_times[i] < event_times[j]:
                            usable_pairs += 1
                            if risk_scores[i] > risk_scores[j]:
                                concordant_pairs += 1
                            elif risk_scores[i] == risk_scores[j]:
                                concordant_pairs += 0.5
                    elif not event_indicators[i] and event_indicators[j]:
                        if event_times[j] < event_times[i]:
                            usable_pairs += 1
                            if risk_scores[j] > risk_scores[i]:
                                concordant_pairs += 1
                            elif risk_scores[j] == risk_scores[i]:
                                concordant_pairs += 0.5
        if usable_pairs == 0:
            return 0.5
        return concordant_pairs / usable_pairs