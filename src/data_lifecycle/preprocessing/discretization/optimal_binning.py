from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class OptimalBinningTransformer(BaseTransformer):

    def __init__(self, feature_indices: Optional[List[int]]=None, objective: str='information_value', max_bins: int=10, min_bin_size: float=0.05, regularization: float=0.0, monotonic: bool=False, name: Optional[str]=None):
        """
        Initialize the OptimalBinningTransformer.
        
        Parameters
        ----------
        feature_indices : Optional[List[int]]
            Indices of features to discretize. If None, all features are processed.
        objective : str
            Objective function to optimize ('information_value', 'gini', 'chi2', etc.)
        max_bins : int
            Maximum number of bins to create
        min_bin_size : float
            Minimum proportion of samples required in each bin
        regularization : float
            Regularization parameter to prevent overfitting
        monotonic : bool
            Whether to enforce monotonicity constraint on binning
        name : Optional[str]
            Name of the transformer
        """
        super().__init__(name=name)
        self.feature_indices = feature_indices
        self.objective = objective
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.regularization = regularization
        self.monotonic = monotonic
        self._bin_boundaries = {}
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'OptimalBinningTransformer':
        """
        Fit the optimal binning transformer to the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to fit the transformer on
        y : Optional[np.ndarray]
            Target values for supervised binning
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        OptimalBinningTransformer
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            X = data.data
            self._feature_names = None
        else:
            X = np.asarray(data)
            self._feature_names = None
        X = np.asarray(X)
        if self.objective in ['information_value', 'gini', 'chi2'] and y is None:
            raise ValueError(f"Target variable 'y' is required for objective '{self.objective}'")
        if y is not None:
            y = np.asarray(y)
            if X.shape[0] != y.shape[0]:
                raise ValueError(f'Number of samples in X ({X.shape[0]}) does not match y ({y.shape[0]})')
        n_features = X.shape[1]
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
            if any((idx < 0 or idx >= n_features for idx in feature_indices)):
                raise ValueError('Feature indices out of bounds')
        self._bin_boundaries = {}
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            if np.all(feature_values == feature_values[0]):
                self._bin_boundaries[feature_idx] = np.array([feature_values[0], feature_values[0]])
                continue
            if self.objective == 'information_value':
                bin_edges = self._compute_information_value_bins(feature_values, y)
            elif self.objective == 'gini':
                bin_edges = self._compute_gini_bins(feature_values, y)
            elif self.objective == 'chi2':
                bin_edges = self._compute_chi2_bins(feature_values, y)
            else:
                raise ValueError(f'Unsupported objective: {self.objective}')
            self._bin_boundaries[feature_idx] = bin_edges
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Apply optimal binning transformation to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Transformed data with binned features
        """
        if not hasattr(self, '_bin_boundaries') or len(self._bin_boundaries) == 0:
            raise RuntimeError('Transformer has not been fitted yet.')
        is_feature_set = False
        is_data_batch = False
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            is_feature_set = True
        elif isinstance(data, DataBatch):
            X = data.data.copy()
            is_data_batch = True
        else:
            X = np.asarray(data).copy()
        for (feature_idx, bin_edges) in self._bin_boundaries.items():
            if feature_idx < X.shape[1]:
                bin_indices = np.digitize(X[:, feature_idx], bin_edges, right=False) - 1
                bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
                X[:, feature_idx] = bin_indices.astype(np.integer)
        if is_feature_set:
            return FeatureSet(features=X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        elif is_data_batch:
            return DataBatch(data=X, sample_ids=data.sample_ids, metadata=data.metadata)
        else:
            return X.astype(np.integer)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply the inverse transformation (not supported for binning).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Original data format (raises NotImplementedError)
        """
        raise NotImplementedError('Inverse transformation is not supported for binning operations')

    def get_bin_boundaries(self, feature_index: int) -> Optional[np.ndarray]:
        """
        Get the bin boundaries for a specific feature.
        
        Parameters
        ----------
        feature_index : int
            Index of the feature
            
        Returns
        -------
        Optional[np.ndarray]
            Array of bin boundaries or None if not fitted
        """
        return self._bin_boundaries.get(feature_index)

    def get_feature_bins(self, feature_name: str) -> Optional[np.ndarray]:
        """
        Get the bin boundaries for a feature by name.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature
            
        Returns
        -------
        Optional[np.ndarray]
            Array of bin boundaries or None if not fitted
        """
        if self._feature_names is None:
            return None
        try:
            idx = self._feature_names.index(feature_name)
            return self._bin_boundaries.get(idx)
        except ValueError:
            return None

    def _compute_information_value_bins(self, feature_values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute bin edges using information value optimization."""
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_y = y[sorted_indices]
        n_samples = len(sorted_values)
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        min_samples_per_bin = max(1, int(self.min_bin_size * n_samples))
        bin_edges = [sorted_values[0]]
        if n_samples <= min_samples_per_bin or self.max_bins <= 1:
            bin_edges.append(sorted_values[-1])
            return np.array(bin_edges)
        candidate_splits = []
        for i in range(1, n_samples):
            if sorted_y[i] != sorted_y[i - 1]:
                candidate_splits.append(i)
        if not candidate_splits:
            n_bins = min(self.max_bins, n_samples // min_samples_per_bin)
            if n_bins <= 1:
                bin_edges.append(sorted_values[-1])
                return np.array(bin_edges)
            for i in range(1, n_bins):
                idx = int(i * n_samples / n_bins)
                if sorted_values[idx] > bin_edges[-1]:
                    bin_edges.append(sorted_values[idx])
            bin_edges.append(sorted_values[-1])
            return np.array(bin_edges)
        best_splits = []
        last_split_idx = 0
        candidate_iv_scores = []
        for split_idx in candidate_splits:
            if split_idx - last_split_idx >= min_samples_per_bin and n_samples - split_idx >= min_samples_per_bin:
                iv_score = self._calculate_information_value(sorted_values, sorted_y, last_split_idx, split_idx)
                candidate_iv_scores.append((iv_score, split_idx))
        candidate_iv_scores.sort(reverse=True)
        splits_added = 0
        for (_, split_idx) in candidate_iv_scores:
            if splits_added >= self.max_bins - 1:
                break
            if split_idx - last_split_idx >= min_samples_per_bin and n_samples - split_idx >= min_samples_per_bin:
                bin_edges.append(sorted_values[split_idx])
                last_split_idx = split_idx
                splits_added += 1
        bin_edges.append(sorted_values[-1])
        if self.regularization > 0 and len(bin_edges) > 2:
            while len(bin_edges) > 2 and self.regularization * (len(bin_edges) - 1) > 0.1:
                bin_edges.pop(1)
        if self.monotonic and len(bin_edges) > 2:
            bin_edges = self._enforce_monotonicity(bin_edges, sorted_values, sorted_y)
        return np.array(bin_edges)

    def _compute_gini_bins(self, feature_values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute bin edges using Gini impurity optimization."""
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_y = y[sorted_indices]
        n_samples = len(sorted_values)
        unique_classes = np.unique(y)
        min_samples_per_bin = max(1, int(self.min_bin_size * n_samples))
        bin_edges = [sorted_values[0]]
        if n_samples <= min_samples_per_bin or self.max_bins <= 1:
            bin_edges.append(sorted_values[-1])
            return np.array(bin_edges)
        candidate_splits = []
        for i in range(1, n_samples):
            if sorted_y[i] != sorted_y[i - 1]:
                candidate_splits.append(i)
        if not candidate_splits:
            n_bins = min(self.max_bins, n_samples // min_samples_per_bin)
            if n_bins <= 1:
                bin_edges.append(sorted_values[-1])
                return np.array(bin_edges)
            for i in range(1, n_bins):
                idx = int(i * n_samples / n_bins)
                if sorted_values[idx] > bin_edges[-1]:
                    bin_edges.append(sorted_values[idx])
            bin_edges.append(sorted_values[-1])
            return np.array(bin_edges)
        best_splits = []
        last_split_idx = 0
        candidate_gini_scores = []
        for split_idx in candidate_splits:
            if split_idx - last_split_idx >= min_samples_per_bin and n_samples - split_idx >= min_samples_per_bin:
                gini_score = self._calculate_gini_gain(sorted_values, sorted_y, last_split_idx, split_idx)
                candidate_gini_scores.append((gini_score, split_idx))
        candidate_gini_scores.sort(reverse=True)
        splits_added = 0
        for (_, split_idx) in candidate_gini_scores:
            if splits_added >= self.max_bins - 1:
                break
            if split_idx - last_split_idx >= min_samples_per_bin and n_samples - split_idx >= min_samples_per_bin:
                bin_edges.append(sorted_values[split_idx])
                last_split_idx = split_idx
                splits_added += 1
        bin_edges.append(sorted_values[-1])
        if self.regularization > 0 and len(bin_edges) > 2:
            while len(bin_edges) > 2 and self.regularization * (len(bin_edges) - 1) > 0.05:
                bin_edges.pop(1)
        if self.monotonic and len(bin_edges) > 2:
            bin_edges = self._enforce_monotonicity(bin_edges, sorted_values, sorted_y)
        return np.array(bin_edges)

    def _compute_chi2_bins(self, feature_values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute bin edges using Chi-square optimization."""
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_y = y[sorted_indices]
        n_samples = len(sorted_values)
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        min_samples_per_bin = max(1, int(self.min_bin_size * n_samples))
        bin_edges = [sorted_values[0]]
        if n_samples <= min_samples_per_bin or self.max_bins <= 1:
            bin_edges.append(sorted_values[-1])
            return np.array(bin_edges)
        candidate_splits = []
        prev_class = sorted_y[0]
        for i in range(1, n_samples):
            if sorted_y[i] != prev_class:
                candidate_splits.append(i)
                prev_class = sorted_y[i]
        if not candidate_splits:
            n_bins = min(self.max_bins, n_samples // min_samples_per_bin)
            if n_bins <= 1:
                bin_edges.append(sorted_values[-1])
                return np.array(bin_edges)
            for i in range(1, n_bins):
                idx = int(i * n_samples / n_bins)
                if sorted_values[idx] > bin_edges[-1]:
                    bin_edges.append(sorted_values[idx])
            bin_edges.append(sorted_values[-1])
            return np.array(bin_edges)
        last_split_idx = 0
        candidate_chi2_scores = []
        for split_idx in candidate_splits:
            if split_idx - last_split_idx >= min_samples_per_bin and n_samples - split_idx >= min_samples_per_bin:
                chi2_score = self._calculate_chi2_gain(sorted_values, sorted_y, unique_classes, last_split_idx, split_idx, n_samples)
                candidate_chi2_scores.append((chi2_score, split_idx))
        candidate_chi2_scores.sort(reverse=True)
        splits_added = 0
        for (_, split_idx) in candidate_chi2_scores:
            if splits_added >= self.max_bins - 1:
                break
            if split_idx - last_split_idx >= min_samples_per_bin and n_samples - split_idx >= min_samples_per_bin:
                bin_edges.append(sorted_values[split_idx])
                last_split_idx = split_idx
                splits_added += 1
        bin_edges.append(sorted_values[-1])
        if self.regularization > 0 and len(bin_edges) > 2:
            while len(bin_edges) > 2 and self.regularization * (len(bin_edges) - 1) > 0.1:
                bin_edges.pop(1)
        if self.monotonic and len(bin_edges) > 2:
            bin_edges = self._enforce_monotonicity(bin_edges, sorted_values, sorted_y)
        return np.array(bin_edges)

    def _calculate_information_value(self, values: np.ndarray, y: np.ndarray, left_start: int, split_idx: int) -> float:
        """Calculate information value for a potential split."""
        left_y = y[left_start:split_idx]
        right_y = y[split_idx:len(y)]
        if len(left_y) == 0 or len(right_y) == 0:
            return 0.0
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            total_iv = 0.0
            for cls in unique_classes:
                left_events = np.sum(left_y == cls)
                left_non_events = len(left_y) - left_events
                right_events = np.sum(right_y == cls)
                right_non_events = len(right_y) - right_events
                if left_events + right_events == 0 or left_non_events + right_non_events == 0:
                    continue
                left_event_rate = left_events / (left_events + right_events)
                left_non_event_rate = left_non_events / (left_non_events + right_non_events)
                if left_event_rate == 0 or left_non_event_rate == 0:
                    continue
                iv = (left_event_rate - left_non_event_rate) * np.log(left_event_rate / left_non_event_rate)
                total_iv += iv
            return total_iv
        left_events = np.sum(left_y == unique_classes[1])
        left_non_events = len(left_y) - left_events
        right_events = np.sum(right_y == unique_classes[1])
        right_non_events = len(right_y) - right_events
        total_events = left_events + right_events
        total_non_events = left_non_events + right_non_events
        if total_events == 0 or total_non_events == 0:
            return 0.0
        left_event_rate = left_events / total_events if total_events > 0 else 0
        left_non_event_rate = left_non_events / total_non_events if total_non_events > 0 else 0
        right_event_rate = right_events / total_events if total_events > 0 else 0
        right_non_event_rate = right_non_events / total_non_events if total_non_events > 0 else 0
        epsilon = 1e-10
        left_event_rate = max(left_event_rate, epsilon)
        left_non_event_rate = max(left_non_event_rate, epsilon)
        right_event_rate = max(right_event_rate, epsilon)
        right_non_event_rate = max(right_non_event_rate, epsilon)
        iv = (left_event_rate - left_non_event_rate) * np.log(left_event_rate / left_non_event_rate) + (right_event_rate - right_non_event_rate) * np.log(right_event_rate / right_non_event_rate)
        return iv

    def _calculate_gini_gain(self, values: np.ndarray, y: np.ndarray, left_start: int, split_idx: int) -> float:
        """Calculate Gini impurity reduction for a potential split."""
        left_y = y[left_start:split_idx]
        right_y = y[split_idx:len(y)]
        if len(left_y) == 0 or len(right_y) == 0:
            return 0.0
        parent_gini = self._gini_impurity(y)
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        left_gini = self._gini_impurity(left_y)
        right_gini = self._gini_impurity(right_y)
        weighted_child_gini = left_weight * left_gini + right_weight * right_gini
        return parent_gini - weighted_child_gini

    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity of a set of labels."""
        if len(y) == 0:
            return 0.0
        (unique_classes, counts) = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _calculate_chi2_gain(self, values: np.ndarray, y: np.ndarray, classes: np.ndarray, left_start: int, split_idx: int, total_samples: int) -> float:
        """Calculate Chi2 statistic for a potential split."""
        left_y = y[left_start:split_idx]
        right_y = y[split_idx:total_samples]
        left_counts = np.array([np.sum(left_y == cls) for cls in classes])
        right_counts = np.array([np.sum(right_y == cls) for cls in classes])
        if np.sum(left_counts) == 0 or np.sum(right_counts) == 0:
            return 0.0
        left_total = np.sum(left_counts)
        right_total = np.sum(right_counts)
        eps = 1e-10
        chi2_stat = 0.0
        for i in range(len(classes)):
            expected_left = max(eps, left_total * (left_counts[i] + right_counts[i]) / (left_total + right_total))
            expected_right = max(eps, right_total * (left_counts[i] + right_counts[i]) / (left_total + right_total))
            observed_left = left_counts[i]
            observed_right = right_counts[i]
            if expected_left > 0:
                chi2_stat += (observed_left - expected_left) ** 2 / expected_left
            if expected_right > 0:
                chi2_stat += (observed_right - expected_right) ** 2 / expected_right
        return chi2_stat

    def _enforce_monotonicity(self, bin_edges: List[float], sorted_values: np.ndarray, sorted_y: np.ndarray) -> List[float]:
        """Enforce monotonic relationship between bin index and target probability."""
        if len(bin_edges) <= 2:
            return bin_edges
        bin_probs = []
        unique_classes = np.unique(sorted_y)
        target_class = unique_classes[1] if len(unique_classes) == 2 else unique_classes[0]
        for i in range(len(bin_edges) - 1):
            left_edge = bin_edges[i]
            right_edge = bin_edges[i + 1]
            if i < len(bin_edges) - 2:
                mask = (sorted_values >= left_edge) & (sorted_values < right_edge)
            else:
                mask = (sorted_values >= left_edge) & (sorted_values <= right_edge)
            if np.sum(mask) == 0:
                bin_probs.append(0.0)
            else:
                prob = np.mean(sorted_y[mask] == target_class)
                bin_probs.append(prob)
        is_increasing = all((bin_probs[i] <= bin_probs[i + 1] for i in range(len(bin_probs) - 1)))
        is_decreasing = all((bin_probs[i] >= bin_probs[i + 1] for i in range(len(bin_probs) - 1)))
        if is_increasing or is_decreasing:
            return bin_edges
        if len(bin_probs) >= 2:
            if bin_probs[0] < bin_probs[-1]:
                should_be_increasing = True
            elif bin_probs[0] > bin_probs[-1]:
                should_be_increasing = False
            elif len(bin_probs) > 2:
                avg_first_half = np.mean(bin_probs[:len(bin_probs) // 2])
                avg_second_half = np.mean(bin_probs[len(bin_probs) // 2:])
                should_be_increasing = avg_first_half <= avg_second_half
            else:
                should_be_increasing = True
        else:
            should_be_increasing = True
        current_edges = bin_edges[:]
        current_probs = bin_probs[:]
        max_iterations = len(current_probs) * 3
        for iteration in range(max_iterations):
            if len(current_probs) <= 1:
                break
            modified = False
            if should_be_increasing:
                for i in range(len(current_probs) - 1):
                    if current_probs[i] > current_probs[i + 1]:
                        new_edges = current_edges[:i + 1] + current_edges[i + 2:]
                        left_edge = current_edges[i]
                        if i + 2 < len(current_edges):
                            right_edge = current_edges[i + 2]
                            mask = (sorted_values >= left_edge) & (sorted_values < right_edge)
                        else:
                            right_edge = current_edges[-1]
                            mask = (sorted_values >= left_edge) & (sorted_values <= right_edge)
                        if np.sum(mask) > 0:
                            new_prob = np.mean(sorted_y[mask] == target_class)
                        else:
                            new_prob = (current_probs[i] + current_probs[i + 1]) / 2.0
                        new_probs = current_probs[:i] + [new_prob] + current_probs[i + 2:]
                        current_edges = new_edges
                        current_probs = new_probs
                        modified = True
                        break
            else:
                for i in range(len(current_probs) - 1):
                    if current_probs[i] < current_probs[i + 1]:
                        new_edges = current_edges[:i + 1] + current_edges[i + 2:]
                        left_edge = current_edges[i]
                        if i + 2 < len(current_edges):
                            right_edge = current_edges[i + 2]
                            mask = (sorted_values >= left_edge) & (sorted_values < right_edge)
                        else:
                            right_edge = current_edges[-1]
                            mask = (sorted_values >= left_edge) & (sorted_values <= right_edge)
                        if np.sum(mask) > 0:
                            new_prob = np.mean(sorted_y[mask] == target_class)
                        else:
                            new_prob = (current_probs[i] + current_probs[i + 1]) / 2.0
                        new_probs = current_probs[:i] + [new_prob] + current_probs[i + 2:]
                        current_edges = new_edges
                        current_probs = new_probs
                        modified = True
                        break
            if not modified:
                break
        if len(current_edges) < 2:
            return [bin_edges[0], bin_edges[-1]]
        return current_edges