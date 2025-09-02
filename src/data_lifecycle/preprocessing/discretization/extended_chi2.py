from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class ExtendedChi2Discretizer(BaseTransformer):

    def __init__(self, n_bins: int=5, min_bin_size: float=0.05, strategy: str='greedy', feature_indices: Optional[List[int]]=None, name: Optional[str]=None):
        """
        Initialize the Extended Chi2 Discretizer.
        
        Args:
            n_bins (int): Maximum number of bins to create. Must be >= 2.
            min_bin_size (float): Minimum proportion of samples per bin (between 0 and 1).
            strategy (str): Discretization strategy ('greedy' or 'balanced').
            feature_indices (Optional[List[int]]): Specific feature indices to process. If None, all features are used.
            name (Optional[str]): Name for the transformer instance.
        """
        super().__init__(name=name)
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.strategy = strategy
        self.feature_indices = feature_indices
        self._bin_edges = {}

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'ExtendedChi2Discretizer':
        """
        Learn discretization thresholds using the Extended Chi2 method.
        
        This method analyzes the relationship between feature values and target labels
        to determine optimal bin boundaries that maximize the Chi2 statistic.
        
        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Input data containing continuous features.
                If FeatureSet or DataBatch, uses their internal data representation.
            y (Optional[np.ndarray]): Target labels for supervised discretization.
                Required for Chi2-based methods.
                
        Returns:
            ExtendedChi2Discretizer: Fitted transformer instance.
            
        Raises:
            ValueError: If y is None or if data dimensions don't match.
        """
        if y is None:
            raise ValueError("Target variable 'y' is required for supervised discretization.")
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = data.data
        else:
            X = data
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X ({X.shape[0]}) does not match y ({y.shape[0]}).')
        n_features = X.shape[1]
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        if any((idx < 0 or idx >= n_features for idx in feature_indices)):
            raise ValueError('Feature indices out of bounds.')
        self._bin_edges = {}
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            bin_edges = self._compute_chi2_bins(feature_values, y)
            self._bin_edges[feature_idx] = bin_edges
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply the learned discretization to transform continuous features into categorical bins.
        
        Each feature is mapped to discrete bins based on the thresholds learned during fitting.
        Values outside the fitted range are assigned to the nearest boundary bin.
        
        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Input data to discretize.
            
        Returns:
            Union[FeatureSet, DataBatch]: Discretized data with same container type as input.
            
        Raises:
            RuntimeError: If called before fitting.
        """
        if not hasattr(self, '_bin_edges') or len(self._bin_edges) == 0:
            raise RuntimeError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            is_feature_set = True
        elif isinstance(data, DataBatch):
            X = data.data.copy()
            is_data_batch = True
        else:
            X = np.asarray(data).copy()
            is_feature_set = False
            is_data_batch = False
        for (feature_idx, bin_edges) in self._bin_edges.items():
            if feature_idx < X.shape[1]:
                bin_indices = np.digitize(X[:, feature_idx], bin_edges, right=False)
                bin_indices = np.clip(bin_indices, 1, len(bin_edges) - 1) - 1
                X[:, feature_idx] = bin_indices
        if is_feature_set:
            return FeatureSet(features=X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        elif is_data_batch:
            return DataBatch(data=X, sample_ids=data.sample_ids, metadata=data.metadata)
        else:
            return X

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Raise NotImplementedError as discretization is not invertible.
        
        Discretization loses information about the original continuous values,
        making exact inversion impossible.
        
        Args:
            data: Discretized input data.
            
        Raises:
            NotImplementedError: Always raised as inversion is not supported.
        """
        pass

    def _compute_chi2_bins(self, feature_values: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute bin edges for a single feature using extended Chi2 method.
        
        Args:
            feature_values: Continuous values of a single feature
            y: Target labels
            
        Returns:
            Array of bin edges
        """
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_y = y[sorted_indices]
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples = len(sorted_values)
        min_bin_size = max(1, int(self.min_bin_size * n_samples))
        max_bins = self.n_bins
        bin_edges = [sorted_values[0]]
        if max_bins <= 1 or n_samples <= min_bin_size:
            bin_edges.append(sorted_values[-1])
            return np.array(bin_edges)
        candidate_splits = self._find_candidate_splits(sorted_values, sorted_y, classes)
        if self.strategy == 'greedy':
            bin_edges = self._greedy_chi2_binning(sorted_values, sorted_y, classes, candidate_splits, max_bins, min_bin_size)
        elif self.strategy == 'balanced':
            bin_edges = self._balanced_chi2_binning(sorted_values, sorted_y, classes, candidate_splits, max_bins, min_bin_size)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Use 'greedy' or 'balanced'.")
        return np.array(bin_edges)

    def _find_candidate_splits(self, sorted_values: np.ndarray, sorted_y: np.ndarray, classes: np.ndarray) -> List[int]:
        """Find candidate split points based on class changes."""
        candidate_splits = []
        prev_class = sorted_y[0]
        for i in range(1, len(sorted_y)):
            if sorted_y[i] != prev_class:
                candidate_splits.append(i)
                prev_class = sorted_y[i]
        return candidate_splits

    def _greedy_chi2_binning(self, values: np.ndarray, y: np.ndarray, classes: np.ndarray, candidate_splits: List[int], max_bins: int, min_bin_size: int) -> List[float]:
        """Perform greedy Chi2-based binning."""
        n_samples = len(values)
        bin_edges = [values[0]]
        if not candidate_splits or n_samples <= min_bin_size:
            bin_edges.append(values[-1])
            return bin_edges
        splits_added = 0
        last_split_idx = 0
        candidate_gains = []
        for split_idx in candidate_splits:
            if split_idx - last_split_idx >= min_bin_size and n_samples - split_idx >= min_bin_size:
                gain = self._calculate_chi2_gain(values, y, classes, last_split_idx, split_idx, n_samples)
                candidate_gains.append((gain, split_idx))
        candidate_gains.sort(reverse=True)
        for (_, split_idx) in candidate_gains:
            if splits_added >= max_bins - 1:
                break
            if split_idx - last_split_idx >= min_bin_size and n_samples - split_idx >= min_bin_size:
                bin_edges.append(values[split_idx])
                last_split_idx = split_idx
                splits_added += 1
        bin_edges.append(values[-1])
        return bin_edges

    def _balanced_chi2_binning(self, values: np.ndarray, y: np.ndarray, classes: np.ndarray, candidate_splits: List[int], max_bins: int, min_bin_size: int) -> List[float]:
        """Perform balanced Chi2-based binning."""
        n_samples = len(values)
        bin_edges = [values[0]]
        if not candidate_splits or n_samples <= min_bin_size:
            bin_edges.append(values[-1])
            return bin_edges
        target_splits = min(max_bins - 1, len(candidate_splits))
        if target_splits <= 0:
            bin_edges.append(values[-1])
            return bin_edges
        candidate_gains = []
        for split_idx in candidate_splits:
            if split_idx >= min_bin_size and n_samples - split_idx >= min_bin_size:
                gain = self._calculate_chi2_gain(values, y, classes, 0, split_idx, n_samples)
                candidate_gains.append((gain, split_idx))
        candidate_gains.sort(reverse=True)
        selected_splits = []
        for (_, split_idx) in candidate_gains:
            if len(selected_splits) >= target_splits:
                break
            conflict = False
            for existing_split in selected_splits:
                if abs(split_idx - existing_split) < min_bin_size:
                    conflict = True
                    break
            if not conflict:
                selected_splits.append(split_idx)
        selected_splits.sort()
        for split_idx in selected_splits:
            bin_edges.append(values[split_idx])
        bin_edges.append(values[-1])
        return bin_edges

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