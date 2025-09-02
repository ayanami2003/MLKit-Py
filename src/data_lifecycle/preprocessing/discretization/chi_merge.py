import numpy as np
from scipy.stats import chi2
from typing import Union, Optional, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class ChiMergeDiscretizer(BaseTransformer):

    def __init__(self, significance_level: float=0.05, max_intervals: Optional[int]=None, min_interval_freq: int=5, feature_indices: Optional[List[int]]=None, name: Optional[str]=None):
        """
        Initialize the ChiMerge discretizer.

        Args:
            significance_level (float): Significance level for chi-square test to determine
                                       whether to merge intervals. Lower values create more bins.
            max_intervals (Optional[int]): Maximum number of intervals to produce. If specified,
                                         will override significance_level when reached.
            min_interval_freq (int): Minimum number of samples required in each interval.
                                   Intervals with fewer samples will be merged.
            feature_indices (Optional[List[int]]): Specific feature indices to discretize.
                                                 If None, all features will be processed.
            name (Optional[str]): Custom name for the transformer instance.
        """
        super().__init__(name=name)
        self.significance_level = significance_level
        self.max_intervals = max_intervals
        self.min_interval_freq = min_interval_freq
        self.feature_indices = feature_indices
        self._interval_bounds = {}
        self._fitted = False

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], labels: Optional[np.ndarray]=None, **kwargs) -> 'ChiMergeDiscretizer':
        """
        Fit the discretizer to the input data using the Chi-Merge algorithm.

        This method computes the optimal interval boundaries for discretization based on
        the chi-square statistical test between adjacent intervals and class labels.

        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Input data containing continuous features.
                                                           Can be a FeatureSet, DataBatch, or numpy array.
            labels (Optional[np.ndarray]): Target class labels for supervised discretization.
                                         Required for chi-merge algorithm.
            **kwargs: Additional fitting parameters (ignored).

        Returns:
            ChiMergeDiscretizer: Returns self for method chaining.

        Raises:
            ValueError: If labels are not provided or data format is unsupported.
        """
        if labels is None:
            if isinstance(data, DataBatch) and data.labels is not None:
                labels = np.array(data.labels)
            else:
                raise ValueError('Labels are required for supervised Chi-Merge discretization')
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Unsupported data type. Must be FeatureSet, DataBatch, or numpy array.')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_labels = np.array(labels)[sorted_indices]
            (unique_vals, unique_indices) = np.unique(sorted_values, return_index=True)
            intervals = []
            class_labels = np.unique(sorted_labels)
            for i in range(len(unique_vals)):
                start_idx = unique_indices[i]
                end_idx = unique_indices[i + 1] if i + 1 < len(unique_indices) else len(sorted_values)
                interval_labels = sorted_labels[start_idx:end_idx]
                label_counts = np.array([np.sum(interval_labels == label) for label in class_labels])
                intervals.append({'start': unique_vals[i], 'end': unique_vals[i], 'label_counts': label_counts, 'total_count': len(interval_labels)})
            merged_intervals = []
            i = 0
            while i < len(intervals):
                current_interval = intervals[i].copy()
                while i + 1 < len(intervals) and intervals[i + 1]['start'] == current_interval['start']:
                    current_interval['label_counts'] += intervals[i + 1]['label_counts']
                    current_interval['total_count'] += intervals[i + 1]['total_count']
                    i += 1
                merged_intervals.append(current_interval)
                i += 1
            intervals = merged_intervals
            changed = True
            while changed and len(intervals) > 1:
                changed = False
                i = 0
                while i < len(intervals):
                    if intervals[i]['total_count'] < self.min_interval_freq:
                        merge_with_next = False
                        merge_with_prev = False
                        if i < len(intervals) - 1:
                            merge_with_next = True
                        if i > 0:
                            merge_with_prev = True
                        if merge_with_next and merge_with_prev:
                            next_total = intervals[i]['total_count'] + intervals[i + 1]['total_count']
                            prev_total = intervals[i - 1]['total_count'] + intervals[i]['total_count']
                            if prev_total >= self.min_interval_freq and next_total >= self.min_interval_freq:
                                try:
                                    chi_next = self._calculate_chi_square(np.array([intervals[i]['label_counts'], intervals[i + 1]['label_counts']]))
                                except:
                                    chi_next = np.inf
                                try:
                                    chi_prev = self._calculate_chi_square(np.array([intervals[i - 1]['label_counts'], intervals[i]['label_counts']]))
                                except:
                                    chi_prev = np.inf
                                if chi_prev <= chi_next:
                                    merge_with_prev = True
                                    merge_with_next = False
                                else:
                                    merge_with_prev = False
                                    merge_with_next = True
                            elif prev_total >= self.min_interval_freq:
                                merge_with_prev = True
                                merge_with_next = False
                            elif next_total >= self.min_interval_freq:
                                merge_with_prev = False
                                merge_with_next = True
                            elif prev_total >= next_total:
                                merge_with_prev = True
                                merge_with_next = False
                            else:
                                merge_with_prev = False
                                merge_with_next = True
                        if merge_with_next and i < len(intervals) - 1:
                            left_interval = intervals[i]
                            right_interval = intervals[i + 1]
                            merged_label_counts = left_interval['label_counts'] + right_interval['label_counts']
                            merged_total_count = left_interval['total_count'] + right_interval['total_count']
                            merged_interval = {'start': left_interval['start'], 'end': right_interval['end'], 'label_counts': merged_label_counts, 'total_count': merged_total_count}
                            intervals[i] = merged_interval
                            del intervals[i + 1]
                            changed = True
                        elif merge_with_prev and i > 0:
                            left_interval = intervals[i - 1]
                            right_interval = intervals[i]
                            merged_label_counts = left_interval['label_counts'] + right_interval['label_counts']
                            merged_total_count = left_interval['total_count'] + right_interval['total_count']
                            merged_interval = {'start': left_interval['start'], 'end': right_interval['end'], 'label_counts': merged_label_counts, 'total_count': merged_total_count}
                            intervals[i - 1] = merged_interval
                            del intervals[i]
                            changed = True
                            i -= 1
                    i += 1
            while len(intervals) > 1:
                if self.max_intervals is not None and len(intervals) <= self.max_intervals:
                    break
                min_chi2 = np.inf
                merge_idx = -1
                for i in range(len(intervals) - 1):
                    counts1 = intervals[i]['label_counts']
                    counts2 = intervals[i + 1]['label_counts']
                    if np.sum(counts1) == 0 or np.sum(counts2) == 0:
                        continue
                    contingency_table = np.array([counts1, counts2])
                    try:
                        chi2_val = self._calculate_chi_square(contingency_table)
                        if chi2_val < min_chi2:
                            min_chi2 = chi2_val
                            merge_idx = i
                    except:
                        continue
                if merge_idx == -1:
                    break
                should_merge = True
                chi2_threshold = chi2.ppf(1 - self.significance_level, df=max(1, len(class_labels) - 1))
                if min_chi2 > chi2_threshold:
                    should_merge = False
                if self.max_intervals is not None and len(intervals) <= self.max_intervals:
                    should_merge = False
                if should_merge:
                    merged_count = intervals[merge_idx]['total_count'] + intervals[merge_idx + 1]['total_count']
                    if merged_count < self.min_interval_freq:
                        should_merge = False
                if not should_merge:
                    break
                left_interval = intervals[merge_idx]
                right_interval = intervals[merge_idx + 1]
                merged_label_counts = left_interval['label_counts'] + right_interval['label_counts']
                merged_total_count = left_interval['total_count'] + right_interval['total_count']
                merged_interval = {'start': left_interval['start'], 'end': right_interval['end'], 'label_counts': merged_label_counts, 'total_count': merged_total_count}
                intervals[merge_idx] = merged_interval
                del intervals[merge_idx + 1]
            boundaries = [interval['start'] for interval in intervals]
            boundaries.append(intervals[-1]['end'])
            self._interval_bounds[feature_idx] = np.array(boundaries)
        self._fitted = True
        return self

    def _calculate_chi_square(self, contingency_table: np.ndarray) -> float:
        """
        Calculate the chi-square statistic for a 2xk contingency table.
        
        Args:
            contingency_table (np.ndarray): 2xk contingency table
            
        Returns:
            float: Chi-square statistic
        """
        row_totals = np.sum(contingency_table, axis=1)
        col_totals = np.sum(contingency_table, axis=0)
        grand_total = np.sum(contingency_table)
        if grand_total == 0:
            return 0.0
        expected = np.outer(row_totals, col_totals) / grand_total
        epsilon = 1e-10
        expected = np.where(expected == 0, epsilon, expected)
        chi_square = np.sum((contingency_table - expected) ** 2 / expected)
        return chi_square

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Apply discretization to the input data using computed interval boundaries.

        Transforms continuous features into discrete ordinal values based on fitted interval bounds.

        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Input data to discretize.
            **kwargs: Additional transformation parameters (ignored).

        Returns:
            Union[FeatureSet, DataBatch]: Discretized data with same container type as input.

        Raises:
            RuntimeError: If transformer has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        original_type = type(data)
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            result_container = FeatureSet
        elif isinstance(data, DataBatch):
            X = np.array(data.data).copy()
            result_container = DataBatch
        elif isinstance(data, np.ndarray):
            X = data.copy()
            result_container = np.ndarray
        else:
            raise ValueError('Unsupported data type. Must be FeatureSet, DataBatch, or numpy array.')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        if self.feature_indices is None:
            feature_indices = list(range(n_features))
        else:
            feature_indices = self.feature_indices
        for feature_idx in feature_indices:
            if feature_idx in self._interval_bounds:
                boundaries = self._interval_bounds[feature_idx]
                if len(boundaries) > 1:
                    X[:, feature_idx] = np.digitize(X[:, feature_idx], boundaries, right=False) - 1
                    X[X[:, feature_idx] >= len(boundaries) - 1, feature_idx] = len(boundaries) - 2
                else:
                    X[:, feature_idx] = 0
        X = X.astype(int)
        if original_type == FeatureSet:
            return FeatureSet(features=X, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
        elif original_type == DataBatch:
            return DataBatch(data=X, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        else:
            return X

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Reverse the discretization transformation (not supported for chi-merge).

        As chi-merge discretization is a lossy transformation that maps continuous values
        to discrete intervals, exact inverse transformation is not possible.

        Args:
            data (Union[FeatureSet, DataBatch, np.ndarray]): Discretized data.
            **kwargs: Additional parameters (ignored).

        Returns:
            Union[FeatureSet, DataBatch]: The same data passed in (identity operation).

        Note:
            This method performs an identity operation as chi-merge discretization
            cannot be exactly inverted due to information loss.
        """
        return data

    def get_interval_boundaries(self, feature_index: int) -> Optional[np.ndarray]:
        """
        Retrieve the computed interval boundaries for a specific feature.

        Returns the boundaries that define the discrete intervals for a feature
        after fitting the discretizer.

        Args:
            feature_index (int): Index of the feature to retrieve boundaries for.

        Returns:
            Optional[np.ndarray]: Array of interval boundaries or None if not fitted.
        """
        if not self._fitted:
            return None
        return self._interval_bounds.get(feature_index, None)