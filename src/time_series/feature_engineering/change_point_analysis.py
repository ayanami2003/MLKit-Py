from typing import Union, Optional, List, Tuple
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from scipy import stats

class StructuralBreakDetector(BaseTransformer):
    """
    Identify structural breaks in time series data where statistical properties change.
    
    This transformer detects points in time where the underlying data generating process
    changes, affecting parameters like mean, variance, or regression coefficients.
    
    Attributes
    ----------
    method : str, default='chow'
        Algorithm to use for structural break detection ('chow', 'cusum', ' Andrews-F-test').
    significance_level : float, default=0.05
        Significance level for detecting structural breaks.
    max_breaks : int, default=5
        Maximum number of breaks to detect.
    """

    def __init__(self, method: str='chow', significance_level: float=0.05, max_breaks: int=5, name: Optional[str]=None):
        super().__init__(name=name)
        self.method = method
        self.significance_level = significance_level
        self.max_breaks = max_breaks
        self._break_points = []

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'StructuralBreakDetector':
        """
        Fit the structural break detector to the time series data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data where each row represents a time point.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        StructuralBreakDetector
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            values = data.get_feature_matrix()
        elif isinstance(data, FeatureSet):
            values = data.get_feature_matrix()
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.ndim > 1:
            if values.shape[1] == 1:
                values = values.flatten()
            else:
                values = values[:, 0]
        if self.method == 'chow':
            self._break_points = self._detect_chow_breaks(values)
        elif self.method == 'cusum':
            self._break_points = self._detect_cusum_breaks(values)
        elif self.method == 'Andrews-F-test':
            self._break_points = self._detect_andrews_f_breaks(values)
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        return self

    def _detect_chow_breaks(self, values: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detect structural breaks using Chow test.
        
        Parameters
        ----------
        values : np.ndarray
            Time series data
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (index, test_statistic) tuples for detected breaks
        """
        breaks = []
        n = len(values)
        min_obs = 10
        for i in range(min_obs, n - min_obs):
            x1 = np.arange(i)
            y1 = values[:i]
            x2 = np.arange(i, n)
            y2 = values[i:]
            X1 = np.column_stack([np.ones(len(x1)), x1])
            X2 = np.column_stack([np.ones(len(x2)), x2])
            try:
                beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                res1 = y1 - X1 @ beta1
                res2 = y2 - X2 @ beta2
                sse_pooled = np.sum(res1 ** 2) + np.sum(res2 ** 2)
                X_combined = np.column_stack([np.ones(n), np.arange(n)])
                beta_combined = np.linalg.lstsq(X_combined, values, rcond=None)[0]
                res_combined = values - X_combined @ beta_combined
                sse_combined = np.sum(res_combined ** 2)
                k = X1.shape[1]
                df1 = 2 * k
                df2 = n - 2 * k
                chow_stat = (sse_combined - sse_pooled) / df1 / (sse_pooled / df2)
                f_critical = stats.f.ppf(1 - self.significance_level, df1, df2)
                if chow_stat > f_critical:
                    breaks.append((i, chow_stat))
            except np.linalg.LinAlgError:
                continue
        breaks.sort(key=lambda x: x[1], reverse=True)
        return breaks[:self.max_breaks]

    def _detect_cusum_breaks(self, values: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detect structural breaks using CUSUM method.
        
        Parameters
        ----------
        values : np.ndarray
            Time series data
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (index, test_statistic) tuples for detected breaks
        """
        breaks = []
        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return breaks
        cusum_stats = np.cumsum(values - mean_val) / std_val
        max_deviation = np.max(np.abs(cusum_stats))
        h = 1.143 * np.sqrt(n)
        if max_deviation > h:
            max_idx = np.argmax(np.abs(cusum_stats))
            breaks.append((max_idx, max_deviation))
            if self.max_breaks > 1:
                if max_idx > 10:
                    left_values = values[:max_idx]
                    left_breaks = self._detect_cusum_breaks(left_values)
                    for (idx, stat) in left_breaks:
                        breaks.append((idx, stat))
                if max_idx < n - 10:
                    right_values = values[max_idx:]
                    right_breaks = self._detect_cusum_breaks(right_values)
                    for (idx, stat) in right_breaks:
                        breaks.append((max_idx + idx, stat))
        breaks.sort(key=lambda x: x[1], reverse=True)
        return breaks[:self.max_breaks]

    def _detect_andrews_f_breaks(self, values: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detect structural breaks using Andrews' F-test.
        
        Parameters
        ----------
        values : np.ndarray
            Time series data
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (index, test_statistic) tuples for detected breaks
        """
        breaks = []
        n = len(values)
        min_obs = 10
        X = np.column_stack([np.ones(n), np.arange(n)])
        try:
            beta = np.linalg.lstsq(X, values, rcond=None)[0]
            residuals = values - X @ beta
            sigma_sq = np.sum(residuals ** 2) / (n - 2)
            f_stats = []
            for i in range(min_obs, n - min_obs):
                x1 = np.arange(i)
                y1 = values[:i]
                x2 = np.arange(i, n)
                y2 = values[i:]
                X1 = np.column_stack([np.ones(len(x1)), x1])
                X2 = np.column_stack([np.ones(len(x2)), x2])
                try:
                    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                    res1 = y1 - X1 @ beta1
                    res2 = y2 - X2 @ beta2
                    sse_segments = np.sum(res1 ** 2) + np.sum(res2 ** 2)
                    f_stat = (np.sum(residuals ** 2) - sse_segments) / 2 / sigma_sq
                    f_stats.append((i, f_stat))
                except np.linalg.LinAlgError:
                    continue
            if f_stats:
                (df1, df2) = (2, n - 4)
                f_critical = stats.f.ppf(1 - self.significance_level, df1, df2)
                for (idx, f_stat) in f_stats:
                    if f_stat > f_critical:
                        breaks.append((idx, f_stat))
        except np.linalg.LinAlgError:
            return []
        breaks.sort(key=lambda x: x[1], reverse=True)
        return breaks[:self.max_breaks]

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Detect structural breaks and generate features indicating their locations.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data to analyze.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Features indicating structural breaks and their characteristics.
        """
        feature_names = []
        for i in range(self.max_breaks):
            feature_names.extend([f'break_{i}_index', f'break_{i}_statistic', f'break_{i}_detected'])
        features = np.zeros((1, len(feature_names)))
        for (i, (idx, stat)) in enumerate(self._break_points):
            if i >= self.max_breaks:
                break
            features[0, i * 3] = idx
            features[0, i * 3 + 1] = stat
            features[0, i * 3 + 2] = 1
        return FeatureSet(feature_matrix=features, feature_names=feature_names, sample_ids=['structural_breaks'] if not hasattr(data, 'sample_ids') else data.sample_ids[:1])

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation (not implemented for this transformer).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original data structure.
        """
        raise NotImplementedError('Inverse transform not supported for StructuralBreakDetector')

class TrendChangePointDetector(BaseTransformer):

    def __init__(self, method: str='pettitt', significance_level: float=0.05, min_distance: int=10, name: Optional[str]=None):
        super().__init__(name=name)
        if method not in ['pettitt', 'mann-kendall', 'cusum']:
            raise ValueError("Method must be one of 'pettitt', 'mann-kendall', 'cusum'")
        self.method = method
        if not 0 < significance_level < 1:
            raise ValueError('Significance level must be between 0 and 1')
        self.significance_level = significance_level
        if min_distance <= 0:
            raise ValueError('Minimum distance must be positive')
        self.min_distance = min_distance
        self._change_points = []

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'TrendChangePointDetector':
        """
        Fit the trend change point detector to the time series data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data where each row represents a time point.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        TrendChangePointDetector
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            values = data.data
        elif isinstance(data, FeatureSet):
            values = data.features
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.ndim > 1:
            if values.shape[1] == 1:
                values = values.flatten()
            else:
                values = values[:, 0]
        if len(values) < 2 * self.min_distance:
            self._change_points = []
            return self
        if self.method == 'pettitt':
            self._change_points = self._detect_pettitt_changes(values)
        elif self.method == 'mann-kendall':
            self._change_points = self._detect_mann_kendall_changes(values)
        elif self.method == 'cusum':
            self._change_points = self._detect_cusum_changes(values)
        return self

    def _detect_pettitt_changes(self, values: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detect change points using the Pettitt test.
        
        Parameters
        ----------
        values : np.ndarray
            Time series data
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (index, test_statistic) tuples for detected change points
        """
        n = len(values)
        change_points = []
        if n < 4:
            return change_points
        u_stat = np.zeros(n)
        for t in range(n):
            u_stat[t] = np.sum(np.sign(values[:t + 1, None] - values[None, t + 1:]))
        max_u = np.max(np.abs(u_stat))
        critical_value = np.sqrt(n * (n + 1) * (2 * n + 1) / 6) * np.sqrt(-np.log(self.significance_level / 2))
        if max_u > critical_value:
            change_idx = np.argmax(np.abs(u_stat))
            if self.min_distance <= change_idx < n - self.min_distance:
                change_points.append((change_idx, max_u))
        return change_points

    def _detect_mann_kendall_changes(self, values: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detect change points using the Mann-Kendall test approach.
        
        Parameters
        ----------
        values : np.ndarray
            Time series data
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (index, test_statistic) tuples for detected change points
        """
        n = len(values)
        change_points = []
        if n < 4:
            return change_points
        min_obs = max(5, self.min_distance)
        critical_z = stats.norm.ppf(1 - self.significance_level / 2)
        potential_changes = []
        for i in range(min_obs, n - min_obs):
            x1 = values[:i]
            x2 = values[i:]
            s1 = self._mann_kendall_s_statistic(x1)
            s2 = self._mann_kendall_s_statistic(x2)
            var1 = self._mann_kendall_variance(len(x1))
            var2 = self._mann_kendall_variance(len(x2))
            if var1 > 0 and var2 > 0:
                z_diff = abs(s1 - s2) / np.sqrt(var1 + var2)
                if z_diff > critical_z:
                    potential_changes.append((i, z_diff))
        if potential_changes:
            potential_changes.sort(key=lambda x: x[1], reverse=True)
            for cp in potential_changes:
                if self.min_distance <= cp[0] < n - self.min_distance:
                    change_points.append(cp)
                    break
        return change_points

    def _mann_kendall_s_statistic(self, x: np.ndarray) -> float:
        """Calculate the S statistic for Mann-Kendall test."""
        n = len(x)
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(x[j] - x[i])
        return s

    def _mann_kendall_variance(self, n: int) -> float:
        """Calculate the variance for Mann-Kendall test."""
        return n * (n - 1) * (2 * n + 5) / 18

    def _detect_cusum_changes(self, values: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detect change points using the CUSUM method.
        
        Parameters
        ----------
        values : np.ndarray
            Time series data
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (index, test_statistic) tuples for detected change points
        """
        n = len(values)
        change_points = []
        if n < 2 * self.min_distance:
            return change_points
        global_mean = np.mean(values)
        cusum = np.zeros(n)
        for i in range(1, n):
            cusum[i] = cusum[i - 1] + (values[i] - global_mean)
        max_deviation = np.max(np.abs(cusum))
        h = 1.143 * np.sqrt(n) * np.sqrt(-np.log(self.significance_level))
        if max_deviation > h:
            change_idx = np.argmax(np.abs(cusum))
            if self.min_distance <= change_idx < n - self.min_distance:
                change_points.append((change_idx, max_deviation))
        return change_points

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Detect trend change points and generate features indicating their locations.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data to analyze.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Features indicating trend change points and their characteristics.
        """
        feature_names = []
        max_changes = len(self._change_points) if self._change_points else 1
        for i in range(max_changes):
            feature_names.extend([f'change_point_{i}_index', f'change_point_{i}_statistic', f'change_point_{i}_detected'])
        features = np.zeros((1, len(feature_names)))
        for (i, (idx, stat)) in enumerate(self._change_points):
            if i >= max_changes:
                break
            features[0, i * 3] = idx
            features[0, i * 3 + 1] = stat
            features[0, i * 3 + 2] = 1
        if hasattr(data, 'sample_ids') and data.sample_ids:
            sample_ids = data.sample_ids[:1]
        else:
            sample_ids = ['trend_change_points']
        return FeatureSet(features=features, feature_names=feature_names, sample_ids=sample_ids)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation (not implemented for this transformer).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original data structure.
        """
        raise NotImplementedError('Inverse transform not supported for TrendChangePointDetector')

class ChangePointFeatureExtractor(BaseTransformer):

    def __init__(self, method: str='pelt', cost_function: str='l2', min_size: int=2, jump: int=5, penalty: float=10, name: Optional[str]=None):
        super().__init__(name=name)
        if method not in ['pelt', 'binseg', 'window']:
            raise ValueError("Method must be one of 'pelt', 'binseg', 'window'")
        if cost_function not in ['l1', 'l2', 'rbf']:
            raise ValueError("Cost function must be one of 'l1', 'l2', 'rbf'")
        if min_size < 1:
            raise ValueError('min_size must be positive')
        if jump < 1:
            raise ValueError('jump must be positive')
        if penalty <= 0:
            raise ValueError('penalty must be positive')
        self.method = method
        self.cost_function = cost_function
        self.min_size = min_size
        self.jump = jump
        self.penalty = penalty
        self._change_points = []
        self._fitted = False

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'ChangePointFeatureExtractor':
        """
        Fit the change point detector to the time series data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data where each row represents a time point.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        ChangePointFeatureExtractor
            Self instance for method chaining.
        """
        if isinstance(data, DataBatch):
            values = data.data
        elif isinstance(data, FeatureSet):
            values = data.features
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.ndim > 1:
            if values.shape[1] == 1:
                values = values.flatten()
            else:
                values = values[:, 0]
        self._data_length = len(values)
        self._change_points = self._detect_change_points(values)
        self._fitted = True
        return self

    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        """
        Detect change points in the time series using the specified method.
        
        Parameters
        ----------
        values : np.ndarray
            1D array of time series values
            
        Returns
        -------
        List[int]
            List of indices where change points occur
        """
        n = len(values)
        if n < 2 * self.min_size:
            return []
        change_points = []
        method = self.method
        if method not in ['pelt', 'binseg', 'window']:
            method = 'pelt'
        if method == 'pelt':
            change_points = self._pelt_detection(values)
        elif method == 'binseg':
            change_points = self._binary_segmentation(values)
        elif method == 'window':
            change_points = self._sliding_window_detection(values)
        return sorted(change_points)

    def _pelt_detection(self, values: np.ndarray) -> List[int]:
        """Simple PELT-like change point detection."""
        n = len(values)
        change_points = []

        def calculate_cost(start, end):
            if end <= start:
                return 0
            segment = values[start:end]
            if self.cost_function == 'l2':
                return np.sum((segment - np.mean(segment)) ** 2)
            elif self.cost_function == 'l1':
                return np.sum(np.abs(segment - np.median(segment)))
            else:
                return np.sum((segment - np.mean(segment)) ** 2)
        current_start = 0
        while current_start + 2 * self.min_size < n:
            best_cost_reduction = 0
            best_split = None
            for split in range(current_start + self.min_size, n - self.min_size, self.jump):
                cost_no_split = calculate_cost(current_start, n)
                cost_with_split = calculate_cost(current_start, split) + calculate_cost(split, n)
                cost_reduction = cost_no_split - cost_with_split - self.penalty
                if cost_reduction > best_cost_reduction:
                    best_cost_reduction = cost_reduction
                    best_split = split
            if best_split is not None:
                change_points.append(best_split)
                current_start = best_split
            else:
                break
        return change_points

    def _binary_segmentation(self, values: np.ndarray) -> List[int]:
        """Binary segmentation change point detection."""
        n = len(values)
        change_points = []

        def calculate_cost(start, end):
            if end <= start:
                return 0
            segment = values[start:end]
            if self.cost_function == 'l2':
                return np.sum((segment - np.mean(segment)) ** 2)
            elif self.cost_function == 'l1':
                return np.sum(np.abs(segment - np.median(segment)))
            else:
                return np.sum((segment - np.mean(segment)) ** 2)

        def find_best_split(start, end):
            if end - start < 2 * self.min_size:
                return (None, 0)
            best_cost_reduction = 0
            best_split = None
            for split in range(start + self.min_size, end - self.min_size, self.jump):
                cost_no_split = calculate_cost(start, end)
                cost_with_split = calculate_cost(start, split) + calculate_cost(split, end)
                cost_reduction = cost_no_split - cost_with_split
                if cost_reduction > best_cost_reduction:
                    best_cost_reduction = cost_reduction
                    best_split = split
            return (best_split, best_cost_reduction)
        segments = [(0, n)]
        while segments and len(change_points) < n // (2 * self.min_size):
            best_segment = None
            best_split = None
            best_reduction = 0
            for (start, end) in segments:
                (split, reduction) = find_best_split(start, end)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_split = split
                    best_segment = (start, end)
            if best_split is not None and best_reduction > self.penalty:
                change_points.append(best_split)
                segments.remove(best_segment)
                segments.append((best_segment[0], best_split))
                segments.append((best_split, best_segment[1]))
            else:
                break
        return change_points

    def _sliding_window_detection(self, values: np.ndarray) -> List[int]:
        """Sliding window change point detection."""
        n = len(values)
        change_points = []
        window_size = max(2 * self.min_size, 20)
        step_size = max(window_size // 4, 5)
        for i in range(0, n - window_size, step_size):
            window_start = i
            window_end = i + window_size
            mid = (window_start + window_end) // 2
            left_segment = values[window_start:mid]
            right_segment = values[mid:window_end]
            if self.cost_function == 'l2':
                (left_mean, left_var) = (np.mean(left_segment), np.var(left_segment))
                (right_mean, right_var) = (np.mean(right_segment), np.var(right_segment))
                pooled_std = np.sqrt(((len(left_segment) - 1) * left_var + (len(right_segment) - 1) * right_var) / (len(left_segment) + len(right_segment) - 2))
                if pooled_std > 0:
                    t_stat = abs(left_mean - right_mean) / (pooled_std * np.sqrt(1 / len(left_segment) + 1 / len(right_segment)))
                    if t_stat > 2.0:
                        change_points.append(mid)
            elif self.cost_function == 'l1':
                left_median = np.median(left_segment)
                right_median = np.median(right_segment)
                left_mad = np.median(np.abs(left_segment - left_median))
                right_mad = np.median(np.abs(right_segment - right_median))
                if left_mad + right_mad > 0:
                    z_stat = abs(left_median - right_median) / (left_mad + right_mad)
                    if z_stat > 1.0:
                        change_points.append(mid)
        return sorted(list(set(change_points)))

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Extract features based on identified change points.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Extracted features including segment statistics and change point indicators.
        """
        if not self._fitted:
            raise ValueError('Transformer must be fitted before transform')
        if isinstance(data, DataBatch):
            values = data.data
            sample_ids = data.sample_ids if hasattr(data, 'sample_ids') else None
        elif isinstance(data, FeatureSet):
            values = data.features
            sample_ids = data.sample_ids
        else:
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.ndim > 1:
            if values.shape[1] == 1:
                values = values.flatten()
            else:
                values = values[:, 0]
        segments = self._create_segments(values)
        (features, feature_names) = self._extract_segment_features(segments, values)
        if sample_ids is None or len(sample_ids) != features.shape[0]:
            sample_ids = [f'segment_{i}' for i in range(features.shape[0])]
        return FeatureSet(features=features, feature_names=feature_names, sample_ids=sample_ids)

    def _create_segments(self, values: np.ndarray) -> List[tuple]:
        """
        Create segments based on change points.
        
        Parameters
        ----------
        values : np.ndarray
            Time series values
            
        Returns
        -------
        List[tuple]
            List of (start_idx, end_idx, data) tuples for each segment
        """
        boundaries = [0] + sorted(self._change_points) + [len(values)]
        segments = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            if end_idx > start_idx:
                segment_data = values[start_idx:end_idx]
                segments.append((start_idx, end_idx, segment_data))
        return segments

    def _extract_segment_features(self, segments: List[tuple], values: np.ndarray) -> tuple:
        """
        Extract features from segments.
        
        Parameters
        ----------
        segments : List[tuple]
            List of (start_idx, end_idx, data) tuples
        values : np.ndarray
            Original time series data
            
        Returns
        -------
        tuple
            (features_array, feature_names) tuple
        """
        feature_list = []
        feature_names = []
        for (i, (start_idx, end_idx, segment_data)) in enumerate(segments):
            segment_mean = np.mean(segment_data)
            segment_var = np.var(segment_data)
            segment_std = np.std(segment_data)
            segment_min = np.min(segment_data)
            segment_max = np.max(segment_data)
            segment_duration = end_idx - start_idx
            if len(segment_data) > 1:
                x = np.arange(len(segment_data))
                (slope, _) = np.polyfit(x, segment_data, 1)
            else:
                slope = 0.0
            is_change_point = 1 if i < len(segments) - 1 else 0
            feature_list.extend([segment_mean, segment_var, segment_std, segment_min, segment_max, segment_duration, slope, is_change_point])
            segment_feature_names = [f'segment_{i}_mean', f'segment_{i}_variance', f'segment_{i}_std', f'segment_{i}_min', f'segment_{i}_max', f'segment_{i}_duration', f'segment_{i}_slope', f'segment_{i}_is_change_point']
            feature_names.extend(segment_feature_names)
        features = np.array(feature_list).reshape(1, -1)
        return (features, feature_names)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Inverse transformation (no-op for this transformer).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Original data structure unchanged.
        """
        raise NotImplementedError('Inverse transform not supported for ChangePointFeatureExtractor')

def calculate_hurst_exponent(data: Union[np.ndarray, DataBatch, FeatureSet], max_lag: Optional[int]=None, method: str='rs') -> float:
    """
    Calculate the Hurst exponent of a time series to assess its long-term memory.
    
    The Hurst exponent indicates whether a time series is mean-reverting (H < 0.5),
    random (H = 0.5), or trending (H > 0.5). This function supports multiple calculation
    methods for robust estimation.
    
    Parameters
    ----------
    data : Union[np.ndarray, DataBatch, FeatureSet]
        Time series data for which to compute the Hurst exponent.
        If DataBatch or FeatureSet, uses the primary data attribute.
    max_lag : Optional[int], default=None
        Maximum lag to consider in calculations. If None, defaults to len(data)//2.
    method : str, default='rs'
        Method for calculating the Hurst exponent ('rs', 'dfa', 'welch').
        
    Returns
    -------
    float
        Estimated Hurst exponent value between 0 and 1.
        
    Raises
    ------
    ValueError
        If the input data is empty or if an unsupported method is specified.
    """
    if isinstance(data, DataBatch):
        ts = data.data
    elif isinstance(data, FeatureSet):
        ts = data.features
    else:
        ts = data
    ts = np.asarray(ts)
    if ts.size == 0:
        raise ValueError('Input data is empty.')
    supported_methods = ['rs']
    if method not in supported_methods:
        raise ValueError(f"Unsupported method '{method}'. Supported methods: {supported_methods}")
    if max_lag is None:
        max_lag = len(ts) // 2
    if method == 'rs':
        mean_adjusted = ts - np.mean(ts)
        cumulative_deviation = np.cumsum(mean_adjusted)
        lags = []
        rs_values = []
        lag_list = np.unique(np.logspace(1, np.log10(max_lag), num=min(50, max_lag // 2), dtype=int))
        lag_list = lag_list[lag_list <= max_lag]
        for lag in lag_list:
            if lag < 10:
                continue
            n_segments = len(ts) // lag
            if n_segments == 0:
                continue
            rs_segment_values = []
            for i in range(n_segments):
                segment = ts[i * lag:(i + 1) * lag]
                if len(segment) < lag:
                    continue
                mean_adj_seg = segment - np.mean(segment)
                cum_dev_seg = np.cumsum(mean_adj_seg)
                r = np.max(cum_dev_seg) - np.min(cum_dev_seg)
                s = np.std(segment)
                if s > 0:
                    rs_segment_values.append(r / s)
            if len(rs_segment_values) > 0:
                lags.append(lag)
                rs_values.append(np.mean(rs_segment_values))
        if len(lags) < 2:
            raise ValueError('Not enough valid lags to compute Hurst exponent.')
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        (slope, _) = np.polyfit(log_lags, log_rs, 1)
        return float(slope)
    raise ValueError(f"Method '{method}' is not implemented.")