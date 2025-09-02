from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class ChangePointDetector(BaseTransformer):

    def __init__(self, method: str='pelt', cost_function: str='l2', min_size: int=2, jump: int=5, penalty: float=10, name: Optional[str]=None):
        """
        Initialize the ChangePointDetector.
        
        Parameters
        ----------
        method : str, default='pelt'
            Algorithm to use for change point detection. Options:
            - 'pelt': Pruned Exact Linear Time
            - 'binseg': Binary Segmentation
            - 'bottomup': Bottom-Up Segmentation
        cost_function : str, default='l2'
            Cost function for evaluating changes. Options:
            - 'l1': Mean absolute deviation
            - 'l2': Mean squared deviation
            - 'rbf': Radial basis function
            - 'mahalanobis': Mahalanobis distance
        min_size : int, default=2
            Minimum segment size between change points
        jump : int, default=5
            Spacing between potential change points for computational efficiency
        penalty : float, default=10
            Penalty value to regulate number of change points detected
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.cost_function = cost_function
        self.min_size = min_size
        self.jump = jump
        self.penalty = penalty
        if self.method not in ['pelt', 'binseg', 'bottomup']:
            raise ValueError("Method must be one of 'pelt', 'binseg', 'bottomup'")
        if self.cost_function not in ['l1', 'l2', 'rbf', 'mahalanobis']:
            raise ValueError("Cost function must be one of 'l1', 'l2', 'rbf', 'mahalanobis'")
        if self.min_size < 1:
            raise ValueError('min_size must be at least 1')
        if self.jump < 1:
            raise ValueError('jump must be at least 1')

    def _extract_array(self, data: Union[DataBatch, FeatureSet, np.ndarray]) -> np.ndarray:
        """Extract numpy array from various input types."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, FeatureSet):
            return data.features.values
        elif isinstance(data, DataBatch):
            if hasattr(data, 'data'):
                return np.array(data.data)
            else:
                raise ValueError("DataBatch must have a 'data' attribute")
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')

    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        """Validate and reshape data if necessary."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError('Data must be 1D or 2D')
        return data

    def _compute_cost_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute cost matrix based on selected cost function."""
        (n_samples, n_features) = data.shape
        costs = np.zeros(n_samples)
        if self.cost_function == 'l1':
            for i in range(n_samples):
                if i > 0:
                    costs[i] = np.mean(np.abs(data[:i] - np.mean(data[:i], axis=0)))
        elif self.cost_function == 'l2':
            for i in range(n_samples):
                if i > 0:
                    mean_val = np.mean(data[:i], axis=0)
                    costs[i] = np.mean((data[:i] - mean_val) ** 2)
        elif self.cost_function == 'rbf':
            for i in range(n_samples):
                if i > 0:
                    diff = data[:i] - np.mean(data[:i], axis=0)
                    costs[i] = np.mean(np.exp(-np.sum(diff ** 2, axis=1)))
        elif self.cost_function == 'mahalanobis':
            for i in range(n_samples):
                if i > self.min_size:
                    try:
                        cov = np.cov(data[:i].T)
                        if np.linalg.det(cov) > 1e-10:
                            inv_cov = np.linalg.inv(cov)
                            mean_val = np.mean(data[:i], axis=0)
                            diff = data[:i] - mean_val
                            mahal_dist = np.mean([np.sqrt(np.dot(np.dot(d, inv_cov), d.T)) for d in diff])
                            costs[i] = mahal_dist
                        else:
                            costs[i] = np.inf
                    except np.linalg.LinAlgError:
                        costs[i] = np.inf
        return costs

    def _detect_changes_pelt(self, data: np.ndarray) -> List[int]:
        """Detect change points using PELT algorithm."""
        n_samples = data.shape[0]
        if n_samples < 2 * self.min_size:
            return []
        F = np.full(n_samples + 1, np.inf)
        F[0] = -self.penalty
        R = [0]
        change_points = []
        costs = np.zeros((n_samples, n_samples))
        for t1 in range(n_samples):
            for t2 in range(t1 + self.min_size, min(t1 + n_samples, n_samples) + 1):
                if t2 - t1 >= self.min_size:
                    segment_data = data[t1:t2]
                    if self.cost_function == 'l2':
                        if len(segment_data) > 0:
                            mean_val = np.mean(segment_data, axis=0)
                            costs[t1, t2 - 1] = np.sum((segment_data - mean_val) ** 2)
                    elif self.cost_function == 'l1':
                        if len(segment_data) > 0:
                            mean_val = np.mean(segment_data, axis=0)
                            costs[t1, t2 - 1] = np.sum(np.abs(segment_data - mean_val))
        for t in range(self.min_size, n_samples + 1):
            min_val = np.inf
            best_s = None
            for s in R:
                if t - s >= self.min_size:
                    segment_cost = costs[s, t - 1]
                    val = F[s] + segment_cost + self.penalty
                    if val < min_val:
                        min_val = val
                        best_s = s
            if min_val != np.inf:
                F[t] = min_val
                R_new = [s for s in R if F[s] + self.penalty <= F[t]]
                R_new.append(t)
                R = R_new
                if best_s is not None and t == n_samples:
                    temp_t = t
                    temp_s = best_s
                    temp_change_points = []
                    while temp_s > 0:
                        temp_change_points.append(temp_s)
                        found_prev = False
                        for prev_s in range(temp_s):
                            if F[prev_s] + costs[prev_s, temp_s - 1] + self.penalty == F[temp_s]:
                                temp_s = prev_s
                                found_prev = True
                                break
                        if not found_prev:
                            break
                    change_points = sorted(temp_change_points)
        change_points = [cp for cp in change_points if 0 <= cp < n_samples]
        return change_points

    def _detect_changes_binseg(self, data: np.ndarray) -> List[int]:
        """Detect change points using Binary Segmentation."""
        change_points = []
        n_samples = data.shape[0]
        segments = [(0, n_samples)]
        while segments:
            (start, end) = segments.pop(0)
            if end - start < 2 * self.min_size:
                continue
            best_cost_reduction = -np.inf
            best_idx = -1
            for i in range(start + self.min_size, end - self.min_size + 1, self.jump):
                left_data = data[start:i]
                right_data = data[i:end]
                if self.cost_function == 'l2':
                    left_mean = np.mean(left_data, axis=0)
                    right_mean = np.mean(right_data, axis=0)
                    left_cost = np.sum((left_data - left_mean) ** 2)
                    right_cost = np.sum((right_data - right_mean) ** 2)
                    total_split_cost = left_cost + right_cost
                    full_data = data[start:end]
                    full_mean = np.mean(full_data, axis=0)
                    no_split_cost = np.sum((full_data - full_mean) ** 2)
                    cost_reduction = no_split_cost - total_split_cost
                    if cost_reduction > best_cost_reduction:
                        best_cost_reduction = cost_reduction
                        best_idx = i
                elif self.cost_function == 'l1':
                    left_mean = np.mean(left_data, axis=0)
                    right_mean = np.mean(right_data, axis=0)
                    left_cost = np.sum(np.abs(left_data - left_mean))
                    right_cost = np.sum(np.abs(right_data - right_mean))
                    total_split_cost = left_cost + right_cost
                    full_data = data[start:end]
                    full_mean = np.mean(full_data, axis=0)
                    no_split_cost = np.sum(np.abs(full_data - full_mean))
                    cost_reduction = no_split_cost - total_split_cost
                    if cost_reduction > best_cost_reduction:
                        best_cost_reduction = cost_reduction
                        best_idx = i
            if best_idx != -1 and best_cost_reduction > self.penalty:
                change_points.append(best_idx)
                segments.append((start, best_idx))
                segments.append((best_idx, end))
        return sorted([cp for cp in change_points if 0 <= cp < n_samples])

    def _detect_changes_bottomup(self, data: np.ndarray) -> List[int]:
        """Detect change points using Bottom-Up segmentation."""
        n_samples = data.shape[0]
        change_points = []
        segments = list(range(0, n_samples, self.min_size))
        if segments[-1] != n_samples:
            segments.append(n_samples)
        segment_costs = {}
        for i in range(len(segments) - 1):
            (start, end) = (segments[i], segments[i + 1])
            segment_data = data[start:end]
            if self.cost_function == 'l2':
                mean_val = np.mean(segment_data, axis=0)
                cost = np.sum((segment_data - mean_val) ** 2)
            else:
                costs = self._compute_cost_matrix(segment_data)
                cost = np.sum(costs)
            segment_costs[start, end] = cost
        while len(segments) > 2:
            min_merge_cost = np.inf
            merge_idx = -1
            for i in range(len(segments) - 2):
                (left_start, mid, right_end) = (segments[i], segments[i + 1], segments[i + 2])
                separate_cost = segment_costs.get((left_start, mid), 0) + segment_costs.get((mid, right_end), 0)
                merged_data = data[left_start:right_end]
                if self.cost_function == 'l2':
                    mean_val = np.mean(merged_data, axis=0)
                    merge_cost = np.sum((merged_data - mean_val) ** 2)
                else:
                    costs = self._compute_cost_matrix(merged_data)
                    merge_cost = np.sum(costs)
                if merge_cost - separate_cost < min_merge_cost:
                    min_merge_cost = merge_cost - separate_cost
                    merge_idx = i + 1
            if min_merge_cost > self.penalty:
                break
            merge_point = segments[merge_idx]
            change_points.append(merge_point)
            segments.pop(merge_idx)
        return sorted(change_points)

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'ChangePointDetector':
        """
        Fit the change point detector to training data.
        
        This method prepares the detector by analyzing the statistical properties
        of the input time series data. For some methods, this might involve
        precomputing cost matrices or other algorithm-specific preparations.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Time series data to fit on. Can be:
            - DataBatch with time series data
            - FeatureSet with time series features
            - numpy array of shape (n_samples,) or (n_samples, n_features)
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        ChangePointDetector
            Self instance for method chaining
        """
        raw_data = self._extract_array(data)
        self._data = self._validate_data(raw_data)
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Detect change points in new time series data.
        
        Applies the fitted change point detection algorithm to identify points
        where statistical properties of the time series change significantly.
        The output includes the original data augmented with change point indicators.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Time series data to analyze for change points. Can be:
            - DataBatch with time series data
            - FeatureSet with time series features
            - numpy array of shape (n_samples,) or (n_samples, n_features)
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Feature set containing the original data with additional columns:
            - change_point_flags: Binary indicators for detected change points
            - segment_ids: Unique identifiers for segments between change points
        """
        raw_data = self._extract_array(data)
        processed_data = self._validate_data(raw_data)
        if self.method == 'pelt':
            change_points = self._detect_changes_pelt(processed_data)
        elif self.method == 'binseg':
            change_points = self._detect_changes_binseg(processed_data)
        elif self.method == 'bottomup':
            change_points = self._detect_changes_bottomup(processed_data)
        else:
            raise ValueError(f'Unknown method: {self.method}')
        self._change_points = change_points
        n_samples = processed_data.shape[0]
        change_point_flags = np.zeros(n_samples, dtype=int)
        change_point_flags[change_points] = 1
        segment_ids = np.zeros(n_samples, dtype=int)
        current_segment = 0
        for i in range(n_samples):
            if i in change_points:
                current_segment += 1
            segment_ids[i] = current_segment
        import pandas as pd
        feature_dict = {}
        if raw_data.ndim == 1:
            feature_dict['original_data'] = raw_data
        else:
            for i in range(raw_data.shape[1]):
                feature_dict[f'feature_{i}'] = raw_data[:, i]
        feature_dict['change_point_flags'] = change_point_flags
        feature_dict['segment_ids'] = segment_ids
        features_df = pd.DataFrame(feature_dict)
        features_array = features_df.values
        feature_names = list(features_df.columns)
        return FeatureSet(features=features_array, feature_names=feature_names)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the transformation (returns original data without change point annotations).
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data with change point annotations
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        FeatureSet
            Original time series data without change point annotations
        """
        if isinstance(data, FeatureSet):
            if hasattr(data.features, 'columns'):
                columns_to_remove = ['change_point_flags', 'segment_ids']
                original_columns = [col for col in data.features.columns if col not in columns_to_remove]
                original_features = data.features[original_columns]
                features_array = original_features.values
                feature_names = list(original_features.columns)
            elif data.features.shape[1] >= 2:
                original_features = data.features[:, :-2]
                features_array = original_features
                feature_names = data.feature_names[:-2] if data.feature_names else [f'feature_{i}' for i in range(original_features.shape[1])]
            else:
                features_array = data.features
                feature_names = data.feature_names if data.feature_names else [f'feature_{i}' for i in range(data.features.shape[1])]
            return FeatureSet(features=features_array, feature_names=feature_names, target=data.target)
        else:
            raw_data = self._extract_array(data)
            import pandas as pd
            if raw_data.ndim == 1:
                df = pd.DataFrame({'original_data': raw_data})
            else:
                df = pd.DataFrame(raw_data, columns=[f'feature_{i}' for i in range(raw_data.shape[1])])
            features_array = df.values
            feature_names = list(df.columns)
            return FeatureSet(features=features_array, feature_names=feature_names)

    def get_change_points(self) -> List[int]:
        """
        Retrieve the indices of detected change points.
        
        Returns
        -------
        List[int]
            List of indices where change points were detected in the last transformed data
        """
        if hasattr(self, '_change_points'):
            return self._change_points.copy()
        else:
            return []