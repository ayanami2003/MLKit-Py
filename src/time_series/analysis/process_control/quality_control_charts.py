from typing import Optional, Union, List, Dict, Any
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import numpy as np
import math


class ControlChartGenerator(BaseTransformer):

    def __init__(self, chart_type: str='x_bar_r', sample_size: int=5, confidence_level: float=0.9973, target_column: str='value', name: Optional[str]=None):
        """
        Initialize the ControlChartGenerator.
        
        Parameters
        ----------
        chart_type : str, optional
            Type of control chart to generate. Options:
            - 'x_bar_r': X-bar and R chart (default)
            - 'x_bar_s': X-bar and S chart
            - 'individuals': Individuals and Moving Range chart
            - 'p': p-chart for fraction defective
            - 'np': np-chart for number defective
            - 'c': c-chart for count of defects
            - 'u': u-chart for defects per unit
        sample_size : int, optional
            Number of observations per subgroup (default=5)
        confidence_level : float, optional
            Confidence level for control limits (default=0.9973, ~3-sigma)
        target_column : str, optional
            Name of column to monitor (default='value')
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.chart_type = chart_type
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.target_column = target_column
        self._control_limits: Dict[str, float] = {}
        self._chart_data: Dict[str, np.ndarray] = {}

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'ControlChartGenerator':
        """
        Fit the control chart generator to the data.
        
        Calculates control limits based on the provided data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data to fit control limits
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        ControlChartGenerator
            Self instance for method chaining
        """
        if isinstance(data, DataBatch):
            values = data.get_feature(self.target_column).values
        else:
            values = data.get_feature(self.target_column).values
        if self.chart_type in ['x_bar_r', 'x_bar_s']:
            n_subgroups = len(values) // self.sample_size
            trimmed_values = values[:n_subgroups * self.sample_size]
            subgroups = trimmed_values.reshape(n_subgroups, self.sample_size)
            subgroup_means = np.mean(subgroups, axis=1)
            subgroup_ranges = np.max(subgroups, axis=1) - np.min(subgroups, axis=1)
            x_bar = np.mean(subgroup_means)
            r_bar = np.mean(subgroup_ranges)
            A2 = self._get_control_chart_constant('A2', self.sample_size)
            D3 = self._get_control_chart_constant('D3', self.sample_size)
            D4 = self._get_control_chart_constant('D4', self.sample_size)
            self._control_limits['x_bar_center'] = x_bar
            self._control_limits['x_bar_ucl'] = x_bar + A2 * r_bar
            self._control_limits['x_bar_lcl'] = x_bar - A2 * r_bar
            self._control_limits['r_center'] = r_bar
            self._control_limits['r_ucl'] = D4 * r_bar
            self._control_limits['r_lcl'] = D3 * r_bar
            self._chart_data['subgroup_means'] = subgroup_means
            self._chart_data['subgroup_ranges'] = subgroup_ranges
            if self.chart_type == 'x_bar_s':
                subgroup_stds = np.std(subgroups, axis=1, ddof=1)
                s_bar = np.mean(subgroup_stds)
                B3 = self._get_control_chart_constant('B3', self.sample_size)
                B4 = self._get_control_chart_constant('B4', self.sample_size)
                self._control_limits['s_center'] = s_bar
                self._control_limits['s_ucl'] = B4 * s_bar
                self._control_limits['s_lcl'] = B3 * s_bar
                self._chart_data['subgroup_stds'] = subgroup_stds
        elif self.chart_type == 'individuals':
            mr = np.abs(np.diff(values))
            mr_bar = np.mean(mr)
            x_bar = np.mean(values)
            self._control_limits['i_center'] = x_bar
            self._control_limits['i_ucl'] = x_bar + 3 * mr_bar / 1.128
            self._control_limits['i_lcl'] = x_bar - 3 * mr_bar / 1.128
            self._control_limits['mr_center'] = mr_bar
            self._control_limits['mr_ucl'] = 3.267 * mr_bar
            self._control_limits['mr_lcl'] = 0
            self._chart_data['values'] = values
            self._chart_data['moving_ranges'] = mr
        elif self.chart_type in ['p', 'np']:
            n_subgroups = len(values) // self.sample_size
            trimmed_values = values[:n_subgroups * self.sample_size]
            subgroups = trimmed_values.reshape(n_subgroups, self.sample_size)
            if self.chart_type == 'p':
                subgroup_proportions = np.mean(subgroups, axis=1)
                p_bar = np.mean(subgroup_proportions)
                self._control_limits['p_center'] = p_bar
                std_p = np.sqrt(p_bar * (1 - p_bar) / self.sample_size)
                self._control_limits['p_ucl'] = p_bar + 3 * std_p
                self._control_limits['p_lcl'] = max(0, p_bar - 3 * std_p)
                self._chart_data['subgroup_proportions'] = subgroup_proportions
            else:
                subgroup_defects = np.sum(subgroups, axis=1)
                np_bar = np.mean(subgroup_defects)
                self._control_limits['np_center'] = np_bar
                std_np = np.sqrt(np_bar * (1 - np_bar / self.sample_size))
                self._control_limits['np_ucl'] = np_bar + 3 * std_np
                self._control_limits['np_lcl'] = max(0, np_bar - 3 * std_np)
                self._chart_data['subgroup_defects'] = subgroup_defects
        elif self.chart_type in ['c', 'u']:
            if self.chart_type == 'c':
                c_bar = np.mean(values)
                self._control_limits['c_center'] = c_bar
                self._control_limits['c_ucl'] = c_bar + 3 * np.sqrt(c_bar)
                self._control_limits['c_lcl'] = max(0, c_bar - 3 * np.sqrt(c_bar))
                self._chart_data['defect_counts'] = values
            else:
                u_bar = np.mean(values)
                self._control_limits['u_center'] = u_bar
                self._control_limits['u_ucl'] = u_bar + 3 * np.sqrt(u_bar / self.sample_size)
                self._control_limits['u_lcl'] = max(0, u_bar - 3 * np.sqrt(u_bar / self.sample_size))
                self._chart_data['defect_rates'] = values
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Generate control chart statistics and signals.
        
        Creates control chart data including center lines, control limits,
        and out-of-control signals.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data to analyze
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        FeatureSet
            Control chart data with statistics and signals
        """
        ooc_indices = self.detect_out_of_control_points()
        result_features = {}
        for (key, value) in self._chart_data.items():
            result_features[key] = value
        for (key, value) in self._control_limits.items():
            result_features[f'limit_{key}'] = np.full(len(list(self._chart_data.values())[0]), value)
        signals = np.zeros(len(list(self._chart_data.values())[0]))
        signals[ooc_indices] = 1
        result_features['out_of_control_signals'] = signals
        return FeatureSet(features=result_features)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transform is not applicable for control charts.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Unmodified input data
        """
        if isinstance(data, DataBatch):
            features = {}
            for feature_name in data.get_feature_names():
                features[feature_name] = data.get_feature(feature_name).values
            return FeatureSet(features=features)
        return data

    def get_control_limits(self) -> Dict[str, float]:
        """
        Get the calculated control limits.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing control limit values
        """
        return self._control_limits.copy()

    def get_chart_data(self) -> Dict[str, np.ndarray]:
        """
        Get the generated control chart data.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing chart data arrays
        """
        return {k: v.copy() for (k, v) in self._chart_data.items()}

    def detect_out_of_control_points(self) -> List[int]:
        """
        Detect points that are out of control.
        
        Uses Western Electric rules for detecting out-of-control conditions.
        
        Returns
        -------
        List[int]
            Indices of out-of-control points
        """
        if not self._control_limits:
            raise ValueError('Control limits not calculated. Call fit() first.')
        ooc_indices = []
        if self.chart_type in ['x_bar_r', 'x_bar_s']:
            values = self._chart_data['subgroup_means']
            center = self._control_limits['x_bar_center']
            ucl = self._control_limits['x_bar_ucl']
            lcl = self._control_limits['x_bar_lcl']
        elif self.chart_type == 'individuals':
            values = self._chart_data['values']
            center = self._control_limits['i_center']
            ucl = self._control_limits['i_ucl']
            lcl = self._control_limits['i_lcl']
        elif self.chart_type == 'p':
            values = self._chart_data['subgroup_proportions']
            center = self._control_limits['p_center']
            ucl = self._control_limits['p_ucl']
            lcl = self._control_limits['p_lcl']
        elif self.chart_type == 'np':
            values = self._chart_data['subgroup_defects']
            center = self._control_limits['np_center']
            ucl = self._control_limits['np_ucl']
            lcl = self._control_limits['np_lcl']
        elif self.chart_type == 'c':
            values = self._chart_data['defect_counts']
            center = self._control_limits['c_center']
            ucl = self._control_limits['c_ucl']
            lcl = self._control_limits['c_lcl']
        elif self.chart_type == 'u':
            values = self._chart_data['defect_rates']
            center = self._control_limits['u_center']
            ucl = self._control_limits['u_ucl']
            lcl = self._control_limits['u_lcl']
        else:
            return []
        for (i, val) in enumerate(values):
            if val > ucl or val < lcl:
                ooc_indices.append(i)
        return sorted(list(set(ooc_indices)))

    def _get_control_chart_constant(self, constant_name: str, sample_size: int) -> float:
        """
        Get control chart constants based on sample size.
        
        Parameters
        ----------
        constant_name : str
            Name of the constant (A2, D3, D4, B3, B4)
        sample_size : int
            Sample size for the subgroup
            
        Returns
        -------
        float
            Control chart constant value
        """
        constants = {'A2': {2: 1.88, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}, 'D3': {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}, 'D4': {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}, 'B3': {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.03, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284}, 'B4': {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.97, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}}
        if constant_name in constants and sample_size in constants[constant_name]:
            return constants[constant_name][sample_size]
        elif constant_name == 'A2':
            return 3 / (np.sqrt(sample_size) * self._get_d2_constant(sample_size))
        elif constant_name == 'D3':
            d2 = self._get_d2_constant(sample_size)
            d3 = self._get_d3_constant(sample_size)
            return max(0, 1 - 3 * d3 / d2)
        elif constant_name == 'D4':
            d2 = self._get_d2_constant(sample_size)
            d3 = self._get_d3_constant(sample_size)
            return 1 + 3 * d3 / d2
        elif constant_name == 'B3':
            c4 = self._get_c4_constant(sample_size)
            return max(0, 1 - 3 * np.sqrt(1 - c4 ** 2) / c4)
        elif constant_name == 'B4':
            c4 = self._get_c4_constant(sample_size)
            return 1 + 3 * np.sqrt(1 - c4 ** 2) / c4
        else:
            return 0

    def _get_d2_constant(self, sample_size: int) -> float:
        """Approximate d2 constant for sample size."""
        if sample_size <= 10:
            d2_values = {1: 0, 2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.97, 10: 3.078}
            return d2_values[sample_size]
        else:
            return np.sqrt(2) * np.math.gamma((sample_size + 1) / 2) / np.math.gamma(sample_size / 2)

    def _get_d3_constant(self, sample_size: int) -> float:
        """Approximate d3 constant for sample size."""
        if sample_size <= 10:
            d3_values = {1: 0, 2: 0.853, 3: 0.888, 4: 0.88, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.82, 9: 0.808, 10: 0.797}
            return d3_values[sample_size]
        else:
            d2 = self._get_d2_constant(sample_size)
            return np.sqrt(np.math.gamma(sample_size / 2) * np.math.gamma((sample_size - 2) / 2) / (2 * np.math.gamma((sample_size - 1) / 2) ** 2) - d2 ** 2)

    def _get_c4_constant(self, sample_size: int) -> float:
        """Calculate c4 constant for sample size."""
        if sample_size <= 1:
            return 1.0
        return np.sqrt(2 / (sample_size - 1)) * np.math.gamma(sample_size / 2) / np.math.gamma((sample_size - 1) / 2)