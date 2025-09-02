from typing import Union, Optional, Tuple
import numpy as np
from general.structures.data_batch import DataBatch
from scipy import stats

class ZScoreOutlierDetector:
    """
    Detect outliers using various z-score based methods.
    
    This class implements multiple z-score approaches for identifying outliers:
    - Standard z-score (based on mean and standard deviation)
    - Modified z-score (based on median and MAD)
    - Advanced z-score variations
    
    The detector can identify outliers and optionally remove them from the dataset.
    
    Attributes:
        threshold (float): Threshold value for identifying outliers (default: 3.0)
        method (str): Z-score method to use ('standard', 'modified', 'advanced')
        axis (Optional[int]): Axis along which to compute z-scores
        
    Methods:
        fit: Compute necessary statistics for z-score calculation
        detect: Identify outliers in the data
        remove_outliers: Remove detected outliers from the data
    """

    def __init__(self, threshold: float=3.0, method: str='standard', axis: Optional[int]=None):
        """
        Initialize the ZScoreOutlierDetector.
        
        Args:
            threshold (float): Threshold for identifying outliers. Points with 
                             |z-score| > threshold are considered outliers.
            method (str): Method to use for z-score calculation:
                         'standard' - uses mean and standard deviation
                         'modified' - uses median and median absolute deviation
                         'advanced' - uses robust estimation techniques
            axis (Optional[int]): Axis along which to compute z-scores.
                                If None, computes over the flattened array.
                                
        Raises:
            ValueError: If method is not one of 'standard', 'modified', or 'advanced'
        """
        self.threshold = threshold
        self.method = method
        self.axis = axis
        self._fit_statistics = {}
        if method not in ['standard', 'modified', 'advanced']:
            raise ValueError("Method must be one of 'standard', 'modified', or 'advanced'")

    def fit(self, data: Union[np.ndarray, DataBatch]) -> 'ZScoreOutlierDetector':
        """
        Compute statistics needed for z-score calculations.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to fit statistics on
            
        Returns:
            ZScoreOutlierDetector: Returns self for method chaining
            
        Raises:
            ValueError: If data is empty or has incompatible dimensions
        """
        if isinstance(data, DataBatch):
            data_array = data.data
        else:
            data_array = data
        if not isinstance(data_array, np.ndarray):
            data_array = np.asarray(data_array)
        if data_array.size == 0:
            raise ValueError('Input data is empty')
        if self.method == 'standard':
            mean = np.mean(data_array, axis=self.axis, keepdims=True)
            std = np.std(data_array, axis=self.axis, keepdims=True)
            std = np.where(std == 0, 1e-10, std)
            self._fit_statistics = {'mean': mean, 'std': std}
        elif self.method == 'modified':
            median = np.median(data_array, axis=self.axis, keepdims=True)
            mad = np.median(np.abs(data_array - median), axis=self.axis, keepdims=True)
            mad_scaled = np.where(mad == 0, 1e-10, mad)
            self._fit_statistics = {'median': median, 'mad': mad_scaled}
        elif self.method == 'advanced':
            median = np.median(data_array, axis=self.axis, keepdims=True)
            mad = np.median(np.abs(data_array - median), axis=self.axis, keepdims=True)
            q75 = np.percentile(data_array, 75, axis=self.axis, keepdims=True)
            q25 = np.percentile(data_array, 25, axis=self.axis, keepdims=True)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1e-10, iqr)
            mad_scaled = np.where(mad == 0, 1e-10, mad)
            self._fit_statistics = {'median': median, 'mad': mad_scaled, 'iqr': iqr, 'q75': q75, 'q25': q25}
        return self

    def detect(self, data: Union[np.ndarray, DataBatch], return_scores: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect outliers in the data using the configured z-score method.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to analyze for outliers
            return_scores (bool): If True, return both outliers and z-scores
            
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Boolean array indicating 
            which points are outliers, or tuple of (outliers, z_scores) if return_scores=True
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if not self._fit_statistics:
            raise RuntimeError("Detector has not been fitted. Call 'fit' first.")
        if isinstance(data, DataBatch):
            data_array = data.data
        else:
            data_array = data
        if not isinstance(data_array, np.ndarray):
            data_array = np.asarray(data_array)
        if self.method == 'standard':
            mean = self._fit_statistics['mean']
            std = self._fit_statistics['std']
            z_scores = (data_array - mean) / std
        elif self.method == 'modified':
            median = self._fit_statistics['median']
            mad = self._fit_statistics['mad']
            z_scores = 0.6745 * (data_array - median) / mad
        elif self.method == 'advanced':
            median = self._fit_statistics['median']
            mad = self._fit_statistics['mad']
            z_scores = 0.6745 * (data_array - median) / mad
        outliers = np.abs(z_scores) > self.threshold
        if return_scores:
            return (outliers, z_scores)
        else:
            return outliers

    def remove_outliers(self, data: Union[np.ndarray, DataBatch]) -> Union[np.ndarray, DataBatch]:
        """
        Remove detected outliers from the data.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to clean
            
        Returns:
            Union[np.ndarray, DataBatch]: Data with outliers removed
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if not self._fit_statistics:
            raise RuntimeError("Detector has not been fitted. Call 'fit' first.")
        if isinstance(data, DataBatch):
            data_array = data.data
            is_databatch = True
            sample_ids = data.sample_ids
            feature_names = data.feature_names
            metadata = data.metadata
        else:
            data_array = data
            is_databatch = False
            sample_ids = None
            feature_names = None
            metadata = None
        if not isinstance(data_array, np.ndarray):
            data_array = np.asarray(data_array)
        outliers = self.detect(data)
        flat_data = data_array.reshape(-1, data_array.shape[-1]) if data_array.ndim > 1 else data_array.flatten()
        flat_outliers = outliers.flatten()
        mask = ~flat_outliers
        cleaned_data = flat_data[mask]
        if data_array.ndim > 1 and len(cleaned_data) > 0:
            try:
                if data_array.shape[-1] > 0:
                    cleaned_data = cleaned_data.reshape(-1, data_array.shape[-1])
            except ValueError:
                pass
        if is_databatch:
            new_sample_ids = None
            if sample_ids is not None:
                flat_sample_ids = np.array(sample_ids).flatten() if hasattr(sample_ids, '__len__') else np.array([sample_ids])
                new_sample_ids = flat_sample_ids[mask][:len(cleaned_data)] if len(flat_sample_ids) == len(flat_outliers) else None
            return DataBatch(data=cleaned_data, sample_ids=new_sample_ids, feature_names=feature_names, metadata=metadata)
        else:
            return cleaned_data


# ...(code omitted)...


class GrubbsOutlierDetector:
    """
    Detect outliers using Grubbs' test (Extreme Studentized Deviate test).
    
    Grubbs' test is used to detect a single outlier in a univariate dataset 
    that follows an approximately normal distribution. The test can be repeated 
    to detect multiple outliers, but this implementation focuses on single outlier detection.
    
    Attributes:
        alpha (float): Significance level for the test (default: 0.05)
        two_sided (bool): Whether to test for outliers on both tails (default: True)
        
    Methods:
        fit: Compute necessary statistics for Grubbs' test
        detect: Identify outliers using Grubbs' test
        transform: Remove detected outliers from the data
        get_critical_value: Calculate the critical value for the test
    """

    def __init__(self, alpha: float=0.05, two_sided: bool=True):
        """
        Initialize the GrubbsOutlierDetector.
        
        Args:
            alpha (float): Significance level for the test (default: 0.05)
            two_sided (bool): Whether to perform a two-sided test (default: True)
        """
        self.alpha = alpha
        self.two_sided = two_sided
        self.mean_ = None
        self.std_ = None
        self.n_ = None

    def _extract_values(self, data: Union[np.ndarray, DataBatch]) -> np.ndarray:
        """
        Extract values from either numpy array or DataBatch.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data
            
        Returns:
            np.ndarray: Extracted values as numpy array
        """
        if isinstance(data, DataBatch):
            if hasattr(data, 'data'):
                return data.data
            elif hasattr(data, 'values'):
                return data.values
            elif hasattr(data, 'array'):
                return data.array
            else:
                return np.asarray(data)
        else:
            return data

    def fit(self, data: Union[np.ndarray, DataBatch]) -> 'GrubbsOutlierDetector':
        """
        Compute statistics needed for Grubbs' test.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to fit statistics on
            
        Returns:
            GrubbsOutlierDetector: Returns self for method chaining
            
        Raises:
            ValueError: If standard deviation is zero or data has less than 3 points
        """
        values = self._extract_values(data)
        flat_data = np.asarray(values).flatten()
        if len(flat_data) < 3:
            raise ValueError("Grubbs' test requires at least 3 data points")
        self.mean_ = np.mean(flat_data)
        self.std_ = np.std(flat_data, ddof=1)
        self.n_ = len(flat_data)
        if self.std_ == 0:
            raise ValueError('Standard deviation is zero. All values are identical.')
        return self

    def get_critical_value(self) -> float:
        """
        Calculate the critical value for Grubbs' test.
        
        Returns:
            float: Critical value for Grubbs' test
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.mean_ is None or self.std_ is None or self.n_ is None:
            raise RuntimeError('not fitted')
        df = self.n_ - 2
        if self.two_sided:
            t_critical = stats.t.isf(self.alpha / (2 * self.n_), df)
        else:
            t_critical = stats.t.isf(self.alpha / self.n_, df)
        numerator = (self.n_ - 1) * np.sqrt(t_critical ** 2)
        denominator = np.sqrt(self.n_ * (df + t_critical ** 2))
        critical_value = numerator / denominator
        return critical_value

    def detect(self, data: Union[np.ndarray, DataBatch]) -> np.ndarray:
        """
        Detect outliers in the data using Grubbs' test.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to analyze for outliers
            
        Returns:
            np.ndarray: Boolean array indicating which points are outliers
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.mean_ is None or self.std_ is None or self.n_ is None:
            raise RuntimeError('not fitted')
        values = self._extract_values(data)
        flat_data = np.asarray(values).flatten()
        if self.two_sided:
            deviations = np.abs(flat_data - self.mean_)
            test_statistic = np.max(deviations) / self.std_
        else:
            normalized_data = (flat_data - self.mean_) / self.std_
            if np.max(normalized_data) > np.abs(np.min(normalized_data)):
                test_statistic = np.max(normalized_data)
            else:
                test_statistic = np.abs(np.min(normalized_data))
        critical_value = self.get_critical_value()
        is_outlier = test_statistic > critical_value
        if is_outlier:
            if self.two_sided:
                outlier_mask = np.abs(flat_data - self.mean_) == np.max(np.abs(flat_data - self.mean_))
            else:
                normalized_data = (flat_data - self.mean_) / self.std_
                max_deviation = np.max(normalized_data)
                min_deviation = np.min(normalized_data)
                if max_deviation > np.abs(min_deviation):
                    outlier_mask = normalized_data == max_deviation
                else:
                    outlier_mask = normalized_data == min_deviation
        else:
            outlier_mask = np.zeros_like(flat_data, dtype=bool)
        return outlier_mask.reshape(values.shape)

    def transform(self, data: Union[np.ndarray, DataBatch]) -> Union[np.ndarray, DataBatch]:
        """
        Remove detected outliers from the data.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to clean
            
        Returns:
            Union[np.ndarray, DataBatch]: Data with outliers removed
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.mean_ is None or self.std_ is None or self.n_ is None:
            raise RuntimeError('not fitted')
        outlier_mask = self.detect(data)
        if isinstance(data, DataBatch):
            values = self._extract_values(data)
            is_databatch = True
        else:
            values = data
            is_databatch = False
        flat_data = np.asarray(values).flatten()
        flat_mask = outlier_mask.flatten()
        filtered_values = flat_data[~flat_mask]
        if is_databatch:
            return DataBatch(filtered_values)
        elif len(filtered_values) == 0:
            return np.array([])
        elif values.ndim == 1:
            return filtered_values
        else:
            return filtered_values.reshape(-1) if len(filtered_values) < flat_data.size else filtered_values.reshape(values.shape)

class DixonsQOutlierDetector:
    """
    Detect outliers using Dixon's Q test.
    
    Dixon's Q test is a statistical test for identifying outliers in small datasets
    (typically less than 30 observations). It tests the significance of the gap 
    between the suspected outlier and its nearest neighbor compared to the range 
    of the entire dataset.
    
    Attributes:
        confidence_level (float): Confidence level for the test (default: 0.95)
        left_tail (bool): Test for outliers on the left tail (minimum values)
        right_tail (bool): Test for outliers on the right tail (maximum values)
        
    Methods:
        fit: Prepare the detector for outlier testing
        detect: Identify outliers using Dixon's Q test
        transform: Remove detected outliers from the data
        get_q_statistic: Calculate the Q statistic for the data
        get_critical_q_value: Get the critical Q value for the specified confidence level
    """

    def __init__(self, confidence_level: float=0.95, left_tail: bool=True, right_tail: bool=True):
        """
        Initialize the DixonsQOutlierDetector.
        
        Args:
            confidence_level (float): Confidence level for the test (between 0 and 1)
            left_tail (bool): If True, test minimum values for outliers
            right_tail (bool): If True, test maximum values for outliers
        """
        self.confidence_level = confidence_level
        self.left_tail = left_tail
        self.right_tail = right_tail
        self.n_ = None

    def fit(self, data: Union[np.ndarray, DataBatch]) -> 'DixonsQOutlierDetector':
        """
        Fit the detector to the data by recording the sample size.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to fit on
            
        Returns:
            DixonsQOutlierDetector: Returns self for method chaining
            
        Raises:
            ValueError: If data has less than 3 points or more than 30 points
        """
        if isinstance(data, DataBatch):
            data_array = data.data.flatten() if hasattr(data.data, 'flatten') else np.asarray(data.data).flatten()
        else:
            data_array = np.asarray(data).flatten()
        n = len(data_array)
        if n < 3:
            raise ValueError("Dixon's Q test requires at least 3 data points")
        if n > 30:
            raise ValueError("Dixon's Q test is not recommended for more than 30 data points")
        self.n_ = n
        return self

    def _extract_values(self, data: Union[np.ndarray, DataBatch]) -> np.ndarray:
        """
        Extract values from either numpy array or DataBatch.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data
            
        Returns:
            np.ndarray: Extracted values as numpy array
        """
        if isinstance(data, DataBatch):
            if hasattr(data, 'data'):
                return data.data
            elif hasattr(data, 'values'):
                return data.values
            elif hasattr(data, 'array'):
                return data.array
            else:
                return np.asarray(data)
        else:
            return data

    def _calculate_q_left(self, sorted_data: np.ndarray) -> float:
        """
        Calculate Q statistic for the leftmost (minimum) value.
        
        Args:
            sorted_data (np.ndarray): Sorted data array
            
        Returns:
            float: Q statistic for the left tail
        """
        if len(sorted_data) < 2:
            return 0.0
        gap = sorted_data[1] - sorted_data[0]
        range_data = sorted_data[-1] - sorted_data[0]
        return gap / range_data if range_data > 0 else 0.0

    def _calculate_q_right(self, sorted_data: np.ndarray) -> float:
        """
        Calculate Q statistic for the rightmost (maximum) value.
        
        Args:
            sorted_data (np.ndarray): Sorted data array
            
        Returns:
            float: Q statistic for the right tail
        """
        if len(sorted_data) < 2:
            return 0.0
        gap = sorted_data[-1] - sorted_data[-2]
        range_data = sorted_data[-1] - sorted_data[0]
        return gap / range_data if range_data > 0 else 0.0

    def get_q_statistic(self, data: Union[np.ndarray, DataBatch]) -> Union[float, Tuple[float, float]]:
        """
        Calculate the Q statistic for the data.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data
            
        Returns:
            Union[float, Tuple[float, float]]: Q statistic(s) for the specified tail(s)
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.n_ is None:
            raise RuntimeError("Detector not fitted. Call 'fit()' before using this method.")
        values = self._extract_values(data)
        flat_data = np.asarray(values).flatten()
        sorted_data = np.sort(flat_data)
        if self.left_tail and self.right_tail:
            q_left = self._calculate_q_left(sorted_data)
            q_right = self._calculate_q_right(sorted_data)
            return (q_left, q_right)
        elif self.left_tail:
            return self._calculate_q_left(sorted_data)
        elif self.right_tail:
            return self._calculate_q_right(sorted_data)
        else:
            raise ValueError('At least one tail must be enabled for testing')

    def get_critical_q_value(self) -> float:
        """
        Get the critical Q value for the specified confidence level.
        
        Returns:
            float: Critical Q value
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.n_ is None:
            raise RuntimeError("Detector not fitted. Call 'fit()' before using this method.")
        q_table = {0.9: {3: 0.941, 4: 0.765, 5: 0.642, 6: 0.56, 7: 0.507, 8: 0.468, 9: 0.437, 10: 0.409, 11: 0.386, 12: 0.366, 13: 0.349, 14: 0.334, 15: 0.321, 16: 0.309, 17: 0.298, 18: 0.288, 19: 0.279, 20: 0.27, 25: 0.239, 30: 0.218}, 0.95: {3: 0.97, 4: 0.829, 5: 0.71, 6: 0.625, 7: 0.568, 8: 0.526, 9: 0.493, 10: 0.466, 11: 0.443, 12: 0.423, 13: 0.404, 14: 0.388, 15: 0.374, 16: 0.361, 17: 0.349, 18: 0.338, 19: 0.328, 20: 0.319, 25: 0.285, 30: 0.259}, 0.99: {3: 0.994, 4: 0.926, 5: 0.821, 6: 0.74, 7: 0.68, 8: 0.634, 9: 0.598, 10: 0.568, 11: 0.542, 12: 0.52, 13: 0.5, 14: 0.483, 15: 0.467, 16: 0.453, 17: 0.44, 18: 0.428, 19: 0.417, 20: 0.407, 25: 0.361, 30: 0.327}}
        alpha = 1 - self.confidence_level
        standard_levels = [0.9, 0.95, 0.99]
        closest_level = min(standard_levels, key=lambda x: abs(1 - x - alpha))
        table = q_table[closest_level]
        if self.n_ in table:
            return table[self.n_]
        elif self.n_ < 3:
            return table[3]
        elif self.n_ > 30:
            return table[30]
        else:
            keys = sorted(table.keys())
            for i in range(len(keys) - 1):
                if keys[i] <= self.n_ <= keys[i + 1]:
                    (x1, x2) = (keys[i], keys[i + 1])
                    (y1, y2) = (table[x1], table[x2])
                    return y1 + (y2 - y1) * (self.n_ - x1) / (x2 - x1)
            closest_n = min(keys, key=lambda x: abs(x - self.n_))
            return table[closest_n]

    def detect(self, data: Union[np.ndarray, DataBatch]) -> np.ndarray:
        """
        Detect outliers in the data using Dixon's Q test.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to analyze for outliers
            
        Returns:
            np.ndarray: Boolean array indicating which points are outliers
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.n_ is None:
            raise RuntimeError("Detector not fitted. Call 'fit()' before using this method.")
        values = self._extract_values(data)
        original_shape = values.shape
        flat_data = np.asarray(values).flatten()
        sorted_indices = np.argsort(flat_data)
        sorted_data = flat_data[sorted_indices]
        outlier_mask = np.zeros(len(flat_data), dtype=bool)
        critical_q = self.get_critical_q_value()
        if self.left_tail:
            q_left = self._calculate_q_left(sorted_data)
            if q_left > critical_q:
                outlier_mask[sorted_indices[0]] = True
        if self.right_tail:
            q_right = self._calculate_q_right(sorted_data)
            if q_right > critical_q:
                outlier_mask[sorted_indices[-1]] = True
        return outlier_mask.reshape(original_shape)

    def transform(self, data: Union[np.ndarray, DataBatch]) -> Union[np.ndarray, DataBatch]:
        """
        Remove detected outliers from the data.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to clean
            
        Returns:
            Union[np.ndarray, DataBatch]: Data with outliers removed
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.n_ is None:
            raise RuntimeError("Detector not fitted. Call 'fit()' before using this method.")
        outlier_mask = self.detect(data)
        if isinstance(data, DataBatch):
            values = self._extract_values(data)
            is_databatch = True
        else:
            values = data
            is_databatch = False
        flat_data = np.asarray(values).flatten()
        flat_mask = outlier_mask.flatten()
        filtered_values = flat_data[~flat_mask]
        if is_databatch:
            if len(filtered_values) == 0:
                result_data = np.array([]).reshape(0, 1) if values.ndim > 1 else np.array([])
            elif values.ndim == 1:
                result_data = filtered_values
            else:
                result_data = filtered_values.reshape(-1, values.shape[1]) if values.shape[1] > 1 else filtered_values.reshape(-1, 1)
            return DataBatch(result_data)
        elif len(filtered_values) == 0:
            return np.array([])
        elif values.ndim == 1:
            return filtered_values
        elif values.shape[1] > 1:
            return filtered_values.reshape(-1, values.shape[1])
        else:
            return filtered_values.reshape(-1, 1)

class ChauvenetsCriterionOutlierDetector:
    """
    Detect outliers using Chauvenet's criterion.
    
    Chauvenet's criterion is a statistical test that states that a data point 
    is an outlier if the probability of observing a value as extreme as that 
    data point is less than 1/(2n), where n is the sample size. It assumes 
    the data follows a normal distribution.
    
    Attributes:
        discard_opposite (bool): Whether to discard outliers on both tails (default: True)
        
    Methods:
        fit: Compute statistics needed for Chauvenet's criterion
        detect: Identify outliers using Chauvenet's criterion
        transform: Remove detected outliers from the data
        get_probability_threshold: Calculate the probability threshold for outlier detection
    """

    def __init__(self, discard_opposite: bool=True):
        """
        Initialize the ChauvenetsCriterionOutlierDetector.
        
        Args:
            discard_opposite (bool): If True, consider outliers on both tails;
                                   if False, only consider one tail
        """
        self.discard_opposite = discard_opposite
        self.mean_ = None
        self.std_ = None
        self.n_ = None

    def fit(self, data: Union[np.ndarray, DataBatch]) -> 'ChauvenetsCriterionOutlierDetector':
        """
        Compute statistics needed for Chauvenet's criterion.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to fit statistics on
            
        Returns:
            ChauvenetsCriterionOutlierDetector: Returns self for method chaining
        """
        if isinstance(data, DataBatch):
            raw_data = data.data
        else:
            raw_data = data
        flat_data = raw_data.flatten()
        self.mean_ = np.mean(flat_data)
        self.std_ = np.std(flat_data)
        self.n_ = len(flat_data)
        return self

    def detect(self, data: Union[np.ndarray, DataBatch]) -> np.ndarray:
        """
        Detect outliers in the data using Chauvenet's criterion.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to analyze for outliers
            
        Returns:
            np.ndarray: Boolean array indicating which points are outliers
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.mean_ is None or self.std_ is None or self.n_ is None:
            raise RuntimeError("Detector has not been fitted. Call 'fit' method first.")
        if isinstance(data, DataBatch):
            raw_data = data.data
        else:
            raw_data = data
        z_scores = np.abs((raw_data - self.mean_) / self.std_)
        prob_threshold = self.get_probability_threshold()
        if self.discard_opposite:
            probabilities = 2 * (1 - stats.norm.cdf(z_scores))
        else:
            probabilities = 1 - stats.norm.cdf(z_scores)
        outliers = probabilities < prob_threshold
        return outliers

    def transform(self, data: Union[np.ndarray, DataBatch]) -> Union[np.ndarray, DataBatch]:
        """
        Remove detected outliers from the data.
        
        Args:
            data (Union[np.ndarray, DataBatch]): Input data to clean
            
        Returns:
            Union[np.ndarray, DataBatch]: Data with outliers removed
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.mean_ is None or self.std_ is None or self.n_ is None:
            raise RuntimeError("Detector has not been fitted. Call 'fit' method first.")
        outliers = self.detect(data)
        if isinstance(data, DataBatch):
            cleaned_data = DataBatch(data=data.data[~outliers], labels=data.labels[~outliers] if data.labels is not None else None, metadata=data.metadata)
            return cleaned_data
        else:
            return data[~outliers]

    def get_probability_threshold(self) -> float:
        """
        Calculate the probability threshold for outlier detection based on sample size.
        
        Returns:
            float: Probability threshold (1/(2*n))
            
        Raises:
            RuntimeError: If detector has not been fitted
        """
        if self.n_ is None:
            raise RuntimeError("Detector has not been fitted. Call 'fit' method first.")
        return 1.0 / (2.0 * self.n_)