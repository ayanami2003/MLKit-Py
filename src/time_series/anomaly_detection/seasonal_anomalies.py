from typing import Optional, Union, List, Tuple
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class SeasonalAnomalyDetector(BaseTransformer):

    def __init__(self, method: str='stl', seasonal_period: int=12, threshold: float=3.0, contamination: float=0.1, feature_columns: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the SeasonalAnomalyDetector.
        
        Parameters
        ----------
        method : str, default='stl'
            The seasonal anomaly detection method to use. Options:
            - 'stl': Uses STL decomposition to separate seasonal components
            - 'seasonal_decomposition': General seasonal decomposition approach
            - 'periodic': Periodic pattern-based anomaly detection
        seasonal_period : int, default=12
            The length of the seasonal period in observations (e.g., 12 for monthly data with yearly seasonality)
        threshold : float, default=3.0
            Threshold for anomaly detection. Interpretation depends on method:
            - For statistical methods: Number of standard deviations from expected value
            - For percentile-based methods: Percentile threshold (0-100)
        contamination : float, default=0.1
            Expected proportion of anomalies in the data (between 0 and 0.5)
        feature_columns : Optional[List[str]], optional
            Specific feature columns to analyze for multivariate data. If None, all features are used
        name : Optional[str], optional
            Name of the transformer instance
            
        Raises
        ------
        ValueError
            If seasonal_period is not positive or threshold is negative
        """
        super().__init__(name=name)
        if seasonal_period <= 0:
            raise ValueError('seasonal_period must be positive')
        if threshold < 0:
            raise ValueError('threshold must be non-negative')
        if not 0 <= contamination <= 0.5:
            raise ValueError('contamination must be between 0 and 0.5')
        self.method = method
        self.seasonal_period = seasonal_period
        self.threshold = threshold
        self.contamination = contamination
        self.feature_columns = feature_columns
        self._anomaly_scores = None
        self._is_fitted = False

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'SeasonalAnomalyDetector':
        """
        Fit the seasonal anomaly detector to training data.
        
        This method learns the seasonal patterns and baseline behavior from the provided data,
        which is then used to detect anomalies in subsequent transform calls.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Training time series data. Can be:
            - FeatureSet: With time series features
            - DataBatch: With time series data
            - np.ndarray: 1D or 2D array of time series observations
        **kwargs : dict
            Additional fitting parameters (method-specific)
            
        Returns
        -------
        SeasonalAnomalyDetector
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data is empty or incompatible with the seasonal period
        """
        if isinstance(data, DataBatch):
            data_array = data.data
        elif isinstance(data, FeatureSet):
            if self.feature_columns is not None:
                data_array = data.features[:, [data.feature_names.index(col) for col in self.feature_columns]]
            else:
                data_array = data.features
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            raise ValueError('Input data must be DataBatch, FeatureSet, or numpy array')
        if data_array.size == 0:
            raise ValueError('Input data is empty')
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        elif data_array.ndim > 2:
            raise ValueError('Input data must be 1D or 2D array')
        if data_array.shape[0] < 2 * self.seasonal_period:
            raise ValueError(f'Time series data too short for seasonal period {self.seasonal_period}')
        self._data_shape = data_array.shape
        self._fitted_params = {}
        for col_idx in range(data_array.shape[1]):
            col_data = data_array[:, col_idx]
            if self.method == 'stl':
                self._fitted_params[col_idx] = {'data': col_data.copy(), 'mean': np.mean(col_data), 'std': np.std(col_data)}
            elif self.method == 'seasonal_decomposition':
                self._fitted_params[col_idx] = self._fit_seasonal_decomposition(col_data)
            elif self.method == 'periodic':
                self._fitted_params[col_idx] = self._fit_periodic_detection(col_data)
            else:
                raise ValueError(f"Unknown method '{self.method}'. Supported methods: 'stl', 'seasonal_decomposition', 'periodic'")
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], return_scores: bool=False, **kwargs) -> Union[FeatureSet, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply seasonal anomaly detection to new data.
        
        Identifies anomalies in the provided time series data based on learned seasonal patterns.
        Returns binary anomaly flags and optionally anomaly scores.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Time series data to analyze for anomalies. Must have same structure as training data.
        return_scores : bool, default=False
            If True, returns both anomaly flags and anomaly scores
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        Union[FeatureSet, np.ndarray, Tuple[np.ndarray, np.ndarray]]
            If return_scores is False:
                - np.ndarray: Binary array where 1 indicates anomaly, 0 indicates normal
            If return_scores is True:
                - Tuple[np.ndarray, np.ndarray]: (anomaly_flags, anomaly_scores)
                
        Raises
        ------
        RuntimeError
            If called before fitting the detector
        """
        if not self._fitted:
            raise RuntimeError('SeasonalAnomalyDetector must be fitted before transform')
        if isinstance(data, DataBatch):
            data_array = data.data
            is_databatch = True
        elif isinstance(data, FeatureSet):
            if self.feature_columns is not None:
                data_array = data.features[:, [data.feature_names.index(col) for col in self.feature_columns]]
            else:
                data_array = data.features
            is_feature_set = True
        elif isinstance(data, np.ndarray):
            data_array = data
            is_databatch = False
            is_feature_set = False
        else:
            raise ValueError('Input data must be DataBatch, FeatureSet, or numpy array')
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        elif data_array.ndim > 2:
            raise ValueError('Input data must be 1D or 2D array')
        if data_array.shape[1] != self._data_shape[1]:
            raise ValueError(f'Number of features in transform data ({data_array.shape[1]}) does not match fitted data ({self._data_shape[1]})')
        n_samples = data_array.shape[0]
        anomaly_scores = np.zeros((n_samples, data_array.shape[1]))
        anomaly_flags = np.zeros((n_samples, data_array.shape[1]), dtype=bool)
        for col_idx in range(data_array.shape[1]):
            col_data = data_array[:, col_idx]
            if self.method == 'stl':
                (scores, flags) = self._detect_anomalies_stl(col_data, col_idx)
            elif self.method == 'seasonal_decomposition':
                (scores, flags) = self._detect_anomalies_seasonal_decomposition(col_data, col_idx)
            elif self.method == 'periodic':
                (scores, flags) = self._detect_anomalies_periodic(col_data, col_idx)
            else:
                raise ValueError(f"Unknown method '{self.method}'")
            anomaly_scores[:, col_idx] = scores
            anomaly_flags[:, col_idx] = flags
        combined_scores = np.max(anomaly_scores, axis=1)
        combined_flags = np.any(anomaly_flags, axis=1)
        self._anomaly_scores = combined_scores
        if return_scores:
            return (combined_flags.astype(int), combined_scores)
        else:
            return combined_flags.astype(int)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Not implemented for anomaly detectors.
        
        Anomaly detectors are not reversible transformations, so this method raises NotImplementedError.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Data to inverse transform (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray]
            Never returns successfully
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for anomaly detection
        """
        raise NotImplementedError('SeasonalAnomalyDetector does not support inverse transformation')

    def get_anomaly_scores(self) -> Optional[np.ndarray]:
        """
        Get the computed anomaly scores from the last transform operation.
        
        Returns anomaly scores that quantify the degree of anomalousness for each observation.
        Higher scores indicate higher likelihood of being an anomaly.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of anomaly scores, or None if no scores are available
        """
        return self._anomaly_scores

    def _fit_seasonal_decomposition(self, data: np.ndarray) -> dict:
        """Fit seasonal decomposition parameters."""
        n = len(data)
        period = self.seasonal_period
        n_seasons = n // period
        seasonal_indices = np.zeros(period)
        for i in range(period):
            seasonal_positions = np.arange(i, n, period)
            if len(seasonal_positions) > 0:
                seasonal_indices[i] = np.mean(data[seasonal_positions])
        overall_mean = np.mean(data)
        seasonal_indices = seasonal_indices - np.mean(seasonal_indices)
        residuals = np.zeros(n)
        for i in range(n):
            seasonal_idx = i % period
            residuals[i] = data[i] - overall_mean - seasonal_indices[seasonal_idx]
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        return {'overall_mean': overall_mean, 'seasonal_indices': seasonal_indices, 'residual_mean': residual_mean, 'residual_std': residual_std}

    def _fit_periodic_detection(self, data: np.ndarray) -> dict:
        """Fit periodic pattern detection parameters."""
        n = len(data)
        period = self.seasonal_period
        patterns = []
        for i in range(period):
            positions = np.arange(i, n, period)
            if len(positions) > 0:
                pattern_stats = {'mean': np.mean(data[positions]), 'std': np.std(data[positions])}
                patterns.append(pattern_stats)
            else:
                patterns.append({'mean': 0, 'std': 1})
        return {'patterns': patterns}

    def _detect_anomalies_stl(self, data: np.ndarray, col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using STL-inspired approach."""
        params = self._fitted_params[col_idx]
        mean_val = params['mean']
        std_val = params['std']
        if std_val == 0:
            std_val = 1e-08
        z_scores = np.abs(data - mean_val) / std_val
        anomaly_flags = z_scores > self.threshold
        return (z_scores, anomaly_flags)

    def _detect_anomalies_seasonal_decomposition(self, data: np.ndarray, col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using seasonal decomposition approach."""
        params = self._fitted_params[col_idx]
        period = self.seasonal_period
        expected_values = np.zeros(len(data))
        residuals = np.zeros(len(data))
        overall_mean = params['overall_mean']
        seasonal_indices = params['seasonal_indices']
        for i in range(len(data)):
            seasonal_idx = i % period
            expected_values[i] = overall_mean + seasonal_indices[seasonal_idx]
            residuals[i] = data[i] - expected_values[i]
        residual_mean = params['residual_mean']
        residual_std = params['residual_std']
        if residual_std == 0:
            residual_std = 1e-08
        anomaly_scores = np.abs(residuals - residual_mean) / residual_std
        anomaly_flags = anomaly_scores > self.threshold
        return (anomaly_scores, anomaly_flags)

    def _detect_anomalies_periodic(self, data: np.ndarray, col_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using periodic pattern approach."""
        params = self._fitted_params[col_idx]
        period = self.seasonal_period
        patterns = params['patterns']
        anomaly_scores = np.zeros(len(data))
        for i in range(len(data)):
            pos_idx = i % period
            if pos_idx < len(patterns):
                pattern = patterns[pos_idx]
                mean_val = pattern['mean']
                std_val = pattern['std']
                if std_val == 0:
                    std_val = 1e-08
                anomaly_scores[i] = np.abs(data[i] - mean_val) / std_val
            else:
                anomaly_scores[i] = 0
        anomaly_flags = anomaly_scores > self.threshold
        return (anomaly_scores, anomaly_flags)