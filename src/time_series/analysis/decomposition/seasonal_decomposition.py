from typing import Optional, Dict, Any, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class STLDecomposer(BaseTransformer):

    def __init__(self, period: int, seasonal: int=7, trend: Optional[int]=None, low_pass: Optional[int]=None, seasonal_deg: int=1, trend_deg: int=1, low_pass_deg: int=1, robust: bool=False, seasonal_jump: int=1, trend_jump: int=1, low_pass_jump: int=1, name: Optional[str]=None):
        """
        Initialize the STL decomposer.
        
        Parameters
        ----------
        period : int
            The period of the seasonal component
        seasonal : int, default=7
            Length of the seasonal smoother
        trend : int, optional
            Length of the trend smoother
        low_pass : int, optional
            Length of the low-pass filter
        seasonal_deg : int, default=1
            Degree of seasonal smoothing polynomial
        trend_deg : int, default=1
            Degree of trend smoothing polynomial
        low_pass_deg : int, default=1
            Degree of low-pass smoothing polynomial
        robust : bool, default=False
            Use robust fitting (fewer outliers)
        seasonal_jump : int, default=1
            Skipping interval for seasonal smoothing
        trend_jump : int, default=1
            Skipping interval for trend smoothing
        low_pass_jump : int, default=1
            Skipping interval for low-pass smoothing
        name : str, optional
            Name of the transformer
        """
        super().__init__(name=name)
        self.period = period
        self.seasonal = seasonal
        self.trend = trend or next_odd(period)
        self.low_pass = low_pass or next_odd(period)
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        self._fitted_components = {}
        self._fitted = False
        self._transformed = False

    def fit(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> 'STLDecomposer':
        """
        Fit the STL decomposition to the time series data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Time series data to decompose. If DataBatch or FeatureSet, expects
            a single time series in the data attribute.
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        STLDecomposer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data is not in the expected format or is incompatible
        """
        if isinstance(data, DataBatch):
            ts_data = data.data
        elif isinstance(data, FeatureSet):
            ts_data = data.features
        elif isinstance(data, np.ndarray):
            ts_data = data
        else:
            raise ValueError('Input data must be DataBatch, FeatureSet, or numpy array')
        if ts_data.ndim > 1 and ts_data.shape[1] > 1:
            raise ValueError('STL decomposition supports only univariate time series')
        if ts_data.ndim > 1:
            ts_data = ts_data.flatten()
        if len(ts_data) < 2 * self.period:
            raise ValueError('Time series data too short for decomposition')
        result = self._stl_decompose(ts_data)
        self._fitted_components = {'observed': ts_data, 'trend': result['trend'], 'seasonal': result['seasonal'], 'residual': result['residual']}
        self._fitted = True
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """
        Apply the STL decomposition to extract components.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, np.ndarray]
            Time series data to decompose
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing the decomposed components:
            - 'trend': Trend component
            - 'seasonal': Seasonal component
            - 'residual': Residual component
            - 'observed': Original data
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if not self._fitted:
            raise ValueError('STLDecomposer must be fitted before transform')
        if isinstance(data, DataBatch):
            ts_data = data.data
        elif isinstance(data, FeatureSet):
            ts_data = data.features
        elif isinstance(data, np.ndarray):
            ts_data = data
        else:
            raise ValueError('Input data must be DataBatch, FeatureSet, or numpy array')
        if ts_data.ndim > 1 and ts_data.shape[1] > 1:
            raise ValueError('STL decomposition supports only univariate time series')
        if ts_data.ndim > 1:
            ts_data = ts_data.flatten()
        if len(ts_data) < 2 * self.period:
            raise ValueError('Time series data too short for decomposition')
        result = self._stl_decompose(ts_data)
        self._transformed = True
        self._fitted_components = {'observed': ts_data, 'trend': result['trend'], 'seasonal': result['seasonal'], 'residual': result['residual']}
        return self._fitted_components.copy()

    def inverse_transform(self, data: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Reconstruct the original time series from components.
        
        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dictionary containing at least 'trend' and 'seasonal' components
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Reconstructed time series (trend + seasonal [+ residual if present])
        """
        if 'trend' not in data or 'seasonal' not in data:
            raise ValueError("Both 'trend' and 'seasonal' components required for reconstruction")
        reconstructed = data['trend'] + data['seasonal']
        if 'residual' in data:
            reconstructed += data['residual']
        return reconstructed

    def get_seasonal_component(self) -> np.ndarray:
        """
        Extract the seasonal component from the last transformation.
        
        Returns
        -------
        np.ndarray
            The seasonal component of the time series
            
        Raises
        ------
        RuntimeError
            If no decomposition has been performed yet or transform not called
        """
        if not self._fitted or 'seasonal' not in self._fitted_components:
            raise RuntimeError('No decomposition has been performed yet')
        if not self._transformed:
            raise RuntimeError('Transform must be called before accessing components')
        return self._fitted_components['seasonal']

    def get_trend_component(self) -> np.ndarray:
        """
        Extract the trend component from the last transformation.
        
        Returns
        -------
        np.ndarray
            The trend component of the time series
            
        Raises
        ------
        RuntimeError
            If no decomposition has been performed yet or transform not called
        """
        if not self._fitted or 'trend' not in self._fitted_components:
            raise RuntimeError('No decomposition has been performed yet')
        if not self._transformed:
            raise RuntimeError('Transform must be called before accessing components')
        return self._fitted_components['trend']

    def get_residual_component(self) -> np.ndarray:
        """
        Extract the residual component from the last transformation.
        
        Returns
        -------
        np.ndarray
            The residual component of the time series
            
        Raises
        ------
        RuntimeError
            If no decomposition has been performed yet or transform not called
        """
        if not self._fitted or 'residual' not in self._fitted_components:
            raise RuntimeError('No decomposition has been performed yet')
        if not self._transformed:
            raise RuntimeError('Transform must be called before accessing components')
        return self._fitted_components['residual']

    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Get all fitted components.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing all components
            
        Raises
        ------
        RuntimeError
            If no decomposition has been performed yet or transform not called
        """
        if not self._fitted:
            raise RuntimeError('No decomposition has been performed yet')
        if not self._transformed:
            raise RuntimeError('Transform must be called before accessing components')
        return self._fitted_components.copy()

    def _stl_decompose(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform STL decomposition using LOESS smoothing.
        
        Parameters
        ----------
        x : np.ndarray
            Time series data
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with trend, seasonal, and residual components
        """
        n = len(x)
        weights = np.ones(n)
        n_outer = 1 if not self.robust else 15
        for _ in range(n_outer):
            if np.sum(weights) == 0:
                weights = np.ones(n)
            trend = self._smooth_trend(x, weights)
            detrended = x - trend
            seasonal = self._smooth_seasonal(detrended, weights)
            seasonal = self._adjust_seasonal(seasonal)
            residual = x - trend - seasonal
            if self.robust:
                weights = self._update_weights(residual)
        return {'trend': trend, 'seasonal': seasonal, 'residual': residual}

    def _smooth_trend(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Smooth the trend component."""
        return self._loess_smooth(x, weights, self.trend, self.trend_deg, self.trend_jump)

    def _smooth_seasonal(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Smooth the seasonal component."""
        n = len(x)
        seasonal = np.zeros(n)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            cycle_data = x[indices]
            cycle_weights = weights[indices]
            smoothed_cycle = self._loess_smooth(cycle_data, cycle_weights, self.seasonal, self.seasonal_deg, self.seasonal_jump)
            seasonal[indices] = smoothed_cycle
        return seasonal

    def _adjust_seasonal(self, seasonal: np.ndarray) -> np.ndarray:
        """Adjust seasonal component to ensure it sums to zero over each period."""
        n = len(seasonal)
        adjusted = seasonal.astype(float)
        n_periods = n // self.period
        for i in range(n_periods):
            start_idx = i * self.period
            end_idx = start_idx + self.period
            period_mean = np.mean(adjusted[start_idx:end_idx])
            adjusted[start_idx:end_idx] -= period_mean
        if n % self.period != 0:
            start_idx = n_periods * self.period
            period_mean = np.mean(adjusted[start_idx:])
            adjusted[start_idx:] -= period_mean
        return adjusted

    def _loess_smooth(self, x: np.ndarray, weights: np.ndarray, window: int, degree: int, jump: int) -> np.ndarray:
        """
        Apply LOESS smoothing to a time series.
        
        Parameters
        ----------
        x : np.ndarray
            Time series data
        weights : np.ndarray
            Weights for each point
        window : int
            Smoothing window size
        degree : int
            Degree of polynomial
        jump : int
            Jump interval for computation
            
        Returns
        -------
        np.ndarray
            Smoothed time series
        """
        n = len(x)
        smoothed = np.zeros(n)
        if window % 2 == 0:
            window += 1
        half_window = window // 2
        for i in range(0, n, jump):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            x_window = np.arange(start, end)
            y_window = x[start:end]
            w_window = weights[start:end]
            if len(y_window) == 0 or np.sum(w_window) == 0:
                smoothed[i] = x[i] if i < n else 0
                continue
            w_window = w_window / np.sum(w_window)
            X_design = np.vander(x_window - i, degree + 1, increasing=True)
            try:
                W = np.diag(w_window)
                XtW = X_design.T @ W
                XtWX_inv = np.linalg.pinv(XtW @ X_design)
                coefs = XtWX_inv @ XtW @ y_window
                smoothed[i] = X_design[half_window if i == half_window else min(half_window, len(X_design) - 1)] @ coefs
            except np.linalg.LinAlgError:
                smoothed[i] = np.average(y_window, weights=w_window)
        if jump > 1:
            for i in range(1, n):
                if i % jump != 0:
                    prev_i = i // jump * jump
                    next_i = min((i // jump + 1) * jump, n - 1)
                    if next_i != prev_i:
                        frac = (i - prev_i) / (next_i - prev_i)
                        smoothed[i] = smoothed[prev_i] + frac * (smoothed[next_i] - smoothed[prev_i])
                    else:
                        smoothed[i] = smoothed[prev_i]
        return smoothed

    def _update_weights(self, residuals: np.ndarray) -> np.ndarray:
        """Update weights for robust fitting based on residuals."""
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad == 0:
            return np.ones_like(residuals)
        h = 1.4826 * mad
        std_resid = np.abs(residuals) / h
        weights = np.where(std_resid <= 4.685, (1 - (std_resid / 4.685) ** 2) ** 2, 0)
        return weights