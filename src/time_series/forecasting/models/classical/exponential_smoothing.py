from typing import Optional, Union, Dict, Any
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
import numpy as np

class ExponentialSmoothingModel(BaseModel):
    """
    Single exponential smoothing model for univariate time series forecasting.
    
    Implements exponential smoothing for time series data without trend or seasonal components.
    Uses a weighted average of past observations, with exponentially decreasing weights.
    
    Attributes
    ----------
    alpha : float
        Smoothing parameter (0 < alpha < 1). Higher values give more weight to recent observations.
    level : Optional[float]
        The estimated level component after fitting.
    fitted_values : Optional[np.ndarray]
        In-sample forecasts (fitted values) from the model.
    residuals : Optional[np.ndarray]
        Residuals from the fitted model.
    """

    def __init__(self, alpha: float=0.3, name: Optional[str]=None):
        """
        Initialize the exponential smoothing model.
        
        Parameters
        ----------
        alpha : float, optional (default=0.3)
            Smoothing parameter between 0 and 1. Controls the rate at which
            the influence of observations decreases over time.
        name : str, optional
            Name for the model instance.
        """
        super().__init__(name=name)
        if not 0 < alpha < 1:
            raise ValueError('Alpha must be between 0 and 1.')
        self.alpha = alpha
        self.level: Optional[float] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, FeatureSet], y: Optional[np.ndarray]=None, **kwargs) -> 'ExponentialSmoothingModel':
        """
        Fit the exponential smoothing model to the training data.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Time series data to fit the model on. Should be a 1D array or a FeatureSet
            with a single feature column.
        y : None
            Ignored in this implementation.
        **kwargs : dict
            Additional fitting parameters (not used).
            
        Returns
        -------
        ExponentialSmoothingModel
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is not valid (e.g., wrong dimensions).
        """
        if isinstance(X, FeatureSet):
            if X.features.shape[1] != 1:
                raise ValueError('FeatureSet must contain exactly one feature for univariate time series.')
            data = X.features[:, 0]
        elif isinstance(X, np.ndarray):
            if X.ndim != 1:
                raise ValueError('Input array must be 1-dimensional.')
            data = X
        else:
            raise ValueError('Input must be a numpy array or FeatureSet.')
        n = len(data)
        if n == 0:
            raise ValueError('Input data cannot be empty.')
        self.level = data[0]
        self.fitted_values = np.zeros(n)
        self.residuals = np.zeros(n)
        for t in range(n):
            if t == 0:
                self.fitted_values[t] = self.level
            else:
                self.level = self.alpha * data[t - 1] + (1 - self.alpha) * self.level
                self.fitted_values[t] = self.level
            self.residuals[t] = data[t] - self.fitted_values[t]
        return self

    def predict(self, X: Union[int, np.ndarray], **kwargs) -> np.ndarray:
        """
        Make forecasts using the fitted exponential smoothing model.
        
        Parameters
        ----------
        X : int or np.ndarray
            If int, number of steps ahead to forecast.
            If array, ignored (used for compatibility with BaseModel interface).
        **kwargs : dict
            Additional prediction parameters (not used).
            
        Returns
        -------
        np.ndarray
            Forecasted values for the specified horizon.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.level is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before making predictions.")
        if isinstance(X, int):
            steps = X
        elif isinstance(X, np.ndarray):
            steps = len(X)
        else:
            raise ValueError('X must be either an integer or a numpy array.')
        if steps <= 0:
            return np.array([])
        return np.full(steps, self.level)

    def score(self, X: Union[np.ndarray, FeatureSet], y: np.ndarray, **kwargs) -> float:
        """
        Calculate the mean squared error of the model on the given data.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Time series data for which to generate forecasts.
        y : np.ndarray
            Actual values to compare against forecasts.
        **kwargs : dict
            Additional scoring parameters (not used).
            
        Returns
        -------
        float
            Mean squared error of the forecasts.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.level is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before scoring.")
        forecasts = self.predict(len(y))
        mse = np.mean((y - forecasts) ** 2)
        return float(mse)

class HoltWintersModel(BaseModel):

    def __init__(self, alpha: float=0.3, beta: float=0.1, gamma: float=0.1, seasonal_periods: int=12, trend: str='add', seasonal: str='add', damped_trend: bool=False, phi: float=0.98, name: Optional[str]=None):
        """
        Initialize the Holt-Winters model.
        
        Parameters
        ----------
        alpha : float, optional (default=0.3)
            Level smoothing parameter (0 < alpha < 1).
        beta : float, optional (default=0.1)
            Trend smoothing parameter (0 < beta < 1).
        gamma : float, optional (default=0.1)
            Seasonal smoothing parameter (0 < gamma < 1).
        seasonal_periods : int, optional (default=12)
            Number of periods in a complete seasonal cycle.
        trend : str, optional (default='add')
            Type of trend component ('add' for additive, 'mul' for multiplicative).
        seasonal : str, optional (default='add')
            Type of seasonal component ('add' for additive, 'mul' for multiplicative).
        damped_trend : bool, optional (default=False)
            Whether to dampen the trend projection.
        phi : float, optional (default=0.98)
            Damping parameter for trend (only used if damped_trend=True).
        name : str, optional
            Name for the model instance.
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal_type = seasonal
        self.damped_trend = damped_trend
        self.phi = phi
        self.level: Optional[np.ndarray] = None
        self.trend_component: Optional[np.ndarray] = None
        self.seasonal: Optional[np.ndarray] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, FeatureSet], y: Optional[np.ndarray]=None, **kwargs) -> 'HoltWintersModel':
        """
        Fit the Holt-Winters model to the training data.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Time series data to fit the model on. Should be a 1D array or a FeatureSet
            with a single feature column.
        y : None
            Ignored in this implementation.
        **kwargs : dict
            Additional fitting parameters (not used).
            
        Returns
        -------
        HoltWintersModel
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is not valid (e.g., wrong dimensions) or if seasonal_periods
            is not compatible with the data length.
        """
        if isinstance(X, FeatureSet):
            if X.features.shape[1] != 1:
                raise ValueError('FeatureSet must contain exactly one feature for univariate time series.')
            data = X.features[:, 0]
        elif isinstance(X, np.ndarray):
            if X.ndim != 1:
                raise ValueError('Input array must be 1-dimensional.')
            data = X
        else:
            raise ValueError('Input must be a numpy array or FeatureSet.')
        n = len(data)
        if n == 0:
            raise ValueError('Input data cannot be empty.')
        if self.seasonal_periods <= 0 or self.seasonal_periods >= n:
            raise ValueError('seasonal_periods must be positive and less than the length of the data.')
        if n < 2 * self.seasonal_periods:
            raise ValueError('Not enough data to initialize seasonal components. Need at least 2*seasonal_periods.')
        self.level = np.zeros(n)
        self.trend_component = np.zeros(n)
        self.seasonal = np.zeros(self.seasonal_periods)
        self.fitted_values = np.zeros(n)
        self.residuals = np.zeros(n)
        initial_level = np.mean(data[:self.seasonal_periods])
        initial_trend = 0
        for i in range(self.seasonal_periods):
            if i + self.seasonal_periods < n:
                initial_trend += data[i + self.seasonal_periods] - data[i]
        initial_trend /= self.seasonal_periods
        seasonal_averages = np.zeros(self.seasonal_periods)
        num_cycles = n // self.seasonal_periods
        for i in range(self.seasonal_periods):
            seasonal_sum = 0
            count = 0
            for j in range(num_cycles):
                idx = i + j * self.seasonal_periods
                if idx < n:
                    seasonal_sum += data[idx]
                    count += 1
            if count > 0:
                seasonal_averages[i] = seasonal_sum / count
            else:
                seasonal_averages[i] = 0
        if self.seasonal_type == 'add':
            seasonal_adjustment = np.mean(seasonal_averages)
            seasonal_indices = seasonal_averages - seasonal_adjustment
        else:
            seasonal_adjustment = np.mean(seasonal_averages)
            if seasonal_adjustment == 0:
                seasonal_indices = np.ones(self.seasonal_periods)
            else:
                seasonal_indices = seasonal_averages / seasonal_adjustment
        self.level[0] = initial_level
        self.trend_component[0] = initial_trend
        for i in range(self.seasonal_periods):
            self.seasonal[i] = seasonal_indices[i]
        for t in range(1, n):
            prev_level = self.level[t - 1]
            prev_trend = self.trend_component[t - 1]
            prev_seasonal = self.seasonal[(t - 1) % self.seasonal_periods]
            seasonal_index = self.seasonal[t % self.seasonal_periods]
            phi = self.phi if self.damped_trend else 1.0
            if self.trend == 'add':
                if self.seasonal_type == 'add':
                    forecast = prev_level + phi * prev_trend + seasonal_index
                else:
                    forecast = (prev_level + phi * prev_trend) * seasonal_index
            elif self.seasonal_type == 'add':
                forecast = prev_level * (1 + phi * prev_trend) + seasonal_index
            else:
                forecast = prev_level * (1 + phi * prev_trend) * seasonal_index
            if not np.isfinite(forecast):
                forecast = data[t - 1] if t > 0 else data[0]
            self.fitted_values[t] = forecast
            if self.seasonal_type == 'add':
                if self.trend == 'add':
                    seasonal_update = self.gamma * (data[t] - (prev_level + phi * prev_trend)) + (1 - self.gamma) * prev_seasonal
                else:
                    denom = prev_level * (1 + phi * prev_trend)
                    if denom != 0 and np.isfinite(denom):
                        seasonal_update = self.gamma * (data[t] - denom) + (1 - self.gamma) * prev_seasonal
                    else:
                        seasonal_update = prev_seasonal
            elif self.trend == 'add':
                denom = prev_level + phi * prev_trend
                if denom != 0 and np.isfinite(denom):
                    seasonal_update = self.gamma * (data[t] / denom) + (1 - self.gamma) * prev_seasonal
                else:
                    seasonal_update = prev_seasonal
            else:
                denom = prev_level * (1 + phi * prev_trend)
                if denom != 0 and np.isfinite(denom):
                    seasonal_update = self.gamma * (data[t] / denom) + (1 - self.gamma) * prev_seasonal
                else:
                    seasonal_update = prev_seasonal
            self.seasonal[t % self.seasonal_periods] = seasonal_update
            if self.seasonal_type == 'add':
                detrended = data[t] - self.seasonal[t % self.seasonal_periods]
            else:
                seasonal_val = self.seasonal[t % self.seasonal_periods]
                if seasonal_val != 0 and np.isfinite(seasonal_val):
                    detrended = data[t] / seasonal_val
                else:
                    detrended = data[t]
            if self.trend == 'add':
                self.level[t] = self.alpha * detrended + (1 - self.alpha) * (prev_level + phi * prev_trend)
            elif prev_level != 0 and np.isfinite(prev_level) and (1 + phi * prev_trend != 0) and np.isfinite(1 + phi * prev_trend):
                self.level[t] = self.alpha * detrended + (1 - self.alpha) * (prev_level * (1 + phi * prev_trend))
            else:
                self.level[t] = prev_level
            if self.trend == 'add':
                self.trend_component[t] = self.beta * (self.level[t] - prev_level) + (1 - self.beta) * phi * prev_trend
            elif prev_level != 0 and np.isfinite(prev_level):
                trend_ratio = self.level[t] / prev_level if prev_level != 0 else 1
                self.trend_component[t] = self.beta * (trend_ratio - 1) + (1 - self.beta) * phi * prev_trend
            else:
                self.trend_component[t] = phi * prev_trend
            if not np.isfinite(self.level[t]):
                self.level[t] = prev_level
            if not np.isfinite(self.trend_component[t]):
                self.trend_component[t] = phi * prev_trend
            if not np.isfinite(self.seasonal[t % self.seasonal_periods]):
                self.seasonal[t % self.seasonal_periods] = prev_seasonal
            self.residuals[t] = data[t] - self.fitted_values[t]
        if self.trend == 'add':
            if self.seasonal_type == 'add':
                self.fitted_values[0] = self.level[0] + self.trend_component[0] + self.seasonal[0]
            else:
                self.fitted_values[0] = (self.level[0] + self.trend_component[0]) * self.seasonal[0]
        elif self.seasonal_type == 'add':
            self.fitted_values[0] = self.level[0] * (1 + self.trend_component[0]) + self.seasonal[0]
        else:
            self.fitted_values[0] = self.level[0] * (1 + self.trend_component[0]) * self.seasonal[0]
        self.residuals[0] = data[0] - self.fitted_values[0]
        self.is_fitted = True
        return self

    def predict(self, X: Union[int, np.ndarray], **kwargs) -> np.ndarray:
        """
        Make forecasts using the fitted Holt-Winters model.
        
        Parameters
        ----------
        X : int or np.ndarray
            If int, number of steps ahead to forecast.
            If array, ignored (used for compatibility with BaseModel interface).
        **kwargs : dict
            Additional prediction parameters (not used).
            
        Returns
        -------
        np.ndarray
            Forecasted values for the specified horizon.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before making predictions.")
        if isinstance(X, int):
            steps = X
        elif isinstance(X, np.ndarray):
            steps = len(X)
        else:
            raise ValueError('X must be either an integer or a numpy array.')
        if steps <= 0:
            return np.array([])
        n = len(self.level)
        forecasts = np.zeros(steps)
        for h in range(1, steps + 1):
            prev_level = self.level[-1]
            prev_trend = self.trend_component[-1]
            seasonal_index = self.seasonal[(n + h - 1) % self.seasonal_periods]
            if self.damped_trend:
                phi_h = sum((self.phi ** i for i in range(1, h + 1))) if h > 0 else 0
            else:
                phi_h = h
            if self.trend == 'add':
                if self.seasonal_type == 'add':
                    forecasts[h - 1] = prev_level + phi_h * prev_trend + seasonal_index
                else:
                    forecasts[h - 1] = (prev_level + phi_h * prev_trend) * seasonal_index
            elif self.seasonal_type == 'add':
                forecasts[h - 1] = prev_level * (1 + phi_h * prev_trend) + seasonal_index
            else:
                forecasts[h - 1] = prev_level * (1 + phi_h * prev_trend) * seasonal_index
            if not np.isfinite(forecasts[h - 1]):
                forecasts[h - 1] = prev_level
        return forecasts

    def score(self, X: Union[np.ndarray, FeatureSet], y: np.ndarray, **kwargs) -> float:
        """
        Calculate the mean squared error of the model on the given data.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Time series data for which to generate forecasts.
        y : np.ndarray
            Actual values to compare against forecasts.
        **kwargs : dict
            Additional scoring parameters (not used).
            
        Returns
        -------
        float
            Mean squared error of the forecasts.
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before scoring.")
        forecasts = self.predict(len(y))
        mse = np.mean((forecasts - y) ** 2)
        return float(mse)