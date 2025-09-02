from typing import Optional, Dict, Any, List, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch

class SeasonalAnomalyDetector(BaseModel):

    def __init__(self, seasonal_period: int=12, threshold: float=3.0, method: str='stl', name: Optional[str]=None):
        """
        Initialize the SeasonalAnomalyDetector.
        
        Parameters
        ----------
        seasonal_period : int, default=12
            The period of the seasonal component (e.g., 12 for monthly data with yearly seasonality)
        threshold : float, default=3.0
            Number of standard deviations to consider a point anomalous
        method : str, default='stl'
            Method for seasonal decomposition ('stl', 'seasonal_decomposition')
        name : str, optional
            Name of the detector
        """
        super().__init__(name)
        self.seasonal_period = seasonal_period
        self.threshold = threshold
        self.method = method

    def fit(self, X: Union[DataBatch, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'SeasonalAnomalyDetector':
        """
        Fit the seasonal anomaly detector to the time series data.
        
        Parameters
        ----------
        X : DataBatch or np.ndarray
            Time series data to fit on. If DataBatch, uses the data attribute.
        y : np.ndarray, optional
            Not used in unsupervised anomaly detection
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        SeasonalAnomalyDetector
            Self instance for method chaining
        """
        if isinstance(X, DataBatch):
            data = X.data
        else:
            data = X
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError('Time series data must be 1-dimensional')
        if len(data) < self.seasonal_period and self.method in ['stl', 'seasonal_decomposition']:
            self.trend_ = np.zeros_like(data)
            self.seasonal_ = np.zeros_like(data)
            self.residuals_ = data.copy()
        elif self.method == 'stl':
            trend_window = self.seasonal_period if self.seasonal_period % 2 == 1 else self.seasonal_period + 1
            trend_window = min(trend_window, len(data))
            pad_width = trend_window // 2
            padded_data = np.pad(data, (pad_width, pad_width), mode='edge')
            trend = np.convolve(padded_data, np.ones(trend_window) / trend_window, mode='valid')
            if len(trend) > len(data):
                trend = trend[:len(data)]
            elif len(trend) < len(data):
                trend = np.pad(trend, (0, len(data) - len(trend)), mode='edge')
            detrended = data - trend
            seasonal = np.zeros_like(data)
            for i in range(self.seasonal_period):
                indices = np.arange(i, len(data), self.seasonal_period)
                if len(indices) > 0:
                    seasonal[indices] = np.mean(detrended[indices])
            residuals = data - trend - seasonal
            self.trend_ = trend
            self.seasonal_ = seasonal
            self.residuals_ = residuals
        elif self.method == 'seasonal_decomposition':
            trend_window = self.seasonal_period if self.seasonal_period % 2 == 1 else self.seasonal_period + 1
            trend_window = min(trend_window, len(data))
            if self.seasonal_period % 2 == 0:
                pad_width = trend_window // 2
                padded_data = np.pad(data, (pad_width, pad_width), mode='edge')
                temp_trend = np.convolve(padded_data, np.ones(trend_window) / trend_window, mode='valid')
                if len(temp_trend) >= 2:
                    temp_trend = np.convolve(temp_trend, np.ones(2) / 2, mode='same')
                trend = temp_trend[:len(data)] if len(temp_trend) > len(data) else np.pad(temp_trend, (0, len(data) - len(temp_trend)), mode='edge')
            else:
                pad_width = trend_window // 2
                padded_data = np.pad(data, (pad_width, pad_width), mode='edge')
                trend = np.convolve(padded_data, np.ones(trend_window) / trend_window, mode='valid')
                if len(trend) > len(data):
                    trend = trend[:len(data)]
                elif len(trend) < len(data):
                    trend = np.pad(trend, (0, len(data) - len(trend)), mode='edge')
            detrended = data - trend
            seasonal = np.zeros_like(data)
            for i in range(self.seasonal_period):
                indices = np.arange(i, len(data), self.seasonal_period)
                if len(indices) > 0:
                    seasonal_indices = detrended[indices]
                    median_val = np.median(seasonal_indices)
                    mad = np.median(np.abs(seasonal_indices - median_val))
                    if mad > 0:
                        mask = np.abs(seasonal_indices - median_val) <= 3 * mad
                        if np.any(mask):
                            seasonal_mean = np.mean(seasonal_indices[mask])
                        else:
                            seasonal_mean = median_val
                    else:
                        seasonal_mean = median_val
                    seasonal[indices] = seasonal_mean
            residuals = data - trend - seasonal
            self.trend_ = trend
            self.seasonal_ = seasonal
            self.residuals_ = residuals
        else:
            raise ValueError(f'Unsupported decomposition method: {self.method}')
        valid_mask = ~np.isnan(self.residuals_)
        if np.sum(valid_mask) == 0:
            raise ValueError('No valid residuals found after decomposition')
        valid_residuals = self.residuals_[valid_mask]
        self.residual_mean_ = np.mean(valid_residuals)
        self.residual_std_ = np.std(valid_residuals)
        if self.residual_std_ == 0:
            self.residual_std_ = 1.0
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[DataBatch, np.ndarray], **kwargs) -> np.ndarray:
        """
        Detect anomalies in the time series data.
        
        Parameters
        ----------
        X : DataBatch or np.ndarray
            Time series data to analyze for anomalies
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        np.ndarray
            Binary array where 1 indicates an anomaly and 0 indicates normal observation
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        if isinstance(X, DataBatch):
            data = X.data
        else:
            data = X
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError('Time series data must be 1-dimensional')
        n = len(data)
        if n != len(self.trend_):
            if n < len(self.trend_):
                trend_component = self.trend_[:n]
                seasonal_component = self.seasonal_[:n]
            else:
                trend_component = np.interp(np.linspace(0, len(self.trend_) - 1, n), np.arange(len(self.trend_)), self.trend_)
                seasonal_extended = np.tile(self.seasonal_[:self.seasonal_period], n // self.seasonal_period + 1)
                seasonal_component = seasonal_extended[:n]
        else:
            trend_component = self.trend_
            seasonal_component = self.seasonal_
        residuals = data - trend_component - seasonal_component
        std_dev = self.residual_std_ if self.residual_std_ > 0 else 1.0
        anomaly_scores = np.abs(residuals - self.residual_mean_) / std_dev
        predictions = (anomaly_scores > self.threshold).astype(int)
        return predictions

    def score(self, X: Union[DataBatch, np.ndarray], y: np.ndarray, **kwargs) -> float:
        """
        Evaluate model performance on labeled data.
        
        Parameters
        ----------
        X : DataBatch or np.ndarray
            Time series data
        y : np.ndarray
            Ground truth labels (1 for anomaly, 0 for normal)
        **kwargs : dict
            Additional evaluation parameters
            
        Returns
        -------
        float
            Accuracy score of anomaly detection
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        predictions = self.predict(X)
        if len(predictions) != len(y):
            raise ValueError('Length of predictions and ground truth labels must match.')
        accuracy = np.mean(predictions == y)
        return float(accuracy)

    def get_anomaly_scores(self, X: Union[DataBatch, np.ndarray], **kwargs) -> np.ndarray:
        """
        Get anomaly scores for each observation.
        
        Parameters
        ----------
        X : DataBatch or np.ndarray
            Time series data to score
            
        Returns
        -------
        np.ndarray
            Anomaly scores for each observation (higher means more anomalous)
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        if isinstance(X, DataBatch):
            data = X.data
        else:
            data = X
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError('Time series data must be 1-dimensional')
        n = len(data)
        if n != len(self.trend_):
            if n < len(self.trend_):
                trend_component = self.trend_[:n]
                seasonal_component = self.seasonal_[:n]
            else:
                trend_component = np.interp(np.linspace(0, len(self.trend_) - 1, n), np.arange(len(self.trend_)), self.trend_)
                seasonal_extended = np.tile(self.seasonal_[:self.seasonal_period], n // self.seasonal_period + 1)
                seasonal_component = seasonal_extended[:n]
        else:
            trend_component = self.trend_
            seasonal_component = self.seasonal_
        residuals = data - trend_component - seasonal_component
        std_dev = self.residual_std_ if self.residual_std_ > 0 else 1.0
        anomaly_scores = np.abs(residuals - self.residual_mean_) / std_dev
        return anomaly_scores

    def get_seasonal_components(self) -> Dict[str, np.ndarray]:
        """
        Get the decomposed seasonal components.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing trend, seasonal, and residual components
        """
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet. Call 'fit' before using this method.")
        return {'trend': self.trend_, 'seasonal': self.seasonal_, 'residual': self.residuals_}