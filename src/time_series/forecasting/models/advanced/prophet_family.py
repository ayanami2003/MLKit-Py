from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, Dict, Any, Union, List
import pandas as pd

class ProphetModel(BaseModel):

    def __init__(self, growth: str='linear', changepoints: Optional[List]=None, n_changepoints: int=25, changepoint_range: float=0.8, yearly_seasonality: Union[str, bool, int]='auto', weekly_seasonality: Union[str, bool, int]='auto', daily_seasonality: Union[str, bool, int]='auto', holidays: Optional[Dict]=None, seasonality_mode: str='additive', seasonality_prior_scale: float=10.0, holidays_prior_scale: float=10.0, changepoint_prior_scale: float=0.05, mcmc_samples: int=0, interval_width: float=0.8, uncertainty_samples: int=1000, name: Optional[str]=None):
        """
        Initialize the Prophet model with specified parameters.
        
        Args:
            growth (str): Growth function type ('linear' or 'logistic')
            changepoints (Optional[List]): List of dates at which to include potential changepoints
            n_changepoints (int): Number of potential changepoints to include automatically
            changepoint_range (float): Proportion of history in which trend changepoints will be estimated
            yearly_seasonality (Union[str, bool, int]): Fit yearly seasonality ('auto', True, False, or number of Fourier terms)
            weekly_seasonality (Union[str, bool, int]): Fit weekly seasonality ('auto', True, False, or number of Fourier terms)
            daily_seasonality (Union[str, bool, int]): Fit daily seasonality ('auto', True, False, or number of Fourier terms)
            holidays (Optional[Dict]): DataFrame with holiday names and dates
            seasonality_mode (str): Seasonality mode ('additive' or 'multiplicative')
            seasonality_prior_scale (float): Scale parameter for seasonality model
            holidays_prior_scale (float): Scale parameter for holiday effects
            changepoint_prior_scale (float): Scale parameter for trend changes
            mcmc_samples (int): Number of MCMC samples if > 0
            interval_width (float): Width of the uncertainty intervals provided for the forecast
            uncertainty_samples (int): Number of simulated draws used to estimate uncertainty
            name (Optional[str]): Name for the model instance
        """
        super().__init__(name)
        self.model_params = {'growth': growth, 'changepoints': changepoints, 'n_changepoints': n_changepoints, 'changepoint_range': changepoint_range, 'yearly_seasonality': yearly_seasonality, 'weekly_seasonality': weekly_seasonality, 'daily_seasonality': daily_seasonality, 'holidays': holidays, 'seasonality_mode': seasonality_mode, 'seasonality_prior_scale': seasonality_prior_scale, 'holidays_prior_scale': holidays_prior_scale, 'changepoint_prior_scale': changepoint_prior_scale, 'mcmc_samples': mcmc_samples, 'interval_width': interval_width, 'uncertainty_samples': uncertainty_samples}
        self._model = None

    def fit(self, X: Union[DataBatch, FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'ProphetModel':
        """
        Fit the Prophet model to the time series data.
        
        Args:
            X: Time series data with timestamps and values. Expected to have columns 'ds' (dates) and 'y' (values)
               or be convertible to this format.
            y: Not used for Prophet (included for compatibility with BaseModel)
            **kwargs: Additional fitting parameters
            
        Returns:
            ProphetModel: Self instance for method chaining
            
        Raises:
            ValueError: If required columns are missing from input data
        """
        df = self._convert_to_dataframe(X)
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("Input data must contain 'ds' (timestamp) and 'y' (value) columns")
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet library is required but not installed. Please install it with 'pip install prophet'")
        self._model = Prophet(**self.model_params)
        self._model.fit(df)
        self.is_fitted = True
        return self

    def predict(self, X: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> pd.DataFrame:
        """
        Generate forecasts for future time points.
        
        Args:
            X: Future timestamps for which to generate forecasts. Should contain 'ds' column.
            **kwargs: Additional prediction parameters
            
        Returns:
            np.ndarray: Array of predicted values with uncertainty intervals
            
        Raises:
            RuntimeError: If model has not been fitted yet
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before making predictions.")
        df = self._convert_to_dataframe(X)
        if 'ds' not in df.columns:
            raise ValueError("Input data must contain 'ds' (timestamp) column")
        forecast = self._model.predict(df)
        return forecast

    def score(self, X: Union[DataBatch, FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> float:
        """
        Evaluate model performance using mean absolute error.
        
        Args:
            X: Test data with timestamps
            y: True values for comparison
            **kwargs: Additional scoring parameters
            
        Returns:
            float: Mean absolute error of predictions
            
        Raises:
            RuntimeError: If model has not been fitted yet
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' before evaluating.")
        df = self._convert_to_dataframe(X)
        if 'ds' not in df.columns:
            raise ValueError("Input data must contain 'ds' (timestamp) column")
        forecast = self._model.predict(df)
        y_pred = forecast['yhat'].values
        mae = np.mean(np.abs(y - y_pred))
        return mae

    def _convert_to_dataframe(self, data: Union[DataBatch, FeatureSet, np.ndarray]) -> pd.DataFrame:
        """
        Convert various data formats to pandas DataFrame compatible with Prophet.
        
        Args:
            data: Input data in various formats
            
        Returns:
            pd.DataFrame: DataFrame with 'ds' and optionally 'y' columns
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
            if 'y' in df.columns:
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
            return df
        elif isinstance(data, DataBatch):
            if isinstance(data.data, pd.DataFrame):
                df = data.data.copy()
                if 'ds' in df.columns:
                    df['ds'] = pd.to_datetime(df['ds'])
                if 'y' in df.columns:
                    df['y'] = pd.to_numeric(df['y'], errors='coerce')
                return df
            elif isinstance(data.data, np.ndarray):
                if data.feature_names and 'ds' in data.feature_names:
                    ds_idx = data.feature_names.index('ds')
                    df = pd.DataFrame({'ds': pd.to_datetime(data.data[:, ds_idx])})
                    if 'y' in data.feature_names:
                        y_idx = data.feature_names.index('y')
                        df['y'] = pd.to_numeric(data.data[:, y_idx], errors='coerce')
                    return df
                elif data.data.shape[1] >= 2:
                    return pd.DataFrame({'ds': pd.to_datetime(data.data[:, 0]), 'y': pd.to_numeric(data.data[:, 1], errors='coerce')})
                else:
                    return pd.DataFrame({'ds': pd.to_datetime(data.data[:, 0])})
            else:
                raise ValueError('Unsupported DataBatch data format')
        elif isinstance(data, FeatureSet):
            if data.feature_names and 'ds' in data.feature_names:
                ds_idx = data.feature_names.index('ds')
                df = pd.DataFrame({'ds': pd.to_datetime(data.features[:, ds_idx])})
                if 'y' in data.feature_names:
                    y_idx = data.feature_names.index('y')
                    df['y'] = pd.to_numeric(data.features[:, y_idx], errors='coerce')
                return df
            elif data.features.shape[1] >= 2:
                return pd.DataFrame({'ds': pd.to_datetime(data.features[:, 0]), 'y': pd.to_numeric(data.features[:, 1], errors='coerce')})
            else:
                return pd.DataFrame({'ds': pd.to_datetime(data.features[:, 0])})
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({'ds': pd.to_datetime(data)})
            elif data.shape[1] >= 2:
                return pd.DataFrame({'ds': pd.to_datetime(data[:, 0]), 'y': pd.to_numeric(data[:, 1], errors='coerce')})
            else:
                return pd.DataFrame({'ds': pd.to_datetime(data[:, 0])})
        else:
            raise ValueError(f'Unsupported data type: {type(data)}')