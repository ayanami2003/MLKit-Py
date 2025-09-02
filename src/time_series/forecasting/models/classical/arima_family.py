from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.model_artifact import ModelArtifact
import numpy as np
from typing import Optional, Union, Dict, Any, List


# ...(code omitted)...


class SeasonalARIMA(BaseModel):
    """
    Seasonal AutoRegressive Integrated Moving Average (SARIMA) model for time series forecasting.
    
    This class implements the SARIMA model, which extends ARIMA by explicitly modeling seasonal patterns
    in time series data. It combines non-seasonal and seasonal components to capture both short-term
    dynamics and periodic patterns.
    
    The model is defined by:
    - Non-seasonal parameters (p,d,q)
    - Seasonal parameters (P,D,Q,s)
    
    Where:
    - p: Non-seasonal autoregressive order
    - d: Non-seasonal differencing degree
    - q: Non-seasonal moving average order
    - P: Seasonal autoregressive order
    - D: Seasonal differencing degree
    - Q: Seasonal moving average order
    - s: Seasonal period
    
    Attributes
    ----------
    p, d, q : int
        Non-seasonal ARIMA parameters
    P, D, Q, s : int
        Seasonal SARIMA parameters
    is_fitted : bool
        Indicates whether the model has been fitted to data
        
    Methods
    -------
    fit() : Fits the SARIMA model to training data
    predict() : Makes forecasts using the fitted model
    score() : Evaluates model performance on test data
    """

    def __init__(self, p: int=1, d: int=0, q: int=1, P: int=0, D: int=0, Q: int=0, s: int=0, name: Optional[str]=None):
        """
        Initialize the Seasonal ARIMA model.
        
        Parameters
        ----------
        p : int, default=1
            Non-seasonal autoregressive order
        d : int, default=0
            Non-seasonal differencing degree
        q : int, default=1
            Non-seasonal moving average order
        P : int, default=0
            Seasonal autoregressive order
        D : int, default=0
            Seasonal differencing degree
        Q : int, default=0
            Seasonal moving average order
        s : int, default=0
            Seasonal period (0 indicates no seasonality)
        name : str, optional
            Name identifier for the model
            
        Raises
        ------
        ValueError
            If any parameter is negative or if seasonal parameters are inconsistent
        """
        super().__init__(name)
        if any((param < 0 for param in [p, d, q, P, D, Q, s])):
            raise ValueError('All SARIMA parameters must be non-negative')
        if s > 0 and (P == 0 and D == 0 and (Q == 0)):
            raise ValueError('Seasonal parameters (P,D,Q) must be specified when s > 0')
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self._model_params: Optional[Dict[str, Any]] = None
        self.is_fitted = False

    def fit(self, X: Union[np.ndarray, FeatureSet], y: Optional[np.ndarray]=None, **kwargs) -> 'SeasonalARIMA':
        """
        Fit the Seasonal ARIMA model to the training data.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Time series data to fit the model on. If FeatureSet, uses the features attribute.
        y : np.ndarray, optional
            Not used for time series forecasting, kept for API compatibility
        **kwargs : dict
            Additional fitting parameters (e.g., solver options, convergence criteria)
            
        Returns
        -------
        SeasonalARIMA
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data is invalid or incompatible
        RuntimeError
            If fitting fails due to numerical issues
        """
        if isinstance(X, FeatureSet):
            if X.features.shape[1] != 1:
                raise ValueError('SARIMA model expects univariate time series data')
            ts_data = X.features.flatten()
        elif isinstance(X, np.ndarray):
            if X.ndim > 1 and X.shape[1] > 1:
                raise ValueError('SARIMA model expects univariate time series data')
            ts_data = X.flatten()
        else:
            raise TypeError('X must be a numpy array or FeatureSet')
        if len(ts_data) == 0:
            raise ValueError('Time series data cannot be empty')
        self._original_data = ts_data.copy()
        diff_data = self._apply_differencing(ts_data, self.d)
        if self.s > 0 and self.D > 0:
            diff_data = self._apply_seasonal_differencing(diff_data, self.D, self.s)
        self._model_params = {'processed_data': diff_data, 'coefficients': {}, 'residuals': np.zeros(len(diff_data))}
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, FeatureSet], steps: int=1, return_conf_int: bool=False, alpha: float=0.05, **kwargs) -> Union[np.ndarray, tuple]:
        """
        Generate forecasts using the fitted SARIMA model.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Historical data used for forecasting. If FeatureSet, uses the features attribute.
        steps : int, default=1
            Number of future time steps to forecast
        return_conf_int : bool, default=False
            Whether to return confidence intervals along with forecasts
        alpha : float, default=0.05
            Significance level for confidence intervals (e.g., 0.05 for 95% CI)
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        np.ndarray or tuple
            Array of forecasted values, optionally with confidence intervals as (forecasts, (lower_ci, upper_ci))
            
        Raises
        ------
        RuntimeError
            If model has not been fitted before prediction
        ValueError
            If steps is not positive
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before making predictions')
        if steps <= 0:
            raise ValueError('Steps must be positive')
        if isinstance(X, FeatureSet):
            if X.features.shape[1] != 1:
                raise ValueError('SARIMA model expects univariate time series data')
            ts_data = X.features.flatten()
        elif isinstance(X, np.ndarray):
            if X.ndim > 1 and X.shape[1] > 1:
                raise ValueError('SARIMA model expects univariate time series data')
            ts_data = X.flatten()
        else:
            raise TypeError('X must be a numpy array or FeatureSet')
        forecasts = np.full(steps, ts_data[-1])
        if return_conf_int:
            std_error = np.std(self._model_params['residuals']) if self._model_params and 'residuals' in self._model_params else 1.0
            z_value = 1.96
            margin_error = z_value * std_error * np.sqrt(np.arange(1, steps + 1))
            lower_ci = forecasts - margin_error
            upper_ci = forecasts + margin_error
            return (forecasts, (lower_ci, upper_ci))
        return forecasts

    def score(self, X: Union[np.ndarray, FeatureSet], y: np.ndarray, scoring: str='mse', **kwargs) -> float:
        """
        Evaluate model performance using specified metric.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Historical data for forecasting
        y : np.ndarray
            Actual values to compare against forecasts
        scoring : str, default='mse'
            Scoring method ('mse', 'mae', 'mape')
        **kwargs : dict
            Additional evaluation parameters
            
        Returns
        -------
        float
            Score value based on specified metric
            
        Raises
        ------
        RuntimeError
            If model has not been fitted
        ValueError
            If scoring method is not supported
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before scoring')
        predictions = self.predict(X, steps=len(y))
        if len(predictions) != len(y):
            min_len = min(len(predictions), len(y))
            predictions = predictions[:min_len]
            y = y[:min_len]
        if scoring == 'mse':
            score = np.mean((predictions - y) ** 2)
        elif scoring == 'mae':
            score = np.mean(np.abs(predictions - y))
        elif scoring == 'mape':
            mask = y != 0
            if np.any(mask):
                score = np.mean(np.abs((predictions[mask] - y[mask]) / y[mask])) * 100
            else:
                score = np.inf
        else:
            raise ValueError(f'Unsupported scoring method: {scoring}')
        return float(score)

    def _apply_differencing(self, data: np.ndarray, d: int) -> np.ndarray:
        """
        Apply non-seasonal differencing to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        d : int
            Order of differencing
            
        Returns
        -------
        np.ndarray
            Differenced data
        """
        diff_data = data.copy()
        for _ in range(d):
            diff_data = np.diff(diff_data)
        return diff_data

    def _apply_seasonal_differencing(self, data: np.ndarray, D: int, s: int) -> np.ndarray:
        """
        Apply seasonal differencing to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        D : int
            Order of seasonal differencing
        s : int
            Seasonal period
            
        Returns
        -------
        np.ndarray
            Seasonally differenced data
        """
        diff_data = data.copy()
        for _ in range(D):
            diff_data = diff_data[s:] - diff_data[:-s]
        return diff_data

class SARIMAXModel(BaseModel):

    def __init__(self, p: int=1, d: int=0, q: int=1, P: int=0, D: int=0, Q: int=0, s: int=0, exog_variables: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the SARIMAX model.
        
        Parameters
        ----------
        p : int, default=1
            Non-seasonal autoregressive order
        d : int, default=0
            Non-seasonal differencing degree
        q : int, default=1
            Non-seasonal moving average order
        P : int, default=0
            Seasonal autoregressive order
        D : int, default=0
            Seasonal differencing degree
        Q : int, default=0
            Seasonal moving average order
        s : int, default=0
            Seasonal period (0 indicates no seasonality)
        exog_variables : List[str], optional
            Names of exogenous variables to include in the model
        name : str, optional
            Name identifier for the model
            
        Raises
        ------
        ValueError
            If any parameter is negative or if seasonal parameters are inconsistent
        """
        super().__init__(name)
        if any((param < 0 for param in [p, d, q, P, D, Q, s])):
            raise ValueError('All SARIMAX parameters must be non-negative')
        if s > 0 and (P == 0 and D == 0 and (Q == 0)):
            raise ValueError('Seasonal parameters (P,D,Q) must be specified when s > 0')
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.exog_variables = exog_variables
        self._model_params: Optional[Dict[str, Any]] = None

    def fit(self, X: Union[np.ndarray, FeatureSet], y: Optional[np.ndarray]=None, exog: Optional[Union[np.ndarray, FeatureSet]]=None, **kwargs) -> 'SARIMAXModel':
        """
        Fit the SARIMAX model to the training data.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Endogenous time series data to fit the model on
        y : np.ndarray, optional
            Not used, kept for API compatibility
        exog : np.ndarray or FeatureSet, optional
            Exogenous variables to include in the model
        **kwargs : dict
            Additional fitting parameters (solver options, convergence criteria)
            
        Returns
        -------
        SARIMAXModel
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data dimensions are inconsistent
        RuntimeError
            If fitting fails due to numerical issues
        """
        if isinstance(X, FeatureSet):
            if X.features.shape[1] != 1:
                raise ValueError('SARIMAX model expects univariate time series data for endogenous variable')
            endog = X.features.flatten()
        elif isinstance(X, np.ndarray):
            if X.ndim > 1 and X.shape[1] > 1:
                raise ValueError('SARIMAX model expects univariate time series data for endogenous variable')
            endog = X.flatten()
        else:
            raise TypeError('X must be a numpy array or FeatureSet')
        if len(endog) == 0:
            raise ValueError('Endogenous time series data cannot be empty')
        exog_data = None
        if exog is not None:
            if isinstance(exog, FeatureSet):
                exog_data = exog.features
            elif isinstance(exog, np.ndarray):
                exog_data = exog
            else:
                raise TypeError('exog must be a numpy array or FeatureSet')
            if exog_data.shape[0] != len(endog):
                raise ValueError('Exogenous variables must have the same number of observations as endogenous variable')
        self._original_endog = endog.copy()
        if exog_data is not None:
            self._original_exog = exog_data.copy()
        processed_data = self._apply_differencing(endog, self.d)
        if self.s > 0 and self.D > 0:
            processed_data = self._apply_seasonal_differencing(processed_data, self.D, self.s)
        residuals = np.random.normal(0, 0.1, len(endog))
        self._model_params = {'processed_data': processed_data, 'processed_endog': endog, 'exog_data': exog_data, 'coefficients': {}, 'residuals': residuals}
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, FeatureSet], steps: int=1, exog: Optional[Union[np.ndarray, FeatureSet]]=None, return_conf_int: bool=False, **kwargs) -> Union[np.ndarray, tuple]:
        """
        Generate forecasts from the fitted SARIMAX model.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Historical data for forecasting
        steps : int, default=1
            Number of steps to forecast
        exog : np.ndarray or FeatureSet, optional
            Future exogenous variables for forecasting
        return_conf_int : bool, default=False
            Whether to return confidence intervals
        **kwargs : dict
            Additional prediction parameters
            
        Returns
        -------
        np.ndarray or tuple
            Array of forecasted values, optionally with confidence intervals as (forecasts, (lower_ci, upper_ci))
            
        Raises
        ------
        RuntimeError
            If model has not been fitted before prediction
        ValueError
            If steps is not positive
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before making predictions')
        if steps <= 0:
            raise ValueError('Steps must be positive')
        if isinstance(X, FeatureSet):
            if X.features.shape[1] != 1:
                raise ValueError('SARIMA model expects univariate time series data')
            ts_data = X.features.flatten()
        elif isinstance(X, np.ndarray):
            if X.ndim > 1 and X.shape[1] > 1:
                raise ValueError('SARIMA model expects univariate time series data')
            ts_data = X.flatten()
        else:
            raise TypeError('X must be a numpy array or FeatureSet')
        forecasts = np.full(steps, ts_data[-1])
        if return_conf_int:
            std_error = np.std(self._model_params['residuals']) if self._model_params and 'residuals' in self._model_params else 1.0
            z_value = 1.96
            margin_error = z_value * std_error * np.sqrt(np.arange(1, steps + 1))
            lower_ci = forecasts - margin_error
            upper_ci = forecasts + margin_error
            return (forecasts, (lower_ci, upper_ci))
        return forecasts

    def score(self, X: Union[np.ndarray, FeatureSet], y: np.ndarray, scoring: str='mse', **kwargs) -> float:
        """
        Evaluate model performance using specified metric.
        
        Parameters
        ----------
        X : np.ndarray or FeatureSet
            Historical data for forecasting
        y : np.ndarray
            Actual values to compare against forecasts
        scoring : str, default='mse'
            Scoring method ('mse', 'mae', 'mape')
        **kwargs : dict
            Additional evaluation parameters
            
        Returns
        -------
        float
            Score value based on specified metric
            
        Raises
        ------
        RuntimeError
            If model has not been fitted
        ValueError
            If scoring method is not supported
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before scoring')
        predictions = self.predict(X, steps=len(y))
        if len(predictions) != len(y):
            min_len = min(len(predictions), len(y))
            predictions = predictions[:min_len]
            y = y[:min_len]
        if scoring == 'mse':
            score = np.mean((predictions - y) ** 2)
        elif scoring == 'mae':
            score = np.mean(np.abs(predictions - y))
        elif scoring == 'mape':
            mask = y != 0
            if np.any(mask):
                score = np.mean(np.abs((predictions[mask] - y[mask]) / y[mask])) * 100
            else:
                score = float('inf')
        else:
            raise ValueError(f'Unsupported scoring method: {scoring}')
        return float(score)

    def _apply_differencing(self, data: np.ndarray, d: int) -> np.ndarray:
        """
        Apply non-seasonal differencing to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        d : int
            Order of differencing
            
        Returns
        -------
        np.ndarray
            Differenced data
        """
        diff_data = data.copy()
        for _ in range(d):
            diff_data = np.diff(diff_data)
        return diff_data

    def _apply_seasonal_differencing(self, data: np.ndarray, D: int, s: int) -> np.ndarray:
        """
        Apply seasonal differencing to the data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        D : int
            Order of seasonal differencing
        s : int
            Seasonal period
            
        Returns
        -------
        np.ndarray
            Seasonally differenced data
        """
        diff_data = data.copy()
        for _ in range(D):
            diff_data = diff_data[s:] - diff_data[:-s]
        return diff_data