import numpy as np
from typing import Union, Optional, Dict, Any
from general.structures.data_batch import DataBatch

def kpss_test(data: Union[np.ndarray, DataBatch], regression: str='c', lags: Optional[int]=None) -> Dict[str, Any]:
    """
    Perform the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test for stationarity.

    The KPSS test is used to test the null hypothesis that a univariate time series 
    is stationary around a deterministic trend. It is complementary to the ADF test,
    which tests for the presence of a unit root.

    Args:
        data (Union[np.ndarray, DataBatch]): Input time series data to test for stationarity.
            If DataBatch, expects a single column of time series data.
        regression (str, optional): The null hypothesis for the test. 
            'c' for constant-only (default), 'ct' for constant+trend.
        lags (Optional[int], optional): Number of lags to use in the Newey-West estimator.
            If None, uses the default lag selection method.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'statistic': The test statistic value
            - 'p_value': The p-value of the test
            - 'critical_values': Critical values at various significance levels
            - 'lags': The number of lags used
            - 'method': Name of the test performed

    Raises:
        ValueError: If regression parameter is not 'c' or 'ct'
        ValueError: If data is empty or not one-dimensional
    """
    if regression not in ['c', 'ct']:
        raise ValueError("regression must be 'c' (constant) or 'ct' (constant and trend)")
    if isinstance(data, DataBatch):
        if hasattr(data.data, 'shape') and len(data.data.shape) > 1 and (data.data.shape[1] > 1):
            raise ValueError('Data must be univariate (single column)')
        ts_data = np.asarray(data.data).flatten()
    else:
        ts_data = np.asarray(data)
        if ts_data.ndim > 1 and ts_data.shape[1] > 1:
            raise ValueError('Data must be univariate (single column)')
        ts_data = ts_data.flatten()
    if ts_data.size == 0:
        raise ValueError('Input data cannot be empty')
    n_obs = len(ts_data)
    if lags is None:
        lags = int(4 * (n_obs / 100) ** 0.25)
        lags = max(lags, 1)
    elif lags < 0:
        raise ValueError('lags must be non-negative')
    if regression == 'c':
        demeaned_data = ts_data - np.mean(ts_data)
        s = np.concatenate([[0], np.cumsum(demeaned_data)])
    else:
        t = np.arange(1, n_obs + 1)
        X = np.column_stack([np.ones(n_obs), t])
        try:
            beta = np.linalg.lstsq(X, ts_data, rcond=None)[0]
            detrended_data = ts_data - X @ beta
        except np.linalg.LinAlgError:
            detrended_data = ts_data - np.mean(ts_data)
        s = np.concatenate([[0], np.cumsum(detrended_data)])
    sum_s2 = np.sum(s[1:] ** 2)
    residuals = np.diff(s)
    sigma2 = np.var(residuals)
    if lags > 0:
        gamma = np.zeros(lags + 1)
        for i in range(lags + 1):
            if i == 0:
                gamma[i] = sigma2
            else:
                gamma[i] = np.mean(residuals[i:] * residuals[:-i])
        weights = 1 - np.arange(lags + 1) / (lags + 1)
        sigma2 = gamma[0] + 2 * np.sum(weights[1:] * gamma[1:])
        sigma2 = max(sigma2, 1e-10)
    eta_squared = sum_s2 / n_obs ** 2
    test_statistic = eta_squared / sigma2
    if regression == 'c':
        critical_values = {'1%': 0.739, '5%': 0.463, '10%': 0.347}
    else:
        critical_values = {'1%': 0.216, '5%': 0.146, '10%': 0.119}
    if regression == 'c':
        p_value = np.exp(-7.7269 * test_statistic + 0.8577)
        p_value = min(p_value, 1.0)
    else:
        p_value = np.exp(-12.0713 * test_statistic + 1.4857)
        p_value = min(p_value, 1.0)
    p_value = max(p_value, 0.0)
    return {'statistic': float(test_statistic), 'p_value': float(p_value), 'critical_values': critical_values, 'lags': lags, 'method': 'KPSS'}


# ...(code omitted)...


def phillips_perron_test(data: Union[np.ndarray, DataBatch], regression: str='c', lags: Optional[int]=None) -> Dict[str, Any]:
    """
    Perform the Phillips-Perron (PP) test for unit roots in a time series.

    The Phillips-Perron test is a non-parametric alternative to the ADF test that 
    accounts for serial correlation and heteroskedasticity in the error terms 
    without explicitly modeling them.

    Args:
        data (Union[np.ndarray, DataBatch]): Input time series data to test for unit roots.
            If DataBatch, expects a single column of time series data.
        regression (str, optional): The type of regression to use. 
            'c' for constant-only (default), 'ct' for constant+trend.
        lags (Optional[int], optional): Number of lags to use in the Newey-West estimator.
            If None, uses a data-driven method to select the lag length.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'statistic': The test statistic value
            - 'p_value': The p-value of the test
            - 'critical_values': Critical values at various significance levels
            - 'lags': The number of lags used
            - 'method': Name of the test performed

    Raises:
        ValueError: If regression parameter is not 'c' or 'ct'
        ValueError: If data is empty or not one-dimensional
    """
    if regression not in ['c', 'ct']:
        raise ValueError("regression must be 'c' (constant) or 'ct' (constant and trend)")
    if isinstance(data, DataBatch):
        if hasattr(data.data, 'shape') and len(data.data.shape) > 1 and (data.data.shape[1] > 1):
            raise ValueError('Data must be univariate (single column)')
        ts_data = np.asarray(data.data).flatten()
    else:
        ts_data = np.asarray(data)
        if ts_data.ndim > 1 and ts_data.shape[1] > 1:
            raise ValueError('Data must be univariate (single column)')
        ts_data = ts_data.flatten()
    if ts_data.size == 0:
        raise ValueError('Input data cannot be empty')
    n_obs = len(ts_data)
    if lags is None:
        lags = int(4 * (n_obs / 100) ** 0.25)
        lags = max(lags, 1)
    elif lags < 0:
        raise ValueError('lags must be non-negative')
    y = ts_data
    dy = np.diff(y)
    y_lag = y[:-1]
    if regression == 'c':
        X = np.column_stack([np.ones(n_obs - 1), y_lag])
    else:
        t = np.arange(1, n_obs)
        X = np.column_stack([np.ones(n_obs - 1), t, y_lag])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        residuals = dy - X @ beta
    except np.linalg.LinAlgError:
        residuals = dy - np.mean(dy)
    sigma_squared = np.var(residuals)
    if lags > 0:
        gamma = np.zeros(lags + 1)
        for i in range(lags + 1):
            if i == 0:
                gamma[i] = sigma_squared
            else:
                gamma[i] = np.mean(residuals[i:] * residuals[:-i])
        weights = 1 - np.arange(lags + 1) / (lags + 1)
        sigma_squared = gamma[0] + 2 * np.sum(weights[1:] * gamma[1:])
        sigma_squared = max(sigma_squared, 1e-10)
    try:
        XTX_inv = np.linalg.inv(X.T @ X)
        se_coefficient = np.sqrt(sigma_squared * XTX_inv[-1, -1])
        coeff_lag = beta[-1]
        test_statistic = coeff_lag / se_coefficient
    except np.linalg.LinAlgError:
        test_statistic = -10.0
    if regression == 'c':
        critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
        p_value = np.exp(-0.5349 * test_statistic - 2.8607)
        p_value = min(p_value, 1.0)
    else:
        critical_values = {'1%': -3.96, '5%': -3.41, '10%': -3.13}
        p_value = np.exp(-0.4791 * test_statistic - 3.1674)
        p_value = min(p_value, 1.0)
    p_value = max(p_value, 0.0)
    return {'statistic': float(test_statistic), 'p_value': float(p_value), 'critical_values': critical_values, 'lags': lags, 'method': 'Phillips-Perron'}

def zivot_andrews_test(data: Union[np.ndarray, DataBatch], regression: str='c', autolag: Optional[str]='AIC', trim: float=0.15) -> Dict[str, Any]:
    """
    Perform the Zivot-Andrews test for unit roots with a single structural break.

    The Zivot-Andrews test is an extension of the ADF test that allows for a single 
    structural break in the intercept, trend, or both. It tests for a unit root 
    against the alternative of stationarity with a structural break.

    Args:
        data (Union[np.ndarray, DataBatch]): Input time series data to test for unit roots.
            If DataBatch, expects a single column of time series data.
        regression (str, optional): The type of regression to use. 
            'c' for constant-only (default), 'ct' for constant+trend.
        autolag (Optional[str], optional): Method to automatically determine lag length.
            'AIC', 'BIC', or None. If None, no lag selection is performed.
        trim (float, optional): Percentage of series to exclude from the beginning and end
            when searching for the breakpoint. Defaults to 0.15 (15%).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'statistic': The test statistic value
            - 'p_value': The p-value of the test
            - 'critical_values': Critical values at various significance levels
            - 'used_lag': The number of lags used in the test
            - 'break_point': Location of the estimated break point
            - 'method': Name of the test performed

    Raises:
        ValueError: If regression parameter is not 'c' or 'ct'
        ValueError: If autolag parameter is not one of 'AIC', 'BIC', None
        ValueError: If trim is not between 0 and 0.5
        ValueError: If data is empty or not one-dimensional
    """
    if regression not in ['c', 'ct']:
        raise ValueError("regression must be 'c' (constant) or 'ct' (constant and trend)")
    if autolag not in ['AIC', 'BIC', None]:
        raise ValueError("autolag must be 'AIC', 'BIC', or None")
    if not 0 <= trim <= 0.5:
        raise ValueError('trim must be between 0 and 0.5')
    if isinstance(data, DataBatch):
        if hasattr(data.data, 'shape') and len(data.data.shape) > 1 and (data.data.shape[1] > 1):
            raise ValueError('Data must be univariate (single column)')
        ts_data = np.asarray(data.data).flatten()
    else:
        ts_data = np.asarray(data)
        if ts_data.ndim > 1 and ts_data.shape[1] > 1:
            raise ValueError('Data must be univariate (single column)')
        ts_data = ts_data.flatten()
    if ts_data.size == 0:
        raise ValueError('Input data cannot be empty')
    n_obs = len(ts_data)
    if autolag is not None:
        max_lags = int(12 * (n_obs / 100) ** 0.25)
    else:
        max_lags = 0
    trim_start = int(trim * n_obs) + 1
    trim_end = n_obs - int(trim * n_obs)
    if trim_end <= trim_start:
        raise ValueError('Trimming too aggressive - no observations left for break point estimation')
    best_statistic = np.inf
    best_lag = 0
    best_break_point = trim_start
    best_ic = np.inf
    for tb in range(trim_start, trim_end):
        for lag in range(max_lags + 1):
            try:
                y_diff = np.diff(ts_data)
                y_lagged = ts_data[lag:n_obs - 1]
                if regression == 'c':
                    n_reg_obs = n_obs - 1 - lag
                    X = np.ones((n_reg_obs, 3 + lag))
                    X[:, 1] = (np.arange(n_reg_obs) + 1 + lag >= tb).astype(int)
                    if lag > 0:
                        for i in range(lag):
                            X[:, 2 + i] = y_diff[lag - 1 - i:n_obs - 2 - i]
                    X[:, -1] = y_lagged
                else:
                    n_reg_obs = n_obs - 1 - lag
                    X = np.ones((n_reg_obs, 5 + lag))
                    X[:, 1] = np.arange(lag + 1, n_obs)
                    X[:, 2] = (np.arange(n_reg_obs) + 1 + lag >= tb).astype(int)
                    time_after_break = np.maximum(0, np.arange(n_reg_obs) + 1 + lag - tb)
                    X[:, 3] = time_after_break
                    if lag > 0:
                        for i in range(lag):
                            X[:, 4 + i] = y_diff[lag - 1 - i:n_obs - 2 - i]
                    X[:, -1] = y_lagged
                y = y_diff[lag:]
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    residuals = y - X @ beta
                except np.linalg.LinAlgError:
                    continue
                sigma2 = np.var(residuals)
                if sigma2 <= 0:
                    continue
                try:
                    XtX_inv = np.linalg.inv(X.T @ X)
                    coef_idx = -1
                    var_coef = sigma2 * XtX_inv[coef_idx, coef_idx]
                    if var_coef <= 0:
                        continue
                    se_coef = np.sqrt(var_coef)
                except np.linalg.LinAlgError:
                    continue
                t_stat = beta[coef_idx] / se_coef
                if autolag is not None:
                    if autolag == 'AIC':
                        ic = np.log(sigma2) + 2 * X.shape[1] / (n_obs - 1 - lag)
                    else:
                        ic = np.log(sigma2) + X.shape[1] * np.log(n_obs - 1 - lag) / (n_obs - 1 - lag)
                    if ic < best_ic:
                        best_ic = ic
                        best_statistic = t_stat
                        best_lag = lag
                        best_break_point = tb
                elif t_stat < best_statistic:
                    best_statistic = t_stat
                    best_lag = lag
                    best_break_point = tb
            except Exception:
                continue
    if best_statistic == np.inf:
        raise ValueError('Could not compute test statistic - data may be insufficient')
    if regression == 'c':
        critical_values = {'1%': -5.34, '5%': -4.8, '10%': -4.58}
        p_value = np.exp(-0.5 * (best_statistic + 5.0) ** 2)
    else:
        critical_values = {'1%': -4.93, '5%': -4.42, '10%': -4.11}
        p_value = np.exp(-0.4 * (best_statistic + 4.5) ** 2)
    p_value = max(min(p_value, 1.0), 0.0)
    return {'statistic': float(best_statistic), 'p_value': float(p_value), 'critical_values': critical_values, 'used_lag': best_lag, 'break_point': best_break_point, 'method': 'Zivot-Andrews'}