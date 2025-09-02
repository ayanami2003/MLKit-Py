from typing import Union, Dict, Any, Optional
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kstest

def perform_statistical_tests(model: BaseModel, X: Union[FeatureSet, DataBatch, np.ndarray], y: Union[np.ndarray, list], test_types: Optional[list]=None, alpha: float=0.05, alternative: str='two-sided', **kwargs) -> Dict[str, Any]:
    """
    Perform statistical hypothesis tests to validate model performance and assumptions.

    This function conducts various statistical tests on a trained model to assess:
    - Model significance (F-test, likelihood ratio test)
    - Coefficient significance (t-tests, z-tests)
    - Residual analysis (normality, homoscedasticity)
    - Goodness of fit tests
    - Non-parametric alternatives when assumptions are violated

    The specific tests performed depend on the model type and the test_types parameter.
    Common tests include:
    - F-test for overall model significance
    - T-test/Z-test for individual coefficients
    - Shapiro-Wilk test for normality of residuals
    - Breusch-Pagan test for heteroscedasticity
    - Durbin-Watson test for autocorrelation
    - Kolmogorov-Smirnov test for distribution comparison

    Args:
        model (BaseModel): A trained model implementing the BaseModel interface.
                          Must have a predict method and potentially coef_/intercept_ attributes.
        X (Union[FeatureSet, DataBatch, np.ndarray]): Input features used for predictions.
                                                      Can be in various formats depending on the model.
        y (Union[np.ndarray, list]): True target values for comparison with predictions.
        test_types (Optional[list]): List of specific test types to perform.
                                   If None, automatically selects relevant tests based on model type.
                                   Supported tests include: ['f_test', 't_test', 'shapiro_wilk',
                                   'breusch_pagan', 'durbin_watson', 'ks_test', 'likelihood_ratio']
        alpha (float): Significance level for hypothesis tests. Defaults to 0.05.
        alternative (str): Alternative hypothesis type ('two-sided', 'less', 'greater').
                          May not apply to all test types. Defaults to 'two-sided'.
        **kwargs: Additional parameters for specific tests:
                 - 'residuals': Precomputed residuals to use instead of calculating from model
                 - 'n_bootstraps': Number of bootstrap samples for bootstrap-based tests
                 - 'random_state': Random seed for reproducibility of stochastic tests

    Returns:
        Dict[str, Any]: A dictionary containing test results with the following structure:
                       {
                           'test_name': {
                               'statistic': float,     # Test statistic value
                               'p_value': float,       # P-value of the test
                               'significant': bool,    # Whether result is significant at alpha level
                               'confidence_interval': tuple,  # CI for the test statistic (if applicable)
                               'assumption_met': bool, # Whether underlying assumption is met
                               'recommendation': str   # Interpretation and recommendations
                           },
                           ...
                       }

    Raises:
        ValueError: If the model is not fitted, if X and y shapes don't match,
                   or if unsupported test types are requested.
        TypeError: If inputs are not of supported types.
        RuntimeError: If a required model attribute is missing for a specific test.
    """
    if not hasattr(model, 'predict'):
        raise ValueError("Model must have a 'predict' method")
    if isinstance(X, FeatureSet):
        X_array = X.features
    elif isinstance(X, DataBatch):
        X_array = X.data
    elif isinstance(X, np.ndarray):
        X_array = X
    else:
        raise TypeError('X must be FeatureSet, DataBatch, or numpy array')
    y_array = np.asarray(y)
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    residuals = kwargs.get('residuals')
    if residuals is None:
        y_pred = model.predict(X)
        residuals = y_array - y_pred
    residuals = np.asarray(residuals)
    if test_types is None:
        test_types = ['shapiro_wilk', 'durbin_watson']
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            test_types.extend(['t_test', 'f_test'])
    supported_tests = ['f_test', 't_test', 'shapiro_wilk', 'breusch_pagan', 'durbin_watson', 'ks_test', 'likelihood_ratio']
    for test in test_types:
        if test not in supported_tests:
            raise ValueError(f'Unsupported test type: {test}')
    results = {}
    if 'shapiro_wilk' in test_types:
        try:
            if len(residuals) < 3:
                raise ValueError('Shapiro-Wilk test requires at least 3 samples')
            if len(residuals) > 5000:
                indices = np.random.RandomState(kwargs.get('random_state', 42)).choice(len(residuals), size=5000, replace=False)
                sw_residuals = residuals[indices]
            else:
                sw_residuals = residuals
            (stat, p_value) = shapiro(sw_residuals)
            significant = p_value < alpha
            assumption_met = not significant
            if not assumption_met:
                recommendation = 'Residuals are not normally distributed. Consider transforming the target variable or using non-parametric methods.'
            else:
                recommendation = 'Residuals appear to be normally distributed.'
            results['shapiro_wilk'] = {'statistic': float(stat), 'p_value': float(p_value), 'significant': significant, 'confidence_interval': None, 'assumption_met': assumption_met, 'recommendation': recommendation}
        except Exception as e:
            results['shapiro_wilk'] = {'statistic': None, 'p_value': None, 'significant': None, 'confidence_interval': None, 'assumption_met': None, 'recommendation': f'Test failed: {str(e)}'}
    if 'durbin_watson' in test_types:
        try:
            diff = np.diff(residuals)
            dw_stat = np.sum(diff ** 2) / np.sum(residuals ** 2) if np.sum(residuals ** 2) != 0 else 0
            assumption_met = 1.5 <= dw_stat <= 2.5
            significant = not assumption_met
            if dw_stat < 1.5:
                recommendation = 'Evidence of positive autocorrelation. Consider adding lag features or using time series models.'
            elif dw_stat > 2.5:
                recommendation = 'Evidence of negative autocorrelation. Investigate model specification.'
            else:
                recommendation = 'No significant autocorrelation detected.'
            results['durbin_watson'] = {'statistic': float(dw_stat), 'p_value': None, 'significant': significant, 'confidence_interval': None, 'assumption_met': assumption_met, 'recommendation': recommendation}
        except Exception as e:
            results['durbin_watson'] = {'statistic': None, 'p_value': None, 'significant': None, 'confidence_interval': None, 'assumption_met': None, 'recommendation': f'Test failed: {str(e)}'}
    if 't_test' in test_types:
        try:
            if not (hasattr(model, 'coef_') and hasattr(model, 'intercept_')):
                raise RuntimeError('Model must have coef_ and intercept_ attributes for t-test')
            coef = np.asarray(model.coef_)
            (n_samples, n_features) = X_array.shape
            mse = np.sum(residuals ** 2) / (n_samples - n_features - 1)
            X_with_intercept = np.column_stack([np.ones(n_samples), X_array])
            try:
                cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                std_errors = np.sqrt(np.diag(cov_matrix)[1:])
            except np.linalg.LinAlgError:
                cov_matrix = mse * np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
                std_errors = np.sqrt(np.diag(cov_matrix)[1:])
            t_stats = coef / std_errors
            dof = n_samples - n_features - 1
            if alternative == 'two-sided':
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))
            elif alternative == 'greater':
                p_values = 1 - stats.t.cdf(t_stats, df=dof)
            else:
                p_values = stats.t.cdf(t_stats, df=dof)
            significant = p_values < alpha
            n_significant = np.sum(significant)
            if n_significant == 0:
                recommendation = 'No coefficients are statistically significant. Consider collecting more data or revising feature selection.'
            elif n_significant < len(coef) // 2:
                recommendation = f'Only {n_significant} out of {len(coef)} coefficients are statistically significant. Consider feature selection.'
            else:
                recommendation = f'{n_significant} out of {len(coef)} coefficients are statistically significant.'
            results['t_test'] = {'statistic': float(np.mean(np.abs(t_stats))), 'p_value': float(np.mean(p_values)), 'significant': np.any(significant), 'confidence_interval': None, 'assumption_met': True, 'recommendation': recommendation}
        except Exception as e:
            results['t_test'] = {'statistic': None, 'p_value': None, 'significant': None, 'confidence_interval': None, 'assumption_met': None, 'recommendation': f'Test failed: {str(e)}'}
    if 'ks_test' in test_types:
        try:
            mean_resid = np.mean(residuals)
            std_resid = np.std(residuals, ddof=1)
            (ks_stat, p_value) = kstest(residuals, lambda x: stats.norm.cdf(x, mean_resid, std_resid))
            significant = p_value < alpha
            assumption_met = not significant
            if not assumption_met:
                recommendation = 'Residuals do not follow the expected distribution. Model assumptions may be violated.'
            else:
                recommendation = 'Residuals follow the expected distribution.'
            results['ks_test'] = {'statistic': float(ks_stat), 'p_value': float(p_value), 'significant': significant, 'confidence_interval': None, 'assumption_met': assumption_met, 'recommendation': recommendation}
        except Exception as e:
            results['ks_test'] = {'statistic': None, 'p_value': None, 'significant': None, 'confidence_interval': None, 'assumption_met': None, 'recommendation': f'Test failed: {str(e)}'}
    if 'breusch_pagan' in test_types:
        try:
            n_samples = len(residuals)
            residuals_squared = residuals ** 2
            X_with_intercept = np.column_stack([np.ones(n_samples), X_array])
            try:
                coef_aux = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ residuals_squared)
            except np.linalg.LinAlgError:
                coef_aux = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ residuals_squared
            y_aux_pred = X_with_intercept @ coef_aux
            ss_res = np.sum((residuals_squared - y_aux_pred) ** 2)
            ss_tot = np.sum((residuals_squared - np.mean(residuals_squared)) ** 2)
            r_squared_aux = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            lm_stat = n_samples * r_squared_aux
            df = X_array.shape[1]
            p_value = 1 - stats.chi2.cdf(lm_stat, df=df)
            significant = p_value < alpha
            assumption_met = not significant
            if not assumption_met:
                recommendation = 'Evidence of heteroscedasticity. Consider weighted least squares or variance-stabilizing transformations.'
            else:
                recommendation = 'No evidence of heteroscedasticity.'
            results['breusch_pagan'] = {'statistic': float(lm_stat), 'p_value': float(p_value), 'significant': significant, 'confidence_interval': None, 'assumption_met': assumption_met, 'recommendation': recommendation}
        except Exception as e:
            results['breusch_pagan'] = {'statistic': None, 'p_value': None, 'significant': None, 'confidence_interval': None, 'assumption_met': None, 'recommendation': f'Test failed: {str(e)}'}
    if 'f_test' in test_types:
        try:
            if not (hasattr(model, 'coef_') and hasattr(model, 'intercept_')):
                raise RuntimeError('Model must have coef_ and intercept_ attributes for F-test')
            (n_samples, n_features) = X_array.shape
            y_pred = model.predict(X)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            ss_res = np.sum(residuals ** 2)
            ss_reg = ss_tot - ss_res
            df_reg = n_features
            df_res = n_samples - n_features - 1
            ms_reg = ss_reg / df_reg if df_reg > 0 else 0
            ms_res = ss_res / df_res if df_res > 0 else 0
            f_stat = ms_reg / ms_res if ms_res > 0 else 0
            p_value = 1 - stats.f.cdf(f_stat, df_reg, df_res)
            significant = p_value < alpha
            assumption_met = True
            if significant:
                recommendation = 'Model is statistically significant. At least one predictor has a significant relationship with the response variable.'
            else:
                recommendation = 'Model is not statistically significant. No predictors have a significant relationship with the response variable.'
            results['f_test'] = {'statistic': float(f_stat), 'p_value': float(p_value), 'significant': significant, 'confidence_interval': None, 'assumption_met': assumption_met, 'recommendation': recommendation}
        except Exception as e:
            results['f_test'] = {'statistic': None, 'p_value': None, 'significant': None, 'confidence_interval': None, 'assumption_met': None, 'recommendation': f'Test failed: {str(e)}'}
    if 'likelihood_ratio' in test_types:
        results['likelihood_ratio'] = {'statistic': None, 'p_value': None, 'significant': None, 'confidence_interval': None, 'assumption_met': None, 'recommendation': 'Likelihood ratio test not implemented in this version. Requires nested models.'}
    return results