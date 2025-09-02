from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import numpy as np
from typing import Optional, Union, List, Dict, Any
from scipy.stats import f as f_dist
from scipy.stats import chi2 as chi2_dist
import pandas as pd

class GrangerCausalityTester(BaseTransformer):

    def __init__(self, max_lag: int=5, target_column: str='target', candidate_cause_column: str='cause', conditioning_columns: Optional[List[str]]=None, significance_level: float=0.05, test_type: str='f-test', name: Optional[str]=None):
        """
        Initialize the Granger causality tester.
        
        Parameters
        ----------
        max_lag : int, default=5
            Maximum number of lags to consider in the test
        target_column : str, default='target'
            Name of the dependent variable
        candidate_cause_column : str, default='cause'
            Name of the variable being tested for causality
        conditioning_columns : Optional[List[str]], default=None
            Additional variables to condition on in the test
        significance_level : float, default=0.05
            Significance level for hypothesis testing
        test_type : str, default='f-test'
            Type of statistical test to perform ('f-test' or 'chi2')
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.max_lag = max_lag
        self.target_column = target_column
        self.candidate_cause_column = candidate_cause_column
        self.conditioning_columns = conditioning_columns
        self.significance_level = significance_level
        self.test_type = test_type
        self._results = {}
        self._fitted = False

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'GrangerCausalityTester':
        """
        Fit the Granger causality tester to the input data.
        
        This method validates the input data structure and prepares
        internal state for conducting causality tests.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data containing target and potential cause variables
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        GrangerCausalityTester
            Self instance for method chaining
        """
        if isinstance(data, DataBatch):
            if hasattr(data, 'features') and isinstance(data.features, pd.DataFrame):
                df = data.features
            elif hasattr(data, 'data') and isinstance(data.data, pd.DataFrame):
                df = data.data
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, FeatureSet):
            if hasattr(data, 'data') and isinstance(data.data, pd.DataFrame):
                df = data.data
            else:
                df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Extracted data must be a pandas DataFrame')
        required_columns = [self.target_column, self.candidate_cause_column]
        if self.conditioning_columns:
            required_columns.extend(self.conditioning_columns)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns: {missing_cols}')
        self._data = df.copy()
        self._fitted = True
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Apply Granger causality testing to the input data.
        
        Conducts the causality test and returns results including test statistics,
        p-values, and causality determination.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Time series data to analyze
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Results of the causality test including:
            - test_statistic: Value of the test statistic
            - p_value: P-value of the test
            - is_causal: Boolean indicating if causality is detected
            - optimal_lag: Lag that minimizes information criterion
            - critical_value: Critical value at specified significance level
        """
        if not self._fitted:
            raise RuntimeError('Transformer must be fitted before transform')
        optimal_lag = self._select_optimal_lag()
        test_results = self._perform_granger_test(optimal_lag)
        self._results = {'test_statistic': test_results['test_statistic'], 'p_value': test_results['p_value'], 'is_significant': test_results['p_value'] < self.significance_level, 'optimal_lag': optimal_lag, 'critical_value': test_results['critical_value'], 'degrees_of_freedom': test_results['degrees_of_freedom']}
        result_dict = {'test_statistic': test_results['test_statistic'], 'p_value': test_results['p_value'], 'is_causal': test_results['p_value'] < self.significance_level, 'optimal_lag': optimal_lag, 'critical_value': test_results['critical_value']}
        return FeatureSet(data=pd.DataFrame([result_dict]), name=f'{self.name}_causality_results')

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not applicable for causality testing.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Empty feature set as inverse transformation is not meaningful
        """
        return FeatureSet(data=pd.DataFrame(), name=f'{self.name}_inverse')

    def get_causality_result(self) -> dict:
        """
        Retrieve detailed results of the last causality test.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'test_statistic': Test statistic value
            - 'p_value': P-value of the test
            - 'is_significant': Whether result is statistically significant
            - 'optimal_lag': Optimal lag order selected
            - 'critical_value': Critical value at significance level
            - 'degrees_of_freedom': Degrees of freedom in the test
        """
        if not self._results:
            raise RuntimeError('No test results available. Run transform first.')
        return self._results.copy()

    def _select_optimal_lag(self) -> int:
        """
        Select optimal lag order using Akaike Information Criterion (AIC).
        
        Returns
        -------
        int
            Optimal lag order
        """
        restricted_vars = [self.target_column]
        if self.conditioning_columns:
            restricted_vars.extend(self.conditioning_columns)
        restricted_data = self._data[restricted_vars].dropna()
        aic_values = []
        lag_range = range(1, min(self.max_lag + 1, len(restricted_data)))
        for lag in lag_range:
            try:
                (X_restricted, y_restricted) = self._create_lagged_matrices(restricted_data, [self.target_column], self.conditioning_columns or [], lag)
                if len(y_restricted) == 0:
                    continue
                coef_restricted = np.linalg.lstsq(X_restricted, y_restricted, rcond=None)[0]
                residuals_restricted = y_restricted - X_restricted @ coef_restricted
                ssr_restricted = np.sum(residuals_restricted ** 2)
                n = len(y_restricted)
                k = X_restricted.shape[1]
                aic = n * np.log(ssr_restricted / n) + 2 * k
                aic_values.append((lag, aic))
            except np.linalg.LinAlgError:
                continue
        if not aic_values:
            return 1
        return min(aic_values, key=lambda x: x[1])[0]

    def _create_lagged_matrices(self, data: pd.DataFrame, target_vars: List[str], conditioning_vars: List[str], lag: int) -> tuple:
        """
        Create lagged data matrices for regression.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        target_vars : List[str]
            Target variables
        conditioning_vars : List[str]
            Conditioning variables
        lag : int
            Number of lags to include
            
        Returns
        -------
        tuple
            (X, y) matrices for regression
        """
        all_vars = target_vars + conditioning_vars
        n = len(data)
        if n <= lag:
            return (np.array([]).reshape(0, 0), np.array([]))
        y_indices = [data.columns.get_loc(var) for var in target_vars]
        y_data = []
        X_data = []
        for i in range(lag, n):
            y_data.append(data.iloc[i, y_indices[0]])
            row = []
            for l in range(1, lag + 1):
                for var in all_vars:
                    row.append(data.iloc[i - l, data.columns.get_loc(var)])
            X_data.append(row)
        if len(y_data) == 0:
            return (np.array([]).reshape(0, 0), np.array([]))
        return (np.array(X_data), np.array(y_data))

    def _perform_granger_test(self, lag: int) -> Dict[str, Any]:
        """
        Perform Granger causality test with specified lag order.
        
        Parameters
        ----------
        lag : int
            Lag order to use in the test
            
        Returns
        -------
        Dict[str, Any]
            Test results including statistic, p-value, and critical value
        """
        all_vars = [self.target_column, self.candidate_cause_column]
        if self.conditioning_columns:
            all_vars.extend(self.conditioning_columns)
        data_subset = self._data[all_vars].dropna()
        if len(data_subset) <= lag:
            raise ValueError('Not enough data points for specified lag order')
        restricted_vars = [self.target_column]
        if self.conditioning_columns:
            restricted_vars.extend(self.conditioning_columns)
        restricted_data = self._data[restricted_vars].dropna()
        (X_r, y_r) = self._create_lagged_matrices(restricted_data, [self.target_column], self.conditioning_columns or [], lag)
        (X_ur, y_ur) = self._create_lagged_matrices(data_subset, [self.target_column], [self.candidate_cause_column] + (self.conditioning_columns or []), lag)
        min_len = min(len(y_r), len(y_ur))
        if min_len == 0:
            raise ValueError('Unable to create valid regression data')
        y_r = y_r[-min_len:]
        y_ur = y_ur[-min_len:]
        X_r = X_r[-min_len:]
        X_ur = X_ur[-min_len:]
        coef_r = np.linalg.lstsq(X_r, y_r, rcond=None)[0]
        coef_ur = np.linalg.lstsq(X_ur, y_ur, rcond=None)[0]
        residuals_r = y_r - X_r @ coef_r
        residuals_ur = y_ur - X_ur @ coef_ur
        ssr_r = np.sum(residuals_r ** 2)
        ssr_ur = np.sum(residuals_ur ** 2)
        n = len(y_ur)
        k_ur = X_ur.shape[1]
        k_r = X_r.shape[1]
        if self.test_type == 'f-test':
            df1 = k_ur - k_r
            df2 = n - k_ur
            if df1 <= 0 or df2 <= 0:
                raise ValueError('Invalid degrees of freedom for F-test')
            f_stat = (ssr_r - ssr_ur) / df1 / (ssr_ur / df2)
            p_value = 1 - f_dist.cdf(f_stat, df1, df2)
            critical_value = f_dist.ppf(1 - self.significance_level, df1, df2)
            return {'test_statistic': f_stat, 'p_value': p_value, 'critical_value': critical_value, 'degrees_of_freedom': (df1, df2)}
        elif self.test_type == 'chi2':
            df = k_ur - k_r
            chi2_stat = n * (np.log(ssr_r) - np.log(ssr_ur))
            p_value = 1 - chi2_dist.cdf(chi2_stat, df)
            critical_value = chi2_dist.ppf(1 - self.significance_level, df)
            return {'test_statistic': chi2_stat, 'p_value': p_value, 'critical_value': critical_value, 'degrees_of_freedom': df}
        else:
            raise ValueError(f'Unsupported test type: {self.test_type}')