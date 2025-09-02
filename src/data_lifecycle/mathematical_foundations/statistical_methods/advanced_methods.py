from typing import Optional, Dict, Any, List, Tuple, Union
from general.base_classes.model_base import BaseModel
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from scipy import stats

class GoodnessOfFitTester(BaseValidator):

    def __init__(self, test_type: str, distribution: str, significance_level: float=0.05, name: Optional[str]=None):
        """
        Initialize the GoodnessOfFitTester.
        
        Parameters
        ----------
        test_type : str
            Statistical test to apply ('ks', 'chi2', 'anderson').
        distribution : str
            Target distribution name ('normal', 'exponential', etc.).
        significance_level : float
            Threshold for rejecting null hypothesis (default: 0.05).
        name : Optional[str]
            Validator name identifier.
        """
        super().__init__(name=name)
        if test_type not in ['ks', 'chi2', 'anderson']:
            raise ValueError("Unsupported test type. Choose from 'ks', 'chi2', or 'anderson'.")
        if distribution not in ['normal', 'exponential']:
            raise ValueError("Unsupported distribution. Currently supports 'normal' or 'exponential'.")
        self.test_type = test_type
        self.distribution = distribution
        self.significance_level = significance_level
        self.test_results: dict = {}
        self._params: dict = {}

    def validate(self, data: Union[DataBatch, np.ndarray], **kwargs) -> bool:
        """
        Execute the goodness-of-fit test on provided data.
        
        Args:
            data (Union[DataBatch, np.ndarray]): Sample data to test.
            **kwargs: Additional test configuration parameters.
            
        Returns:
            bool: True if data fits the distribution, False otherwise.
        """
        if isinstance(data, DataBatch):
            data_array = data.data.flatten()
        else:
            data_array = np.asarray(data).flatten()
        if self.distribution in ['normal', 'exponential'] and (not self._params):
            raise RuntimeError('Parameters not fitted. Call fit() before validate() for this distribution.')
        if self.test_type == 'ks':
            result = self._perform_ks_test(data_array)
        elif self.test_type == 'chi2':
            result = self._perform_chi2_test(data_array)
        elif self.test_type == 'anderson':
            result = self._perform_anderson_test(data_array)
        else:
            raise ValueError(f'Unsupported test type: {self.test_type}')
        return bool(result)

    def fit(self, data: Union[DataBatch, np.ndarray], **kwargs) -> 'GoodnessOfFitTester':
        """
        Pre-compute distribution parameters from reference data.
        
        Args:
            data (Union[DataBatch, np.ndarray]): Reference data for fitting.
            **kwargs: Fitting parameters.
            
        Returns:
            GoodnessOfFitTester: Instance with fitted parameters.
        """
        if isinstance(data, DataBatch):
            data_array = data.data.flatten()
        else:
            data_array = np.asarray(data).flatten()
        if self.distribution == 'normal':
            (mean, std) = stats.norm.fit(data_array)
            self._params['mean'] = float(mean)
            self._params['std'] = float(std)
        elif self.distribution == 'exponential':
            (_, scale) = stats.expon.fit(data_array, floc=0)
            self._params['scale'] = float(scale)
        return self

    def get_test_statistics(self) -> dict:
        """
        Retrieve detailed results from the most recent test.
        
        Returns:
            dict: Test statistics including p-values and critical values.
        """
        if not self.test_results:
            result = {'test_type': self.test_type, 'distribution': self.distribution}
            if self._params:
                params_list = []
                if self.distribution == 'normal':
                    params_list = [self._params['mean'], self._params['std']]
                elif self.distribution == 'exponential':
                    params_list = [self._params['scale']]
                result['params'] = params_list
            return result
        return self.test_results

    def _perform_ks_test(self, data: np.ndarray) -> bool:
        """Perform Kolmogorov-Smirnov test."""
        if self.distribution == 'normal':
            (ks_stat, p_value) = stats.kstest(data, 'norm', args=(self._params['mean'], self._params['std']))
        elif self.distribution == 'exponential':
            (ks_stat, p_value) = stats.kstest(data, 'expon', args=(0, self._params['scale']))
        else:
            raise ValueError(f'KS test not supported for distribution: {self.distribution}')
        params_list = []
        if self.distribution == 'normal':
            params_list = [self._params['mean'], self._params['std']]
        elif self.distribution == 'exponential':
            params_list = [self._params['scale']]
        self.test_results = {'test_type': self.test_type, 'distribution': self.distribution, 'statistic': float(ks_stat), 'p_value': float(p_value), 'critical_value': None, 'critical_values': None, 'params': params_list}
        return bool(p_value > self.significance_level)

    def _perform_chi2_test(self, data: np.ndarray) -> bool:
        """Perform Chi-Square test."""
        if len(data) < 10:
            params_list = []
            if self.distribution == 'normal':
                params_list = [self._params['mean'], self._params['std']]
            elif self.distribution == 'exponential':
                params_list = [self._params['scale']]
            self.test_results = {'test_type': self.test_type, 'distribution': self.distribution, 'statistic': None, 'p_value': None, 'critical_value': None, 'critical_values': None, 'params': params_list}
            return False
        (hist, bin_edges) = np.histogram(data, bins=min(10, max(3, len(data) // 5)))
        if self.distribution == 'normal':
            expected_prob = np.diff(stats.norm.cdf(bin_edges, self._params['mean'], self._params['std']))
        elif self.distribution == 'exponential':
            expected_prob = np.diff(stats.expon.cdf(bin_edges, scale=self._params['scale']))
        else:
            raise ValueError(f'Chi-square test not supported for distribution: {self.distribution}')
        expected_freq = expected_prob * len(data)
        combined_hist = []
        combined_expected = []
        i = 0
        while i < len(hist):
            h_sum = hist[i]
            e_sum = expected_freq[i]
            j = i + 1
            while e_sum < 5 and j < len(hist):
                h_sum += hist[j]
                e_sum += expected_freq[j]
                j += 1
            combined_hist.append(float(h_sum))
            combined_expected.append(float(e_sum))
            i = j if j > i + 1 else i + 1
        if len(combined_hist) < 2:
            params_list = []
            if self.distribution == 'normal':
                params_list = [self._params['mean'], self._params['std']]
            elif self.distribution == 'exponential':
                params_list = [self._params['scale']]
            self.test_results = {'test_type': self.test_type, 'distribution': self.distribution, 'statistic': None, 'p_value': None, 'critical_value': None, 'critical_values': None, 'params': params_list}
            return False
        hist_sum = sum(combined_hist)
        exp_sum = sum(combined_expected)
        if exp_sum > 0:
            combined_expected = [e * hist_sum / exp_sum for e in combined_expected]
        try:
            (chi2_stat, p_value) = stats.chisquare(combined_hist, combined_expected)
        except Exception:
            params_list = []
            if self.distribution == 'normal':
                params_list = [self._params['mean'], self._params['std']]
            elif self.distribution == 'exponential':
                params_list = [self._params['scale']]
            self.test_results = {'test_type': self.test_type, 'distribution': self.distribution, 'statistic': None, 'p_value': None, 'critical_value': None, 'critical_values': None, 'params': params_list}
            return False
        params_list = []
        if self.distribution == 'normal':
            params_list = [self._params['mean'], self._params['std']]
        elif self.distribution == 'exponential':
            params_list = [self._params['scale']]
        self.test_results = {'test_type': self.test_type, 'distribution': self.distribution, 'statistic': float(chi2_stat), 'p_value': float(p_value), 'critical_value': None, 'critical_values': None, 'params': params_list}
        return bool(p_value > self.significance_level) if p_value is not None else False

    def _perform_anderson_test(self, data: np.ndarray) -> bool:
        """Perform Anderson-Darling test."""
        if self.distribution == 'normal':
            result = stats.anderson(data, dist='norm')
            statistic = result.statistic
            critical_values = result.critical_values
            significance_levels = result.significance_level
            idx = np.argmin(np.abs(significance_levels - self.significance_level * 100))
            critical_value = critical_values[idx]
            params_list = [self._params['mean'], self._params['std']]
            self.test_results = {'test_type': self.test_type, 'distribution': self.distribution, 'statistic': float(statistic), 'p_value': None, 'critical_value': float(critical_value), 'critical_values': [float(cv) for cv in critical_values], 'significance_levels': [float(sl) for sl in significance_levels], 'params': params_list}
            return bool(statistic < critical_value)
        else:
            raise ValueError(f'Anderson-Darling test not supported for distribution: {self.distribution}')


# ...(code omitted)...


class BayesianParameterEstimator(BaseModel):
    """
    Estimates model parameters using Bayesian inference techniques.

    This class provides a framework for incorporating prior knowledge and updating
    beliefs about model parameters based on observed data. It supports conjugate
    priors for common distributions and Markov Chain Monte Carlo (MCMC) methods
    for complex posterior sampling.

    Attributes:
        model_type (str): Statistical model family ('normal', 'bernoulli', etc.).
        prior_params (Dict[str, Any]): Distribution parameters for prior beliefs.
        posterior_params (Dict[str, Any]): Updated parameters after observing data.
        samples (np.ndarray): Posterior samples if MCMC is used.

    Args:
        model_type (str): Family of distributions to model.
        prior_params (Dict[str, Any]): Initial parameter assumptions.
        method (str): Estimation technique ('conjugate', 'mcmc').
        name (Optional[str]): Model identifier.
    """

    def __init__(self, model_type: str, prior_params: Dict[str, Any], method: str='conjugate', name: Optional[str]=None):
        super().__init__(name=name)
        if model_type not in ['normal', 'bernoulli']:
            raise ValueError("Unsupported model type. Choose from 'normal' or 'bernoulli'.")
        if method not in ['conjugate', 'mcmc']:
            raise ValueError("Unsupported method. Choose from 'conjugate' or 'mcmc'.")
        self.model_type = model_type
        self.prior_params = prior_params
        self.method = method
        self.posterior_params: Dict[str, Any] = {}
        self.samples: Optional[np.ndarray] = None
        self._validate_prior_params()

    def _validate_prior_params(self):
        """Validate that prior parameters are appropriate for the model type."""
        if self.model_type == 'normal':
            provided_keys = set(self.prior_params.keys())
            has_mu_sigma2 = {'mu', 'sigma2'}.issubset(provided_keys)
            has_mean_var = {'mean', 'var'}.issubset(provided_keys)
            if not (has_mu_sigma2 or has_mean_var):
                raise ValueError("Normal model requires 'mu' and 'sigma2' (or 'mean' and 'var') in prior_params")
            if has_mu_sigma2:
                if not isinstance(self.prior_params['mu'], (int, float)):
                    raise ValueError("'mu' must be a numeric value")
                if not isinstance(self.prior_params['sigma2'], (int, float)) or self.prior_params['sigma2'] <= 0:
                    raise ValueError("'sigma2' must be a positive numeric value")
            elif has_mean_var:
                if not isinstance(self.prior_params['mean'], (int, float)):
                    raise ValueError("'mean' must be a numeric value")
                if not isinstance(self.prior_params['var'], (int, float)) or self.prior_params['var'] <= 0:
                    raise ValueError("'var' must be a positive numeric value")
        elif self.model_type == 'bernoulli':
            required_keys = {'alpha', 'beta'}
            provided_keys = set(self.prior_params.keys())
            if not required_keys.issubset(provided_keys):
                raise ValueError("Bernoulli model requires 'alpha' and 'beta' in prior_params")
            if not isinstance(self.prior_params['alpha'], (int, float)) or self.prior_params['alpha'] <= 0:
                raise ValueError("'alpha' must be a positive numeric value")
            if not isinstance(self.prior_params['beta'], (int, float)) or self.prior_params['beta'] <= 0:
                raise ValueError("'beta' must be a positive numeric value")

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'BayesianParameterEstimator':
        """
        Update parameter estimates using observed data.

        Args:
            X (Union[FeatureSet, np.ndarray]): Input features or observations.
            y (Optional[np.ndarray]): Target values if applicable.
            **kwargs: Method-specific fitting options.

        Returns:
            BayesianParameterEstimator: Instance with updated posteriors.
        """
        if isinstance(X, FeatureSet):
            data = X.features.flatten()
        else:
            data = np.asarray(X).flatten()
        if self.method == 'conjugate':
            self._fit_conjugate(data)
        elif self.method == 'mcmc':
            self._fit_mcmc(data, **kwargs)
        self.is_fitted = True
        return self

    def _fit_conjugate(self, data: np.ndarray):
        """Fit using conjugate prior approach."""
        if self.model_type == 'normal':
            if 'mu' in self.prior_params and 'sigma2' in self.prior_params:
                mu0 = self.prior_params['mu']
                sigma0_sq = self.prior_params['sigma2']
            elif 'mean' in self.prior_params and 'var' in self.prior_params:
                mu0 = self.prior_params['mean']
                sigma0_sq = self.prior_params['var']
            else:
                raise ValueError('Invalid prior parameters for normal model')
            n = len(data)
            x_bar = np.mean(data) if n > 0 else 0.0
            if n > 1:
                sigma_sq = np.var(data, ddof=1)
            elif n == 1:
                sigma_sq = 1.0
            else:
                sigma_sq = 1.0
            if sigma_sq <= 0:
                sigma_sq = 1.0
            if sigma0_sq <= 0:
                sigma0_sq = 1.0
            precision0 = 1.0 / sigma0_sq
            precision_data = n / sigma_sq if n > 0 else 0.0
            precision_post = precision0 + precision_data
            if n > 0 and precision_post > 0:
                mu_post = (precision0 * mu0 + precision_data * x_bar) / precision_post
            else:
                mu_post = mu0
            sigma_post_sq = 1.0 / precision_post if precision_post > 0 else sigma0_sq
            self.posterior_params = {'mu': mu_post, 'sigma2': sigma_post_sq}
        elif self.model_type == 'bernoulli':
            alpha0 = self.prior_params['alpha']
            beta0 = self.prior_params['beta']
            n = len(data)
            successes = np.sum(data) if n > 0 else 0
            alpha_post = alpha0 + successes
            beta_post = beta0 + n - successes
            self.posterior_params = {'alpha': alpha_post, 'beta': beta_post}

    def _fit_mcmc(self, data: np.ndarray, n_samples: int=1000, burn_in: int=100):
        """Fit using MCMC sampling approach."""
        if self.model_type == 'normal':
            if 'mu' in self.prior_params and 'sigma2' in self.prior_params:
                mu0 = self.prior_params['mu']
                sigma0_sq = self.prior_params['sigma2']
            elif 'mean' in self.prior_params and 'var' in self.prior_params:
                mu0 = self.prior_params['mean']
                sigma0_sq = self.prior_params['var']
            else:
                raise ValueError('Invalid prior parameters for normal model')
            if len(data) > 1:
                sigma_sq = np.var(data, ddof=1)
            elif len(data) == 1:
                sigma_sq = 1.0
            else:
                sigma_sq = 1.0
            if sigma_sq <= 0:
                sigma_sq = 1.0
            if sigma0_sq <= 0:
                sigma0_sq = 1.0
            samples = []
            current_mu = mu0
            proposal_std = np.sqrt(sigma0_sq) if sigma0_sq > 0 else 1.0
            for i in range(n_samples + burn_in):
                proposed_mu = np.random.normal(current_mu, proposal_std)
                log_prior_current = stats.norm.logpdf(current_mu, mu0, np.sqrt(sigma0_sq))
                log_prior_proposed = stats.norm.logpdf(proposed_mu, mu0, np.sqrt(sigma0_sq))
                log_likelihood_current = np.sum(stats.norm.logpdf(data, current_mu, np.sqrt(sigma_sq))) if len(data) > 0 else 0
                log_likelihood_proposed = np.sum(stats.norm.logpdf(data, proposed_mu, np.sqrt(sigma_sq))) if len(data) > 0 else 0
                log_accept_ratio = log_prior_proposed + log_likelihood_proposed - (log_prior_current + log_likelihood_current)
                if np.log(np.random.uniform()) < log_accept_ratio:
                    current_mu = proposed_mu
                if i >= burn_in:
                    samples.append(current_mu)
            self.samples = np.array(samples)
            if len(samples) > 0:
                self.posterior_params = {'mu': np.mean(self.samples), 'sigma2': np.var(self.samples) if len(self.samples) > 1 else 1.0}
            else:
                self.posterior_params = {'mu': mu0, 'sigma2': sigma0_sq}
        elif self.model_type == 'bernoulli':
            alpha0 = self.prior_params['alpha']
            beta0 = self.prior_params['beta']
            samples = []
            current_p = np.random.beta(alpha0, beta0) if alpha0 > 0 and beta0 > 0 else 0.5
            proposal_std = 0.1
            for i in range(n_samples + burn_in):
                proposed_p = current_p + np.random.normal(0, proposal_std)
                proposed_p = np.clip(proposed_p, 0.01, 0.99)
                log_prior_current = stats.beta.logpdf(current_p, alpha0, beta0) if alpha0 > 0 and beta0 > 0 else 0
                log_prior_proposed = stats.beta.logpdf(proposed_p, alpha0, beta0) if alpha0 > 0 and beta0 > 0 else 0
                log_likelihood_current = np.sum(stats.bernoulli.logpmf(data, current_p)) if len(data) > 0 else 0
                log_likelihood_proposed = np.sum(stats.bernoulli.logpmf(data, proposed_p)) if len(data) > 0 else 0
                log_accept_ratio = log_prior_proposed + log_likelihood_proposed - (log_prior_current + log_likelihood_current)
                if np.log(np.random.uniform()) < log_accept_ratio:
                    current_p = proposed_p
                if i >= burn_in:
                    samples.append(current_p)
            self.samples = np.array(samples)
            if len(samples) > 0:
                mean_p = np.mean(samples)
                total_samples = len(samples)
                self.posterior_params = {'alpha': mean_p * total_samples, 'beta': (1 - mean_p) * total_samples}
            else:
                self.posterior_params = {'alpha': alpha0, 'beta': beta0}

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Generate predictions using posterior parameter distributions.

        Args:
            X (Union[FeatureSet, np.ndarray]): Input features for prediction.
            **kwargs: Prediction configuration.

        Returns:
            np.ndarray: Predicted values or distribution samples.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before making predictions.')
        if isinstance(X, FeatureSet):
            data = X.features
        else:
            data = np.asarray(X)
        n_predictions = data.shape[0] if data.ndim > 0 else 1
        if self.method == 'conjugate':
            if self.model_type == 'normal':
                mu_post = self.posterior_params['mu']
                sigma_post_sq = self.posterior_params['sigma2']
                sigma_sq = sigma_post_sq if sigma_post_sq > 0 else 1.0
                return np.random.normal(mu_post, np.sqrt(sigma_sq), n_predictions)
            elif self.model_type == 'bernoulli':
                alpha_post = self.posterior_params['alpha']
                beta_post = self.posterior_params['beta']
                p = np.random.beta(alpha_post, beta_post) if alpha_post > 0 and beta_post > 0 else 0.5
                return np.random.binomial(1, p, n_predictions)
        elif self.samples is not None and len(self.samples) > 0:
            if self.model_type == 'normal':
                sampled_mu = np.random.choice(self.samples, size=n_predictions)
                if len(self.samples) > 1:
                    sample_var = np.var(self.samples)
                else:
                    sample_var = self.posterior_params.get('sigma2', 1.0)
                sigma_sq = sample_var if sample_var > 0 else 1.0
                return np.random.normal(sampled_mu, np.sqrt(sigma_sq))
            elif self.model_type == 'bernoulli':
                sampled_p = np.random.choice(self.samples, size=n_predictions)
                return np.random.binomial(1, sampled_p)
        else:
            raise RuntimeError('No posterior information available for prediction.')

    def score(self, X: Union[FeatureSet, np.ndarray], y: np.ndarray, **kwargs) -> float:
        """
        Evaluate model performance using posterior predictive checks.

        Args:
            X (Union[FeatureSet, np.ndarray]): Test features.
            y (np.ndarray): True target values.
            **kwargs: Scoring options.

        Returns:
            float: Log-probability or divergence metric.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before scoring.')
        if isinstance(X, FeatureSet):
            data = X.features.flatten()
        else:
            data = np.asarray(X).flatten()
        y = np.asarray(y).flatten()
        if len(data) != len(y):
            raise ValueError('Length of X and y must match.')
        if self.method == 'conjugate':
            if self.model_type == 'normal':
                mu_post = self.posterior_params['mu']
                sigma_post_sq = self.posterior_params['sigma2']
                sigma_sq = sigma_post_sq if sigma_post_sq > 0 else 1.0
                predictive_sigma_sq = sigma_sq + sigma_post_sq
                log_probs = stats.norm.logpdf(y, mu_post, np.sqrt(predictive_sigma_sq))
                return float(np.mean(log_probs))
            elif self.model_type == 'bernoulli':
                alpha_post = self.posterior_params['alpha']
                beta_post = self.posterior_params['beta']
                expected_p = alpha_post / (alpha_post + beta_post) if alpha_post + beta_post > 0 else 0.5
                log_probs = stats.bernoulli.logpmf(y, expected_p)
                return float(np.mean(log_probs))
        elif self.samples is not None and len(self.samples) > 0:
            if self.model_type == 'normal':
                log_probs = []
                for sample in self.samples[::max(1, len(self.samples) // 100)]:
                    log_prob = np.mean(stats.norm.logpdf(y, sample, np.sqrt(self.posterior_params.get('sigma2', 1.0))))
                    log_probs.append(log_prob)
                return float(np.mean(log_probs)) if log_probs else -np.inf
            elif self.model_type == 'bernoulli':
                log_probs = []
                for sample in self.samples[::max(1, len(self.samples) // 100)]:
                    log_prob = np.mean(stats.bernoulli.logpmf(y, sample))
                    log_probs.append(log_prob)
                return float(np.mean(log_probs)) if log_probs else -np.inf
        else:
            raise RuntimeError('No posterior information available for scoring.')

    def get_credible_interval(self, percentile: float=95.0) -> Dict[str, Tuple[float, float]]:
        """
        Compute credible intervals for model parameters.

        Args:
            percentile (float): Width of the credible interval (default: 95%).

        Returns:
            Dict[str, Tuple[float, float]]: Lower and upper bounds for each parameter.
        """
        if not self.is_fitted:
            raise RuntimeError('Model must be fitted before computing credible intervals.')
        alpha = (100 - percentile) / 2.0
        lower_percentile = alpha
        upper_percentile = 100 - alpha
        if self.method == 'conjugate':
            if self.model_type == 'normal':
                mu = self.posterior_params['mu']
                sigma_sq = self.posterior_params['sigma2']
                sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 1.0
                mu_lower = stats.norm.ppf(lower_percentile / 100.0, mu, sigma)
                mu_upper = stats.norm.ppf(upper_percentile / 100.0, mu, sigma)
                sigma_lower = stats.norm.ppf(lower_percentile / 100.0, sigma_sq, sigma_sq * 0.1)
                sigma_upper = stats.norm.ppf(upper_percentile / 100.0, sigma_sq, sigma_sq * 0.1)
                return {'mu': (mu_lower, mu_upper), 'sigma2': (max(0, sigma_lower), max(0, sigma_upper))}
            elif self.model_type == 'bernoulli':
                alpha_param = self.posterior_params['alpha']
                beta_param = self.posterior_params['beta']
                lower = stats.beta.ppf(lower_percentile / 100.0, alpha_param, beta_param) if alpha_param > 0 and beta_param > 0 else 0.0
                upper = stats.beta.ppf(upper_percentile / 100.0, alpha_param, beta_param) if alpha_param > 0 and beta_param > 0 else 1.0
                return {'p': (lower, upper)}
        elif self.samples is not None and len(self.samples) > 0:
            lower = np.percentile(self.samples, lower_percentile)
            upper = np.percentile(self.samples, upper_percentile)
            if self.model_type == 'normal':
                return {'mu': (lower, upper)}
            elif self.model_type == 'bernoulli':
                return {'p': (lower, upper)}
        else:
            raise RuntimeError('No posterior information available for credible intervals.')