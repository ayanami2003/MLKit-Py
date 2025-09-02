from typing import Optional, Dict, Any, Union
import numpy as np
from scipy.special import psi
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class VariationalBayesClustering(BaseModel):

    def __init__(self, n_components: int=10, *, covariance_type: str='full', tol: float=0.001, reg_covar: float=1e-06, max_iter: int=100, n_init: int=1, init_params: str='kmeans', alpha: float=1.0, beta: float=1.0, degrees_of_freedom: Optional[float]=None, scale_prior: Optional[np.ndarray]=None, random_state: Optional[int]=None, warm_start: bool=False, verbose: int=0, verbose_interval: int=10, name: Optional[str]=None):
        """
        Initialize the VariationalBayesClustering model.
        
        Args:
            n_components: Maximum number of mixture components. Defaults to 10.
            covariance_type: Type of covariance parameters to use. Must be one of:
                           'full' (each component has its own general covariance matrix),
                           'tied' (all components share the same general covariance matrix),
                           'diag' (each component has its own diagonal covariance matrix),
                           'spherical' (each component has its own single variance).
                           Defaults to 'full'.
            tol: Convergence threshold. EM iterations will stop when the lower bound
                 average gain is below this threshold. Defaults to 1e-3.
            reg_covar: Non-negative regularization added to the diagonal of covariance.
                      Allows to assure that the covariance matrices are all positive.
                      Defaults to 1e-6.
            max_iter: Maximum number of EM iterations. Defaults to 100.
            n_init: Number of initializations to perform. The best results are kept.
                  Defaults to 1.
            init_params: Method for initialization. Must be one of:
                        'kmeans' (use KMeans clustering for initialization),
                        'random' (generate randomly initialized parameters).
                        Defaults to 'kmeans'.
            alpha: Concentration parameter for the Dirichlet prior on the mixing weights.
                  Defaults to 1.0.
            beta: Precision prior for the distribution of the mean. Controls how much
                 the means can deviate from the prior mean. Defaults to 1.0.
            degrees_of_freedom: Degrees of freedom for the Wishart prior on the precision.
                               If None, defaults to the dimensionality of the data plus 1.
                               Must be greater than or equal to the dimensionality.
            scale_prior: Scale matrix for the Wishart prior on the precision. If None,
                        defaults to the identity matrix. Must be symmetric and positive definite.
            random_state: Random state for reproducibility. Defaults to None.
            warm_start: If True, reuse the solution from the previous fit as initialization.
                       Otherwise, just erase the previous solution. Defaults to False.
            verbose: Enable verbose output. If 1 then it prints the current initialization
                    and each iteration step. If greater than 1 then it prints also the
                    change and time needed for each step. Defaults to 0.
            verbose_interval: Number of iterations between verbose outputs. Defaults to 10.
            name: Optional name for the model instance. Defaults to class name.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.alpha = alpha
        self.beta = beta
        self.degrees_of_freedom = degrees_of_freedom
        self.scale_prior = scale_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def _validate_inputs(self, X: np.ndarray) -> bool:
        """Validate input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data must be a numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError('Input data must have at least one sample and one feature')
        if self.covariance_type not in ['full', 'tied', 'diag', 'spherical']:
            raise ValueError("covariance_type must be one of 'full', 'tied', 'diag', 'spherical'")
        if self.init_params not in ['kmeans', 'random']:
            raise ValueError("init_params must be one of 'kmeans', 'random'")
        if self.n_components <= 0:
            raise ValueError('n_components must be positive')
        if self.tol <= 0:
            raise ValueError('tol must be positive')
        if self.reg_covar < 0:
            raise ValueError('reg_covar must be non-negative')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if self.n_init <= 0:
            raise ValueError('n_init must be positive')
        return True

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """Initialize model parameters."""
        (n_samples, n_features) = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.init_params == 'kmeans':
            n_components_to_use = min(self.n_components, n_samples)
            indices = np.random.choice(n_samples, size=n_components_to_use, replace=False)
            self.means_ = np.zeros((self.n_components, n_features))
            self.means_[:n_components_to_use] = X[indices].copy()
            if n_components_to_use < self.n_components:
                data_min = X.min(axis=0)
                data_max = X.max(axis=0)
                self.means_[n_components_to_use:] = np.random.uniform(data_min, data_max, (self.n_components - n_components_to_use, n_features))
        else:
            data_min = X.min(axis=0)
            data_max = X.max(axis=0)
            self.means_ = np.random.uniform(data_min, data_max, (self.n_components, n_features))
        if self.covariance_type == 'full':
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances_ = np.eye(n_features)
        elif self.covariance_type == 'diag':
            self.covariances_ = np.ones((self.n_components, n_features))
        else:
            self.covariances_ = np.ones(self.n_components)
        self.weights_ = np.ones(self.n_components) / self.n_components
        if self.degrees_of_freedom is None:
            self.degrees_of_freedom = n_features + 1
        if self.scale_prior is None:
            self.scale_prior = np.eye(n_features)
        self.weight_concentration_ = np.full(self.n_components, self.alpha)
        self.mean_precision_ = np.full(self.n_components, self.beta)
        self.degrees_of_freedom_ = np.full(self.n_components, self.degrees_of_freedom)
        if self.covariance_type == 'full':
            self.covariance_scale_ = np.tile(self.scale_prior, (self.n_components, 1, 1))
        else:
            self.covariance_scale_ = self.scale_prior.copy()

    def _compute_precision_cholesky(self, covariances: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute the Cholesky decomposition of the precisions."""
        if self.covariance_type == 'full':
            (n_components, n_features, _) = covariances.shape
            precisions_chol = np.empty((n_components, n_features, n_features))
            for k in range(n_components):
                try:
                    cov_chol = np.linalg.cholesky(covariances[k])
                    precisions_chol[k] = np.linalg.inv(cov_chol).T
                except np.linalg.LinAlgError:
                    cov_regularized = covariances[k] + self.reg_covar * np.eye(covariances[k].shape[0])
                    cov_chol = np.linalg.cholesky(cov_regularized)
                    precisions_chol[k] = np.linalg.inv(cov_chol).T
            return precisions_chol
        elif self.covariance_type == 'tied':
            n_features = covariances.shape[0]
            try:
                cov_chol = np.linalg.cholesky(covariances)
                return np.linalg.inv(cov_chol).T
            except np.linalg.LinAlgError:
                cov_regularized = covariances + self.reg_covar * np.eye(n_features)
                cov_chol = np.linalg.cholesky(cov_regularized)
                return np.linalg.inv(cov_chol).T
        elif self.covariance_type == 'diag':
            return np.sqrt(1.0 / (covariances + self.reg_covar))
        else:
            return np.sqrt(1.0 / (covariances + self.reg_covar))

    def _estimate_log_weights(self) -> np.ndarray:
        """Estimate log weights."""
        digamma_sum = psi(np.sum(self.weight_concentration_))
        digamma_alpha = psi(self.weight_concentration_)
        return digamma_alpha - digamma_sum

    def _estimate_log_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the log probability of data under each component.
        
        Returns:
            np.ndarray: Array of shape (n_samples, n_components) with log probabilities
        """
        (n_samples, n_features) = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        if self.covariance_type == 'full':
            precisions_chol = self._compute_precision_cholesky(self.covariances_)
            log_det_precisions = np.sum(np.log(np.diagonal(precisions_chol, axis1=1, axis2=2)), axis=1)
            precisions = np.einsum('kij,kjl->kil', precisions_chol, precisions_chol.transpose(0, 2, 1))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                log_prob[:, k] = np.sum(np.dot(diff, precisions[k]) * diff, axis=1)
        elif self.covariance_type == 'tied':
            precisions_chol = self._compute_precision_cholesky(self.covariances_)
            log_det_precisions = np.sum(np.log(np.diag(precisions_chol)))
            precisions = np.dot(precisions_chol, precisions_chol.T)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                log_prob[:, k] = np.sum(np.dot(diff, precisions) * diff, axis=1)
        elif self.covariance_type == 'diag':
            precisions = 1.0 / (self.covariances_ + self.reg_covar)
            log_det_precisions = np.sum(np.log(precisions), axis=1)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                log_prob[:, k] = np.sum(diff ** 2 * precisions[k], axis=1)
        else:
            precisions = 1.0 / (self.covariances_ + self.reg_covar)
            log_det_precisions = n_features * np.log(precisions)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                log_prob[:, k] = np.sum(diff ** 2, axis=1) * precisions[k]
        if self.covariance_type in ['tied', 'spherical']:
            log_det_precisions = np.full(self.n_components, log_det_precisions)
        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det_precisions[np.newaxis, :]

    def _compute_lower_bound(self, log_resp: np.ndarray) -> float:
        """Compute the lower bound of the model evidence."""
        return np.sum(log_resp)

    def _e_step(self, X: np.ndarray):
        """E step: compute responsibilities."""
        log_prob = self._estimate_log_prob(X)
        log_weights = self._estimate_log_weights()
        log_prob_norm = log_prob + log_weights[np.newaxis, :]
        log_prob_sum = np.log(np.sum(np.exp(log_prob_norm), axis=1))
        log_resp = log_prob_norm - log_prob_sum[:, np.newaxis]
        return (log_prob_sum, log_resp)

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray) -> None:
        """M step: update parameters based on responsibilities."""
        (n_samples, n_features) = X.shape
        resp = np.exp(log_resp)
        weights = resp.sum(axis=0) + 1e-10
        self.weights_ = weights / n_samples
        weighted_X_sum = np.dot(resp.T, X)
        self.means_ = weighted_X_sum / weights[:, np.newaxis]
        if self.covariance_type == 'full':
            self.covariances_ = np.empty((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.dot(resp[:, k] * diff.T, diff) / weights[k]
                self.covariances_[k].flat[::n_features + 1] += self.reg_covar
        elif self.covariance_type == 'tied':
            self.covariances_ = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_ += np.dot(resp[:, k] * diff.T, diff)
            self.covariances_ /= n_samples
            self.covariances_.flat[::n_features + 1] += self.reg_covar
        elif self.covariance_type == 'diag':
            self.covariances_ = np.empty((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k][:, np.newaxis] * diff ** 2, axis=0) / weights[k]
                self.covariances_[k] += self.reg_covar
        else:
            self.covariances_ = np.empty(self.n_components)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k] * np.sum(diff ** 2, axis=1)) / (n_features * weights[k])
                self.covariances_[k] += self.reg_covar

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'VariationalBayesClustering':
        """
        Estimate model parameters with the variational inference algorithm.
        
        Args:
            X: Training data. Can be either:
               - FeatureSet: Structured feature data with metadata
               - np.ndarray: Array-like of shape (n_samples, n_features)
            y: Not used, present for API consistency by convention. Defaults to None.
            **kwargs: Additional keyword arguments for fitting
            
        Returns:
            VariationalBayesClustering: Fitted model instance
            
        Raises:
            ValueError: If the input data is invalid or incompatible with model settings
        """
        if isinstance(X, FeatureSet):
            X_array = X.X
        else:
            X_array = X
        self._validate_inputs(X_array)
        if not self.warm_start or not hasattr(self, 'means_'):
            self._initialize_parameters(X_array)
        max_lower_bound = -np.inf
        best_params = None
        self.fit_status_ = 0
        for init in range(self.n_init):
            if init > 0:
                self._initialize_parameters(X_array)
            prev_lower_bound = -np.inf
            for iteration in range(self.max_iter):
                (log_prob_norm, log_resp) = self._e_step(X_array)
                self._m_step(X_array, log_resp)
                lower_bound = self._compute_lower_bound(log_resp)
                change = lower_bound - prev_lower_bound
                if abs(change) < self.tol:
                    if self.verbose:
                        print(f'Converged at iteration {iteration}: change={change}')
                    self.fit_status_ = 1
                    break
                prev_lower_bound = lower_bound
                if self.verbose and iteration % self.verbose_interval == 0:
                    print(f'Iteration {iteration}: lower bound = {lower_bound}')
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = {'weights_': self.weights_.copy(), 'means_': self.means_.copy(), 'covariances_': self.covariances_.copy() if hasattr(self.covariances_, 'copy') else self.covariances_}
        if best_params is not None:
            self.weights_ = best_params['weights_']
            self.means_ = best_params['means_']
            self.covariances_ = best_params['covariances_']
        self.n_features_in_ = X_array.shape[1]
        self.fitted_ = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict the labels for the data samples in X using trained model.
        
        Args:
            X: New data to predict. Can be either:
               - FeatureSet: Structured feature data with metadata
               - np.ndarray: Array-like of shape (n_samples, n_features)
            **kwargs: Additional keyword arguments for prediction
            
        Returns:
            np.ndarray: Array of shape (n_samples,) with predicted cluster labels
            
        Raises:
            RuntimeError: If the model has not been fitted yet
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise RuntimeError('Model must be fitted before making predictions')
        if isinstance(X, FeatureSet):
            X_array = X.X
        else:
            X_array = X
        prob = self.predict_proba(X_array)
        return np.argmax(prob, axis=1)

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Predict posterior probability of each component given the data.
        
        Args:
            X: New data for which to predict probabilities. Can be either:
               - FeatureSet: Structured feature data with metadata
               - np.ndarray: Array-like of shape (n_samples, n_features)
            **kwargs: Additional keyword arguments for prediction
            
        Returns:
            np.ndarray: Array of shape (n_samples, n_components) with posterior probabilities
            
        Raises:
            RuntimeError: If the model has not been fitted yet
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise RuntimeError('Model must be fitted before computing probabilities')
        if isinstance(X, FeatureSet):
            X_array = X.X
        else:
            X_array = X
        log_prob = self._estimate_log_prob(X_array)
        log_weights = self._estimate_log_weights()
        log_prob_norm = log_prob + log_weights[np.newaxis, :]
        log_prob_norm -= np.max(log_prob_norm, axis=1, keepdims=True)
        prob = np.exp(log_prob_norm)
        prob /= np.sum(prob, axis=1, keepdims=True)
        return prob

    def score(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> float:
        """
        Compute the per-sample average log-likelihood of the given data X.
        
        Args:
            X: Test samples. Can be either:
               - FeatureSet: Structured feature data with metadata
               - np.ndarray: Array-like of shape (n_samples, n_features)
            y: Not used, present for API consistency by convention. Defaults to None.
            **kwargs: Additional keyword arguments for scoring
            
        Returns:
            float: Average log-likelihood of the samples under the model
            
        Raises:
            RuntimeError: If the model has not been fitted yet
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise RuntimeError('Model must be fitted before scoring')
        if isinstance(X, FeatureSet):
            X_array = X.X
        else:
            X_array = X
        log_prob = self._estimate_log_prob(X_array)
        log_weights = self._estimate_log_weights()
        weighted_log_prob = log_prob + log_weights[np.newaxis, :]
        max_weighted_log_prob = np.max(weighted_log_prob, axis=1, keepdims=True)
        log_prob_norm = max_weighted_log_prob.squeeze() + np.log(np.sum(np.exp(weighted_log_prob - max_weighted_log_prob), axis=1))
        return float(np.mean(log_prob_norm))