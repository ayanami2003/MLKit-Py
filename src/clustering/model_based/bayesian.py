from typing import Optional, Any
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
import numpy as np
from scipy.special import digamma, logsumexp, gammaln


class VariationalBayesClustering(BaseModel):

    def __init__(self, n_components: int=1, covariance_type: str='full', tol: float=0.001, max_iter: int=100, init_params: str='kmeans', alpha_prior: Optional[float]=None, beta_prior: Optional[float]=None, degrees_of_freedom_prior: Optional[float]=None, covariance_prior: Optional[np.ndarray]=None, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the VariationalBayesClustering model.

        Args:
            n_components (int): Number of mixture components (clusters). Defaults to 1.
            covariance_type (str): Type of covariance - 'full', 'tied', 'diag', 'spherical'. Defaults to 'full'.
            tol (float): Convergence tolerance for ELBO changes. Defaults to 1e-3.
            max_iter (int): Maximum number of EM iterations. Defaults to 100.
            init_params (str): Initialization method ('kmeans', 'random'). Defaults to 'kmeans'.
            alpha_prior (Optional[float]): Dirichlet concentration prior. If None, uses 1/n_components.
            beta_prior (Optional[float]): Precision prior for mean distribution. If None, estimated from data.
            degrees_of_freedom_prior (Optional[float]): Degrees of freedom for Wishart prior. If None, estimated.
            covariance_prior (Optional[np.ndarray]): Prior for covariance matrices. If None, estimated from data.
            random_state (Optional[int]): Random seed for reproducibility. If None, uses random initialization.
            name (Optional[str]): Name of the model instance. If None, uses class name.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.init_params = init_params
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'VariationalBayesClustering':
        """
        Fit the variational Bayes clustering model to the data.

        Args:
            X (FeatureSet): Input features to cluster.
            y (Optional[Any]): Ignored in unsupervised learning.
            **kwargs: Additional fitting parameters.

        Returns:
            VariationalBayesClustering: Fitted model instance.
        """
        if isinstance(X, FeatureSet):
            X_array = X.get_feature_matrix()
        else:
            X_array = np.asarray(X)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        (n_samples, n_features) = X_array.shape
        if self.alpha_prior is None:
            self.alpha_prior = 1.0 / self.n_components
        if self.beta_prior is None:
            self.beta_prior = 1.0
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior = n_features
        if self.init_params == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=10)
            kmeans.fit(X_array)
            means = kmeans.cluster_centers_
        else:
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            means = X_array[indices].copy()
        if self.covariance_prior is None:
            cov = np.cov(X_array.T) + 1e-06 * np.eye(n_features)
            if self.covariance_type == 'full':
                self.covariance_prior = cov
            elif self.covariance_type == 'tied':
                self.covariance_prior = cov
            elif self.covariance_type == 'diag':
                self.covariance_prior = np.diag(cov)
            elif self.covariance_type == 'spherical':
                self.covariance_prior = np.mean(np.diag(cov))
        resp = np.random.dirichlet(np.ones(self.n_components), size=n_samples)
        alpha = np.full(self.n_components, self.alpha_prior)
        beta = np.full(self.n_components, self.beta_prior)
        means_var = means.copy()
        degrees_of_freedom = np.full(self.n_components, self.degrees_of_freedom_prior)
        if self.covariance_type == 'full':
            covariances = np.tile(self.covariance_prior, (self.n_components, 1, 1))
        elif self.covariance_type == 'tied':
            covariances = self.covariance_prior.copy()
        elif self.covariance_type == 'diag':
            covariances = np.tile(self.covariance_prior, (self.n_components, 1))
        elif self.covariance_type == 'spherical':
            covariances = np.full(self.n_components, self.covariance_prior)
        prev_elbo = -np.inf
        elbo_history = []
        self.converged_ = False
        for iteration in range(self.max_iter):
            resp = self._update_responsibilities(X_array, resp, alpha, beta, means_var, degrees_of_freedom, covariances)
            Nk = resp.sum(axis=0) + 1e-10
            alpha = self.alpha_prior + Nk
            beta = self.beta_prior + Nk
            xk = np.dot(resp.T, X_array) / Nk[:, np.newaxis]
            means_var = (self.beta_prior * means + Nk[:, np.newaxis] * xk) / beta[:, np.newaxis]
            if self.covariance_type == 'full':
                for k in range(self.n_components):
                    diff = X_array - means_var[k]
                    covariances[k] = (self.degrees_of_freedom_prior * self.covariance_prior + np.dot(resp[:, k] * diff.T, diff) + self.beta_prior * Nk[k] / (self.beta_prior + Nk[k]) * np.outer(means_var[k] - means[k], means_var[k] - means[k])) / degrees_of_freedom[k]
                degrees_of_freedom = self.degrees_of_freedom_prior + Nk
            elif self.covariance_type == 'tied':
                diff = X_array[:, np.newaxis, :] - means_var[np.newaxis, :, :]
                covariances = (self.degrees_of_freedom_prior * self.covariance_prior + np.sum(resp[:, :, np.newaxis, np.newaxis] * diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :], axis=0).sum(axis=0)) / degrees_of_freedom[0]
                degrees_of_freedom = self.degrees_of_freedom_prior + Nk
            elif self.covariance_type == 'diag':
                for k in range(self.n_components):
                    diff = X_array - means_var[k]
                    covariances[k] = (self.degrees_of_freedom_prior * self.covariance_prior + np.sum(resp[:, k, np.newaxis] * diff ** 2, axis=0)) / degrees_of_freedom[k]
                degrees_of_freedom = self.degrees_of_freedom_prior + Nk
            elif self.covariance_type == 'spherical':
                for k in range(self.n_components):
                    diff = X_array - means_var[k]
                    covariances[k] = (self.degrees_of_freedom_prior * self.covariance_prior + np.sum(resp[:, k] * np.sum(diff ** 2, axis=1))) / (degrees_of_freedom[k] * n_features)
                degrees_of_freedom = self.degrees_of_freedom_prior + Nk
            elbo = self._compute_elbo(X_array, resp, alpha, beta, means_var, degrees_of_freedom, covariances, Nk)
            elbo_history.append(elbo)
            if abs(elbo - prev_elbo) < self.tol:
                self.converged_ = True
                break
            prev_elbo = elbo
        else:
            self.converged_ = False
        self.weights_ = Nk / n_samples
        self.means_ = means_var
        self.covariances_ = covariances
        self.responsibilities_ = resp
        self.elbo_ = elbo
        self.elbo_history_ = np.array(elbo_history)
        self.is_fitted = True
        return self

    def _update_responsibilities(self, X, resp, alpha, beta, means_var, degrees_of_freedom, covariances):
        """
        Update the responsibilities (posterior probabilities) in the E-step.
        """
        (n_samples, n_features) = X.shape
        expectation_log_pi = digamma(alpha) - digamma(alpha.sum())
        log_prob = np.zeros((n_samples, self.n_components))
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - means_var[k]
                cov_k = covariances[k]
                (sign, log_det) = np.linalg.slogdet(cov_k)
                inv_cov = np.linalg.inv(cov_k)
                quadratic_form = np.sum(diff @ inv_cov * diff, axis=1)
                log_prob[:, k] = expectation_log_pi[k] + 0.5 * (sign * log_det) - 0.5 * quadratic_form
        elif self.covariance_type == 'tied':
            (sign, log_det) = np.linalg.slogdet(covariances)
            inv_cov = np.linalg.inv(covariances)
            for k in range(self.n_components):
                diff = X - means_var[k]
                quadratic_form = np.sum(diff @ inv_cov * diff, axis=1)
                log_prob[:, k] = expectation_log_pi[k] + 0.5 * (sign * log_det) - 0.5 * quadratic_form
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                diff = X - means_var[k]
                log_det = np.sum(np.log(covariances[k]))
                inv_cov_diag = 1.0 / covariances[k]
                quadratic_form = np.sum(diff ** 2 * inv_cov_diag, axis=1)
                log_prob[:, k] = expectation_log_pi[k] + 0.5 * log_det - 0.5 * quadratic_form
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - means_var[k]
                log_det = n_features * np.log(covariances[k])
                quadratic_form = np.sum(diff ** 2, axis=1) / covariances[k]
                log_prob[:, k] = expectation_log_pi[k] + 0.5 * log_det - 0.5 * quadratic_form
        log_prob_norm = logsumexp(log_prob, axis=1)
        resp = np.exp(log_prob - log_prob_norm[:, np.newaxis])
        return resp

    def _compute_elbo(self, X, resp, alpha, beta, means_var, degrees_of_freedom, covariances, Nk):
        """
        Compute the evidence lower bound (ELBO) to check for convergence.
        """
        (n_samples, n_features) = X.shape
        entropy_resp = -np.sum(resp * np.log(resp + 1e-10))
        expectation_log_pi = digamma(alpha) - digamma(np.sum(alpha))
        expected_log_pz = np.sum(resp * expectation_log_pi)
        expected_log_px = 0.0
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - means_var[k]
                inv_cov = np.linalg.inv(covariances[k])
                quadratic_form = np.sum(diff @ inv_cov * diff, axis=1)
                (_, log_det_cov) = np.linalg.slogdet(covariances[k])
                expected_log_px += np.sum(resp[:, k] * (-0.5 * n_features * np.log(2 * np.pi) - 0.5 * log_det_cov - 0.5 * quadratic_form))
        elif self.covariance_type == 'tied':
            inv_cov = np.linalg.inv(covariances)
            (_, log_det_cov) = np.linalg.slogdet(covariances)
            for k in range(self.n_components):
                diff = X - means_var[k]
                quadratic_form = np.sum(diff @ inv_cov * diff, axis=1)
                expected_log_px += np.sum(resp[:, k] * (-0.5 * n_features * np.log(2 * np.pi) - 0.5 * log_det_cov - 0.5 * quadratic_form))
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                diff = X - means_var[k]
                log_det_cov = np.sum(np.log(covariances[k]))
                inv_cov_diag = 1.0 / covariances[k]
                quadratic_form = np.sum(diff ** 2 * inv_cov_diag, axis=1)
                expected_log_px += np.sum(resp[:, k] * (-0.5 * n_features * np.log(2 * np.pi) - 0.5 * log_det_cov - 0.5 * quadratic_form))
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - means_var[k]
                log_det_cov = n_features * np.log(covariances[k])
                quadratic_form = np.sum(diff ** 2, axis=1) / covariances[k]
                expected_log_px += np.sum(resp[:, k] * (-0.5 * n_features * np.log(2 * np.pi) - 0.5 * log_det_cov - 0.5 * quadratic_form))
        alpha_sum = np.sum(alpha)
        entropy_pi = lgamma(alpha_sum) - np.sum(lgamma(alpha)) + np.sum((alpha - 1) * expectation_log_pi)
        entropy_gaussian = 0.0
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                expected_log_det_lambda = digamma((degrees_of_freedom[k] - np.arange(n_features)) / 2).sum() + n_features * np.log(2)
                entropy_gaussian += 0.5 * expected_log_det_lambda + 0.5 * n_features * (1 + np.log(2 * np.pi / beta[k]))
        elif self.covariance_type == 'tied':
            expected_log_det_lambda = digamma((degrees_of_freedom[0] - np.arange(n_features)) / 2).sum() + n_features * np.log(2)
            entropy_gaussian += 0.5 * self.n_components * expected_log_det_lambda
            for k in range(self.n_components):
                entropy_gaussian += 0.5 * n_features * (1 + np.log(2 * np.pi / beta[k]))
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                expected_log_det_lambda = np.sum(digamma((degrees_of_freedom[k] - np.arange(n_features)) / 2)) + n_features * np.log(2)
                entropy_gaussian += 0.5 * expected_log_det_lambda + 0.5 * n_features * (1 + np.log(2 * np.pi / beta[k]))
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                expected_log_det_lambda = n_features * digamma(degrees_of_freedom[k] / 2) + n_features * np.log(2)
                entropy_gaussian += 0.5 * expected_log_det_lambda + 0.5 * n_features * (1 + np.log(2 * np.pi / beta[k]))
        elbo = expected_log_pz + expected_log_px + entropy_resp + entropy_pi + entropy_gaussian
        return elbo

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for the input data.

        Args:
            X (FeatureSet): Input features to predict cluster labels for.
            **kwargs: Additional prediction parameters.

        Returns:
            np.ndarray: Array of predicted cluster labels (hard assignments).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        """
        Predict posterior probabilities for each cluster.

        Args:
            X (FeatureSet): Input features to compute cluster probabilities for.

        Returns:
            np.ndarray: Matrix of shape (n_samples, n_components) with posterior probabilities.
        """
        if not hasattr(self, 'means_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        if isinstance(X, FeatureSet):
            X_array = X.get_feature_matrix()
        else:
            X_array = np.asarray(X)
        (n_samples, n_features) = X_array.shape
        resp = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            diff = X_array - self.means_[k]
            if self.covariance_type == 'full':
                cov_k = self.covariances_[k]
                inv_cov = np.linalg.inv(cov_k)
                quadratic_form = np.sum(diff @ inv_cov * diff, axis=1)
                log_prob = -0.5 * quadratic_form
            elif self.covariance_type == 'tied':
                inv_cov = np.linalg.inv(self.covariances_)
                quadratic_form = np.sum(diff @ inv_cov * diff, axis=1)
                log_prob = -0.5 * quadratic_form
            elif self.covariance_type == 'diag':
                inv_cov_diag = 1.0 / self.covariances_[k]
                quadratic_form = np.sum(diff ** 2 * inv_cov_diag, axis=1)
                log_prob = -0.5 * quadratic_form
            elif self.covariance_type == 'spherical':
                quadratic_form = np.sum(diff ** 2, axis=1) / self.covariances_[k]
                log_prob = -0.5 * quadratic_form
            resp[:, k] = np.log(self.weights_[k] + 1e-10) + log_prob
        resp_max = np.max(resp, axis=1, keepdims=True)
        resp = resp - resp_max
        exp_resp = np.exp(resp)
        proba = exp_resp / np.sum(exp_resp, axis=1, keepdims=True)
        return proba

    def score(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> float:
        """
        Compute the evidence lower bound (ELBO) for the model.

        Args:
            X (FeatureSet): Input features to evaluate.
            y (Optional[Any]): Ignored in unsupervised learning.
            **kwargs: Additional scoring parameters.

        Returns:
            float: Evidence lower bound value.
        """
        if not hasattr(self, 'elbo_'):
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        return self.elbo_