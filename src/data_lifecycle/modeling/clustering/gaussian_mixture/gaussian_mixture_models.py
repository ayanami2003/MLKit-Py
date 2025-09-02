from typing import Optional, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

class GaussianMixtureModel(BaseModel):

    def __init__(self, n_components: int=1, covariance_type: str='full', tol: float=0.001, reg_covar: float=1e-06, max_iter: int=100, n_init: int=1, init_params: str='kmeans', weights_init: Optional[np.ndarray]=None, means_init: Optional[np.ndarray]=None, precisions_init: Optional[np.ndarray]=None, random_state: Optional[int]=None, warm_start: bool=False, verbose: int=0, verbose_interval: int=10, name: Optional[str]=None):
        """
        Initialize the Gaussian Mixture Model.
        
        Parameters
        ----------
        n_components : int, default=1
            Number of mixture components (clusters)
        covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
            String describing the type of covariance parameters to use
        tol : float, default=1e-3
            Convergence threshold for the EM algorithm
        reg_covar : float, default=1e-6
            Non-negative regularization added to the diagonal of covariance matrices
        max_iter : int, default=100
            Maximum number of iterations for the EM algorithm
        n_init : int, default=1
            Number of initializations to perform
        init_params : {'kmeans', 'random'}, default='kmeans'
            Method for initialization
        weights_init : array-like of shape (n_components,), optional
            Initial weights for each mixture component
        means_init : array-like of shape (n_components, n_features), optional
            Initial means for each mixture component
        precisions_init : array-like, optional
            Initial precisions (inverse of covariance matrices) for each mixture component
        random_state : int, optional
            Random seed for reproducibility
        warm_start : bool, default=False
            If True, reuse solution from previous fit as initialization
        verbose : int, default=0
            Verbosity level
        verbose_interval : int, default=10
            Number of iterations between verbose outputs
        name : str, optional
            Name of the model instance
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> 'GaussianMixtureModel':
        """
        Estimate model parameters with the EM algorithm.
        
        Parameters
        ----------
        X : FeatureSet
            Training data containing features
        y : Any, optional
            Not used, present for API consistency
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        GaussianMixtureModel
            Fitted model instance
        """
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_data = X.features
        (n_samples, n_features) = X_data.shape
        if self.n_components <= 0:
            raise ValueError('n_components must be positive')
        if self.tol < 0:
            raise ValueError('tol must be non-negative')
        if self.reg_covar < 0:
            raise ValueError('reg_covar must be non-negative')
        if self.max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if self.n_init <= 0:
            raise ValueError('n_init must be positive')
        valid_covariance_types = ['full', 'tied', 'diag', 'spherical']
        if self.covariance_type not in valid_covariance_types:
            raise ValueError(f'covariance_type must be one of {valid_covariance_types}')
        valid_init_params = ['kmeans', 'random']
        if self.init_params not in valid_init_params:
            raise ValueError(f'init_params must be one of {valid_init_params}')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.warm_start and hasattr(self, 'weights_'):
            best_params = (self.weights_, self.means_, self.covariances_)
            best_log_likelihood = -np.inf
        else:
            best_params = None
            best_log_likelihood = -np.inf
        for _ in range(self.n_init):
            if self.warm_start and hasattr(self, 'weights_') and (best_params is not None):
                (weights, means, covariances) = best_params
            elif self.weights_init is not None and self.means_init is not None and (self.precisions_init is not None):
                weights = self.weights_init
                means = self.means_init
                covariances = self._precisions_to_covariances(self.precisions_init)
            else:
                (weights, means, covariances) = self._initialize_parameters(X_data)
            log_likelihood = -np.inf
            for iteration in range(self.max_iter):
                (log_prob_norm, log_resp) = self._e_step(X_data, weights, means, covariances)
                previous_log_likelihood = log_likelihood
                log_likelihood = np.mean(log_prob_norm)
                if abs(log_likelihood - previous_log_likelihood) < self.tol:
                    break
                (weights, means, covariances) = self._m_step(X_data, log_resp)
                covariances = self._regularize_covariances(covariances)
                if self.verbose > 0 and iteration % self.verbose_interval == 0:
                    print(f'Iteration {iteration}, Log-Likelihood: {log_likelihood}')
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = (weights, means, covariances)
        (self.weights_, self.means_, self.covariances_) = best_params
        self.precisions_ = self._compute_precisions()
        self.log_likelihood_ = best_log_likelihood
        self.is_fitted = True
        return self

    def predict(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict cluster labels for samples.
        
        Parameters
        ----------
        X : FeatureSet
            Input data to predict cluster labels for
            
        Returns
        -------
        np.ndarray
            Cluster labels for each data point
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_data = X.features
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: FeatureSet, **kwargs) -> np.ndarray:
        """
        Predict posterior probability of each component given the data.
        
        Parameters
        ----------
        X : FeatureSet
            Input data to predict cluster probabilities for
            
        Returns
        -------
        np.ndarray
            Probability of each data point belonging to each cluster
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict_proba'.")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_data = X.features
        (_, log_resp) = self._e_step(X_data, self.weights_, self.means_, self.covariances_)
        return np.exp(log_resp)

    def score(self, X: FeatureSet, y: Optional[Any]=None, **kwargs) -> float:
        """
        Compute the per-sample average log-likelihood of the data.
        
        Parameters
        ----------
        X : FeatureSet
            Test data
        y : Any, optional
            Not used, present for API consistency
            
        Returns
        -------
        float
            Average log-likelihood of the samples under the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'score'.")
        if not isinstance(X, FeatureSet):
            raise TypeError('X must be a FeatureSet instance')
        X_data = X.features
        (log_prob_norm, _) = self._e_step(X_data, self.weights_, self.means_, self.covariances_)
        return np.mean(log_prob_norm)

    def sample(self, n_samples: int=1, **kwargs) -> np.ndarray:
        """
        Generate random samples from the fitted Gaussian distribution.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
            
        Returns
        -------
        np.ndarray
            Generated samples
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'sample'.")
        component_indices = np.random.choice(self.n_components, size=n_samples, p=self.weights_)
        samples = np.zeros((n_samples, self.means_.shape[1]))
        for i in range(n_samples):
            comp_idx = component_indices[i]
            if self.covariance_type == 'full':
                cov = self.covariances_[comp_idx]
            elif self.covariance_type == 'tied':
                cov = self.covariances_
            elif self.covariance_type == 'diag':
                cov = np.diag(self.covariances_[comp_idx])
            else:
                cov = np.eye(self.means_.shape[1]) * self.covariances_[comp_idx]
            samples[i] = np.random.multivariate_normal(self.means_[comp_idx], cov)
        return samples

    def _initialize_parameters(self, X: np.ndarray) -> tuple:
        """
        Initialize the model parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Training data
            
        Returns
        -------
        tuple
            Tuple containing (weights, means, covariances)
        """
        (n_samples, n_features) = X.shape
        weights = np.full(self.n_components, 1.0 / self.n_components)
        if self.init_params == 'kmeans':
            means = self._simple_kmeans(X, self.n_components)
        else:
            indices = np.random.choice(n_samples, self.n_components, replace=False)
            means = X[indices]
        if self.covariance_type == 'full':
            covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            covariances = np.eye(n_features)
        elif self.covariance_type == 'diag':
            covariances = np.ones((self.n_components, n_features))
        else:
            covariances = np.ones(self.n_components)
        return (weights, means, covariances)

    def _simple_kmeans(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Simple K-means implementation for initialization.
        
        Parameters
        ----------
        X : np.ndarray
            Training data
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        np.ndarray
            Cluster centers
        """
        (n_samples, n_features) = X.shape
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[indices].copy()
        for _ in range(10):
            distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return centroids

    def _e_step(self, X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> tuple:
        """
        E-step of the EM algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        weights : np.ndarray
            Component weights
        means : np.ndarray
            Component means
        covariances : np.ndarray
            Component covariances
            
        Returns
        -------
        tuple
            Tuple containing (log_prob_norm, log_resp)
        """
        (n_samples, _) = X.shape
        log_prob = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            log_prob[:, k] = self._log_gaussian_pdf(X, means[k], covariances, k)
            log_prob[:, k] += np.log(weights[k])
        log_prob_norm = np.log(np.sum(np.exp(log_prob - np.max(log_prob, axis=1, keepdims=True)), axis=1)) + np.max(log_prob, axis=1)
        log_resp = log_prob - log_prob_norm[:, np.newaxis]
        return (log_prob_norm, log_resp)

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray) -> tuple:
        """
        M-step of the EM algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
        log_resp : np.ndarray
            Log responsibilities from E-step
            
        Returns
        -------
        tuple
            Tuple containing (weights, means, covariances)
        """
        (n_samples, n_features) = X.shape
        resp = np.exp(log_resp)
        weights = np.sum(resp, axis=0) / n_samples
        means = np.dot(resp.T, X) / np.sum(resp, axis=0)[:, np.newaxis]
        if self.covariance_type == 'full':
            covariances = np.empty((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = np.dot(resp[:, k] * diff.T, diff) / np.sum(resp[:, k])
        elif self.covariance_type == 'tied':
            covariances = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - means[k]
                covariances += np.dot(resp[:, k] * diff.T, diff)
            covariances /= n_samples
        elif self.covariance_type == 'diag':
            covariances = np.empty((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = np.sum(resp[:, k][:, np.newaxis] * diff ** 2, axis=0) / np.sum(resp[:, k])
        else:
            covariances = np.empty(self.n_components)
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = np.sum(resp[:, k] * np.sum(diff ** 2, axis=1)) / (n_features * np.sum(resp[:, k]))
        return (weights, means, covariances)

    def _log_gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, covariances: np.ndarray, k: int) -> np.ndarray:
        """
        Compute log probability density function of a Gaussian distribution.
        
        Parameters
        ----------
        X : np.ndarray
            Data points
        mean : np.ndarray
            Mean of the Gaussian
        covariances : np.ndarray
            Covariance matrices
        k : int
            Component index
            
        Returns
        -------
        np.ndarray
            Log probabilities for each data point
        """
        (n_samples, n_features) = X.shape
        diff = X - mean
        if self.covariance_type == 'full':
            cov = covariances[k]
            cov_reg = cov + self.reg_covar * np.eye(cov.shape[0])
            try:
                cov_chol = np.linalg.cholesky(cov_reg)
                log_det = 2 * np.sum(np.log(np.diag(cov_chol)))
                mahalanobis = np.sum(np.square(np.dot(diff, np.linalg.inv(cov_chol).T)), axis=1)
            except np.linalg.LinAlgError:
                (eigvals, eigvecs) = np.linalg.eigh(cov_reg)
                eigvals = np.maximum(eigvals, self.reg_covar)
                log_det = np.sum(np.log(eigvals))
                diff_transformed = np.dot(diff, eigvecs)
                mahalanobis = np.sum(diff_transformed ** 2 / eigvals, axis=1)
        elif self.covariance_type == 'tied':
            cov = covariances
            cov_reg = cov + self.reg_covar * np.eye(cov.shape[0])
            try:
                cov_chol = np.linalg.cholesky(cov_reg)
                log_det = 2 * np.sum(np.log(np.diag(cov_chol)))
                mahalanobis = np.sum(np.square(np.dot(diff, np.linalg.inv(cov_chol).T)), axis=1)
            except np.linalg.LinAlgError:
                (eigvals, eigvecs) = np.linalg.eigh(cov_reg)
                eigvals = np.maximum(eigvals, self.reg_covar)
                log_det = np.sum(np.log(eigvals))
                diff_transformed = np.dot(diff, eigvecs)
                mahalanobis = np.sum(diff_transformed ** 2 / eigvals, axis=1)
        elif self.covariance_type == 'diag':
            cov = covariances[k]
            log_det = np.sum(np.log(cov + self.reg_covar))
            mahalanobis = np.sum(diff ** 2 / (cov + self.reg_covar), axis=1)
        else:
            cov = covariances[k]
            log_det = n_features * np.log(cov + self.reg_covar)
            mahalanobis = np.sum(diff ** 2, axis=1) / (cov + self.reg_covar)
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)

    def _regularize_covariances(self, covariances: np.ndarray) -> np.ndarray:
        """
        Regularize covariance matrices to ensure numerical stability.
        
        Parameters
        ----------
        covariances : np.ndarray
            Covariance matrices
            
        Returns
        -------
        np.ndarray
            Regularized covariance matrices
        """
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                covariances[k].flat[::len(covariances[k]) + 1] += self.reg_covar
        elif self.covariance_type == 'tied':
            covariances.flat[::len(covariances) + 1] += self.reg_covar
        elif self.covariance_type == 'diag':
            covariances += self.reg_covar
        else:
            covariances += self.reg_covar
        return covariances

    def _compute_precisions(self) -> np.ndarray:
        """
        Compute precisions (inverse of covariance matrices).
        
        Returns
        -------
        np.ndarray
            Precisions matrices
        """
        if self.covariance_type == 'full':
            precisions = np.empty_like(self.covariances_)
            for k in range(self.n_components):
                precisions[k] = np.linalg.inv(self.covariances_[k])
        elif self.covariance_type == 'tied':
            precisions = np.linalg.inv(self.covariances_)
        elif self.covariance_type == 'diag':
            precisions = 1.0 / self.covariances_
        else:
            precisions = 1.0 / self.covariances_
        return precisions

    def _precisions_to_covariances(self, precisions: np.ndarray) -> np.ndarray:
        """
        Convert precisions to covariances.
        
        Parameters
        ----------
        precisions : np.ndarray
            Precision matrices
            
        Returns
        -------
        np.ndarray
            Covariance matrices
        """
        if self.covariance_type == 'full':
            covariances = np.empty_like(precisions)
            for k in range(self.n_components):
                covariances[k] = np.linalg.inv(precisions[k])
        elif self.covariance_type == 'tied':
            covariances = np.linalg.inv(precisions)
        elif self.covariance_type == 'diag':
            covariances = 1.0 / precisions
        else:
            covariances = 1.0 / precisions
        return covariances