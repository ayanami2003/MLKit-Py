from typing import Optional, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class RobustPCATransformer(BaseTransformer):

    def __init__(self, n_components: Optional[int]=None, regularization: float=1.0, max_iter: int=1000, tol: float=1e-06, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the RobustPCATransformer.
        
        Parameters
        ----------
        n_components : Optional[int], default=None
            Number of components to keep. If None, it will be determined automatically
            based on the data.
        regularization : float, default=1.0
            Regularization parameter that controls the sparsity of the sparse component.
            Higher values lead to sparser solutions.
        max_iter : int, default=1000
            Maximum number of iterations for the optimization algorithm.
        tol : float, default=1e-6
            Tolerance for convergence of the algorithm. The algorithm stops when the
            change in objective function is less than this value.
        random_state : Optional[int], default=None
            Random seed for reproducibility of results.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, data: FeatureSet, **kwargs) -> 'RobustPCATransformer':
        """
        Fit the RobustPCA model to the input data.
        
        This method decomposes the input data matrix into low-rank and sparse components,
        and stores the learned components for future transformations.
        
        Parameters
        ----------
        data : FeatureSet
            Input data to fit the model on. Must contain a 2D numpy array of features.
        **kwargs : dict
            Additional keyword arguments (ignored in this implementation).
            
        Returns
        -------
        RobustPCATransformer
            Self instance for method chaining.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = data.features
        if X.ndim != 2:
            raise ValueError('Input features must be a 2D array')
        (self.n_samples_, self.n_features_) = X.shape
        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_)
        elif self.n_components > min(self.n_samples_, self.n_features_):
            raise ValueError('n_components must be less than or equal to min(n_samples, n_features)')
        else:
            self.n_components_ = self.n_components
        (L, S) = self._robust_pca_admm(X)
        (U, s, Vt) = np.linalg.svd(L, full_matrices=False)
        self.components_ = Vt[:self.n_components_]
        self.singular_values_ = s[:self.n_components_]
        self.mean_ = np.mean(X, axis=0)
        self.is_fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply dimensionality reduction to the input data using the fitted model.
        
        Projects the input data onto the principal components learned during fitting.
        
        Parameters
        ----------
        data : FeatureSet
            Input data to transform. Must have the same number of features as the
            training data.
        **kwargs : dict
            Additional keyword arguments (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensionality.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("This RobustPCATransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance')
        X = data.features
        if X.ndim != 2:
            raise ValueError('Input features must be a 2D array')
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Input data has {X.shape[1]} features, but the transformer was fitted with {self.n_features_} features.')
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T
        feature_names = self.get_feature_names(data.feature_names)
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=['numeric'] * self.n_components_, sample_ids=data.sample_ids, metadata={'transformation': 'RobustPCA'})

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform data back to its original space.
        
        Projects the reduced data back to the original feature space using the
        learned principal components.
        
        Parameters
        ----------
        data : FeatureSet
            Reduced data to inverse transform.
        **kwargs : dict
            Additional keyword arguments (ignored in this implementation).
            
        Returns
        -------
        FeatureSet
            Data reconstructed in the original feature space.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("This RobustPCATransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance')
        X = data.features
        if X.ndim != 2:
            raise ValueError('Input features must be a 2D array')
        if X.shape[1] != self.n_components_:
            raise ValueError(f'Input data has {X.shape[1]} features, but the transformer was fitted with {self.n_components_} components.')
        X_original = X @ self.components_
        X_reconstructed = X_original + self.mean_
        feature_names = None
        if hasattr(self, '_original_feature_names'):
            feature_names = self._original_feature_names
        elif data.feature_names and len(data.feature_names) == self.n_features_:
            feature_names = data.feature_names
        return FeatureSet(features=X_reconstructed, feature_names=feature_names, feature_types=['numeric'] * self.n_features_, sample_ids=data.sample_ids, metadata={'transformation': 'RobustPCA_inverse'})

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Input feature names. If None, feature names will be generated automatically.
            
        Returns
        -------
        List[str]
            Output feature names.
        """
        if input_features is not None:
            if hasattr(self, 'is_fitted_') and self.is_fitted_:
                if len(input_features) != self.n_features_:
                    raise ValueError(f'input_features should have length {self.n_features_}, but has length {len(input_features)}')
            self._original_feature_names = input_features
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("This RobustPCATransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        return [f'PC{i}' for i in range(self.n_components_)]

    def _soft_threshold(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply soft thresholding operator (shrinkage function).
        
        Parameters
        ----------
        X : np.ndarray
            Input matrix
        threshold : float
            Threshold value
            
        Returns
        -------
        np.ndarray
            Soft-thresholded matrix
        """
        return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)

    def _svd_threshold(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply singular value thresholding.
        
        Parameters
        ----------
        X : np.ndarray
            Input matrix
        threshold : float
            Threshold value
            
        Returns
        -------
        np.ndarray
            Matrix with thresholded singular values
        """
        (U, s, Vt) = np.linalg.svd(X, full_matrices=False)
        s_thresholded = np.maximum(s - threshold, 0)
        return U @ np.diag(s_thresholded) @ Vt

    def _robust_pca_admm(self, X: np.ndarray) -> tuple:
        """
        Perform Robust PCA using ADMM algorithm.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
            
        Returns
        -------
        tuple
            (Low-rank component L, Sparse component S)
        """
        L = np.zeros_like(X)
        S = np.zeros_like(X)
        Y = np.zeros_like(X)
        if self.regularization == 1.0:
            lambda_reg = 1.0 / np.sqrt(max(X.shape))
        else:
            lambda_reg = self.regularization
        mu = 1.25 / np.sqrt(np.linalg.norm(X, ord=2))
        mu = max(mu, 1e-06)
        rho = 1.5
        for iteration in range(self.max_iter):
            L_new = self._svd_threshold(X - S + Y / mu, 1 / mu)
            S_new = self._soft_threshold(X - L_new + Y / mu, lambda_reg / mu)
            Z = X - L_new - S_new
            Y = Y + mu * Z
            primal_residual = np.linalg.norm(Z, 'fro')
            dual_residual = np.linalg.norm(mu * (L_new - L), 'fro')
            if primal_residual < self.tol and dual_residual < self.tol:
                break
            if primal_residual > 10 * dual_residual:
                mu *= rho
            elif dual_residual > 10 * primal_residual:
                mu /= rho
            (L, S) = (L_new, S_new)
        return (L, S)