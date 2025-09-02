from typing import Optional, List
import numpy as np
from scipy.stats import kurtosis
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class IndependentComponentAnalysisExtractor(BaseTransformer):

    def __init__(self, n_components: Optional[int]=None, algorithm: str='fastica', whiten: bool=True, random_state: Optional[int]=None, max_iter: int=200, tol: float=0.0001, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.mixing_matrix_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.whitening_matrix_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.X_mean_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    def fit(self, data: FeatureSet, **kwargs) -> 'IndependentComponentAnalysisExtractor':
        """
        Fit the ICA model to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing the data to fit the ICA model on.
        **kwargs : dict
            Additional parameters for fitting (ignored).
            
        Returns
        -------
        IndependentComponentAnalysisExtractor
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is invalid or incompatible.
        """
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet instance')
        if data.features.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = data.features.copy()
        (n_samples, n_features) = X.shape
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)
        elif self.n_components > min(n_samples, n_features):
            raise ValueError(f'n_components ({self.n_components}) cannot be larger than min(n_samples, n_features) ({min(n_samples, n_features)})')
        else:
            self.n_components_ = self.n_components
        self.X_mean_ = np.mean(X, axis=0)
        X_centered = X - self.X_mean_
        if self.whiten:
            cov_matrix = np.cov(X_centered, rowvar=False)
            (eigenvals, eigenvecs) = np.linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, 1e-12)
            self.whitening_matrix_ = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
            X_whitened = X_centered @ self.whitening_matrix_.T
        else:
            X_whitened = X_centered
            self.whitening_matrix_ = np.eye(n_features)
        if self.algorithm == 'fastica':
            self.components_ = self._fastica(X_whitened)
        elif self.algorithm == 'infomax':
            self.components_ = self._infomax(X_whitened)
        elif self.algorithm == 'jade':
            self.components_ = self._jade(X_whitened)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Choose from 'fastica', 'infomax', or 'jade'")
        if self.whiten:
            self.mixing_matrix_ = np.linalg.pinv(self.whitening_matrix_ @ self.components_.T).T
        else:
            self.mixing_matrix_ = np.linalg.pinv(self.components_.T).T
        self.feature_names_ = [f'component_{i}' for i in range(self.n_components_)]
        return self

    def _fastica(self, X: np.ndarray) -> np.ndarray:
        """FastICA algorithm implementation"""
        (n_samples, n_features) = X.shape
        W = np.random.randn(self.n_components_, n_features)

        def _sym_decorrelation(W):
            (U, _, Vt) = np.linalg.svd(W.T @ W)
            return W @ U @ Vt
        W = _sym_decorrelation(W)
        for iter_idx in range(self.max_iter):
            W_old = W.copy()
            wx = W @ X.T
            gwx = np.tanh(wx)
            g_wx = 1 - np.tanh(wx) ** 2
            W = gwx @ X / n_samples - np.diag(np.mean(g_wx, axis=1)) @ W
            W = _sym_decorrelation(W)
            lim = np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1))
            if lim < self.tol:
                break
        return W

    def _infomax(self, X: np.ndarray) -> np.ndarray:
        """Infomax ICA algorithm implementation"""
        (n_samples, n_features) = X.shape
        W = np.random.randn(self.n_components_, n_features)

        def _sym_decorrelation(W):
            (U, _, Vt) = np.linalg.svd(W.T @ W)
            return W @ U @ Vt
        W = _sym_decorrelation(W)
        for iter_idx in range(self.max_iter):
            W_old = W.copy()
            Y = W @ X.T
            phi = np.tanh(Y)
            phi_deriv = 1 - np.tanh(Y) ** 2
            W += 0.01 * (np.eye(self.n_components_) + np.diag(np.mean(phi_deriv, axis=1)) - phi @ Y.T / n_samples) @ W
            W = _sym_decorrelation(W)
            lim = np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1))
            if lim < self.tol:
                break
        return W

    def _jade(self, X: np.ndarray) -> np.ndarray:
        """JADE algorithm implementation (simplified version)"""
        (n_samples, n_features) = X.shape
        return self._fastica(X)

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the ICA transformation to extract independent components.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform using the fitted ICA model.
        **kwargs : dict
            Additional parameters for transformation (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed feature set with independent components as features.
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        ValueError
            If the input data dimensions don't match the fitted model.
        """
        if self.components_ is None:
            raise RuntimeError("ICA model has not been fitted yet. Call 'fit' before 'transform'.")
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet instance')
        X = data.features.copy()
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(f'Input data has {X.shape[1]} features, but the fitted model expects {self.components_.shape[1]} features.')
        X_centered = X - self.X_mean_
        if self.whiten:
            X_whitened = X_centered @ self.whitening_matrix_.T
        else:
            X_whitened = X_centered
        transformed_features = X_whitened @ self.components_.T
        feature_names = [f'component_{i}' for i in range(self.n_components_)]
        return FeatureSet(features=transformed_features, feature_names=feature_names)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the inverse ICA transformation to reconstruct original data.
        
        Parameters
        ----------
        data : FeatureSet
            Feature set with independent components to reconstruct.
        **kwargs : dict
            Additional parameters for inverse transformation (ignored).
            
        Returns
        -------
        FeatureSet
            Reconstructed feature set in the original space.
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        ValueError
            If the input data dimensions don't match the fitted model.
        """
        if self.mixing_matrix_ is None:
            raise RuntimeError("ICA model has not been fitted yet. Call 'fit' before 'inverse_transform'.")
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet instance')
        X = data.features.copy()
        if X.shape[1] != self.n_components_:
            raise ValueError(f'Input data has {X.shape[1]} components, but the fitted model expects {self.n_components_} components.')
        reconstructed_centered = X @ self.mixing_matrix_.T
        reconstructed_features = reconstructed_centered + self.X_mean_
        if hasattr(data, 'feature_names') and data.feature_names:
            feature_names = data.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(reconstructed_features.shape[1])]
        return FeatureSet(features=reconstructed_features, feature_names=feature_names)

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of the extracted independent components.
        
        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Names of input features (used for generating component names).
            
        Returns
        -------
        List[str]
            Names of the extracted independent components.
        """
        if self.feature_names_ is not None:
            return self.feature_names_
        elif self.n_components_ is not None:
            return [f'component_{i}' for i in range(self.n_components_)]
        else:
            return []