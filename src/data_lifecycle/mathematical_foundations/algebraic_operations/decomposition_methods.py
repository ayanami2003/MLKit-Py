from typing import Optional, Tuple
from general.structures.feature_set import FeatureSet
import numpy as np
from general.base_classes.transformer_base import BaseTransformer


# ...(code omitted)...


class NonNegativeMatrixFactorization(BaseTransformer):

    def __init__(self, n_components: int, init: Optional[str]='random', solver: Optional[str]='cd', max_iter: int=200, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the Non-Negative Matrix Factorization transformer.
        
        Parameters
        ----------
        n_components : int
            Number of components/factors to extract from the data
        init : str, optional
            Method used to initialize the procedure ('random', 'nndsvd', 'nndsvda', 'nndsvdar')
        solver : str, optional
            Numerical solver to use ('cd' for Coordinate Descent, 'mu' for Multiplicative Update)
        max_iter : int, optional
            Maximum number of iterations before stopping
        random_state : int, optional
            Random seed for reproducible results
        name : str, optional
            Name identifier for the transformer instance
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None
        self._H: Optional[np.ndarray] = None
        self._fitted = False

    def _initialize_random(self, X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize W and H with random non-negative values."""
        rng = np.random.RandomState(self.random_state)
        avg = np.sqrt(X.mean() / n_components)
        W = avg * rng.randn(X.shape[0], n_components)
        H = avg * rng.randn(n_components, X.shape[1])
        np.abs(W, out=W)
        np.abs(H, out=H)
        return (W, H)

    def _initialize_nndsvd(self, X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize W and H with NNDSVD (Nonnegative Double Singular Value Decomposition)."""
        (U, S, Vt) = np.linalg.svd(X, full_matrices=False)
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components, :]
        W = np.zeros((X.shape[0], n_components))
        H = np.zeros((n_components, X.shape[1]))
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
        for j in range(1, n_components):
            x = U[:, j]
            y = Vt[j, :]
            x_p = np.maximum(x, 0)
            x_n = np.abs(np.minimum(x, 0))
            y_p = np.maximum(y, 0)
            y_n = np.abs(np.minimum(y, 0))
            x_p_norm = np.linalg.norm(x_p)
            x_n_norm = np.linalg.norm(x_n)
            y_p_norm = np.linalg.norm(y_p)
            y_n_norm = np.linalg.norm(y_n)
            if x_p_norm == 0 or y_p_norm == 0:
                W[:, j] = np.sqrt(S[j] / n_components) * np.ones(X.shape[0])
                H[j, :] = np.sqrt(S[j] / n_components) * np.ones(X.shape[1])
            else:
                scale = np.sqrt(S[j] * x_p_norm * y_p_norm)
                W[:, j] = scale / x_p_norm * x_p
                H[j, :] = scale / y_p_norm * y_p
        if self.init in ['nndsvda', 'nndsvdar']:
            avg = X.mean()
            W[W == 0] = avg / 100
            H[H == 0] = avg / 100
            if self.init == 'nndsvdar':
                rng = np.random.RandomState(self.random_state)
                W += rng.uniform(0, avg / 50, size=W.shape)
                H += rng.uniform(0, avg / 50, size=H.shape)
        return (W, H)

    def _solve_cd(self, X: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve NMF using Coordinate Descent algorithm."""
        for _ in range(self.max_iter):
            numerator = W.T @ X
            denominator = W.T @ W @ H + 1e-10
            H *= numerator / denominator
            H = np.maximum(H, 0)
            numerator = X @ H.T
            denominator = W @ H @ H.T + 1e-10
            W *= numerator / denominator
            W = np.maximum(W, 0)
        return (W, H)

    def _solve_mu(self, X: np.ndarray, W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve NMF using Multiplicative Update algorithm."""
        for _ in range(self.max_iter):
            numerator = W.T @ X
            denominator = W.T @ W @ H + 1e-10
            H *= numerator / denominator
            H = np.maximum(H, 0)
            numerator = X @ H.T
            denominator = W @ H @ H.T + 1e-10
            W *= numerator / denominator
            W = np.maximum(W, 0)
        return (W, H)

    def fit(self, data: FeatureSet, **kwargs) -> 'NonNegativeMatrixFactorization':
        """
        Fit the Non-Negative Matrix Factorization model to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to decompose (must contain only non-negative values)
        **kwargs : dict
            Additional parameters for fitting (ignored)
            
        Returns
        -------
        NonNegativeMatrixFactorization
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data contains negative values
        """
        if hasattr(data, 'features'):
            if hasattr(data.features, 'values'):
                X = data.features.values
            else:
                X = data.features
        else:
            X = data
        if np.any(X < 0):
            raise ValueError('Input data must contain only non-negative values for NMF')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.init == 'random':
            (W, H) = self._initialize_random(X, self.n_components)
        elif self.init in ['nndsvd', 'nndsvda', 'nndsvdar']:
            (W, H) = self._initialize_nndsvd(X, self.n_components)
        else:
            raise ValueError(f'Unknown init method: {self.init}')
        if self.solver == 'cd':
            (W, H) = self._solve_cd(X, W, H)
        elif self.solver == 'mu':
            (W, H) = self._solve_mu(X, W, H)
        else:
            raise ValueError(f'Unknown solver: {self.solver}')
        self._W = W
        self._H = H
        self.components_ = H
        self._fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply dimensionality reduction to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to transform using fitted NMF components
        **kwargs : dict
            Additional parameters for transformation (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed feature set with reduced dimensionality (non-negative values)
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit()' first.")
        if hasattr(data, 'features'):
            if hasattr(data.features, 'values'):
                X = data.features.values
            else:
                X = data.features
        else:
            X = data
        HHt = self._H @ self._H.T
        HHt_inv = np.linalg.pinv(HHt)
        W_new = X @ self._H.T @ HHt_inv
        W_new = np.maximum(W_new, 0)
        component_names = [f'component_{i}' for i in range(self.n_components)]
        sample_ids = getattr(data, 'sample_ids', [f'sample_{i}' for i in range(W_new.shape[0])])
        metadata = getattr(data, 'metadata', {}).copy() if hasattr(data, 'metadata') and data.metadata else {}
        transformed_data = FeatureSet(features=W_new, feature_names=component_names, sample_ids=sample_ids, metadata=metadata)
        return transformed_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform data back to its original space.
        
        Parameters
        ----------
        data : FeatureSet
            Low-dimensional representation to reconstruct
        **kwargs : dict
            Additional parameters for reconstruction (ignored)
            
        Returns
        -------
        FeatureSet
            Reconstructed feature set in original space
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit()' first.")
        if hasattr(data, 'features'):
            if hasattr(data.features, 'values'):
                W_lowdim = data.features.values
            else:
                W_lowdim = data.features
        else:
            W_lowdim = data
        X_reconstructed = W_lowdim @ self._H
        original_feature_count = self._H.shape[1]
        sample_ids = getattr(data, 'sample_ids', [f'sample_{i}' for i in range(W_lowdim.shape[0])])
        metadata = getattr(data, 'metadata', {}).copy() if hasattr(data, 'metadata') and data.metadata else {}
        feature_names = [f'feature_{i}' for i in range(original_feature_count)]
        reconstructed_data = FeatureSet(features=X_reconstructed, feature_names=feature_names, sample_ids=sample_ids, metadata=metadata)
        return reconstructed_data

    def fit_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Fit the model and apply dimensionality reduction in one step.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set to fit and transform
        **kwargs : dict
            Additional parameters for fitting and transformation (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed feature set with reduced dimensionality
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    def get_components(self) -> np.ndarray:
        """
        Get the components matrix from the fitted NMF model.
        
        Returns
        -------
        np.ndarray
            Components matrix (H) of shape (n_components, n_features)
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit()' first.")
        return self.components_.copy()