from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class AdvancedPCAExtractor(BaseTransformer):

    def __init__(self, n_components: Optional[int]=None, whiten: bool=False, svd_solver: str='auto', tol: float=0.0, random_state: Optional[int]=None, robust_covariance: bool=False, explained_variance_threshold: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the AdvancedPCAExtractor.
        
        Parameters
        ----------
        n_components : Optional[int], default=None
            Number of components to keep. If None, automatically determined based on
            explained_variance_threshold or data dimensions.
        whiten : bool, default=False
            When True, the transformed features are normalized to have unit variance.
        svd_solver : str, default='auto'
            Solver to use for the decomposition. Options are 'auto', 'full', 'arpack', 'randomized'.
        tol : float, default=0.0
            Tolerance for singular values computed by svd_solver='arpack'.
        random_state : Optional[int], default=None
            Random seed for reproducible results.
        robust_covariance : bool, default=False
            Whether to use robust covariance estimation to reduce influence of outliers.
        explained_variance_threshold : Optional[float], default=None
            Threshold for cumulative explained variance to automatically determine components.
            Used only when n_components is None.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.random_state = random_state
        self.robust_covariance = robust_covariance
        self.explained_variance_threshold = explained_variance_threshold

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AdvancedPCAExtractor':
        """
        Fit the advanced PCA transformer to the input data.
        
        Computes the principal components and stores them for later use in transformation.
        Optionally uses robust covariance estimation and automatic component selection.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. Should be a 2D array-like structure
            with samples as rows and features as columns.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        AdvancedPCAExtractor
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is not 2-dimensional or if n_components is larger than
            the number of features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        (n_samples, n_features) = X.shape
        if self.n_components is None:
            if self.explained_variance_threshold is not None:
                n_components = min(n_samples, n_features)
            else:
                n_components = min(n_samples, n_features)
        else:
            if self.n_components > min(n_samples, n_features):
                raise ValueError(f'n_components ({self.n_components}) must be between 0 and min(n_samples, n_features) ({min(n_samples, n_features)})')
            n_components = self.n_components
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        if self.robust_covariance:
            median = np.median(X_centered, axis=0)
            mad = np.median(np.abs(X_centered - median), axis=0)
            mad = np.where(mad == 0, 1, mad)
            X_normalized = (X_centered - median) / mad
            cov_matrix = np.cov(X_normalized, rowvar=False)
        else:
            cov_matrix = np.cov(X_centered, rowvar=False)
        if self.svd_solver == 'full' or (self.svd_solver == 'auto' and n_features <= 500):
            (eigenvals, eigenvecs) = np.linalg.eigh(cov_matrix)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            eigenvals = eigenvals[:n_components]
            eigenvecs = eigenvecs[:, :n_components]
        elif self.svd_solver == 'randomized' or (self.svd_solver == 'auto' and n_features > 500):
            rnd_state = np.random.RandomState(self.random_state)
            (U, S, Vt) = self._randomized_svd(X_centered, n_components, random_state=rnd_state)
            eigenvals = S ** 2 / (n_samples - 1)
            eigenvecs = Vt.T
        else:
            (U, S, Vt) = np.linalg.svd(X_centered, full_matrices=False)
            eigenvals = S ** 2 / (n_samples - 1)
            eigenvecs = Vt.T
            eigenvals = eigenvals[:n_components]
            eigenvecs = eigenvecs[:, :n_components]
        if self.n_components is None and self.explained_variance_threshold is not None:
            total_variance = np.sum(eigenvals)
            if total_variance > 0:
                cumsum_variance = np.cumsum(eigenvals) / total_variance
                n_components = np.searchsorted(cumsum_variance, self.explained_variance_threshold) + 1
                n_components = min(n_components, len(eigenvals))
            else:
                n_components = 1
            eigenvals = eigenvals[:n_components]
            eigenvecs = eigenvecs[:, :n_components]
        self.components_ = eigenvecs.T
        self.explained_variance_ = eigenvals
        total_var = np.sum(eigenvals) if np.sum(eigenvals) > 0 else 1
        self.explained_variance_ratio_ = eigenvals / total_var
        self.n_components_ = len(eigenvals)
        self.n_features_ = n_features
        self.n_samples_ = n_samples
        if self.whiten:
            self.scalings_ = np.sqrt(eigenvals)
        else:
            self.scalings_ = np.ones_like(eigenvals)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply dimensionality reduction to the input data using the fitted PCA model.
        
        Projects the input data onto the principal components computed during fitting.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have the same number of features as the
            data used during fitting.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed data with reduced dimensionality.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet or if the input data
            dimensions don't match the fitted model.
        """
        if not hasattr(self, 'components_'):
            raise ValueError("This AdvancedPCAExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Input data has {X.shape[1]} features, but PCA was fitted with {self.n_features_} features.')
        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T
        if self.whiten:
            X_transformed /= self.scalings_
        feature_names = self.get_feature_names()
        return FeatureSet(features=X_transformed, feature_names=feature_names, metadata={'transformer': 'AdvancedPCAExtractor'})

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform data back to its original space.
        
        Inverts the dimensionality reduction by projecting the data back to the
        original feature space using the principal components.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data in the transformed space to invert back to original space.
        **kwargs : dict
            Additional parameters for inverse transformation.
            
        Returns
        -------
        FeatureSet
            Data projected back to the original space.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet or if the input data
            dimensions don't match the transformed space.
        """
        if not hasattr(self, 'components_'):
            raise ValueError("This AdvancedPCAExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if X.shape[1] != self.n_components_:
            raise ValueError(f'Input data has {X.shape[1]} components, but PCA was fitted with {self.n_components_} components.')
        if self.whiten:
            X = X * self.scalings_
        X_original = X @ self.components_
        X_original += self.mean_
        return FeatureSet(features=X_original, feature_names=None, metadata={'transformer': 'AdvancedPCAExtractor', 'inverse_transform': True})

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for the transformed features.
        
        Generates names for the principal components in the format 'PC{i}' where i
        is the component index.
        
        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Not used in this implementation but kept for API consistency.
            
        Returns
        -------
        List[str]
            Names of the transformed features (principal components).
        """
        return [f'PC{i + 1}' for i in range(self.n_components_)] if hasattr(self, 'n_components_') else []

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the explained variance ratio for each principal component.
        
        Returns the proportion of the dataset's variance that lies along each
        principal component.
        
        Returns
        -------
        np.ndarray
            Explained variance ratios for each component.
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'explained_variance_ratio_'):
            raise RuntimeError("This AdvancedPCAExtractor instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        return self.explained_variance_ratio_

    def _randomized_svd(self, M, n_components, n_oversamples=10, n_iter=4, random_state=None):
        """
        Compute randomized SVD
        
        Parameters
        ----------
        M : ndarray
            Input matrix
        n_components : int
            Number of components
        n_oversamples : int
            Additional number of random vectors to sample the range of M
        n_iter : int
            Number of power iterations
        random_state : RandomState
            Random state for reproducibility
            
        Returns
        -------
        U, S, Vt : ndarray
            SVD decomposition
        """
        n_random = n_components + n_oversamples
        (n_samples, n_features) = M.shape
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        Q = random_state.normal(size=(n_features, n_random))
        for _ in range(n_iter):
            Q = M @ Q
            (Q, _) = np.linalg.qr(Q, mode='reduced')
            Q = M.T @ Q
            (Q, _) = np.linalg.qr(Q, mode='reduced')
        Q = M @ Q
        (Q, _) = np.linalg.qr(Q, mode='reduced')
        B = Q.T @ M
        (Uhat, S, Vt) = np.linalg.svd(B, full_matrices=False)
        U = Q @ Uhat
        return (U[:, :n_components], S[:n_components], Vt[:n_components, :])