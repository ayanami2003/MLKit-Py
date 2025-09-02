from typing import Optional, Dict, Any, List
import numpy as np
from scipy.linalg import eigh, svd
from scipy.optimize import minimize
from general.structures.feature_set import FeatureSet
from general.base_classes.transformer_base import BaseTransformer
MAX_ITER = 1000

class ExploratoryFactorAnalyzer(BaseTransformer):

    def __init__(self, n_factors: int=2, rotation: str='varimax', method: str='minres', tolerance: float=1e-05, name: Optional[str]=None):
        super().__init__(name)
        self.n_factors = n_factors
        self.rotation = rotation
        self.method = method
        self.tolerance = tolerance
        self.max_iter = MAX_ITER
        self.tol = tolerance
        self.is_fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'ExploratoryFactorAnalyzer':
        """
        Fit the factor analysis model to the input data.
        
        Args:
            data (FeatureSet): Input feature set with observations as rows and variables as columns.
            
        Returns:
            ExploratoryFactorAnalyzer: Fitted transformer instance.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance')
        X = data.features
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if self.n_factors >= n_features:
            raise ValueError('Number of factors must be less than number of features')
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        corr_matrix = np.corrcoef(X_centered, rowvar=False)
        if self.method == 'principal':
            (loadings, eigenvalues) = self._principal_components_extraction(corr_matrix)
        elif self.method == 'minres':
            (loadings, eigenvalues) = self._minres_extraction(corr_matrix)
        elif self.method == 'ml':
            (loadings, eigenvalues) = self._maximum_likelihood_extraction(corr_matrix, X_centered)
        else:
            raise ValueError(f'Unsupported extraction method: {self.method}')
        if self.rotation and self.rotation != 'none':
            loadings = self._rotate_factors(loadings)
        communalities = np.sum(loadings ** 2, axis=1)
        uniquenesses = 1 - communalities
        self.loadings_ = loadings
        self.eigenvalues_ = eigenvalues
        self.communalities_ = communalities
        self.uniquenesses_ = uniquenesses
        self.n_features_ = n_features
        self.is_fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the fitted factor model to transform the data.
        
        Args:
            data (FeatureSet): Input feature set to transform.
            
        Returns:
            FeatureSet: Transformed data with factor scores.
        """
        if not self.is_fitted:
            raise RuntimeError("This ExploratoryFactorAnalyzer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        if data.features.shape[1] != self.loadings_.shape[0]:
            raise ValueError(f'Input data must have {self.loadings_.shape[0]} features, but got {data.features.shape[1]}.')
        X_centered = data.features - self.mean_
        L = self.loadings_
        factor_scores = X_centered @ L @ np.linalg.inv(L.T @ L)
        return FeatureSet(features=factor_scores, feature_names=[f'Factor_{i + 1}' for i in range(factor_scores.shape[1])])

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Reconstruct original variables from factor scores.
        
        Args:
            data (FeatureSet): Factor scores to reconstruct.
            
        Returns:
            FeatureSet: Reconstructed original variables.
        """
        if not self.is_fitted:
            raise RuntimeError("This ExploratoryFactorAnalyzer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet instance.')
        if data.features.shape[1] != self.n_factors:
            raise ValueError(f'Input data must have {self.n_factors} factors, but got {data.features.shape[1]}.')
        reconstructed_centered = data.features @ self.loadings_.T
        reconstructed = reconstructed_centered + self.mean_
        return FeatureSet(features=reconstructed, feature_names=[f'Var_{i + 1}' for i in range(reconstructed.shape[1])])

    def get_factor_loadings(self) -> Dict[str, Any]:
        """
        Retrieve the factor loading matrix.
        
        Returns:
            Dict[str, Any]: Dictionary containing factor loadings and related statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("This ExploratoryFactorAnalyzer instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        return {'loadings': self.loadings_, 'eigenvalues': self.eigenvalues_, 'communalities': self.communalities_, 'uniquenesses': self.uniquenesses_}

    def _principal_components_extraction(self, corr_matrix):
        """
        Extract initial factor loadings using principal components method.
        """
        (eigenvalues, eigenvectors) = eigh(corr_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        loadings = eigenvectors[:, :self.n_factors] @ np.diag(np.sqrt(eigenvalues[:self.n_factors]))
        return (loadings, eigenvalues[:self.n_factors])

    def _minres_extraction(self, corr_matrix):
        """
        Extract factor loadings using MINRES (Minimum Residual) method.
        """
        n_vars = corr_matrix.shape[0]
        communalities = np.diag(corr_matrix) - 1 / n_vars
        for iteration in range(self.max_iter):
            R_adj = corr_matrix.copy()
            np.fill_diagonal(R_adj, communalities)
            (eigenvalues, eigenvectors) = eigh(R_adj)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            loadings = eigenvectors[:, :self.n_factors] @ np.diag(np.sqrt(np.maximum(eigenvalues[:self.n_factors], 0)))
            new_communalities = np.sum(loadings ** 2, axis=1)
            if np.allclose(communalities, new_communalities, atol=self.tol):
                break
            communalities = new_communalities
        return (loadings, eigenvalues[:self.n_factors])

    def _maximum_likelihood_extraction(self, corr_matrix, X_centered):
        """
        Extract factor loadings using Maximum Likelihood method.
        """
        n_vars = corr_matrix.shape[0]
        n_samples = X_centered.shape[0]
        communalities = np.diag(corr_matrix) - 1 / n_vars
        psi = np.diag(1 - communalities)
        for iteration in range(self.max_iter):
            R_star = corr_matrix + psi
            (eigenvalues, eigenvectors) = eigh(R_star)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            L = eigenvectors[:, :self.n_factors] @ np.diag(np.sqrt(np.maximum(eigenvalues[:self.n_factors] - 1, 0)))
            new_psi = np.diag(np.diag(corr_matrix - L @ L.T))
            new_psi = np.clip(new_psi, 0.005, np.inf)
            psi_diag = np.diag(psi)
            if np.allclose(psi_diag, np.diag(new_psi), atol=self.tol):
                break
            psi = np.diag(np.diag(new_psi))
        return (L, eigenvalues[:self.n_factors])

    def _rotate_factors(self, loadings):
        """
        Apply rotation to factor loadings based on the specified rotation method.
        """
        if self.rotation == 'varimax':
            return self._varimax_rotation(loadings)
        elif self.rotation == 'promax':
            return self._promax_rotation(loadings)
        elif self.rotation == 'oblimin':
            return self._oblimin_rotation(loadings)
        else:
            raise ValueError(f'Unsupported rotation method: {self.rotation}')

    def _varimax_rotation(self, loadings):
        """
        Perform varimax rotation on factor loadings.
        """
        (n_vars, n_factors) = loadings.shape
        loadings = loadings.copy()
        T = np.eye(n_factors)
        for iteration in range(self.max_iter):
            rotated = loadings @ T
            V = rotated ** 3 - 1 / n_vars * rotated @ np.diag(np.sum(rotated ** 2, axis=0))
            (U, _, Vt) = svd(loadings.T @ V)
            T_new = U @ Vt
            if np.allclose(T, T_new, atol=self.tol):
                break
            T = T_new
        return loadings @ T

    def _promax_rotation(self, loadings):
        """
        Perform promax rotation on factor loadings.
        """
        varimax_loadings = self._varimax_rotation(loadings)
        kappa = 4
        P = np.abs(varimax_loadings) ** kappa
        P = np.sign(varimax_loadings) * P
        (U, _, Vt) = svd(loadings.T @ P)
        T = U @ Vt
        rotated = loadings @ T
        return rotated

    def _oblimin_rotation(self, loadings):
        """
        Perform oblimin rotation on factor loadings.
        """
        (n_vars, n_factors) = loadings.shape
        loadings = loadings.copy()
        T = np.eye(n_factors)
        gamma = 0
        for iteration in range(self.max_iter):
            rotated = loadings @ T
            rotated_squared = rotated ** 2
            row_sums = np.sum(rotated_squared, axis=1, keepdims=True)
            col_sums = np.sum(rotated_squared, axis=0, keepdims=True)
            N = row_sums @ np.ones((1, n_factors)) - gamma * np.ones((n_vars, 1)) @ col_sums
            G = rotated.T @ (rotated ** 3 - N * rotated)
            step_size = 0.01
            try:
                T_inv = np.linalg.inv(T)
                T_new = T - step_size * T_inv.T @ G
            except np.linalg.LinAlgError:
                T_inv = np.linalg.pinv(T)
                T_new = T - step_size * T_inv.T @ G
            if np.allclose(T, T_new, atol=self.tol):
                break
            T = T_new
        return loadings @ T

class MultidimensionalScaler(BaseTransformer):

    def __init__(self, n_components: int=2, metric: bool=True, dissimilarity: str='euclidean', max_iter: int=300, eps: float=0.001, name: Optional[str]=None):
        super().__init__(name)
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError('n_components must be a positive integer')
        if not isinstance(metric, bool):
            raise TypeError('metric must be a boolean')
        if dissimilarity not in ['euclidean', 'precomputed']:
            raise ValueError("dissimilarity must be 'euclidean' or 'precomputed'")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError('max_iter must be a positive integer')
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError('eps must be a positive number')
        self.n_components = n_components
        self.metric = metric
        self.dissimilarity = dissimilarity
        self.max_iter = max_iter
        self.eps = eps
        self._is_fitted = False
        self._stress = None
        self._embedding = None

    def fit(self, data: FeatureSet, **kwargs) -> 'MultidimensionalScaler':
        """
        Fit the MDS model to the input data.
        
        Args:
            data (FeatureSet): Input feature set representing dissimilarities or raw data points.
            
        Returns:
            MultidimensionalScaler: Fitted transformer instance.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('data must be a FeatureSet instance')
        if self.dissimilarity == 'precomputed':
            dissimilarities = data.features
            if dissimilarities.shape[0] != dissimilarities.shape[1]:
                raise ValueError('Precomputed dissimilarity matrix must be square')
        else:
            X = data.features
            n_samples = X.shape[0]
            dissimilarities = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.linalg.norm(X[i] - X[j])
                    dissimilarities[i, j] = dist
                    dissimilarities[j, i] = dist
        n_samples = dissimilarities.shape[0]
        embedding = np.random.RandomState(42).randn(n_samples, self.n_components) * 0.1
        if self.metric:
            D_squared = dissimilarities ** 2
            J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
            B = -0.5 * J @ D_squared @ J
            (eigenvals, eigenvecs) = eigh(B)
            idx = np.argsort(eigenvals)[::-1][:self.n_components]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            positive_eigenvals = np.maximum(eigenvals, 0)
            embedding = eigenvecs @ np.diag(np.sqrt(positive_eigenvals))
        (self._embedding, self._stress) = self._smacof_optimization(dissimilarities, embedding)
        self._is_fitted = True
        return self

    def _smacof_optimization(self, dissimilarities, init_embedding):
        """
        Perform SMACOF optimization for MDS.
        
        Args:
            dissimilarities: Dissimilarity matrix
            init_embedding: Initial embedding
            
        Returns:
            tuple: (optimized embedding, final stress)
        """
        n_samples = dissimilarities.shape[0]
        embedding = init_embedding.copy()
        prev_stress = np.inf
        for iteration in range(self.max_iter):
            distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.linalg.norm(embedding[i] - embedding[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            stress = np.sum((dissimilarities - distances) ** 2)
            stress = np.sqrt(stress / np.sum(dissimilarities ** 2))
            if abs(prev_stress - stress) < self.eps:
                break
            prev_stress = stress
            ratios = np.divide(dissimilarities, distances, out=np.zeros_like(dissimilarities), where=distances != 0)
            B = -ratios
            np.fill_diagonal(B, 0)
            np.fill_diagonal(B, -np.sum(B, axis=1))
            embedding = 1 / n_samples * B @ embedding
        return (embedding, stress)

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the fitted MDS model to transform the data.
        
        Args:
            data (FeatureSet): Input feature set to transform.
            
        Returns:
            FeatureSet: Transformed data with coordinates in the reduced dimensional space.
        """
        if not self._is_fitted:
            raise RuntimeError('Transformer must be fitted before calling transform')
        if not isinstance(data, FeatureSet):
            raise TypeError('data must be a FeatureSet instance')
        if self.dissimilarity == 'precomputed':
            dissimilarities = data.features
            if dissimilarities.shape[0] != dissimilarities.shape[1]:
                raise ValueError('Precomputed dissimilarity matrix must be square')
        else:
            X = data.features
            n_samples = X.shape[0]
            dissimilarities = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = np.linalg.norm(X[i] - X[j])
                    dissimilarities[i, j] = dist
                    dissimilarities[j, i] = dist
        n_samples = dissimilarities.shape[0]
        embedding = np.random.RandomState(42).randn(n_samples, self.n_components) * 0.1
        if self.metric:
            D_squared = dissimilarities ** 2
            J = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
            B = -0.5 * J @ D_squared @ J
            (eigenvals, eigenvecs) = eigh(B)
            idx = np.argsort(eigenvals)[::-1][:self.n_components]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            positive_eigenvals = np.maximum(eigenvals, 0)
            embedding = eigenvecs @ np.diag(np.sqrt(positive_eigenvals))
        (final_embedding, _) = self._smacof_optimization(dissimilarities, embedding)
        return FeatureSet(features=final_embedding, feature_names=[f'mds_component_{i}' for i in range(self.n_components)], sample_ids=data.sample_ids)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Not supported for MDS as transformation is not invertible.
        
        Args:
            data (FeatureSet): Input data (ignored).
            
        Returns:
            FeatureSet: Empty feature set.
            
        Raises:
            NotImplementedError: Always raised as MDS does not support inverse transformation.
        """
        raise NotImplementedError('MDS does not support inverse transformation')

    def get_stress(self) -> float:
        """
        Retrieve the final stress value from the MDS optimization.
        
        Returns:
            float: Stress value indicating goodness of fit.
        """
        if not self._is_fitted:
            raise RuntimeError('Transformer must be fitted before getting stress')
        return self._stress