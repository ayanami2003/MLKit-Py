from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy import linalg
from scipy.sparse.linalg import eigsh


# ...(code omitted)...


class PolynomialKernelPCA(BaseTransformer):

    def __init__(self, n_components: Optional[int]=None, degree: int=3, gamma: Optional[float]=None, coef0: float=1, alpha: float=1.0, fit_inverse_transform: bool=False, eigen_solver: str='auto', tol: float=0, max_iter: Optional[int]=None, remove_zero_eig: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state

    def _get_kernel(self, X, Y=None):
        """
        Compute the polynomial kernel between X and Y.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.
            
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Polynomial kernel k(X, Y) = (gamma * <X, Y> + coef0)^degree
        """
        if Y is None:
            Y = X
        K = np.dot(X, Y.T)
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        K = K * self.gamma
        K += self.coef0
        K **= self.degree
        return K

    def _center_kernel(self, K):
        """
        Center the kernel matrix.
        
        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix to center
            
        Returns
        -------
        K_centered : ndarray of shape (n_samples, n_samples)
            Centered kernel matrix
        """
        n_samples = K.shape[0]
        K_copy = K.copy()
        means = np.mean(K_copy, axis=1)
        mean_total = np.mean(means)
        K_centered = K_copy
        K_centered -= means[:, np.newaxis]
        K_centered -= means[np.newaxis, :]
        K_centered += mean_total
        return K_centered

    def _center_kernel_for_transform(self, K):
        """
        Center the kernel matrix for transformation of new data.
        
        Parameters
        ----------
        K : ndarray of shape (n_samples_test, n_samples_train)
            Kernel matrix between test and training data
            
        Returns
        -------
        K_centered : ndarray of shape (n_samples_test, n_samples_train)
            Centered kernel matrix for transformation
        """
        n_samples_train = self.X_fit_.shape[0]
        n_samples_test = K.shape[0]
        K_train_means = np.mean(self._get_kernel(self.X_fit_), axis=1)
        K_test_means = np.mean(K, axis=1)
        K_total_mean = np.mean(K_train_means)
        K_centered = K - K_test_means[:, np.newaxis] - K_train_means[np.newaxis, :] + K_total_mean
        return K_centered

    def _solve_eigen_system(self, K):
        """
        Solve the eigenvalue problem for the kernel matrix.
        
        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix
            
        Returns
        -------
        eigvals : ndarray of shape (n_components,)
            Eigenvalues in descending order
        eigvecs : ndarray of shape (n_samples, n_components)
            Eigenvectors corresponding to eigenvalues
        """
        n_samples = K.shape[0]
        if self.n_components is None:
            n_components = n_samples
        else:
            n_components = min(self.n_components, n_samples)
        if self.eigen_solver == 'auto':
            if n_samples > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver
        if eigen_solver == 'dense':
            (eigvals, eigvecs) = linalg.eigh(K)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            eigvals = eigvals[:n_components]
            eigvecs = eigvecs[:, :n_components]
        elif eigen_solver == 'arpack':
            (eigvals, eigvecs) = eigsh(K, k=n_components, which='LA', tol=self.tol, maxiter=self.max_iter)
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]
        if self.remove_zero_eig:
            mask = eigvals > 0
            eigvals = eigvals[mask]
            eigvecs = eigvecs[:, mask]
        return (eigvals, eigvecs)

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'PolynomialKernelPCA':
        """
        Fit the Polynomial Kernel PCA model with the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the model on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        PolynomialKernelPCA
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
            self._sample_ids = data.sample_ids
            self._metadata = data.metadata
        else:
            X = data
            self._feature_names = None
            self._sample_ids = None
            self._metadata = None
        self.X_fit_ = X.copy()
        K = self._get_kernel(X)
        K_centered = self._center_kernel(K)
        (eigvals, eigvecs) = self._solve_eigen_system(K_centered)
        self.eigenvalues_ = eigvals
        self.eigenvectors_ = eigvecs
        self.eigenvectors_ = eigvecs / np.sqrt(np.maximum(eigvals, 1e-12))
        if self.fit_inverse_transform:
            self._fit_inverse_transform(X, eigvecs, eigvals)
        return self

    def _fit_inverse_transform(self, X, eigvecs, eigvals):
        """
        Fit the inverse transform mapping.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        eigvecs : ndarray of shape (n_samples, n_components)
            Eigenvectors of the kernel matrix
        eigvals : ndarray of shape (n_components,)
            Eigenvalues of the kernel matrix
        """
        self.dual_coef_ = np.dot(X.T, eigvecs) / np.sqrt(np.maximum(eigvals, 1e-12))
        self.X_transformed_fit_ = np.dot(eigvecs, np.sqrt(np.maximum(eigvals, 1e-12)) * np.eye(len(eigvals)))

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply dimensionality reduction to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, transforms the features.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data in lower-dimensional space. Type matches input type.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            input_is_feature_set = True
        else:
            X = data
            input_is_feature_set = False
        K = self._get_kernel(X, self.X_fit_)
        K_centered = self._center_kernel_for_transform(K)
        X_transformed = np.dot(K_centered, self.eigenvectors_)
        if input_is_feature_set:
            return FeatureSet(features=X_transformed, feature_names=[f'component_{i}' for i in range(X_transformed.shape[1])], sample_ids=data.sample_ids if hasattr(data, 'sample_ids') else None, metadata=data.metadata if hasattr(data, 'metadata') else None)
        else:
            return X_transformed

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Transform data back to its original space (approximation).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data in the reduced space to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Approximation of the data in original space. Type matches input type.
            
        Raises
        ------
        NotImplementedError
            If fit_inverse_transform was False during initialization.
        """
        if not self.fit_inverse_transform:
            raise NotImplementedError('Inverse transform is not available when fit_inverse_transform=False')
        if isinstance(data, FeatureSet):
            X = data.features
            input_is_feature_set = True
        else:
            X = data
            input_is_feature_set = False
        X_original = np.dot(X, self.dual_coef_.T)
        if input_is_feature_set:
            return FeatureSet(features=X_original, feature_names=self._feature_names if self._feature_names else [f'feature_{i}' for i in range(X_original.shape[1])], sample_ids=data.sample_ids if hasattr(data, 'sample_ids') else None, metadata=data.metadata if hasattr(data, 'metadata') else None)
        else:
            return X_original

class RBFKernelPCA(BaseTransformer):

    def __init__(self, n_components: Optional[int]=None, gamma: Optional[float]=None, alpha: float=1.0, fit_inverse_transform: bool=False, eigen_solver: str='auto', tol: float=0, max_iter: Optional[int]=None, remove_zero_eig: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'RBFKernelPCA':
        """
        Fit the RBF Kernel PCA model with the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Training data. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        RBFKernelPCA
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
            self._sample_ids = data.sample_ids
            self._metadata = data.metadata
        else:
            X = data
            self._feature_names = None
            self._sample_ids = None
            self._metadata = None
        self.X_fit_ = X.copy()
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        K = self._compute_rbf_kernel(X, X)
        self.K_fit_ = K.copy()
        K_centered = self._center_kernel(K)
        (eigvals, eigvecs) = self._solve_eigen_system(K_centered)
        self.eigenvalues_ = eigvals
        self.eigenvectors_ = eigvecs
        self.eigenvectors_ = eigvecs / np.sqrt(np.maximum(eigvals, 1e-12))
        if self.fit_inverse_transform:
            self._fit_inverse_transform(X, eigvecs, eigvals)
        return self

    def _compute_rbf_kernel(self, X, Y):
        """
        Compute the RBF kernel matrix between X and Y.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            First set of samples
        Y : ndarray of shape (n_samples_Y, n_features)
            Second set of samples
            
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            RBF kernel matrix
        """
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
        XY = np.dot(X, Y.T)
        dist_sq = X_sq + Y_sq - 2 * XY
        K = np.exp(-self.gamma * dist_sq)
        return K

    def _center_kernel(self, K):
        """
        Center the kernel matrix.
        
        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix to center
            
        Returns
        -------
        K_centered : ndarray of shape (n_samples, n_samples)
            Centered kernel matrix
        """
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K_centered = K - np.dot(one_n, K) - np.dot(K, one_n) + np.dot(np.dot(one_n, K), one_n)
        return K_centered

    def _solve_eigen_system(self, K):
        """
        Solve the eigenvalue problem for the kernel matrix.
        
        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix
            
        Returns
        -------
        eigvals : ndarray of shape (n_components,)
            Eigenvalues in descending order
        eigvecs : ndarray of shape (n_samples, n_components)
            Eigenvectors corresponding to eigenvalues
        """
        n_samples = K.shape[0]
        if self.n_components is None:
            n_components = n_samples
        else:
            n_components = min(self.n_components, n_samples)
        if self.eigen_solver == 'auto':
            if n_samples > 1000 and n_components < 10:
                solver = 'arpack'
            else:
                solver = 'dense'
        else:
            solver = self.eigen_solver
        if solver == 'dense':
            (eigvals, eigvecs) = linalg.eigh(K, eigvals_only=False)
            indices = np.argsort(eigvals)[::-1]
            eigvals = eigvals[indices][:n_components]
            eigvecs = eigvecs[:, indices][:, :n_components]
        elif solver == 'arpack':
            random_state = np.random.RandomState(self.random_state)
            v0 = random_state.uniform(-1, 1, K.shape[0])
            (eigvals, eigvecs) = eigsh(K, k=n_components, which='LA', tol=self.tol, maxiter=self.max_iter, v0=v0)
            indices = np.argsort(eigvals)[::-1]
            eigvals = eigvals[indices]
            eigvecs = eigvecs[:, indices]
        else:
            raise ValueError(f'Unsupported eigen_solver: {solver}')
        if self.remove_zero_eig:
            mask = eigvals > self.tol
            eigvals = eigvals[mask]
            eigvecs = eigvecs[:, mask]
        return (eigvals, eigvecs)

    def _fit_inverse_transform(self, X, eigvecs, eigvals):
        """
        Fit the inverse transform mapping.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        eigvecs : ndarray of shape (n_samples, n_components)
            Eigenvectors of the kernel matrix
        eigvals : ndarray of shape (n_components,)
            Eigenvalues of the kernel matrix
        """
        self.dual_coef_ = np.dot(X.T, eigvecs) / np.sqrt(np.maximum(eigvals, 1e-12))
        self.X_transformed_fit_ = np.dot(eigvecs, np.sqrt(np.maximum(eigvals, 1e-12)) * np.eye(len(eigvals)))

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply dimensionality reduction to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, transforms the features.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data in lower-dimensional space. Type matches input type.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            input_is_feature_set = True
        else:
            X = data
            input_is_feature_set = False
        K = self._compute_rbf_kernel(X, self.X_fit_)
        K_centered = self._center_kernel_for_transform(K)
        X_transformed = np.dot(K_centered, self.eigenvectors_)
        if input_is_feature_set:
            return FeatureSet(features=X_transformed, feature_names=[f'component_{i}' for i in range(X_transformed.shape[1])], sample_ids=data.sample_ids if hasattr(data, 'sample_ids') else None, metadata=data.metadata if hasattr(data, 'metadata') else None)
        else:
            return X_transformed

    def _center_kernel_for_transform(self, K):
        """
        Center the kernel matrix for transformation of new data.
        
        Parameters
        ----------
        K : ndarray of shape (n_samples_test, n_samples_train)
            Kernel matrix between test and training data
            
        Returns
        -------
        K_centered : ndarray of shape (n_samples_test, n_samples_train)
            Centered kernel matrix
        """
        N_train = self.K_fit_.shape[0]
        N_test = K.shape[0]
        one_n_train = np.ones((N_train, N_train)) / N_train
        one_n_test = np.ones((N_test, N_train)) / N_train
        K_centered = K - np.dot(one_n_test, self.K_fit_) - np.dot(K, one_n_train) + np.dot(np.dot(one_n_test, self.K_fit_), one_n_train)
        return K_centered

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Transform data back to its original space (approximation).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data in the reduced space to inverse transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Approximation of the data in original space. Type matches input type.
            
        Raises
        ------
        NotImplementedError
            If fit_inverse_transform was False during initialization.
        """
        if not self.fit_inverse_transform:
            raise NotImplementedError('Inverse transform is not available when fit_inverse_transform=False')
        if isinstance(data, FeatureSet):
            X = data.features
            input_is_feature_set = True
        else:
            X = data
            input_is_feature_set = False
        X_original = np.dot(X, self.dual_coef_.T)
        if input_is_feature_set:
            return FeatureSet(features=X_original, feature_names=self._feature_names if self._feature_names else [f'feature_{i}' for i in range(X_original.shape[1])], sample_ids=data.sample_ids if hasattr(data, 'sample_ids') else None, metadata=data.metadata if hasattr(data, 'metadata') else None)
        else:
            return X_original

    def fit_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Fit the model with data and apply the dimensionality reduction on data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Training data to fit and transform.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data in lower-dimensional space. Type matches input type.
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)