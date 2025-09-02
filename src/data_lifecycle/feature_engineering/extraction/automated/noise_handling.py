from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy import ndimage
from sklearn.decomposition import PCA

class NoiseHandlerExtractor(BaseTransformer):

    def __init__(self, method: str='wavelet_denoising', strength: float=0.5, preserve_features: bool=True, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        supported_methods = ['wavelet_denoising', 'median_filtering', 'pca_denoising', 'autoencoder_denoising']
        if method not in supported_methods:
            raise ValueError(f"Method '{method}' is not supported. Choose from: {supported_methods}")
        if not 0 <= strength <= 1:
            raise ValueError('Strength must be between 0 and 1')
        self.method = method
        self.strength = strength
        self.preserve_features = preserve_features
        self.random_state = random_state
        self._is_fitted = False
        self.pca_components_ = None
        self.feature_mean_ = None
        self.feature_std_ = None
        self.n_features_in_ = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'NoiseHandlerExtractor':
        """
        Fit the noise handler to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the noise handler on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for fitting (ignored).
            
        Returns
        -------
        NoiseHandlerExtractor
            Self instance for method chaining.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(data, FeatureSet):
            X = data.features.copy()
        elif isinstance(data, np.ndarray):
            X = data.copy()
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        self.n_features_in_ = X.shape[1]
        if self.method == 'pca_denoising':
            n_components = max(1, int(X.shape[1] * (1 - self.strength * 0.5)))
            n_components = min(n_components, X.shape[1])
            pca = PCA(n_components=n_components, random_state=self.random_state)
            pca.fit(X)
            self.pca_components_ = pca.components_
        elif self.method == 'autoencoder_denoising':
            self.feature_mean_ = np.mean(X, axis=0)
            self.feature_std_ = np.std(X, axis=0)
            self.feature_std_[self.feature_std_ == 0] = 1
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply noise handling to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, transforms the features attribute.
        **kwargs : dict
            Additional parameters for transformation (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed feature set with noise reduced.
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data.copy()
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data must have {self.n_features_in_} features, got {X.shape[1]}')
        if self.method == 'wavelet_denoising':
            X_denoised = self._wavelet_denoising(X)
        elif self.method == 'median_filtering':
            X_denoised = self._median_filtering(X)
        elif self.method == 'pca_denoising':
            X_denoised = self._pca_denoising(X)
        elif self.method == 'autoencoder_denoising':
            X_denoised = self._autoencoder_denoising(X)
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        metadata['noise_handling_method'] = self.method
        metadata['noise_handling_strength'] = self.strength
        return FeatureSet(features=X_denoised, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation if possible (may not be supported by all methods).
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert.
        **kwargs : dict
            Additional parameters for inverse transformation (ignored).
            
        Returns
        -------
        FeatureSet
            Data in original space (if supported) or raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported for the chosen method.
        """
        if not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data.copy()
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if self.method == 'pca_denoising':
            if self.pca_components_ is not None:
                X_reconstructed = X @ self.pca_components_.T @ self.pca_components_
                if 'noise_handling_method' in metadata:
                    del metadata['noise_handling_method']
                if 'noise_handling_strength' in metadata:
                    del metadata['noise_handling_strength']
                return FeatureSet(features=X_reconstructed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
            else:
                raise ValueError('PCA components not available for inverse transformation')
        else:
            raise NotImplementedError(f"Inverse transformation not supported for method '{self.method}'")

    def _wavelet_denoising(self, X: np.ndarray) -> np.ndarray:
        """
        Apply wavelet-based denoising to the data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to denoise.
            
        Returns
        -------
        np.ndarray
            Denoised data.
        """
        sigma = np.median(np.abs(X - np.median(X))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(X.size)) * (1 - self.strength)
        X_denoised = np.sign(X) * np.maximum(np.abs(X) - threshold, 0)
        return X_denoised

    def _median_filtering(self, X: np.ndarray) -> np.ndarray:
        """
        Apply median filtering for noise reduction.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to filter.
            
        Returns
        -------
        np.ndarray
            Filtered data.
        """
        filter_size = max(1, int(3 + self.strength * 4))
        if filter_size % 2 == 0:
            filter_size += 1
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_filtered[i] = ndimage.median_filter(X[i], size=filter_size)
        return X_filtered

    def _pca_denoising(self, X: np.ndarray) -> np.ndarray:
        """
        Use PCA to separate signal from noise.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to denoise.
            
        Returns
        -------
        np.ndarray
            Denoised data.
        """
        if self.pca_components_ is not None:
            X_projected = X @ self.pca_components_.T
            X_reconstructed = X_projected @ self.pca_components_
            return (1 - self.strength) * X + self.strength * X_reconstructed
        else:
            return X

    def _autoencoder_denoising(self, X: np.ndarray) -> np.ndarray:
        """
        Simulate autoencoder reconstruction for denoising.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to denoise.
            
        Returns
        -------
        np.ndarray
            Denoised data.
        """
        if self.feature_mean_ is not None and self.feature_std_ is not None:
            X_normalized = (X - self.feature_mean_) / self.feature_std_
        else:
            X_normalized = X
        noise_factor = 0.1 * (1 - self.strength)
        X_noisy = X_normalized + noise_factor * np.random.normal(size=X_normalized.shape)
        weights = np.random.normal(0, 0.1, (X.shape[1], X.shape[1]))
        weights = (weights + weights.T) / 2
        X_denoised = X_noisy @ weights
        result = (1 - self.strength) * X_normalized + self.strength * X_denoised
        if self.feature_mean_ is not None and self.feature_std_ is not None:
            result = result * self.feature_std_ + self.feature_mean_
        return result