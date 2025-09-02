from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import pandas as pd
from scipy.sparse.linalg import svds
TOLERANCE = 1e-12


# ...(code omitted)...


class IncrementalPrincipalComponentAnalysis(BaseTransformer):

    def __init__(self, n_components: Optional[int]=None, whiten: bool=False, batch_size: Optional[int]=None):
        """
        Initialize the IncrementalPrincipalComponentAnalysis transformer.
        
        Parameters
        ----------
        n_components : Optional[int], default=None
            Number of components to keep. If None, min(n_samples, n_features) is used.
        whiten : bool, default=False
            When True, the transformed features are scaled to have unit variance.
        batch_size : Optional[int], default=None
            Number of samples to process in each batch. If None, inferred from data.
        """
        super().__init__(name='IncrementalPrincipalComponentAnalysis')
        self.n_components = n_components
        self.whiten = whiten
        self.batch_size = batch_size
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0

    def _validate_data(self, X: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """
        Validate and convert input data to numpy array.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input data to validate and convert.
            
        Returns
        -------
        np.ndarray
            Validated input data as numpy array.
        """
        if isinstance(X, FeatureSet):
            X_array = X.features.values
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError(f'Expected 2D array, got {X_array.ndim}D array instead.')
        return X_array

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, check_input: bool=True) -> 'IncrementalPrincipalComponentAnalysis':
        """
        Fit the IPCA model to the input data in batches.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training data where rows represent samples and columns represent features.
        y : Optional[np.ndarray], default=None
            Ignored in unsupervised learning.
        check_input : bool, default=True
            Whether to validate the input data.
            
        Returns
        -------
        IncrementalPrincipalComponentAnalysis
            Self instance for method chaining.
        """
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0
        X_array = self._validate_data(X) if check_input else np.asarray(X)
        (n_samples, n_features) = X_array.shape
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        if self.batch_size is None:
            batch_size = 5 * n_features
        else:
            batch_size = self.batch_size
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_X = X_array[start_idx:end_idx]
            self.partial_fit(batch_X, check_input=False)
        return self

    def partial_fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, check_input: bool=True) -> 'IncrementalPrincipalComponentAnalysis':
        """
        Incrementally fit the IPCA model with a batch of data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Batch of training data.
        y : Optional[np.ndarray], default=None
            Ignored in unsupervised learning.
        check_input : bool, default=True
            Whether to validate the input data.
            
        Returns
        -------
        IncrementalPrincipalComponentAnalysis
            Self instance for method chaining.
        """
        X_array = self._validate_data(X) if check_input else np.asarray(X)
        (n_samples, n_features) = X_array.shape
        if self.n_features_ is None:
            self.n_features_ = n_features
            if self.n_components is None:
                self.n_components = min(n_samples, n_features)
            else:
                self.n_components = min(self.n_components, n_features)
            self.mean_ = np.zeros(n_features, dtype=np.float64)
            self.components_ = np.zeros((self.n_components, n_features), dtype=np.float64)
            self.singular_values_ = np.zeros(self.n_components, dtype=np.float64)
        elif n_features != self.n_features_:
            raise ValueError(f'X has {n_features} features, but IPCA is expecting {self.n_features_} features as input.')
        if n_samples == 0:
            return self
        col_mean = np.mean(X_array, axis=0)
        n_total = self.n_samples_seen_ + n_samples
        new_mean = (self.n_samples_seen_ * self.mean_ + n_samples * col_mean) / n_total
        X_centered = X_array - col_mean
        if self.n_samples_seen_ == 0:
            (U, s, Vt) = np.linalg.svd(X_centered, full_matrices=False)
            U = U[:, :self.n_components]
            s = s[:self.n_components]
            Vt = Vt[:self.n_components]
            self.components_ = Vt
            self.singular_values_ = s * np.sqrt(n_samples)
        else:
            mean_correction = np.sqrt(self.n_samples_seen_ * n_samples / n_total) * (self.mean_ - col_mean)
            prev_components_weighted = self.components_.T * self.singular_values_
            new_components_weighted = np.vstack([prev_components_weighted, X_centered.T * np.sqrt(n_samples), mean_correction[:, np.newaxis]])
            (U_combined, s_combined, Vt_combined) = np.linalg.svd(new_components_weighted, full_matrices=False)
            self.components_ = Vt_combined[:self.n_components, :]
            self.singular_values_ = s_combined[:self.n_components]
        self.mean_ = new_mean
        self.n_samples_seen_ = n_total
        total_variance = np.sum(np.square(self.singular_values_)) / max(1, n_total - 1)
        if total_variance > 0:
            self.explained_variance_ratio_ = np.square(self.singular_values_) / (total_variance * self.n_components)
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        return self

    def transform(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply dimensionality reduction to the input data.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data to transform with rows as samples and columns as features.
        **kwargs : dict
            Additional parameters (not used in this implementation).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data in the reduced dimensional space.
        """
        if self.components_ is None:
            raise ValueError("This IncrementalPrincipalComponentAnalysis instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        was_feature_set = isinstance(X, FeatureSet)
        if was_feature_set:
            X_array = X.features.values
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2 or X_array.shape[1] != self.n_features_:
            raise ValueError(f"Input data has {(X_array.shape[1] if X_array.ndim == 2 else 'wrong')} features, but IPCA expects {self.n_features_}")
        X_centered = X_array - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        if self.whiten and self.n_samples_seen_ > 1:
            explained_variance = np.var(X_transformed, axis=0, ddof=1)
            X_transformed /= np.sqrt(explained_variance)
        if was_feature_set:
            feature_names = [f'PC{i + 1}' for i in range(X_transformed.shape[1])]
            df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index if hasattr(X, 'index') else None)
            return FeatureSet(features=df, name=f'{X.name}_transformed' if hasattr(X, 'name') else 'transformed')
        else:
            return X_transformed

    def inverse_transform(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Transform data back to its original space.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Data in the reduced dimensional space to transform back.
        **kwargs : dict
            Additional parameters (not used in this implementation).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Data reconstructed in the original feature space.
        """
        if self.components_ is None:
            raise ValueError("This IncrementalPrincipalComponentAnalysis instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        was_feature_set = isinstance(X, FeatureSet)
        if was_feature_set:
            X_array = X.features.values
        else:
            X_array = np.asarray(X)
        if X_array.ndim != 2 or X_array.shape[1] != self.n_components:
            raise ValueError(f"Input data has {(X_array.shape[1] if X_array.ndim == 2 else 'wrong')} components, but IPCA expects {self.n_components}")
        X_to_reconstruct = X_array.copy()
        if self.whiten and self.n_samples_seen_ > 1:
            explained_variance = np.var(np.dot(X_array, self.components_), axis=0, ddof=1)
            X_to_reconstruct *= np.sqrt(explained_variance)
        X_original = np.dot(X_to_reconstruct, self.components_)
        X_reconstructed = X_original + self.mean_
        if was_feature_set:
            feature_names = [f'feature_{i}' for i in range(X_reconstructed.shape[1])]
            df = pd.DataFrame(X_reconstructed, columns=feature_names, index=X.index if hasattr(X, 'index') else None)
            return FeatureSet(features=df, name=f'{X.name}_reconstructed' if hasattr(X, 'name') else 'reconstructed')
        else:
            return X_reconstructed