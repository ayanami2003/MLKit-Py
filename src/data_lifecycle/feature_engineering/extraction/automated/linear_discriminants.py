from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, List
from scipy.linalg import eigh

class LinearDiscriminantExtractor(BaseTransformer):
    """
    Extracts linear discriminant features for classification tasks.
    
    This transformer performs Linear Discriminant Analysis (LDA) to project data
    into a lower-dimensional space that maximizes class separability. It's primarily
    used as a supervised dimensionality reduction technique for classification.
    
    The extractor computes the linear discriminants (components) that best separate
    the classes in the training data, which can then be used to transform new data.
    
    Attributes
    ----------
    n_components : Optional[int]
        Number of components to keep. If None, defaults to (n_classes - 1)
    feature_names : Optional[List[str]]
        Names of the extracted features
    
    Methods
    -------
    fit() : Fits the LDA model to the data
    transform() : Applies the LDA transformation
    inverse_transform() : Approximates original data from LDA components (limited)
    """

    def __init__(self, n_components: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the LinearDiscriminantExtractor.
        
        Parameters
        ----------
        n_components : Optional[int], default=None
            Number of linear discriminants to extract. Must be <= (n_classes - 1).
            If None, all possible components are retained.
        name : Optional[str], default=None
            Name for the transformer instance
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.feature_names: Optional[List[str]] = None
        self._is_fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'LinearDiscriminantExtractor':
        """
        Fit the linear discriminant analysis model to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input features with labels required for supervised LDA
        **kwargs : dict
            Additional fitting parameters (not used)
            
        Returns
        -------
        LinearDiscriminantExtractor
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data does not contain labels or has insufficient samples
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet')
        X = data.features
        if X.ndim != 2:
            raise ValueError('Features must be a 2D array')
        if 'labels' not in data.metadata:
            raise ValueError('FeatureSet must contain labels in metadata for supervised LDA')
        y = np.asarray(data.metadata['labels'])
        if y.ndim != 1:
            raise ValueError('Labels must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in features and labels must match')
        if X.shape[0] < 2:
            raise ValueError('At least 2 samples are required for LDA')
        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError('At least 2 classes are required for LDA')
        n_features = X.shape[1]
        max_components = min(n_classes - 1, n_features)
        if self.n_components is None:
            self.n_components = max_components
        elif self.n_components > max_components:
            raise ValueError(f'n_components cannot be larger than min(n_classes-1, n_features) = {max_components}')
        elif self.n_components <= 0:
            raise ValueError('n_components must be positive')
        class_means = []
        for cls in classes:
            class_means.append(np.mean(X[y == cls], axis=0))
        class_means = np.array(class_means)
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((n_features, n_features))
        for (i, cls) in enumerate(classes):
            n_cls = np.sum(y == cls)
            diff = (class_means[i] - overall_mean).reshape(-1, 1)
            S_B += n_cls * (diff @ diff.T)
        S_W = np.zeros((n_features, n_features))
        for (i, cls) in enumerate(classes):
            X_cls = X[y == cls]
            diff = X_cls - class_means[i]
            S_W += diff.T @ diff
        try:
            (eigenvals, eigenvecs) = eigh(S_B, S_W)
        except np.linalg.LinAlgError:
            S_W_reg = S_W + 1e-09 * np.eye(S_W.shape[0])
            (eigenvals, eigenvecs) = eigh(S_B, S_W_reg)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        self.components_ = eigenvecs[:, :self.n_components].T
        self.explained_variance_ratio_ = eigenvals[:self.n_components] / np.sum(eigenvals)
        self.class_means_ = class_means
        self.overall_mean_ = overall_mean
        self.classes_ = classes
        self.feature_names = [f'LD{i + 1}' for i in range(self.n_components)]
        self._is_fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the linear discriminant transformation to the input data.
        
        Projects the data onto the linear discriminant axes computed during fitting.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform
        **kwargs : dict
            Additional transformation parameters (not used)
            
        Returns
        -------
        FeatureSet
            Transformed features in the discriminant space
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet')
        X = data.features
        if X.ndim != 2:
            raise ValueError('Features must be a 2D array')
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(f'Expected {self.components_.shape[1]} features, got {X.shape[1]}')
        X_transformed = (X - self.overall_mean_) @ self.components_.T
        transformed_fs = FeatureSet(features=X_transformed, feature_names=self.feature_names, metadata=data.metadata.copy() if data.metadata else {})
        return transformed_fs

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Attempt to reconstruct original features from discriminant components.
        
        Note: This is an approximation as information is lost during projection.
        
        Parameters
        ----------
        data : FeatureSet
            Linear discriminant features to reconstruct
        **kwargs : dict
            Additional reconstruction parameters (not used)
            
        Returns
        -------
        FeatureSet
            Approximated original feature space
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted
        NotImplementedError
            If inverse transformation is not supported
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet')
        X_lda = data.features
        if X_lda.ndim != 2:
            raise ValueError('Features must be a 2D array')
        if X_lda.shape[1] != self.n_components:
            raise ValueError(f'Expected {self.n_components} features, got {X_lda.shape[1]}')
        X_reconstructed = X_lda @ self.components_ + self.overall_mean_
        original_feature_names = None
        if hasattr(self, '_original_feature_names'):
            original_feature_names = self._original_feature_names
        elif data.feature_names and len(data.feature_names) == X_reconstructed.shape[1]:
            original_feature_names = data.feature_names
        reconstructed_fs = FeatureSet(features=X_reconstructed, feature_names=original_feature_names, metadata=data.metadata.copy() if data.metadata else {})
        return reconstructed_fs

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get names for the extracted linear discriminant features.
        
        Returns
        -------
        Optional[List[str]]
            Names of the discriminant components, or None if not available
        """
        return self.feature_names