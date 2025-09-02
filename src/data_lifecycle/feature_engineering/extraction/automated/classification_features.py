from typing import Optional, List, Union
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, FastICA
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ClassificationFeatureExtractor(BaseTransformer):

    def __init__(self, extraction_method: str='lda', n_components: Optional[int]=None, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the ClassificationFeatureExtractor.
        
        Parameters
        ----------
        extraction_method : str, default='lda'
            Method to use for feature extraction. Supported methods include:
            - 'lda': Linear Discriminant Analysis
            - 'pca_classifier': Principal Component Analysis optimized for classification
            - 'ica': Independent Component Analysis
        n_components : int, optional
            Number of components to retain. If None, automatically determined
        random_state : int, optional
            Random seed for reproducible results
        name : str, optional
            Name for the transformer instance
        """
        super().__init__(name=name)
        self.extraction_method = extraction_method
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, np.ndarray], labels: Union[np.ndarray, List], **kwargs) -> 'ClassificationFeatureExtractor':
        """
        Fit the feature extractor to the input data and labels.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to extract features from. Should be a FeatureSet or 2D numpy array
            where rows are samples and columns are features
        labels : Union[np.ndarray, List]
            Target class labels for each sample
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        ClassificationFeatureExtractor
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data and labels shapes don't match or if unsupported extraction method is specified
        """
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            self._feature_names = data.feature_names
        else:
            X = np.asarray(data)
            self._feature_names = None
        y = np.asarray(labels)
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if y.ndim != 1:
            raise ValueError('Labels must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in data ({X.shape[0]}) does not match number of labels ({y.shape[0]})')
        supported_methods = ['lda', 'pca_classifier', 'ica']
        if self.extraction_method not in supported_methods:
            raise ValueError(f"Unsupported extraction method '{self.extraction_method}'. Supported methods are: {supported_methods}")
        if self.random_state is not None:
            np.random.seed(self.random_state)
        (n_samples, n_features) = X.shape
        n_classes = len(np.unique(y))
        if self.n_components is None:
            if self.extraction_method == 'lda':
                self._n_components = min(n_classes - 1, n_features)
            else:
                self._n_components = min(n_samples, n_features)
        else:
            if self.n_components <= 0:
                raise ValueError('n_components must be positive')
            if self.extraction_method == 'lda' and self.n_components > min(n_classes - 1, n_features):
                raise ValueError(f'For LDA, n_components cannot be larger than min(n_classes-1, n_features) = {min(n_classes - 1, n_features)}')
            elif self.n_components > min(n_samples, n_features):
                raise ValueError(f'n_components cannot be larger than min(n_samples, n_features) = {min(n_samples, n_features)}')
            self._n_components = self.n_components
        if self.extraction_method == 'lda':
            self._extractor = LinearDiscriminantAnalysis(n_components=self._n_components)
            self._extractor.fit(X, y)
        elif self.extraction_method == 'pca_classifier':
            self._extractor = PCA(n_components=self._n_components, random_state=self.random_state)
            self._extractor.fit(X)
        elif self.extraction_method == 'ica':
            self._extractor = FastICA(n_components=self._n_components, random_state=self.random_state)
            self._extractor.fit(X)
        self._is_fitted = True
        self._n_features_in = n_features
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the fitted feature extraction to new data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Must have same number of features as training data
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed feature set with extracted classification-oriented features
            
        Raises
        ------
        RuntimeError
            If transformer has not been fitted yet
        ValueError
            If input data has incompatible shape
        """
        pass

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation to reconstruct original feature space.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert back to original space
        **kwargs : dict
            Additional inversion parameters
            
        Returns
        -------
        FeatureSet
            Reconstructed features in original space
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported for the extraction method
        """
        pass