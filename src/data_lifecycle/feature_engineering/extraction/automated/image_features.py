import numpy as np
from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from skimage.feature import hog
from skimage.transform import resize
from skimage.color import rgb2gray
import warnings


class ImageFeatureExtractor(BaseTransformer):

    def __init__(self, method: str='hog', resize_dims: Optional[tuple]=None, color_mode: str='grayscale', normalize: bool=True, name: Optional[str]=None):
        """
        Initialize the image feature extractor.
        
        Parameters
        ----------
        method : str, default='hog'
            Feature extraction technique to apply
        resize_dims : tuple of (int, int), optional
            Dimensions to resize images to before extraction
        color_mode : str, default='grayscale'
            Color space for processing images
        normalize : bool, default=True
            Whether to normalize extracted features
        name : str, optional
            Custom name for the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.resize_dims = resize_dims
        self.color_mode = color_mode
        self.normalize = normalize
        self._feature_means = None
        self._feature_stds = None

    def fit(self, data: Union[np.ndarray, List[np.ndarray]], **kwargs) -> 'ImageFeatureExtractor':
        """
        Fit the extractor to the training images.
        
        For some methods, this might involve learning normalization parameters
        or initializing pre-trained models.
        
        Parameters
        ----------
        data : ndarray or list of ndarrays
            Training images as arrays or list of image arrays
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        ImageFeatureExtractor
            Fitted transformer instance
        """
        if self.normalize:
            temp_features = self._extract_features_batch(data)
            self._feature_means = np.mean(temp_features, axis=0)
            self._feature_stds = np.std(temp_features, axis=0)
            self._feature_stds[self._feature_stds == 0] = 1.0
        return self

    def transform(self, data: Union[np.ndarray, List[np.ndarray]], **kwargs) -> FeatureSet:
        """
        Extract features from input images.
        
        Applies the configured feature extraction method to convert images
        into numerical feature representations.
        
        Parameters
        ----------
        data : ndarray or list of ndarrays
            Images to extract features from
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Extracted features with metadata
        """
        features = self._extract_features_batch(data)
        if self.normalize and self._feature_means is not None and (self._feature_stds is not None):
            features = (features - self._feature_means) / self._feature_stds
        metadata = {'method': self.method, 'resize_dims': self.resize_dims, 'color_mode': self.color_mode, 'normalize': self.normalize}
        return FeatureSet(features=features, metadata=metadata)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Attempt to reconstruct images from features (if supported).
        
        Some extraction methods may support approximate reconstruction.
        
        Parameters
        ----------
        data : FeatureSet
            Extracted features to reconstruct from
        **kwargs : dict
            Additional inversion parameters
            
        Returns
        -------
        ndarray or list of ndarrays
            Reconstructed images or list of images
        """
        features = data.features
        if self.normalize and self._feature_means is not None and (self._feature_stds is not None):
            features = features * self._feature_stds + self._feature_means
        return features

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image according to the configuration.
        
        Parameters
        ----------
        image : ndarray
            Input image array
            
        Returns
        -------
        ndarray
            Preprocessed image
        """
        if self.resize_dims is not None:
            image = cv2.resize(image, self.resize_dims)
        if self.color_mode == 'grayscale' and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def _extract_features_single(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a single image.
        
        Parameters
        ----------
        image : ndarray
            Input image array
            
        Returns
        -------
        ndarray
            Extracted features as a 1D array
        """
        processed_image = self._preprocess_image(image)
        if self.method == 'hog':
            if len(processed_image.shape) == 3:
                if self.color_mode != 'grayscale':
                    processed_image = rgb2gray(processed_image)
            features = hog(processed_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        elif self.method == 'sift':
            raise NotImplementedError('SIFT feature extraction requires additional dependencies')
        elif self.method == 'cnn_embedding':
            raise NotImplementedError('CNN embedding requires deep learning framework')
        else:
            raise ValueError(f'Unsupported feature extraction method: {self.method}')
        return features

    def _extract_features_batch(self, data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Parameters
        ----------
        data : ndarray or list of ndarrays
            Input images
            
        Returns
        -------
        ndarray
            Extracted features as a 2D array (n_samples, n_features)
        """
        if isinstance(data, np.ndarray) and (len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[2] in [1, 3, 4])):
            features = self._extract_features_single(data)
            return features.reshape(1, -1)
        if isinstance(data, list):
            features_list = [self._extract_features_single(img) for img in data]
        else:
            features_list = [self._extract_features_single(data[i]) for i in range(data.shape[0])]
        return np.vstack(features_list)