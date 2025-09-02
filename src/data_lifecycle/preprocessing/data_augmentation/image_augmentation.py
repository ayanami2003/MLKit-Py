from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ImageAugmentationTransformer(BaseTransformer):

    def __init__(self, augmentation_types: Optional[List[str]]=None, intensity: float=0.5, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None):
        super().__init__(name=name)
        self.augmentation_types = augmentation_types or ['rotation', 'horizontal_flip', 'vertical_flip', 'scaling', 'translation', 'noise_injection']
        self.intensity = intensity
        self.random_state = random_state
        self.preserve_original = preserve_original
        self.fitted_ = False
        self.augmentation_params_ = {}

    def fit(self, data: FeatureSet, **kwargs) -> 'ImageAugmentationTransformer':
        """
        Fit the transformer to the input data.
        
        This method analyzes the input data to determine appropriate
        parameters for the specified augmentations.
        
        Parameters
        ----------
        data : FeatureSet
            Input image data to fit the transformer on. Expected to contain
            image arrays in the features attribute.
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        ImageAugmentationTransformer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data format is not compatible
        """
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet object')
        if data.features.ndim != 2:
            raise ValueError('Features must be a 2D array')
        (n_samples, n_features) = data.features.shape
        if data.metadata and 'image_shape' in data.metadata:
            image_shape = data.metadata['image_shape']
            if len(image_shape) not in [2, 3]:
                raise ValueError('Image shape must have 2 or 3 dimensions (height, width) or (height, width, channels)')
        else:
            img_size = int(np.sqrt(n_features))
            if img_size * img_size != n_features:
                raise ValueError('Unable to infer image dimensions. Please provide image_shape in metadata.')
            image_shape = (img_size, img_size)
        self.augmentation_params_ = {'image_shape': image_shape, 'n_samples': n_samples, 'n_features': n_features}
        self.rng = np.random.default_rng(self.random_state)
        self.fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply image augmentations to the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input image data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Transformed image data with applied augmentations
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if not self.fitted_:
            raise ValueError('Transformer must be fitted before transform')
        image_shape = self.augmentation_params_['image_shape']
        n_samples = data.features.shape[0]
        images = data.features.reshape((n_samples, *image_shape))
        augmented_images = []
        for img in images:
            aug_img = img.copy()
            for aug_type in self.augmentation_types:
                if aug_type == 'rotation':
                    aug_img = self._rotate_image(aug_img)
                elif aug_type == 'horizontal_flip':
                    if self.rng.random() < self.intensity:
                        aug_img = np.fliplr(aug_img)
                elif aug_type == 'vertical_flip':
                    if self.rng.random() < self.intensity:
                        aug_img = np.flipud(aug_img)
                elif aug_type == 'scaling':
                    aug_img = self._scale_image(aug_img)
                elif aug_type == 'translation':
                    aug_img = self._translate_image(aug_img)
                elif aug_type == 'noise_injection':
                    aug_img = self._add_noise(aug_img)
            augmented_images.append(aug_img)
        augmented_images = np.array(augmented_images)
        if self.preserve_original:
            combined_images = np.vstack([images, augmented_images])
        else:
            combined_images = augmented_images
        n_output_samples = combined_images.shape[0]
        output_features = combined_images.reshape((n_output_samples, -1))
        output_sample_ids = None
        if data.sample_ids is not None:
            if self.preserve_original:
                output_sample_ids = data.sample_ids + [f'aug_{i}' for i in range(len(augmented_images))]
            else:
                output_sample_ids = [f'aug_{i}' for i in range(len(augmented_images))]
        return FeatureSet(features=output_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=output_sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation (not applicable for most augmentations).
        
        As most image augmentations are not invertible, this method will typically
        return the input data unchanged while issuing a warning.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original data (unchanged)
        """
        import warnings
        warnings.warn('Image augmentations are generally not invertible. Returning input data unchanged.', UserWarning)
        return data

    def _rotate_image(self, image):
        """Rotate image by a random angle based on intensity."""
        max_angle = 30 * self.intensity
        angle = self.rng.uniform(-max_angle, max_angle)
        if abs(angle) < 1e-06:
            return image
        try:
            from scipy.ndimage import rotate
            return rotate(image, angle, reshape=False, mode='nearest')
        except ImportError:
            if abs(angle - 90) < 1:
                return np.rot90(image)
            elif abs(angle - 180) < 1:
                return np.rot90(image, 2)
            elif abs(angle - 270) < 1:
                return np.rot90(image, 3)
            else:
                return image

    def _scale_image(self, image):
        """Scale image by a factor based on intensity."""
        if self.intensity <= 0:
            return image
        scale_factor = 1 + self.intensity * 0.5
        try:
            from scipy.ndimage import zoom
            if len(image.shape) == 3:
                zoom_factors = [scale_factor, scale_factor, 1]
            else:
                zoom_factors = [scale_factor, scale_factor]
            scaled_image = zoom(image, zoom_factors, order=1)
            original_shape = image.shape
            result = np.zeros(original_shape)
            min_height = min(scaled_image.shape[0], original_shape[0])
            min_width = min(scaled_image.shape[1], original_shape[1])
            result[:min_height, :min_width] = scaled_image[:min_height, :min_width]
            return result
        except ImportError:
            return image

    def _translate_image(self, image):
        """Translate image by a few pixels based on intensity."""
        if self.intensity <= 0:
            return image
        max_shift = int(5 * self.intensity)
        if max_shift == 0:
            return image
        shift_x = self.rng.integers(-max_shift, max_shift + 1)
        shift_y = self.rng.integers(-max_shift, max_shift + 1)
        if shift_x == 0 and shift_y == 0:
            return image
        try:
            from scipy.ndimage import shift
            if len(image.shape) == 3:
                shifts = [shift_x, shift_y, 0]
            else:
                shifts = [shift_x, shift_y]
            return shift(image, shifts, mode='nearest')
        except ImportError:
            return image

    def _add_noise(self, image):
        """Add random noise to image based on intensity."""
        if self.intensity <= 0:
            return image
        noise_level = 0.1 * self.intensity
        noise = self.rng.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)