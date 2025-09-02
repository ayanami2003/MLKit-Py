from typing import Union, Optional
import numpy as np
from scipy.ndimage import median_filter
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class MedianFilterTransformer(BaseTransformer):

    def __init__(self, kernel_size: int=3, axis: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the MedianFilterTransformer.
        
        Parameters
        ----------
        kernel_size : int, default=3
            Size of the sliding window for median calculation. Must be a positive odd integer.
        axis : int, optional
            Axis along which to apply the filter. If None, filter is applied to all axes.
        name : str, optional
            Name of the transformer instance.
            
        Raises
        ------
        ValueError
            If kernel_size is not a positive odd integer.
        """
        super().__init__(name=name)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        self.kernel_size = kernel_size
        self.axis = axis

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'MedianFilterTransformer':
        """
        Fit the transformer to the input data.
        
        For median filtering, fitting is a no-op since no parameters need to be learned.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        MedianFilterTransformer
            Self instance for method chaining.
        """
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply median filtering to reduce noise in the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Can be a FeatureSet or numpy array.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Filtered data with the same type as input.
        """
        if isinstance(data, FeatureSet):
            original_metadata = data.metadata.copy() if data.metadata else {}
            original_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
            if self.axis is None:
                filtered_data = median_filter(data.features, size=self.kernel_size)
            else:
                filtered_data = median_filter(data.features, size=self.kernel_size, axis=self.axis)
            return FeatureSet(features=filtered_data, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=original_metadata, quality_scores=original_quality_scores)
        elif isinstance(data, np.ndarray):
            if self.axis is None:
                return median_filter(data, size=self.kernel_size)
            else:
                return median_filter(data, size=self.kernel_size, axis=self.axis)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Raise NotImplementedError as median filtering is not invertible.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data to inverse transform (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Raises
        ------
        NotImplementedError
            Always raised as median filtering is not invertible.
        """
        raise NotImplementedError('Median filtering is not invertible')