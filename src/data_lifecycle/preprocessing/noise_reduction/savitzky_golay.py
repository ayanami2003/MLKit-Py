from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class SavitzkyGolayFilterTransformer(BaseTransformer):

    def __init__(self, window_length: int, polyorder: int, deriv: int=0, delta: float=1.0, axis: int=-1, mode: str='interp', cval: float=0.0, name: Optional[str]=None):
        """
        Initialize the Savitzky-Golay Filter Transformer.
        
        Parameters
        ----------
        window_length : int
            The length of the filter window (number of coefficients). Must be positive and odd.
        polyorder : int
            The order of the polynomial used to fit the samples. Must be less than window_length.
        deriv : int, default=0
            The order of the derivative to compute. 0 is smoothing only.
        delta : float, default=1.0
            The spacing of the samples to which the filter will be applied.
        axis : int, default=-1
            The axis of the array along which the filter is to be applied.
        mode : str, default='interp'
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.
        cval : float, default=0.0
            Value to fill past the edges of the input if mode is 'constant'.
        name : Optional[str], default=None
            Name of the transformer instance.
            
        Raises
        ------
        ValueError
            If window_length is not positive or not odd, or if polyorder >= window_length.
        """
        super().__init__(name=name)
        if window_length <= 0 or window_length % 2 == 0:
            raise ValueError('window_length must be a positive odd number')
        if polyorder >= window_length:
            raise ValueError('polyorder must be less than window_length')
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.axis = axis
        self.mode = mode
        self.cval = cval

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SavitzkyGolayFilterTransformer':
        """
        Fit the transformer to the input data.
        
        For the Savitzky-Golay filter, fitting is a no-op since the transformation
        does not require learning any parameters from the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. Can be a FeatureSet or numpy array.
        **kwargs : dict
            Additional parameters (ignored for this transformer).
            
        Returns
        -------
        SavitzkyGolayFilterTransformer
            Self instance for method chaining.
        """
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply Savitzky-Golay filtering to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Can be a FeatureSet or numpy array.
        **kwargs : dict
            Additional parameters (ignored for this transformer).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with the same type as input.
            
        Raises
        ------
        ValueError
            If the data array is smaller than the window length in the filter axis.
        """
        from scipy.signal import savgol_filter
        if isinstance(data, FeatureSet):
            features = data.features
        elif isinstance(data, np.ndarray):
            features = data
        else:
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if features.ndim == 0:
            raise ValueError('Cannot apply Savitzky-Golay filter to 0-dimensional arrays')
        if self.axis < -features.ndim or self.axis >= features.ndim:
            raise ValueError(f'axis ({self.axis}) is out of bounds for array of dimension {features.ndim}')
        axis_for_shape_check = self.axis if self.axis >= 0 else features.ndim + self.axis
        if features.shape[axis_for_shape_check] < self.window_length:
            raise ValueError(f'Data array size ({features.shape[axis_for_shape_check]}) along axis {self.axis} is smaller than window_length ({self.window_length})')
        filtered_features = savgol_filter(features, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv, delta=self.delta, axis=self.axis, mode=self.mode, cval=self.cval)
        if isinstance(data, FeatureSet):
            return FeatureSet(features=filtered_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
        else:
            return filtered_features

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply the inverse transformation (not supported for Savitzky-Golay filtering).
        
        Since Savitzky-Golay filtering is a lossy operation (noise is removed),
        the inverse transformation is not mathematically possible.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            The same data passed as input (identity operation).
            
        Warning
        -------
        This operation is a no-op since Savitzky-Golay filtering is not invertible.
        """
        import warnings
        warnings.warn('Savitzky-Golay filtering is not invertible. Returning input unchanged.', UserWarning)
        return data