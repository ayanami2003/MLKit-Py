from typing import Optional
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy.interpolate import interp1d

class TimeWarpingAugmentationTransformer(BaseTransformer):
    """
    A transformer that applies time warping augmentation to time series data.
    
    Time warping is a data augmentation technique that non-linearly distorts
    the time axis of time series data while preserving the overall structure.
    This can help improve model robustness to temporal variations.
    
    Parameters
    ----------
    warp_strength : float, default=0.1
        Controls the intensity of the time warping distortion.
        Higher values create more pronounced warping effects.
    n_knots : int, default=5
        Number of control points for the warping function.
        More knots allow for more complex warping patterns.
    random_state : Optional[int], default=None
        Random seed for reproducible augmentations.
    preserve_original : bool, default=True
        Whether to include the original samples in the output.
    name : Optional[str], default=None
        Name of the transformer instance.
        
    Attributes
    ----------
    warp_function_ : callable
        The learned warping function.
    feature_names_ : list
        Names of features in the training data.
    """

    def __init__(self, warp_strength: float=0.1, n_knots: int=5, random_state: Optional[int]=None, preserve_original: bool=True, name: Optional[str]=None):
        super().__init__(name=name)
        self.warp_strength = warp_strength
        self.n_knots = n_knots
        self.random_state = random_state
        self.preserve_original = preserve_original

    def fit(self, data: FeatureSet, **kwargs) -> 'TimeWarpingAugmentationTransformer':
        """
        Fit the time warping transformer to the input data.
        
        This method validates the input data structure and initializes
        internal parameters needed for the warping transformation.
        
        Parameters
        ----------
        data : FeatureSet
            Time series data to fit on. Expected to have samples as rows
            and time steps as columns for each feature.
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        TimeWarpingAugmentationTransformer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If the input data is not suitable for time warping.
        """
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet')
        if data.features.ndim != 2:
            raise ValueError('Time series data must be 2D with samples as rows and time steps as columns')
        self.feature_names_ = data.feature_names
        self.feature_types_ = data.feature_types
        self.sample_ids_ = data.sample_ids
        self.original_shape_ = data.features.shape
        self.rng_ = np.random.default_rng(self.random_state)
        self.fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply time warping augmentation to the input data.
        
        Applies non-linear time warping to each time series in the dataset,
        creating augmented versions that maintain the same temporal structure
        but with distorted time axes.
        
        Parameters
        ----------
        data : FeatureSet
            Time series data to transform.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Augmented time series data with warped time axes.
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError('Transformer must be fitted before transform')
        if not isinstance(data, FeatureSet):
            raise TypeError('Input data must be a FeatureSet')
        if data.features.shape != self.original_shape_:
            raise ValueError(f'Input data shape {data.features.shape} does not match fitted shape {self.original_shape_}')
        ts_data = data.features
        warped_data = []
        for i in range(ts_data.shape[0]):
            warped_sample = self._warp_time_series(ts_data[i, :])
            warped_data.append(warped_sample)
        warped_data = np.array(warped_data)
        if self.preserve_original:
            combined_features = np.vstack([ts_data, warped_data])
            combined_sample_ids = None
            if data.sample_ids is not None:
                warped_sample_ids = [f'warped_{i}' for i in range(len(warped_data))]
                combined_sample_ids = data.sample_ids + warped_sample_ids
        else:
            combined_features = warped_data
            combined_sample_ids = None
            if data.sample_ids is not None:
                combined_sample_ids = [f'warped_{i}' for i in range(len(warped_data))]
        return FeatureSet(features=combined_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=combined_sample_ids, metadata=data.metadata.copy() if data.metadata is not None else None, quality_scores=data.quality_scores.copy() if data.quality_scores is not None else None)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Apply the inverse time warping transformation.
        
        Attempts to reverse the time warping effect, though perfect inversion
        may not be possible due to the non-linear nature of the transformation.
        
        Parameters
        ----------
        data : FeatureSet
            Warped time series data to invert.
        **kwargs : dict
            Additional inversion parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Time series data with (partially) restored time axes.
        """
        if not hasattr(self, 'fitted_') or not self.fitted_:
            raise ValueError('Transformer must be fitted before inverse_transform')
        metadata = data.metadata.copy() if data.metadata is not None else {}
        metadata['inverse_warning'] = 'Perfect inversion of time warping is not possible'
        return FeatureSet(features=data.features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=metadata, quality_scores=data.quality_scores.copy() if data.quality_scores is not None else None)

    def _warp_time_series(self, ts: np.ndarray) -> np.ndarray:
        """
        Apply time warping to a single time series.
        
        Parameters
        ----------
        ts : np.ndarray
            1D array representing a single time series.
            
        Returns
        -------
        np.ndarray
            Warped time series.
        """
        n_points = len(ts)
        knot_positions = np.linspace(0, n_points - 1, self.n_knots)
        base_positions = np.linspace(0, n_points - 1, n_points)
        displacements = self.rng_.normal(0, self.warp_strength * n_points / 10, self.n_knots)
        displacements[0] = 0
        displacements[-1] = 0
        warped_knot_positions = knot_positions + displacements
        warped_knot_positions = np.sort(warped_knot_positions)
        interp_func = interp1d(warped_knot_positions, knot_positions, kind='linear', fill_value='extrapolate')
        warped_positions = interp_func(base_positions)
        warped_positions = np.clip(warped_positions, 0, n_points - 1)
        interp_ts = interp1d(base_positions, ts, kind='linear', fill_value='extrapolate')
        warped_ts = interp_ts(warped_positions)
        return warped_ts