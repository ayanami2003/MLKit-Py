from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, List, Dict, Any, Union
from scipy.stats import entropy, linregress
from scipy.signal import periodogram

class TimeSeriesFeatureExtractor(BaseTransformer):

    def __init__(self, feature_types: Optional[List[str]]=None, window_size: Optional[int]=None, sampling_rate: Optional[float]=None, config: Optional[Dict[str, Any]]=None, name: Optional[str]=None):
        """
        Initialize the TimeSeriesFeatureExtractor.

        Args:
            feature_types: List of feature types to compute. Defaults to common time-series features.
            window_size: Size of the sliding window for local feature extraction.
            sampling_rate: Rate at which the time-series data was sampled (important for frequency features).
            config: Additional configuration parameters for specific feature extractors.
            name: Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.feature_types = feature_types or ['trend', 'seasonality', 'autocorrelation', 'entropy']
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.config = config or {}

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'TimeSeriesFeatureExtractor':
        """
        Fit the extractor to the time-series data (e.g., learn parameters for normalization).

        Args:
            data: Input time-series data as a FeatureSet or numpy array.
                  For FeatureSet, assumes rows are samples and columns are time steps or series.
            **kwargs: Additional parameters for fitting.

        Returns:
            TimeSeriesFeatureExtractor: The fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            ts_data = data.features
        elif isinstance(data, np.ndarray):
            ts_data = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if ts_data.ndim == 1:
            ts_data = ts_data.reshape(1, -1)
        elif ts_data.ndim != 2:
            raise ValueError('Time-series data must be a 1D or 2D array with shape (n_samples, n_time_steps) or (n_time_steps,)')
        (self.n_samples_, self.n_timesteps_) = ts_data.shape
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Extract features from the time-series data.

        Args:
            data: Input time-series data to transform.
            **kwargs: Additional transformation parameters.

        Returns:
            FeatureSet: A feature set containing extracted time-series features.
        """
        if isinstance(data, FeatureSet):
            ts_data = data.features
        elif isinstance(data, np.ndarray):
            ts_data = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if ts_data.ndim == 1:
            ts_data = ts_data.reshape(1, -1)
        elif ts_data.ndim != 2:
            raise ValueError('Time-series data must be a 1D or 2D array with shape (n_samples, n_time_steps) or (n_time_steps,)')
        (n_samples, n_timesteps) = ts_data.shape
        if n_timesteps == 1:
            features_list = []
            for i in range(n_samples):
                sample_features = self._extract_features_from_window(ts_data[i, :])
                features_list.append(sample_features)
            features = np.array(features_list)
            feature_names = self.feature_types.copy()
        elif self.window_size and self.window_size > 1:
            n_windows = max(0, n_timesteps - self.window_size + 1)
            if n_windows == 0:
                features_list = []
                for i in range(n_samples):
                    sample_features = [0.0] * len(self.feature_types)
                    features_list.append(sample_features)
                features = np.array(features_list)
                feature_names = self.feature_types.copy()
            else:
                features_list = []
                for i in range(n_samples):
                    window_features_list = []
                    for w in range(n_windows):
                        window_data = ts_data[i, w:w + self.window_size]
                        window_features = self._extract_features_from_window(window_data)
                        window_features_list.append(window_features)
                    if window_features_list:
                        aggregated_features = np.mean(window_features_list, axis=0)
                        features_list.append(aggregated_features)
                    else:
                        features_list.append([0.0] * len(self.feature_types))
                features = np.array(features_list)
                feature_names = self.feature_types.copy()
        else:
            features_list = []
            for i in range(n_samples):
                sample_features = self._extract_features_from_window(ts_data[i, :])
                features_list.append(sample_features)
            features = np.array(features_list)
            feature_names = self.feature_types.copy()
        return FeatureSet(features=features, feature_names=feature_names, feature_types=['numeric'] * len(feature_names) if feature_names else [], metadata={'transformer_name': self.name} if self.name else {})

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for feature extraction.

        Args:
            data: Transformed feature data (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            FeatureSet: An empty FeatureSet as inversion is not applicable.

        Raises:
            NotImplementedError: Always raised as inversion is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for feature extraction.')

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of the extracted features.

        Returns:
            Optional[List[str]]: List of feature names generated by the extractor.
        """
        return self.feature_types.copy()

    def _extract_features_from_window(self, window_data: np.ndarray) -> List[float]:
        """
        Extract all configured features from a single window of data.
        
        Args:
            window_data: 1D array of time-series values
            
        Returns:
            List of feature values
        """
        features = []
        for feature_type in self.feature_types:
            if feature_type == 'trend':
                if len(window_data) < 2:
                    features.append(0.0)
                else:
                    (slope, _, _, _, _) = linregress(np.arange(len(window_data)), window_data)
                    features.append(slope)
            elif feature_type == 'seasonality':
                if len(window_data) < 2 or self.sampling_rate is None:
                    features.append(0.0)
                else:
                    (freqs, psd) = periodogram(window_data, fs=self.sampling_rate)
                    if len(psd) > 1:
                        dominant_freq_idx = np.argmax(psd[1:]) + 1
                        features.append(freqs[dominant_freq_idx])
                    else:
                        features.append(0.0)
            elif feature_type == 'autocorrelation':
                if len(window_data) < 2:
                    features.append(0.0)
                else:
                    mean_val = np.mean(window_data)
                    numerator = np.sum((window_data[:-1] - mean_val) * (window_data[1:] - mean_val))
                    denominator = np.sum((window_data - mean_val) ** 2)
                    if denominator == 0:
                        features.append(0.0)
                    else:
                        features.append(numerator / denominator)
            elif feature_type == 'entropy':
                if len(window_data) < 2:
                    features.append(0.0)
                else:
                    normalized = window_data - np.min(window_data)
                    if np.sum(normalized) == 0:
                        features.append(0.0)
                    else:
                        p = normalized / np.sum(normalized)
                        features.append(entropy(p))
            else:
                features.append(0.0)
        return features