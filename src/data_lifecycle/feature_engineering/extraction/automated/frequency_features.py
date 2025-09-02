from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy import signal


class FrequencyFeatureExtractor(BaseTransformer):
    """
    Extracts frequency-domain features from time-series or signal data.
    
    This transformer computes various frequency-based characteristics such as
    spectral energy, dominant frequencies, power spectral density peaks, and
    other relevant metrics that describe the frequency content of signals.
    It is particularly useful for time-series analysis, audio processing,
    and any domain where periodic patterns are of interest.
    
    Attributes:
        sample_rate (float): Sampling rate of the input signals in Hz.
        n_fft (int): Number of points for FFT computation.
        feature_names (List[str]): Names of computed frequency features.
    """

    def __init__(self, sample_rate: float=1.0, n_fft: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the FrequencyFeatureExtractor.
        
        Args:
            sample_rate (float): Sampling rate of the input signals in Hz. Defaults to 1.0.
            n_fft (Optional[int]): Number of points for FFT computation. If None, it will be inferred
                                 from the input data length.
            name (Optional[str]): Name of the transformer instance.
        """
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.feature_names: List[str] = []

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'FrequencyFeatureExtractor':
        """
        Fit the transformer to the input data.
        
        For frequency feature extraction, fitting primarily involves
        determining the appropriate FFT size if not provided and
        preparing the feature name list based on the input dimensions.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input time-series data as a 2D array
                                                of shape (n_samples, n_time_points) or a FeatureSet.
            **kwargs: Additional fitting parameters (not used).
            
        Returns:
            FrequencyFeatureExtractor: Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            features = data.features
        else:
            features = data
        if features.ndim != 2:
            raise ValueError('Input data must be a 2D array with shape (n_samples, n_time_points)')
        if self.n_fft is None:
            signal_length = features.shape[1]
            self.n_fft = 2 ** int(np.ceil(np.log2(signal_length)))
        n_channels = features.shape[0] if features.shape[0] > 0 else 1
        self.feature_names = []
        for i in range(n_channels):
            self.feature_names.extend([f'channel_{i}_spectral_centroid', f'channel_{i}_spectral_bandwidth', f'channel_{i}_spectral_rolloff', f'channel_{i}_peak_frequency', f'channel_{i}_spectral_flux'])
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Extract frequency features from the input time-series data.
        
        Computes various frequency-domain metrics for each sample in the input data.
        Features typically include spectral centroid, bandwidth, roll-off, flux,
        and peak frequencies.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input time-series data as a 2D array
                                                of shape (n_samples, n_time_points) or a FeatureSet.
            **kwargs: Additional transformation parameters (not used).
            
        Returns:
            FeatureSet: A FeatureSet containing the extracted frequency features
                       with shape (n_samples, n_frequency_features).
        """
        if isinstance(data, FeatureSet):
            features = data.features
        else:
            features = data
        if features.ndim != 2:
            raise ValueError('Input data must be a 2D array with shape (n_samples, n_time_points)')
        if not hasattr(self, 'n_fft') or self.n_fft is None:
            raise ValueError('Transformer must be fitted before transform. Call fit() first.')
        (n_samples, n_time_points) = features.shape
        n_features_per_channel = 5
        n_channels = n_samples
        n_total_features = n_channels * n_features_per_channel
        extracted_features = np.zeros((n_samples, n_total_features))
        for i in range(n_samples):
            sample = features[i, :]
            if len(sample) < self.n_fft:
                padded_sample = np.pad(sample, (0, self.n_fft - len(sample)), mode='constant')
            else:
                padded_sample = sample[:self.n_fft]
            fft_result = np.fft.rfft(padded_sample)
            magnitude_spectrum = np.abs(fft_result)
            power_spectrum = magnitude_spectrum ** 2
            freqs = np.fft.rfftfreq(self.n_fft, d=1 / self.sample_rate)
            if np.sum(power_spectrum) > 0:
                spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
            else:
                spectral_centroid = 0.0
            if np.sum(power_spectrum) > 0:
                spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid) ** 2 * power_spectrum) / np.sum(power_spectrum))
            else:
                spectral_bandwidth = 0.0
            cumulative_energy = np.cumsum(power_spectrum)
            if np.sum(power_spectrum) > 0:
                rolloff_threshold = 0.85 * np.sum(power_spectrum)
                rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            else:
                spectral_rolloff = 0.0
            peak_idx = np.argmax(power_spectrum)
            peak_frequency = freqs[peak_idx]
            if i > 0:
                prev_sample = features[i - 1, :]
                if len(prev_sample) < self.n_fft:
                    padded_prev = np.pad(prev_sample, (0, self.n_fft - len(prev_sample)), mode='constant')
                else:
                    padded_prev = prev_sample[:self.n_fft]
                prev_fft = np.fft.rfft(padded_prev)
                prev_power_spectrum = np.abs(prev_fft) ** 2
                spectral_flux = np.sum((power_spectrum - prev_power_spectrum) ** 2)
            else:
                spectral_flux = 0.0
            start_idx = i * n_features_per_channel
            extracted_features[i, start_idx:start_idx + n_features_per_channel] = [spectral_centroid, spectral_bandwidth, spectral_rolloff, peak_frequency, spectral_flux]
        return FeatureSet(features=extracted_features, feature_names=self.feature_names, feature_types=['numeric'] * len(self.feature_names), metadata={'transformer_name': self.name} if self.name else {})

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for frequency feature extraction.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Transformed data (ignored).
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: The input data unchanged.
            
        Raises:
            NotImplementedError: Always raised as inverse transformation is not meaningful
                               for frequency feature extraction.
        """
        raise NotImplementedError('Inverse transformation is not supported for frequency feature extraction.')

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the extracted frequency features.
        
        Returns:
            List[str]: List of feature names corresponding to the computed frequency metrics.
        """
        return self.feature_names.copy()