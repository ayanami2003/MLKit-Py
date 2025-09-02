from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple


# ...(code omitted)...


class SpectralAnalyzer(BaseTransformer):

    def __init__(self, n_points: Optional[int]=None, window_type: str='hamming', overlap_ratio: float=0.5, feature_types: List[str]=['psd', 'peak_freq'], sampling_rate: float=1.0, name: Optional[str]=None):
        """
        Initialize the SpectralAnalyzer.
        
        Args:
            n_points (Optional[int]): Number of points in the FFT. If None, uses the length of input signal.
            window_type (str): Type of windowing function to apply.
            overlap_ratio (float): Overlap ratio for windowed analysis (between 0 and 1).
            feature_types (List[str]): Types of spectral features to compute.
            sampling_rate (float): Sampling rate of the time series data in Hz.
            name (Optional[str]): Name identifier for the transformer instance.
        """
        super().__init__(name=name)
        self.n_points = n_points
        self.window_type = window_type
        self.overlap_ratio = overlap_ratio
        self.feature_types = feature_types
        self.sampling_rate = sampling_rate
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError('overlap_ratio must be between 0 and 1')
        valid_windows = ['hamming', 'hann', 'blackman', 'none']
        if self.window_type not in valid_windows:
            raise ValueError(f'window_type must be one of {valid_windows}')
        valid_features = ['psd', 'peak_freq', 'spectral_moments']
        for ftype in self.feature_types:
            if ftype not in valid_features:
                raise ValueError(f'Unknown feature type: {ftype}. Must be one of {valid_features}')

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SpectralAnalyzer':
        """
        Fit the analyzer to the input data (configures internal parameters based on data properties).
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input time series data for analysis.
            **kwargs: Additional parameters (not used).
            
        Returns:
            SpectralAnalyzer: Returns self for method chaining.
        """
        if not isinstance(data, (FeatureSet, np.ndarray)):
            raise TypeError('Input data must be either a FeatureSet or numpy array')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (self.n_samples_, self.n_features_) = X.shape
        if self.n_points is not None and self.n_points <= 0:
            raise ValueError('n_points must be positive if specified')
        return self

    def _get_window(self, window_size: int) -> np.ndarray:
        """Generate window function for spectral analysis."""
        if self.window_type == 'hamming':
            return np.hamming(window_size)
        elif self.window_type == 'hann':
            return np.hanning(window_size)
        elif self.window_type == 'blackman':
            return np.blackman(window_size)
        else:
            return np.ones(window_size)

    def _compute_psd_features(self, signal: np.ndarray, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density features."""
        n_points = self.n_points if self.n_points is not None else len(signal)
        if len(signal) < len(window):
            padded_signal = np.pad(signal, (0, len(window) - len(signal)), mode='constant')
            windowed_signal = padded_signal * window
        elif len(signal) > len(window):
            windowed_signal = signal[:len(window)] * window
        else:
            windowed_signal = signal * window
        fft_result = np.fft.fft(windowed_signal, n=n_points)
        psd = np.abs(fft_result) ** 2
        freqs = np.fft.fftfreq(n_points, d=1 / self.sampling_rate)
        half_point = n_points // 2
        psd = psd[:half_point]
        freqs = freqs[:half_point]
        return (psd, freqs)

    def _compute_peak_frequency_features(self, signal: np.ndarray, window: np.ndarray) -> Tuple[float, float]:
        """Compute peak frequency features."""
        (psd, freqs) = self._compute_psd_features(signal, window)
        if len(psd) == 0:
            return (0.0, 0.0)
        peak_idx = np.argmax(psd)
        peak_frequency = freqs[peak_idx]
        peak_power = psd[peak_idx]
        return (peak_frequency, peak_power)

    def _compute_spectral_moment_features(self, signal: np.ndarray, window: np.ndarray) -> Tuple[float, float, float]:
        """Compute spectral moment features."""
        (psd, freqs) = self._compute_psd_features(signal, window)
        total_power = np.sum(psd)
        if total_power == 0 or len(psd) == 0:
            return (0.0, 0.0, 0.0)
        prob = psd / total_power
        spectral_centroid = np.sum(freqs * prob)
        spectral_spread = np.sqrt(np.sum((freqs - spectral_centroid) ** 2 * prob)) if np.sum((freqs - spectral_centroid) ** 2 * prob) >= 0 else 0.0
        spectral_skewness = np.sum((freqs - spectral_centroid) ** 3 * prob) / spectral_spread ** 3 if spectral_spread != 0 else 0.0
        return (spectral_centroid, spectral_spread, spectral_skewness)

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Perform spectral analysis on the input time series data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input time series data for analysis.
            **kwargs: Additional parameters (not used).
            
        Returns:
            FeatureSet: FeatureSet containing computed spectral features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            input_feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        else:
            X = data
            input_feature_names = [f'feature_{i}' for i in range(X.shape[1])] if X.ndim > 1 else ['feature_0']
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        (n_samples, n_features) = X.shape
        fft_size = self.n_points if self.n_points is not None else X.shape[0]
        all_features = []
        all_feature_names = []
        for feat_idx in range(n_features):
            signal = X[:, feat_idx]
            if len(signal) <= 1:
                feature_values = []
                feature_names = []
                if 'psd' in self.feature_types:
                    psd_val = np.abs(signal[0]) ** 2 if len(signal) == 1 else 0.0
                    feature_values.append(psd_val)
                    feature_names.append(f'{input_feature_names[feat_idx]}_psd_freq_0.00Hz')
                if 'peak_freq' in self.feature_types:
                    feature_values.extend([0.0, np.abs(signal[0]) ** 2 if len(signal) == 1 else 0.0])
                    feature_names.extend([f'{input_feature_names[feat_idx]}_peak_frequency_Hz', f'{input_feature_names[feat_idx]}_peak_magnitude'])
                if 'spectral_moments' in self.feature_types:
                    feature_values.extend([0.0, 0.0, 0.0, 0.0])
                    feature_names.extend([f'{input_feature_names[feat_idx]}_spectral_centroid_Hz', f'{input_feature_names[feat_idx]}_spectral_spread_Hz2', f'{input_feature_names[feat_idx]}_spectral_skewness', f'{input_feature_names[feat_idx]}_spectral_kurtosis'])
                all_features.extend(feature_values)
                all_feature_names.extend(feature_names)
                continue
            if fft_size > len(signal):
                padded_signal = np.pad(signal, (0, fft_size - len(signal)), mode='constant')
                signal_processed = padded_signal
            elif fft_size < len(signal):
                signal_processed = signal[:fft_size]
            else:
                signal_processed = signal
            if self.window_type != 'none':
                window = self._get_window(len(signal_processed))
                signal_windowed = signal_processed * window
            else:
                signal_windowed = signal_processed
            fft_result = np.fft.fft(signal_windowed, n=fft_size)
            psd = np.abs(fft_result) ** 2
            freqs = np.fft.fftfreq(fft_size, 1 / self.sampling_rate)
            positive_freq_idx = freqs >= 0
            freqs_positive = freqs[positive_freq_idx]
            psd_positive = psd[positive_freq_idx]
            feature_values = []
            feature_names = []
            if 'psd' in self.feature_types:
                feature_values.extend(psd_positive)
                for (i, freq) in enumerate(freqs_positive):
                    feature_names.append(f'{input_feature_names[feat_idx]}_psd_freq_{freq:.2f}Hz')
            if 'peak_freq' in self.feature_types:
                peak_idx = np.argmax(psd_positive)
                peak_freq = freqs_positive[peak_idx]
                peak_magnitude = psd_positive[peak_idx]
                feature_values.extend([peak_freq, peak_magnitude])
                feature_names.extend([f'{input_feature_names[feat_idx]}_peak_frequency_Hz', f'{input_feature_names[feat_idx]}_peak_magnitude'])
            if 'spectral_moments' in self.feature_types:
                total_power = np.sum(psd_positive)
                if total_power > 0:
                    moment1 = np.sum(freqs_positive * psd_positive) / total_power
                    moment2 = np.sum((freqs_positive - moment1) ** 2 * psd_positive) / total_power
                    moment3 = np.sum((freqs_positive - moment1) ** 3 * psd_positive) / (total_power * moment2 ** 1.5) if moment2 > 0 else 0
                    moment4 = np.sum((freqs_positive - moment1) ** 4 * psd_positive) / (total_power * moment2 ** 2) if moment2 > 0 else 0
                else:
                    moment1 = moment2 = moment3 = moment4 = 0
                feature_values.extend([moment1, moment2, moment3, moment4])
                feature_names.extend([f'{input_feature_names[feat_idx]}_spectral_centroid_Hz', f'{input_feature_names[feat_idx]}_spectral_spread_Hz2', f'{input_feature_names[feat_idx]}_spectral_skewness', f'{input_feature_names[feat_idx]}_spectral_kurtosis'])
            all_features.extend(feature_values)
            all_feature_names.extend(feature_names)
        if all_features:
            features_array = np.array(all_features).reshape(1, -1)
        else:
            features_array = np.array([]).reshape(1, 0)
        return FeatureSet(features=features_array, feature_names=all_feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transform is not applicable for spectral analysis.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data.
            **kwargs: Additional parameters (not used).
            
        Returns:
            Union[FeatureSet, np.ndarray]: Returns the input data unchanged.
        """
        return data