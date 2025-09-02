from general.structures.feature_set import FeatureSet
from general.base_classes.transformer_base import BaseTransformer
import numpy as np
from typing import Optional, List, Union

class FourierCoefficientExtractor(BaseTransformer):

    def __init__(self, n_coefficients: Optional[int]=10, include_magnitude: bool=True, include_phase: bool=False, name: Optional[str]=None):
        """
        Initialize the Fourier coefficient extractor.
        
        Parameters
        ----------
        n_coefficients : int, optional
            Number of coefficients to retain from the Fourier transform.
            If None, all coefficients are kept. Default is 10.
        include_magnitude : bool
            If True, includes magnitude of Fourier coefficients as features.
            Default is True.
        include_phase : bool
            If True, includes phase angles of Fourier coefficients as features.
            Default is False.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_coefficients = n_coefficients
        self.include_magnitude = include_magnitude
        self.include_phase = include_phase
        self.feature_names_: List[str] = []

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'FourierCoefficientExtractor':
        """
        Fit the transformer to the input data.
        
        This method validates the input data and prepares the transformer
        for coefficient extraction. For Fourier transforms, fitting typically
        involves determining the dimensions of the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing time-series or signal information.
            Expected shape is (n_samples, n_timepoints) for 1D signals or
            (n_samples, n_timepoints, n_features) for multi-dimensional signals.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        FourierCoefficientExtractor
            The fitted transformer instance.
            
        Raises
        ------
        ValueError
            If input data has invalid dimensions or types.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if X.ndim not in [2, 3]:
            raise ValueError('Input data must be 2D (n_samples, n_timepoints) or 3D (n_samples, n_timepoints, n_features)')
        self.n_samples_ = X.shape[0]
        self.n_timepoints_ = X.shape[1]
        self.is_fitted_3d_ = X.ndim == 3
        if X.ndim == 3:
            self.n_features_in_ = X.shape[2]
        else:
            self.n_features_in_ = 1
        max_coeffs = self.n_timepoints_ // 2 + 1
        if self.n_coefficients is None:
            self.n_coefficients_ = max_coeffs
        else:
            self.n_coefficients_ = min(self.n_coefficients, max_coeffs)
        self.feature_names_ = self.get_feature_names(getattr(data, 'feature_names', None) if isinstance(data, FeatureSet) else None)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Extract Fourier coefficients from the input data.
        
        Computes the discrete Fourier transform of each sample and extracts
        the specified number of coefficients as new features. Magnitude and/or
        phase components are included based on configuration.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. Should have the same dimensionality
            as the data used for fitting.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        FeatureSet
            A FeatureSet containing the extracted Fourier coefficients
            with appropriate metadata.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or input data is incompatible.
        """
        if not hasattr(self, 'n_samples_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if X.shape[0] != self.n_samples_ or X.shape[1] != self.n_timepoints_:
            raise ValueError(f'Input data shape {X.shape} does not match fitted data shape ({self.n_samples_}, {self.n_timepoints_})')
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        (n_samples, n_timepoints, n_features) = X.shape
        features_list = []
        for i in range(n_features):
            fft_result = np.fft.rfft(X[:, :, i], axis=1)
            truncated_fft = fft_result[:, :self.n_coefficients_]
            feature_components = []
            if self.include_magnitude:
                magnitudes = np.abs(truncated_fft)
                feature_components.append(magnitudes)
            if self.include_phase:
                phases = np.angle(truncated_fft)
                feature_components.append(phases)
            if not feature_components:
                raise ValueError('At least one of include_magnitude or include_phase must be True')
            combined_features = np.concatenate(feature_components, axis=1)
            features_list.append(combined_features)
        if features_list:
            final_features = np.concatenate(features_list, axis=1)
        else:
            final_features = np.empty((n_samples, 0))
        return FeatureSet(features=final_features, feature_names=self.feature_names_, feature_types=['numeric'] * final_features.shape[1] if final_features.shape[1] > 0 else [], metadata={'transformer_name': self.name} if self.name else {})

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reconstruct signals from Fourier coefficients (not fully invertible).
        
        Attempts to reconstruct the original signals from the stored Fourier
        coefficients. Note that reconstruction may not be perfect if coefficients
        were truncated during extraction.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Fourier coefficients to reconstruct signals from.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        FeatureSet
            Reconstructed signals as a FeatureSet.
            
        Raises
        ------
        NotImplementedError
            If attempting to perform inverse transform when phase information
            was not preserved.
        """
        if not self.include_phase:
            raise NotImplementedError('Inverse transform requires phase information which was not preserved during forward transform.')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        coeffs_per_feature = self.n_coefficients_
        if self.include_magnitude and self.include_phase:
            coeffs_per_feature *= 2
        n_features = self.n_features_in_
        n_samples = X.shape[0]
        reconstructed_signals = []
        for i in range(n_features):
            if self.include_magnitude and self.include_phase:
                mag_start = i * 2 * self.n_coefficients_
                mag_end = mag_start + self.n_coefficients_
                phase_start = mag_end
                phase_end = phase_start + self.n_coefficients_
                magnitudes = X[:, mag_start:mag_end]
                phases = X[:, phase_start:phase_end]
                complex_coeffs = magnitudes * np.exp(1j * phases)
            elif self.include_magnitude:
                mag_start = i * self.n_coefficients_
                mag_end = mag_start + self.n_coefficients_
                magnitudes = X[:, mag_start:mag_end]
                complex_coeffs = magnitudes.astype(complex)
            else:
                phase_start = i * self.n_coefficients_
                phase_end = phase_start + self.n_coefficients_
                phases = X[:, phase_start:phase_end]
                complex_coeffs = np.exp(1j * phases)
            padded_coeffs = np.zeros((n_samples, self.n_timepoints_ // 2 + 1), dtype=complex)
            padded_coeffs[:, :self.n_coefficients_] = complex_coeffs
            reconstructed = np.fft.irfft(padded_coeffs, n=self.n_timepoints_, axis=1)
            reconstructed_signals.append(reconstructed)
        if len(reconstructed_signals) > 1:
            final_signal = np.stack(reconstructed_signals, axis=-1)
        else:
            final_signal = reconstructed_signals[0]
        return FeatureSet(features=final_signal, feature_names=[f'reconstructed_feature_{i}' for i in range(final_signal.shape[-1])], feature_types=['numeric'] * final_signal.shape[-1], metadata={'transformer_name': f'inverse_{self.name}'} if self.name else {})

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names for the extracted Fourier coefficient features.
        
        Generates descriptive names for the output features based on whether
        magnitude and/or phase components are included.
        
        Parameters
        ----------
        input_features : List[str], optional
            Names of input features (used as base names for output features).
            
        Returns
        -------
        List[str]
            Names for the extracted Fourier coefficient features.
        """
        feature_names = []
        if self.include_magnitude:
            for i in range(self.n_coefficients_):
                feature_names.append(f'coef_mag_{i}')
        if self.include_phase:
            for i in range(self.n_coefficients_):
                feature_names.append(f'coef_phase_{i}')
        return feature_names