from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Optional, List, Dict, Any, Union

class DomainSpecificFeatureExtractor(BaseTransformer):

    def __init__(self, domain_type: str, config: Optional[Dict[str, Any]]=None, name: Optional[str]=None):
        """
        Initialize the domain-specific feature extractor.
        
        Parameters
        ----------
        domain_type : str
            Type of domain to apply feature extraction for
        config : Optional[Dict[str, Any]]
            Domain-specific configuration parameters
        name : Optional[str]
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.domain_type = domain_type
        self.config = config or {}
        self.feature_names_out_ = None
        self._validate_domain_type()

    def _validate_domain_type(self) -> None:
        """Validate that the domain type is supported."""
        supported_domains = ['financial', 'medical', 'geospatial']
        if self.domain_type not in supported_domains:
            raise ValueError(f"Unsupported domain type '{self.domain_type}'. Supported domains: {supported_domains}")

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'DomainSpecificFeatureExtractor':
        """
        Fit the extractor to the input data.
        
        This method analyzes the input data to determine how domain-specific features
        should be extracted and prepares the transformer for transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the extractor on
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        DomainSpecificFeatureExtractor
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            input_data = data.features
            input_feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            input_data = data
            input_feature_names = None
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if input_data.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.feature_names_out_ = self._generate_feature_names(input_data, input_feature_names)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Extract domain-specific features from the input data.
        
        Applies domain-specific transformations and feature generation techniques
        to create meaningful features based on the configured domain type.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to extract features from
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Extracted domain-specific features
        """
        if self.feature_names_out_ is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            input_data = data.features
        elif isinstance(data, np.ndarray):
            input_data = data
        else:
            raise TypeError('Input data must be a FeatureSet or numpy array')
        if input_data.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        extracted_features = self._extract_domain_features(input_data)
        return FeatureSet(features=extracted_features, feature_names=self.feature_names_out_, feature_types=['numeric'] * len(self.feature_names_out_), metadata={'domain_type': self.domain_type, 'transformer_name': self.name} if self.name else {'domain_type': self.domain_type})

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation if possible.
        
        Note: Inverse transformation may not be supported for all domain types.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original data format if inversion is possible
        """
        raise NotImplementedError('Inverse transformation is not supported for domain-specific feature extraction.')

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get names of extracted features.
        
        Returns
        -------
        Optional[List[str]]
            Names of extracted features or None if not yet fitted
        """
        return self.feature_names_out_

    def _generate_feature_names(self, data: np.ndarray, input_feature_names: Optional[List[str]]) -> List[str]:
        """
        Generate feature names based on domain type and input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        input_feature_names : Optional[List[str]]
            Names of input features
            
        Returns
        -------
        List[str]
            Generated feature names
        """
        n_features = data.shape[1]
        if self.domain_type == 'financial':
            return self._generate_financial_feature_names(n_features, input_feature_names)
        elif self.domain_type == 'medical':
            return self._generate_medical_feature_names(n_features, input_feature_names)
        elif self.domain_type == 'geospatial':
            return self._generate_geospatial_feature_names(n_features, input_feature_names)
        else:
            return [f'{self.domain_type}_feature_{i}' for i in range(n_features)]

    def _extract_domain_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features based on the domain type.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
            
        Returns
        -------
        np.ndarray
            Extracted features
        """
        if self.domain_type == 'financial':
            return self._extract_financial_features(data)
        elif self.domain_type == 'medical':
            return self._extract_medical_features(data)
        elif self.domain_type == 'geospatial':
            return self._extract_geospatial_features(data)
        else:
            return data

    def _generate_financial_feature_names(self, n_features: int, input_feature_names: Optional[List[str]]) -> List[str]:
        """Generate feature names for financial domain."""
        config_features = self.config.get('features', [])
        if config_features:
            return config_features
        base_names = ['mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 'return_mean', 'return_std', 'sharpe_ratio', 'volatility']
        return base_names[:min(len(base_names), n_features)] + [f'financial_feature_{i}' for i in range(max(0, n_features - len(base_names)))]

    def _generate_medical_feature_names(self, n_features: int, input_feature_names: Optional[List[str]]) -> List[str]:
        """Generate feature names for medical domain."""
        config_features = self.config.get('features', [])
        if config_features:
            return config_features
        base_names = ['mean', 'std', 'min', 'max', 'median', 'range', 'heart_rate_variability', 'systolic_bp', 'diastolic_bp', 'bmi']
        return base_names[:min(len(base_names), n_features)] + [f'medical_feature_{i}' for i in range(max(0, n_features - len(base_names)))]

    def _generate_geospatial_feature_names(self, n_features: int, input_feature_names: Optional[List[str]]) -> List[str]:
        """Generate feature names for geospatial domain."""
        config_features = self.config.get('features', [])
        if config_features:
            return config_features
        base_names = ['latitude_mean', 'longitude_mean', 'elevation_mean', 'distance_from_center', 'area', 'perimeter', 'compactness']
        return base_names[:min(len(base_names), n_features)] + [f'geospatial_feature_{i}' for i in range(max(0, n_features - len(base_names)))]

    def _extract_financial_features(self, data: np.ndarray) -> np.ndarray:
        """Extract financial features from data."""
        (n_samples, n_features) = data.shape
        extracted = []
        expected_features = len(self.feature_names_out_) if self.feature_names_out_ is not None else 10
        for i in range(n_samples):
            sample = data[i, :]
            features = []
            features.append(np.mean(sample))
            features.append(np.std(sample))
            features.append(np.min(sample))
            features.append(np.max(sample))
            if len(sample) > 1:
                features.append(self._skewness(sample))
                features.append(self._kurtosis(sample))
            else:
                features.extend([0.0, 0.0])
            if len(sample) > 1:
                returns = np.diff(sample) / sample[:-1]
                features.append(np.mean(returns))
                features.append(np.std(returns))
                if np.std(returns) > 0:
                    features.append(np.mean(returns) / np.std(returns))
                else:
                    features.append(0.0)
                features.append(np.std(returns))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            if len(features) > expected_features:
                features = features[:expected_features]
            elif len(features) < expected_features:
                features.extend([0.0] * (expected_features - len(features)))
            extracted.append(features)
        return np.array(extracted)

    def _extract_medical_features(self, data: np.ndarray) -> np.ndarray:
        """Extract medical features from data."""
        (n_samples, n_features) = data.shape
        extracted = []
        for i in range(n_samples):
            sample = data[i, :]
            features = []
            features.append(np.mean(sample))
            features.append(np.std(sample))
            features.append(np.min(sample))
            features.append(np.max(sample))
            features.append(np.median(sample))
            features.append(np.max(sample) - np.min(sample))
            if len(sample) > 1:
                features.append(np.std(np.diff(sample)))
            else:
                features.append(0.0)
            features.append(np.max(sample))
            features.append(np.min(sample))
            mean_val = np.mean(sample)
            if mean_val > 0:
                features.append(mean_val)
            else:
                features.append(0.0)
            extracted.append(features)
        result = np.array(extracted)
        if self.feature_names_out_ is not None:
            expected_features = len(self.feature_names_out_)
            if result.shape[1] > expected_features:
                result = result[:, :expected_features]
            elif result.shape[1] < expected_features:
                padding = np.zeros((result.shape[0], expected_features - result.shape[1]))
                result = np.hstack([result, padding])
        return result

    def _extract_geospatial_features(self, data: np.ndarray) -> np.ndarray:
        """Extract geospatial features from data."""
        (n_samples, n_features) = data.shape
        extracted = []
        coord_dim = min(n_features, 4)
        for i in range(n_samples):
            sample = data[i, :]
            features = []
            if coord_dim >= 2:
                (lat1, lon1) = (sample[0], sample[1])
                features.append(lat1)
                features.append(lon1)
                if coord_dim >= 4:
                    (lat2, lon2) = (sample[2], sample[3])
                    distance = np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
                    features.append(distance)
                else:
                    features.append(0.0)
            else:
                features.append(np.mean(sample))
                features.append(np.std(sample))
                features.append(0.0)
            if coord_dim >= 3:
                features.append(sample[2])
            else:
                features.append(np.mean(sample))
            features.append(np.std(sample) * len(sample))
            features.append(np.sum(np.abs(np.diff(sample))))
            area = features[-2]
            perimeter = features[-1]
            if area > 0:
                features.append(perimeter ** 2 / area)
            else:
                features.append(0.0)
            extracted.append(features)
        result = np.array(extracted)
        if self.feature_names_out_ is not None:
            expected_features = len(self.feature_names_out_)
            if result.shape[1] > expected_features:
                result = result[:, :expected_features]
            elif result.shape[1] < expected_features:
                padding = np.zeros((result.shape[0], expected_features - result.shape[1]))
                result = np.hstack([result, padding])
        return result

    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        skew = np.mean(((data - mean) / std) ** 3)
        skew = skew * (n / ((n - 1) * (n - 2)))
        return skew

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=0)
        if std == 0:
            return 0.0
        moment4 = np.mean(((data - mean) / std) ** 4)
        return moment4