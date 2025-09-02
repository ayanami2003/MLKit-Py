from typing import Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from scipy.stats import norm
import copy

class NonLinearFeatureTransformer(BaseTransformer):
    """
    A transformer that applies various nonlinear mathematical transformations to features.
    
    This class supports multiple nonlinear transformations including Gaussian transformation
    and inverse hyperbolic sine transformation. It can be used to transform features to 
    achieve normality, stabilize variance, or improve model performance.
    
    Supported transformations:
    - Gaussian transformation (quantile transformation to normal distribution)
    - Inverse hyperbolic sine transformation
    - Other customizable nonlinear transformations
    
    Attributes
    ----------
    method : str
        The transformation method to apply ('gaussian', 'inverse_hyperbolic_sine', or 'custom')
    epsilon : float, default=1e-8
        Small constant added to avoid division by zero or log of zero
    output_distribution : str, default='normal'
        Desired distribution of transformed features when method='gaussian'
    """

    def __init__(self, method: str='gaussian', epsilon: float=1e-08, output_distribution: str='normal', name: Optional[str]=None):
        """
        Initialize the NonLinearFeatureTransformer.
        
        Parameters
        ----------
        method : str, default='gaussian'
            The transformation method to apply. Options are:
            - 'gaussian': Quantile transformation to normal distribution
            - 'inverse_hyperbolic_sine': Inverse hyperbolic sine transformation
            - 'custom': Placeholder for custom transformation
        epsilon : float, default=1e-8
            Small constant added to avoid numerical issues
        output_distribution : str, default='normal'
            Target distribution when using 'gaussian' method. Currently only 'normal' is supported.
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.epsilon = epsilon
        self.output_distribution = output_distribution
        valid_methods = ['gaussian', 'inverse_hyperbolic_sine', 'custom']
        if self.method not in valid_methods:
            raise ValueError(f"Method '{self.method}' is not supported. Valid options are: {valid_methods}")
        if self.output_distribution != 'normal':
            raise ValueError(f"Output distribution '{self.output_distribution}' is not supported. Only 'normal' is currently supported.")

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'NonLinearFeatureTransformer':
        """
        Fit the transformer to the input data.
        
        For Gaussian transformation, this computes the empirical CDF of each feature.
        For inverse hyperbolic sine transformation, no fitting is required.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to fit the transformer on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters for fitting (not used in current implementation)
            
        Returns
        -------
        NonLinearFeatureTransformer
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet with numpy array features')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_ = X.shape[1]
        if self.method == 'gaussian':
            self._empirical_cdfs = []
            self.quantiles_ = []
            for i in range(self.n_features_):
                feature_values = np.sort(X[:, i])
                n = len(feature_values)
                empirical_cdf = np.arange(1, n + 1) / (n + 1)
                self._empirical_cdfs.append((feature_values, empirical_cdf))
                self.quantiles_.append(np.column_stack([feature_values, empirical_cdf]))
        elif self.method == 'custom':
            pass
        return self

    def _get_features_from_data(self, data: Union[FeatureSet, np.ndarray]) -> np.ndarray:
        """Extract features array from input data."""
        if isinstance(data, FeatureSet):
            return data.features
        else:
            return data

    def _create_output_data(self, original_data: Union[FeatureSet, np.ndarray], transformed_features: np.ndarray) -> Union[FeatureSet, np.ndarray]:
        """Create output data in the same format as input."""
        if isinstance(original_data, FeatureSet):
            result = copy.deepcopy(original_data)
            result.features = transformed_features
            return result
        else:
            return transformed_features

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply the nonlinear transformation to input data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data to transform. If FeatureSet, transforms the features attribute.
        **kwargs : dict
            Additional parameters for transformation (not used in current implementation)
            
        Returns
        -------
        FeatureSet or np.ndarray
            Transformed data in the same format as input
        """
        if self.method == 'gaussian' and (not hasattr(self, '_empirical_cdfs')):
            raise ValueError("This transformer has not been fitted yet. Call 'fit' before using this method.")
        X = self._get_features_from_data(data)
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet with numpy array features')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if not hasattr(self, 'n_features_'):
            self.n_features_ = X.shape[1]
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features but got {X.shape[1]}')
        if self.method == 'gaussian':
            X_transformed = np.empty_like(X)
            for i in range(self.n_features_):
                (feature_values, empirical_cdf) = self._empirical_cdfs[i]
                cdf_values = np.interp(X[:, i], feature_values, empirical_cdf, left=self.epsilon, right=1 - self.epsilon)
                X_transformed[:, i] = norm.ppf(cdf_values)
        elif self.method == 'inverse_hyperbolic_sine':
            X_transformed = np.arcsinh(X)
        elif self.method == 'custom':
            X_transformed = X.copy()
        return self._create_output_data(data, X_transformed)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply the inverse transformation if possible.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Transformed data to invert
        **kwargs : dict
            Additional parameters for inverse transformation (not used in current implementation)
            
        Returns
        -------
        FeatureSet or np.ndarray
            Original data format
        """
        if self.method == 'gaussian' and (not hasattr(self, '_empirical_cdfs')):
            raise ValueError("This transformer has not been fitted yet. Call 'fit' before using this method.")
        X = self._get_features_from_data(data)
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet with numpy array features')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if not hasattr(self, 'n_features_'):
            self.n_features_ = X.shape[1]
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features but got {X.shape[1]}')
        if self.method == 'gaussian':
            X_original = np.empty_like(X)
            for i in range(self.n_features_):
                cdf_values = norm.cdf(X[:, i])
                (feature_values, empirical_cdf) = self._empirical_cdfs[i]
                X_original[:, i] = np.interp(cdf_values, empirical_cdf, feature_values, left=feature_values[0], right=feature_values[-1])
        elif self.method == 'inverse_hyperbolic_sine':
            X_original = np.sinh(X)
        elif self.method == 'custom':
            X_original = X.copy()
        return self._create_output_data(data, X_original)