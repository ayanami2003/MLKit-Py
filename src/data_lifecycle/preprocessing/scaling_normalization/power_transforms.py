import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import Union, Optional, Tuple
from scipy import stats

class PowerTransformer(BaseTransformer):

    def __init__(self, exponent: float=1.0, name: Optional[str]=None):
        """
        Initialize the PowerTransformer.
        
        Args:
            exponent (float): The exponent to use for the power transformation. Defaults to 1.0.
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.exponent = exponent
        self.inverse_exponent = None
        self._fitted = False

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'PowerTransformer':
        """
        Fit method for the transformer (computes inverse exponent for inversion).
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            PowerTransformer: Returns self for method chaining.
        """
        if self.exponent == 0:
            self.inverse_exponent = float('inf')
        else:
            self.inverse_exponent = 1.0 / self.exponent
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply power transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Transformed feature set with power transformation applied.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        with np.errstate(invalid='ignore'):
            X_transformed = np.sign(X) * np.abs(X) ** self.exponent
        metadata['power_transform'] = {'exponent': self.exponent, 'method': 'power_function'}
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Revert the power transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Power-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Data in original scale.
        """
        if not self._fitted:
            if self.exponent == 0:
                self.inverse_exponent = float('inf')
            else:
                self.inverse_exponent = 1.0 / self.exponent
            self._fitted = True
        if self.exponent == 0:
            if isinstance(data, FeatureSet):
                X = data.features
                feature_names = data.feature_names
                feature_types = data.feature_types
                sample_ids = data.sample_ids
                metadata = data.metadata.copy() if data.metadata else {}
                quality_scores = data.quality_scores.copy() if data.quality_scores else {}
            elif isinstance(data, np.ndarray):
                X = data
                feature_names = None
                feature_types = None
                sample_ids = None
                metadata = {}
                quality_scores = {}
            else:
                raise ValueError('Input data must be either a FeatureSet or numpy array')
            metadata['inverse_power_transform'] = {'warning': 'Inverse transformation not meaningful for exponent=0 (all values mapped to 1)', 'original_exponent': self.exponent}
            return FeatureSet(features=X, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        with np.errstate(invalid='ignore'):
            X_inverted = np.sign(X) * np.abs(X) ** self.inverse_exponent
        metadata['inverse_power_transform'] = {'inverse_exponent': self.inverse_exponent, 'method': 'inverse_power_function'}
        return FeatureSet(features=X_inverted, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class LogTransformer(BaseTransformer):

    def __init__(self, base: float=np.e, shift: float=1e-08, name: Optional[str]=None):
        """
        Initialize the LogTransformer.
        
        Args:
            base (float): The base of the logarithm. Defaults to e (natural log).
            shift (float): Small constant to add to features to ensure positivity. Defaults to 1e-8.
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.base = base
        self.shift = shift
        self.min_values_ = None
        self.is_fitted_ = False
        self._feature_names: Optional[List[str]] = None
        self.shift_amounts_ = None
        self.n_features_in_ = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'LogTransformer':
        """
        Fit the transformer to the input data by computing minimum values for shifting.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            LogTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names.copy() if data.feature_names is not None else None
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names = None
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.min_values_ = np.min(X, axis=0)
        self.shift_amounts_ = np.maximum(1 - self.min_values_, self.shift)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply logarithmic transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Transformed feature set with logarithmic scaling applied.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        X_shifted = X + self.shift_amounts_
        if self.base == np.e:
            X_transformed = np.log(X_shifted)
        else:
            X_transformed = np.log(X_shifted) / np.log(self.base)
        metadata['log_transform_base'] = self.base
        metadata['log_transform_shift'] = self.shift_amounts_
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Revert the logarithmic transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Log-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Data in original scale.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        if self.base == np.e:
            X_exp = np.exp(X)
        else:
            X_exp = np.power(self.base, X)
        X_original = X_exp - self.shift_amounts_
        if 'log_transform_base' in metadata:
            del metadata['log_transform_base']
        if 'log_transform_shift' in metadata:
            del metadata['log_transform_shift']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class Log1PTransformer(BaseTransformer):
    """
    Applies log(1+x) transformation to features, useful for data containing zero or small positive values.
    
    This transformer computes the natural logarithm of (1 + x) for each feature value. It is especially
    useful when dealing with data that includes zeros or very small positive numbers, where a standard
    log transformation would fail.
    
    Attributes:
        name (str): Name identifier for the transformer.
        
    Methods:
        fit: Placeholder method (stateless transformation).
        transform: Applies log(1+x) transformation to input data.
        inverse_transform: Reverts the log(1+x) transformation.
    """

    def __init__(self, name: Optional[str]=None):
        """
        Initialize the Log1PTransformer.
        
        Args:
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'Log1PTransformer':
        """
        Fit method for the transformer (no state required for log1p).
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Log1PTransformer: Returns self for method chaining.
        """
        if not isinstance(data, (FeatureSet, np.ndarray)):
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if isinstance(data, np.ndarray) and data.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if isinstance(data, FeatureSet) and data.features.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if np.any(X <= -1):
            raise ValueError('All values must be greater than -1 for log(1+x) transformation')
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply log(1+x) transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Transformed feature set with log(1+x) applied.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if np.any(X <= -1):
            raise ValueError('All values must be greater than -1 for log(1+x) transformation')
        X_transformed = np.log1p(X)
        metadata['log1p_transform'] = True
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Revert the log(1+x) transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Log(1+x)-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Data in original scale.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        X_original = np.expm1(X)
        if 'log1p_transform' in metadata:
            del metadata['log1p_transform']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class SqrtTransformer(BaseTransformer):
    """
    Applies square root transformation to features to reduce right-skewness.
    
    This transformer computes the square root of each feature value, which helps stabilize
    variance and make data distributions more symmetric. It is commonly used for count data
    or other right-skewed distributions.
    
    Attributes:
        name (str): Name identifier for the transformer.
        
    Methods:
        fit: Placeholder method (stateless transformation).
        transform: Applies square root transformation to input data.
        inverse_transform: Reverts the square root transformation.
    """

    def __init__(self, name: Optional[str]=None):
        """
        Initialize the SqrtTransformer.
        
        Args:
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SqrtTransformer':
        """
        Fit method for the transformer (no state required for sqrt).
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            SqrtTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names.copy() if data.feature_names is not None else None
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names = None
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if np.any(X < 0):
            raise ValueError('Square root transformation cannot be applied to negative values')
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply square root transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Transformed feature set with square root applied.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        if np.any(X < 0):
            raise ValueError('Square root transformation cannot be applied to negative values')
        X_transformed = np.sqrt(X)
        metadata['sqrt_transformed'] = True
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Revert the square root transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Square root-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Data in original scale.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        X_original = np.square(X)
        if 'sqrt_transformed' in metadata:
            del metadata['sqrt_transformed']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class ArcsinhTransformer(BaseTransformer):
    """
    Applies the inverse hyperbolic sine (arcsinh) transformation to features.
    
    This transformer is particularly effective for heavy-tailed data distributions as it behaves
    like a logarithmic transform for large positive values and like a negative logarithmic transform
    for large negative values, while remaining defined at zero. It's useful for stabilizing variance
    and making data more Gaussian-like.
    
    Attributes:
        name (str): Name identifier for the transformer.
        
    Methods:
        fit: Placeholder method (stateless transformation).
        transform: Applies arcsinh transformation to input data.
        inverse_transform: Reverts the arcsinh transformation.
    """

    def __init__(self, name: Optional[str]=None):
        """
        Initialize the ArcsinhTransformer.
        
        Args:
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.is_fitted_ = False
        self.n_features_in_ = None
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ArcsinhTransformer':
        """
        Fit method for the transformer (no state required for arcsinh).
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            ArcsinhTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names.copy() if data.feature_names is not None else None
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names = None
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply arcsinh transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Transformed feature set with arcsinh applied.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        X_transformed = np.arcsinh(X)
        metadata['arcsinh_transformed'] = True
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Revert the arcsinh transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Arcsinh-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Data in original scale.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Input data has {X.shape[1]} features, but transformer was fitted on {self.n_features_in_} features')
        X_original = np.sinh(X)
        if 'arcsinh_transformed' in metadata:
            del metadata['arcsinh_transformed']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class BoxCoxTransformer(BaseTransformer):

    def __init__(self, lmbda: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the BoxCoxTransformer.
        
        Args:
            lmbda (Optional[float]): The lambda parameter for the transformation. 
                                   If None, it will be estimated during fitting.
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.lmbda = lmbda
        self._fitted_lambdas = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'BoxCoxTransformer':
        """
        Fit the transformer by estimating optimal lambda parameters for each feature.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on. Must be positive.
            **kwargs: Additional keyword arguments.
            
        Returns:
            BoxCoxTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if np.any(X <= 0):
            raise ValueError('All values must be positive for Box-Cox transformation')
        (n_samples, n_features) = X.shape
        if self.lmbda is not None:
            self._fitted_lambdas = np.full(n_features, self.lmbda)
        else:
            self._fitted_lambdas = np.zeros(n_features)
            for i in range(n_features):
                try:
                    (_, lmbda_opt) = stats.boxcox(X[:, i])
                    self._fitted_lambdas[i] = lmbda_opt
                except Exception as e:
                    raise RuntimeError(f'Failed to estimate lambda for feature {i}: {str(e)}')
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply Box-Cox transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform. Must be positive.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Transformed data in same format as input.
        """
        if self._fitted_lambdas is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            is_feature_set = True
        else:
            X = data
            is_feature_set = False
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if np.any(X <= 0):
            raise ValueError('All values must be positive for Box-Cox transformation')
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_lambdas):
            raise ValueError('Number of features in data does not match fitted parameters')
        transformed_X = np.zeros_like(X)
        for i in range(n_features):
            lmbda = self._fitted_lambdas[i]
            if lmbda == 0:
                transformed_X[:, i] = np.log(X[:, i])
            else:
                transformed_X[:, i] = (np.power(X[:, i], lmbda) - 1) / lmbda
        if is_feature_set and isinstance(data, FeatureSet):
            new_metadata = data.metadata.copy() if data.metadata else {}
            new_quality_scores = data.quality_scores.copy() if data.quality_scores else {}
            return FeatureSet(features=transformed_X, sample_ids=data.sample_ids, feature_names=data.feature_names, feature_types=data.feature_types, metadata=new_metadata, quality_scores=new_quality_scores)
        else:
            return transformed_X

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Revert the Box-Cox transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Box-Cox-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Data in original scale in same format as input.
        """
        if self._fitted_lambdas is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            is_feature_set = True
            original_data = data
        else:
            X = data
            is_feature_set = False
            original_data = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_lambdas):
            raise ValueError('Number of features in data does not match fitted parameters')
        inverted_X = np.zeros_like(X)
        for i in range(n_features):
            lmbda = self._fitted_lambdas[i]
            if lmbda == 0:
                inverted_X[:, i] = np.exp(X[:, i])
            else:
                inverted_X[:, i] = np.power(X[:, i] * lmbda + 1, 1 / lmbda)
        if is_feature_set and isinstance(original_data, FeatureSet):
            new_metadata = original_data.metadata.copy() if original_data.metadata else {}
            new_quality_scores = original_data.quality_scores.copy() if original_data.quality_scores else {}
            return FeatureSet(features=inverted_X, sample_ids=original_data.sample_ids, feature_names=original_data.feature_names, feature_types=original_data.feature_types, metadata=new_metadata, quality_scores=new_quality_scores)
        else:
            return inverted_X

class YeoJohnsonTransformer(BaseTransformer):

    def __init__(self, lmbda: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the YeoJohnsonTransformer.
        
        Args:
            lmbda (Optional[float]): The lambda parameter for the transformation.
                                   If None, it will be estimated during fitting.
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.lmbda = lmbda
        self._fitted_lambdas = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'YeoJohnsonTransformer':
        """
        Fit the transformer by estimating optimal lambda parameters for each feature.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            YeoJohnsonTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if self.lmbda is not None:
            self._fitted_lambdas = np.full(n_features, self.lmbda)
        else:
            self._fitted_lambdas = np.zeros(n_features)
            for i in range(n_features):
                try:
                    (_, lmbda_opt) = stats.yeojohnson(X[:, i])
                    self._fitted_lambdas[i] = lmbda_opt
                except Exception as e:
                    raise RuntimeError(f'Failed to estimate lambda for feature {i}: {str(e)}')
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply Yeo-Johnson transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Transformed feature set with Yeo-Johnson transformation applied.
        """
        if self._fitted_lambdas is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_lambdas):
            raise ValueError('Number of features in data does not match fitted parameters')
        X_transformed = np.zeros_like(X)
        for i in range(n_features):
            lmbda = self._fitted_lambdas[i]
            x = X[:, i]
            pos = x >= 0
            if lmbda != 0:
                X_transformed[pos, i] = ((x[pos] + 1) ** lmbda - 1) / lmbda
            else:
                X_transformed[pos, i] = np.log(x[pos] + 1)
            neg = ~pos
            if lmbda != 2:
                X_transformed[neg, i] = -((-x[neg] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
            else:
                X_transformed[neg, i] = -np.log(-x[neg] + 1)
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Revert the Yeo-Johnson transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Yeo-Johnson-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            FeatureSet: Data in original scale.
        """
        if self._fitted_lambdas is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_lambdas):
            raise ValueError('Number of features in data does not match fitted parameters')
        X_inverted = np.zeros_like(X)
        for i in range(n_features):
            lmbda = self._fitted_lambdas[i]
            x = X[:, i]
            if lmbda != 0:
                pos = x >= 0
                X_inverted[pos, i] = (x[pos] * lmbda + 1) ** (1 / lmbda) - 1
            else:
                pos = x >= 0
                X_inverted[pos, i] = np.exp(x[pos]) - 1
            if lmbda != 2:
                neg = x < 0
                X_inverted[neg, i] = 1 - (-(2 - lmbda) * x[neg] + 1) ** (1 / (2 - lmbda))
            else:
                neg = x < 0
                X_inverted[neg, i] = 1 - np.exp(-x[neg])
        if 'yeo_johnson_transform' in metadata:
            del metadata['yeo_johnson_transform']
        if 'inverse_yeo_johnson_transform' in metadata:
            del metadata['inverse_yeo_johnson_transform']
        return FeatureSet(features=X_inverted, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class JohnsonTransformer(BaseTransformer):
    """
    Applies Johnson transformation to make data follow a normal distribution.
    
    The Johnson transformation system is a family of transformations that can
    transform any continuous distribution to a normal distribution. It includes
    three types: SB (bounded), SL (log-normal), and SU (unbounded).
    
    Attributes:
        transformation_type (str): Type of Johnson transformation ('su', 'sb', 'sl').
        gamma (float): Shape parameter gamma.
        delta (float): Shape parameter delta (must be positive).
        xi (float): Location parameter.
        lambda_ (float): Scale parameter (must be positive).
        name (str): Name identifier for the transformer.
        
    Methods:
        fit: Estimates optimal Johnson transformation parameters from data.
        transform: Applies Johnson transformation to input data.
        inverse_transform: Reverts the Johnson transformation.
    """

    def __init__(self, transformation_type: str='su', gamma: Optional[float]=None, delta: Optional[float]=None, xi: Optional[float]=None, lambda_: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the JohnsonTransformer.
        
        Args:
            transformation_type (str): Type of Johnson transformation ('su', 'sb', 'sl').
            gamma (Optional[float]): Shape parameter gamma. If None, it will be estimated.
            delta (Optional[float]): Shape parameter delta. If None, it will be estimated.
            xi (Optional[float]): Location parameter. If None, it will be estimated.
            lambda_ (Optional[float]): Scale parameter. If None, it will be estimated.
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.transformation_type = transformation_type.lower()
        if self.transformation_type not in ['su', 'sb', 'sl']:
            raise ValueError("transformation_type must be one of 'su', 'sb', or 'sl'")
        self.gamma = gamma
        self.delta = delta
        self.xi = xi
        self.lambda_ = lambda_
        self._fitted_params = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'JohnsonTransformer':
        """
        Fit the transformer by estimating optimal Johnson transformation parameters.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit on.
            **kwargs: Additional keyword arguments.
            
        Returns:
            JohnsonTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        self._fitted_params = []
        all_params_provided = self.gamma is not None and self.delta is not None and (self.xi is not None) and (self.lambda_ is not None)
        if all_params_provided:
            if self.delta <= 0:
                raise ValueError('delta must be positive')
            if self.lambda_ <= 0:
                raise ValueError('lambda_ must be positive')
            params = {'gamma': self.gamma, 'delta': self.delta, 'xi': self.xi, 'lambda_': self.lambda_}
            for i in range(n_features):
                self._fitted_params.append(params.copy())
        elif n_samples == 0:
            default_params = {'gamma': self.gamma if self.gamma is not None else 0.0, 'delta': self.delta if self.delta is not None else 1.0, 'xi': self.xi if self.xi is not None else 0.0, 'lambda_': self.lambda_ if self.lambda_ is not None else 1.0}
            if default_params['delta'] <= 0:
                raise ValueError('delta must be positive')
            if default_params['lambda_'] <= 0:
                raise ValueError('lambda_ must be positive')
            for i in range(n_features):
                self._fitted_params.append(default_params.copy())
        else:
            for i in range(n_features):
                feature_data = X[:, i]
                if len(feature_data) == 0:
                    raise ValueError(f'Cannot fit transformer on empty feature {i}')
                try:
                    if self.transformation_type == 'su':
                        params_scipy = stats.johnsonsu.fit(feature_data)
                        params = {'gamma': params_scipy[0], 'delta': params_scipy[1], 'xi': params_scipy[2], 'lambda_': params_scipy[3]}
                    elif self.transformation_type == 'sb':
                        params_scipy = stats.johnsonsb.fit(feature_data)
                        params = {'gamma': params_scipy[0], 'delta': params_scipy[1], 'xi': params_scipy[2], 'lambda_': params_scipy[3]}
                    elif self.transformation_type == 'sl':
                        shifted_data = feature_data - np.min(feature_data) + 1e-08
                        params_scipy = stats.lognorm.fit(shifted_data)
                        params = {'gamma': -np.log(params_scipy[2]) / params_scipy[0], 'delta': 1 / params_scipy[0], 'xi': np.min(feature_data) - 1e-08, 'lambda_': 1.0}
                    if params['delta'] <= 0:
                        raise ValueError('Estimated delta must be positive')
                    if params['lambda_'] <= 0:
                        raise ValueError('Estimated lambda_ must be positive')
                    self._fitted_params.append(params)
                except Exception as e:
                    raise RuntimeError(f'Failed to estimate parameters for feature {i}: {str(e)}')
        if n_features > 0:
            self.gamma = np.array([params['gamma'] for params in self._fitted_params])
            self.delta = np.array([params['delta'] for params in self._fitted_params])
            self.xi = np.array([params['xi'] for params in self._fitted_params])
            self.lambda_ = np.array([params['lambda_'] for params in self._fitted_params])
        else:
            self.gamma = np.array([])
            self.delta = np.array([])
            self.xi = np.array([])
            self.lambda_ = np.array([])
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply Johnson transformation to the input data.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Transformed data in same format as input.
        """
        if self._fitted_params is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            is_feature_set = True
            original_data = data
        else:
            X = data
            is_feature_set = False
            original_data = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_params):
            raise ValueError('Number of features in data does not match fitted parameters')
        transformed_X = np.zeros_like(X)
        for i in range(n_features):
            params = self._fitted_params[i]
            gamma = params['gamma']
            delta = params['delta']
            xi = params['xi']
            lambda_ = params['lambda_']
            x = X[:, i]
            if self.transformation_type == 'su':
                transformed_X[:, i] = gamma + delta * np.arcsinh((x - xi) / lambda_)
            elif self.transformation_type == 'sb':
                denominator = xi + lambda_ - x
                if np.any(denominator <= 0) or np.any(x - xi <= 0):
                    raise ValueError(f'Data for feature {i} is outside the valid range for SB transformation')
                transformed_X[:, i] = gamma + delta * np.log((x - xi) / denominator)
            elif self.transformation_type == 'sl':
                numerator = x - xi
                if np.any(numerator <= 0):
                    raise ValueError(f'Data for feature {i} is outside the valid range for SL transformation')
                transformed_X[:, i] = gamma + delta * np.log(numerator / lambda_)
        if is_feature_set and isinstance(original_data, FeatureSet):
            new_metadata = original_data.metadata.copy() if original_data.metadata else {}
            new_quality_scores = original_data.quality_scores.copy() if original_data.quality_scores else {}
            new_metadata['johnson_transform'] = {'type': self.transformation_type, 'params': self._fitted_params}
            return FeatureSet(features=transformed_X, sample_ids=original_data.sample_ids, feature_names=original_data.feature_names, feature_types=original_data.feature_types, metadata=new_metadata, quality_scores=new_quality_scores)
        else:
            return transformed_X

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Revert the Johnson transformation to recover original scale.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Johnson-transformed data to invert.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Union[FeatureSet, np.ndarray]: Data in original scale.
        """
        if self._fitted_params is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            is_feature_set = True
            original_data = data
        else:
            X = data
            is_feature_set = False
            original_data = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if n_features != len(self._fitted_params):
            raise ValueError('Number of features in data does not match fitted parameters')
        inverted_X = np.zeros_like(X)
        for i in range(n_features):
            params = self._fitted_params[i]
            gamma = params['gamma']
            delta = params['delta']
            xi = params['xi']
            lambda_ = params['lambda_']
            z = X[:, i]
            if self.transformation_type == 'su':
                inverted_X[:, i] = xi + lambda_ * np.sinh((z - gamma) / delta)
            elif self.transformation_type == 'sb':
                exp_term = np.exp((z - gamma) / delta)
                inverted_X[:, i] = xi + lambda_ * exp_term / (1 + exp_term)
            elif self.transformation_type == 'sl':
                inverted_X[:, i] = xi + lambda_ * np.exp((z - gamma) / delta)
        if is_feature_set and isinstance(original_data, FeatureSet):
            new_metadata = original_data.metadata.copy() if original_data.metadata else {}
            new_quality_scores = original_data.quality_scores.copy() if original_data.quality_scores else {}
            if 'johnson_transform' in new_metadata:
                del new_metadata['johnson_transform']
            if 'inverse_johnson_transform' in new_metadata:
                del new_metadata['inverse_johnson_transform']
            return FeatureSet(features=inverted_X, sample_ids=original_data.sample_ids, feature_names=original_data.feature_names, feature_types=original_data.feature_types, metadata=new_metadata, quality_scores=new_quality_scores)
        else:
            return inverted_X