from typing import Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy import stats

class DataNormalizer(BaseTransformer):

    def __init__(self, strategy: str='yeo-johnson', normalize_skewness: bool=True, target_distribution: str='normal', clip_bounds: Optional[tuple]=None, name: Optional[str]=None):
        """
        Initialize the DataNormalizer.
        
        Args:
            strategy (str): Normalization technique to apply. Options: 'yeo-johnson', 'box-cox', 'quantile', 'z-score', 'min-max'
            normalize_skewness (bool): Whether to prioritize skewness reduction in transformation
            target_distribution (str): For quantile transformation, target distribution ('uniform' or 'normal')
            clip_bounds (Optional[tuple]): Optional (min, max) bounds to clip transformed values
            name (Optional[str]): Optional name for the transformer instance
        """
        super().__init__(name=name)
        self.strategy = strategy
        self.normalize_skewness = normalize_skewness
        self.target_distribution = target_distribution
        self.clip_bounds = clip_bounds

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'DataNormalizer':
        """
        Compute normalization parameters from the input data.
        
        For example, with 'yeo-johnson' strategy, computes optimal lambda parameters for each feature.
        With 'quantile' strategy, computes empirical cumulative distribution functions.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit normalization parameters.
                                                Should be 2D array-like with samples as rows and features as columns.
            **kwargs: Additional keyword arguments for fitting process
            
        Returns:
            DataNormalizer: Returns self for method chaining
            
        Raises:
            ValueError: If data contains invalid values for the selected strategy (e.g., negative values for box-cox)
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        self.n_features_in_ = X.shape[1]
        if self.strategy == 'yeo-johnson':
            self.lambdas_ = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                try:
                    (_, lambda_opt) = stats.yeojohnson(X[:, i])
                    self.lambdas_[i] = lambda_opt
                except Exception:
                    self.lambdas_[i] = 1.0
        elif self.strategy == 'box-cox':
            if np.any(X <= 0):
                raise ValueError('Box-Cox transformation requires strictly positive values')
            self.lambdas_ = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                try:
                    (_, lambda_opt) = stats.boxcox(X[:, i])
                    self.lambdas_[i] = lambda_opt
                except Exception:
                    self.lambdas_[i] = 1.0
        elif self.strategy == 'quantile':
            self.references_ = np.linspace(0, 1, X.shape[0])
            self.quantiles_ = []
            self.quantiles_inverse_ = []
            for i in range(X.shape[1]):
                sorted_vals = np.sort(X[:, i])
                empirical_cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                self.quantiles_.append((sorted_vals, empirical_cdf))
                if self.target_distribution == 'uniform':
                    self.quantiles_inverse_.append((empirical_cdf, sorted_vals))
                else:
                    normal_vals = stats.norm.ppf(empirical_cdf)
                    normal_vals = np.clip(normal_vals, -10, 10)
                    self.quantiles_inverse_.append((normal_vals, sorted_vals))
        elif self.strategy == 'z-score':
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        elif self.strategy == 'min-max':
            self.data_min_ = np.min(X, axis=0)
            self.data_max_ = np.max(X, axis=0)
            self.data_range_ = self.data_max_ - self.data_min_
            self.data_range_[self.data_range_ == 0] = 1.0
            self.scale_ = self.data_range_
            self.min_ = self.data_min_
        else:
            raise ValueError(f'Unsupported strategy: {self.strategy}')
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply normalization transformation to the input data.
        
        Transforms the data according to the fitted parameters and specified strategy.
        Handles both FeatureSet and raw numpy array inputs.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
                                                Should be 2D array-like with samples as rows and features as columns.
            **kwargs: Additional keyword arguments for transformation process
            
        Returns:
            FeatureSet: Transformed data with normalization applied.
                       Preserves feature names and metadata when input is FeatureSet.
                        
        Raises:
            RuntimeError: If transform is called before fit
            ValueError: If data dimensions don't match fitted parameters
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
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
        if self.strategy == 'yeo-johnson':
            X_transformed = np.zeros_like(X)
            for i in range(X.shape[1]):
                X_transformed[:, i] = stats.yeojohnson(X[:, i], lmbda=self.lambdas_[i])
        elif self.strategy == 'box-cox':
            X_transformed = np.zeros_like(X)
            for i in range(X.shape[1]):
                X_transformed[:, i] = stats.boxcox(X[:, i], lmbda=self.lambdas_[i])
        elif self.strategy == 'quantile':
            X_transformed = np.zeros_like(X)
            for i in range(X.shape[1]):
                (sorted_vals, empirical_cdf) = self.quantiles_[i]
                if self.target_distribution == 'uniform':
                    X_transformed[:, i] = np.interp(X[:, i], sorted_vals, empirical_cdf)
                else:
                    uniform_vals = np.interp(X[:, i], sorted_vals, empirical_cdf)
                    X_transformed[:, i] = stats.norm.ppf(uniform_vals)
        elif self.strategy == 'z-score':
            X_transformed = (X - self.mean_) / self.scale_
        elif self.strategy == 'min-max':
            X_transformed = (X - self.data_min_) / self.data_range_
        else:
            raise ValueError(f'Unsupported strategy: {self.strategy}')
        if self.clip_bounds is not None:
            X_transformed = np.clip(X_transformed, self.clip_bounds[0], self.clip_bounds[1])
        metadata['scaling_method'] = self.strategy
        if self.strategy in ['yeo-johnson', 'box-cox']:
            metadata['transformation_lambdas'] = self.lambdas_.tolist()
        if self.clip_bounds is not None:
            metadata['clip_bounds'] = self.clip_bounds
        return FeatureSet(features=X_transformed, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse normalization transformation.
        
        Reverses the normalization to recover the original data scale where possible.
        Not all transformations support perfect inversion (e.g., clipped quantile transforms).
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Normalized data to inverse transform.
                                                Should be 2D array-like with samples as rows and features as columns.
            **kwargs: Additional keyword arguments for inverse transformation process
            
        Returns:
            FeatureSet: Data restored to original scale.
                       Preserves feature names and metadata when input is FeatureSet.
                        
        Raises:
            RuntimeError: If inverse_transform is called before fit
            ValueError: If inverse transformation is not supported for the current strategy
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
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
        if self.strategy == 'yeo-johnson':
            X_original = np.zeros_like(X)
            for i in range(X.shape[1]):
                X_original[:, i] = self._yeo_johnson_inverse(X[:, i], self.lambdas_[i])
        elif self.strategy == 'box-cox':
            X_original = np.zeros_like(X)
            for i in range(X.shape[1]):
                X_original[:, i] = self._box_cox_inverse(X[:, i], self.lambdas_[i])
        elif self.strategy == 'quantile':
            X_original = np.zeros_like(X)
            for i in range(X.shape[1]):
                (transformed_vals, original_vals) = self.quantiles_inverse_[i]
                X_original[:, i] = np.interp(X[:, i], transformed_vals, original_vals)
        elif self.strategy == 'z-score':
            X_original = X * self.scale_ + self.mean_
        elif self.strategy == 'min-max':
            X_original = X * self.data_range_ + self.data_min_
        else:
            raise ValueError(f'Unsupported strategy: {self.strategy}')
        if 'scaling_method' in metadata:
            del metadata['scaling_method']
        if 'transformation_lambdas' in metadata:
            del metadata['transformation_lambdas']
        if 'clip_bounds' in metadata:
            del metadata['clip_bounds']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def _yeo_johnson_inverse(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply inverse Yeo-Johnson transformation."""
        result = np.zeros_like(x)
        pos = x >= 0
        if lmbda != 0:
            result[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda)
        else:
            result[pos] = np.exp(x[pos])
        neg = x < 0
        if lmbda != 2:
            result[neg] = -np.power(-x[neg] * (2 - lmbda) + 1, 1 / (2 - lmbda))
        else:
            result[neg] = -np.exp(-x[neg])
        return result

    def _box_cox_inverse(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply inverse Box-Cox transformation."""
        if lmbda == 0:
            return np.exp(x)
        else:
            return np.power(x * lmbda + 1, 1 / lmbda)