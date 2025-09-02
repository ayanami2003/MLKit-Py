from typing import Optional, Union
from general.structures.feature_set import FeatureSet
from general.base_classes.transformer_base import BaseTransformer
import numpy as np

class DataScaler(BaseTransformer):

    def __init__(self, strategy: str='standard', name: Optional[str]=None, **strategy_params):
        """
        Initialize the DataScaler.

        Args:
            strategy (str): Scaling strategy to use. Options include:
                - 'standard': Zero-mean, unit-variance scaling (z-score)
                - 'minmax': Scale to fixed range, usually [0,1]
                - 'robust': Use median and IQR for scaling (outlier resistant)
            name (Optional[str]): Custom name for the transformer
            **strategy_params: Additional parameters specific to the chosen strategy
        """
        super().__init__(name=name)
        self.strategy = strategy
        self.strategy_params = strategy_params
        self.scale_params = {}

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'DataScaler':
        """
        Fit the scaler to the input data to compute scaling parameters.

        This method analyzes the input data to determine the necessary parameters
        (like mean, standard deviation, min/max values) needed for the selected
        scaling strategy. These parameters are stored internally and used in
        subsequent transform operations.

        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to fit the scaler on.
                Can be either a FeatureSet object or a raw numpy array.
            **kwargs: Additional keyword arguments for fitting.

        Returns:
            DataScaler: Returns self for method chaining.

        Raises:
            ValueError: If the input data format is not supported.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        if self.strategy == 'standard':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std = np.where(std == 0, 1.0, std)
            self.scale_params = {'mean': mean, 'std': std, 'n_features': n_features}
        elif self.strategy == 'minmax':
            feature_range = self.strategy_params.get('feature_range', (0, 1))
            (min_val, max_val) = feature_range
            data_min = np.min(X, axis=0)
            data_max = np.max(X, axis=0)
            data_range = np.where(data_max == data_min, 1.0, data_max - data_min)
            self.scale_params = {'min': data_min, 'max': data_max, 'data_range': data_range, 'scale_min': min_val, 'scale_max': max_val, 'scale_range': max_val - min_val, 'n_features': n_features}
        elif self.strategy == 'robust':
            median = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1.0, iqr)
            self.scale_params = {'median': median, 'iqr': iqr, 'n_features': n_features}
        else:
            raise ValueError(f'Unsupported scaling strategy: {self.strategy}')
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the learned scaling transformation to the input data.

        Uses the scaling parameters computed during the fit phase to transform
        the input data according to the selected scaling strategy.

        Args:
            data (Union[FeatureSet, np.ndarray]): Input data to transform.
            **kwargs: Additional keyword arguments for transformation.

        Returns:
            FeatureSet: Transformed data as a FeatureSet with scaling metadata.

        Raises:
            RuntimeError: If the transformer has not been fitted yet.
            ValueError: If the input data dimensions don't match the fitted parameters.
        """
        if not self.scale_params:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' first.")
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
        if X.shape[1] != self.scale_params['n_features']:
            raise ValueError(f"Input data has {X.shape[1]} features, but scaler was fitted on {self.scale_params['n_features']} features")
        if self.strategy == 'standard':
            X_scaled = (X - self.scale_params['mean']) / self.scale_params['std']
        elif self.strategy == 'minmax':
            X_std = (X - self.scale_params['min']) / self.scale_params['data_range']
            X_scaled = X_std * self.scale_params['scale_range'] + self.scale_params['scale_min']
        elif self.strategy == 'robust':
            X_scaled = (X - self.scale_params['median']) / self.scale_params['iqr']
        else:
            raise ValueError(f'Unsupported scaling strategy: {self.strategy}')
        metadata['scaling_method'] = self.strategy
        metadata['scaling_params'] = self.scale_params
        return FeatureSet(features=X_scaled, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse scaling transformation to revert scaled data.

        This method applies the inverse of the scaling transformation to
        convert scaled data back to its original scale, using the parameters
        learned during the fit phase.

        Args:
            data (Union[FeatureSet, np.ndarray]): Scaled input data to inverse transform.
            **kwargs: Additional keyword arguments for inverse transformation.

        Returns:
            FeatureSet: Data transformed back to original scale.

        Raises:
            RuntimeError: If the transformer has not been fitted yet.
            ValueError: If inverse transformation is not possible for the current strategy.
        """
        if not self.scale_params:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' first.")
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
        if X.shape[1] != self.scale_params['n_features']:
            raise ValueError(f"Input data has {X.shape[1]} features, but scaler was fitted on {self.scale_params['n_features']} features")
        if self.strategy == 'standard':
            X_original = X * self.scale_params['std'] + self.scale_params['mean']
        elif self.strategy == 'minmax':
            X_std = (X - self.scale_params['scale_min']) / self.scale_params['scale_range']
            X_original = X_std * self.scale_params['data_range'] + self.scale_params['min']
        elif self.strategy == 'robust':
            X_original = X * self.scale_params['iqr'] + self.scale_params['median']
        else:
            raise ValueError(f'Unsupported scaling strategy: {self.strategy}')
        if 'scaling_method' in metadata:
            del metadata['scaling_method']
        if 'scaling_params' in metadata:
            del metadata['scaling_params']
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)