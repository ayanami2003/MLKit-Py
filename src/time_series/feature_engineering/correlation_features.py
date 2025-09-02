from typing import Union, Optional, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np

class AutoCorrelationFeatures(BaseTransformer):

    def __init__(self, lags: Union[int, List[int]]=[1, 2, 3, 5, 10], feature_columns: Optional[List[str]]=None, threshold: float=0.1, name: Optional[str]=None):
        """
        Initialize the AutoCorrelationFeatures transformer.
        
        Parameters
        ----------
        lags : Union[int, List[int]], default=[1, 2, 3, 5, 10]
            Lag value(s) to compute autocorrelation for. Can be a single integer or list of integers.
        feature_columns : Optional[List[str]], default=None
            Specific columns to compute autocorrelation features for. If None, uses all columns.
        threshold : float, default=0.1
            Threshold for determining significant autocorrelations.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.lags = [lags] if isinstance(lags, int) else lags
        self.feature_columns = feature_columns
        self.threshold = threshold

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'AutoCorrelationFeatures':
        """
        Fit the transformer to the input data.
        
        This method validates the input data and prepares for transformation.
        No actual computation is performed here.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input time series data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        AutoCorrelationFeatures
            Self instance for method chaining.
        """
        if not isinstance(data, (DataBatch, FeatureSet)):
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if isinstance(data, DataBatch):
            if not hasattr(data, 'features') or data.features is None:
                raise ValueError('DataBatch must contain features attribute')
            features = data.features
        else:
            features = data.features
        if not isinstance(features, np.ndarray):
            raise TypeError('Features must be a numpy array')
        if features.ndim != 2:
            raise ValueError('Features must be a 2D array')
        max_lag = max(self.lags) if self.lags else 0
        if features.shape[0] <= max_lag:
            raise ValueError(f'Insufficient data length ({features.shape[0]}) for maximum lag ({max_lag})')
        if hasattr(data, 'feature_names') and data.feature_names is not None:
            self._feature_names = data.feature_names
        else:
            self._feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        if self.feature_columns is not None:
            invalid_columns = [col for col in self.feature_columns if col not in self._feature_names]
            if invalid_columns:
                raise ValueError(f'Specified feature columns not found in data: {invalid_columns}')
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Compute autocorrelation features for the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input time series data to compute autocorrelation features for.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            FeatureSet with original features plus computed autocorrelation features.
        """
        if isinstance(data, DataBatch):
            features = data.features
            feature_names = data.feature_names if hasattr(data, 'feature_names') and data.feature_names is not None else self._feature_names
            sample_ids = data.sample_ids if hasattr(data, 'sample_ids') else None
            metadata = data.metadata.copy() if hasattr(data, 'metadata') and data.metadata else {}
        else:
            features = data.features
            feature_names = data.feature_names if data.feature_names is not None else self._feature_names
            sample_ids = data.sample_ids if hasattr(data, 'sample_ids') else None
            metadata = data.metadata.copy() if data.metadata else {}
        if self.feature_columns is not None:
            column_indices = [feature_names.index(col) for col in self.feature_columns]
            process_columns = self.feature_columns
        else:
            column_indices = list(range(len(feature_names)))
            process_columns = feature_names
        (n_samples, n_features) = features.shape
        max_lag = max(self.lags) if self.lags else 0
        new_features_list = [features]
        new_feature_names = list(feature_names)
        for (col_idx, col_name) in zip(column_indices, process_columns):
            series = features[:, col_idx]
            acf_values = []
            for lag in self.lags:
                if lag >= n_samples:
                    acf_values.append(0.0)
                else:
                    mean_val = np.mean(series)
                    numerator = np.sum((series[:-lag] - mean_val) * (series[lag:] - mean_val)) if lag > 0 else np.sum((series - mean_val) ** 2)
                    denominator = np.sum((series - mean_val) ** 2)
                    if denominator == 0:
                        acf_values.append(0.0)
                    else:
                        acf_values.append(numerator / denominator)
            for (lag, acf_val) in zip(self.lags, acf_values):
                new_features_list.append(acf_val * np.ones((n_samples, 1)))
                new_feature_names.append(f'{col_name}_acf_lag_{lag}')
            first_below_threshold = next((i for (i, val) in enumerate(acf_values) if abs(val) < self.threshold), len(self.lags))
            new_features_list.append(first_below_threshold * np.ones((n_samples, 1)))
            new_feature_names.append(f'{col_name}_acf_first_below_threshold')
            if len(self.lags) > 1 and len(acf_values) > 1:
                slope = np.polyfit(self.lags, acf_values, 1)[0]
            else:
                slope = 0.0
            new_features_list.append(slope * np.ones((n_samples, 1)))
            new_feature_names.append(f'{col_name}_acf_decay_rate')
        new_features = np.hstack(new_features_list)
        original_feature_types = data.feature_types if hasattr(data, 'feature_types') and data.feature_types else ['numeric'] * len(feature_names)
        new_feature_types = list(original_feature_types)
        new_feature_types.extend(['numeric'] * (len(new_feature_names) - len(feature_names)))
        return FeatureSet(features=new_features, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for this transformer.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data to invert (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            The input data unchanged.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        pass

class PartialCorrelationAnalysis(BaseTransformer):

    def __init__(self, feature_columns: Optional[List[str]]=None, max_conditioning_vars: int=5, significance_level: float=0.05, name: Optional[str]=None):
        """
        Initialize the PartialCorrelationAnalysis transformer.
        
        Parameters
        ----------
        feature_columns : Optional[List[str]], default=None
            Specific columns to compute partial correlations for. If None, uses all columns.
        max_conditioning_vars : int, default=5
            Maximum number of conditioning variables to consider.
        significance_level : float, default=0.05
            Significance level for partial correlation tests.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.feature_columns = feature_columns
        self.max_conditioning_vars = max_conditioning_vars
        self.significance_level = significance_level

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'PartialCorrelationAnalysis':
        """
        Fit the transformer to the input data.
        
        This method validates the input data and prepares for transformation.
        No actual computation is performed here.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input multivariate time series data to fit the transformer on.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        PartialCorrelationAnalysis
            Self instance for method chaining.
        """
        if not isinstance(data, (DataBatch, FeatureSet)):
            raise TypeError('Input data must be either DataBatch or FeatureSet')
        if isinstance(data, DataBatch):
            feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(data.features.shape[1])]
        else:
            feature_names = data.feature_names
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(feature_names)
            if missing_cols:
                raise ValueError(f'Specified feature columns not found in data: {missing_cols}')
        else:
            self.feature_columns = feature_names
        self._feature_names = feature_names
        self._n_features = len(self.feature_columns)
        if self.max_conditioning_vars < 0:
            raise ValueError('max_conditioning_vars must be non-negative')
        if self.max_conditioning_vars >= self._n_features:
            raise ValueError('max_conditioning_vars must be less than the number of features')
        if not 0 < self.significance_level < 1:
            raise ValueError('significance_level must be between 0 and 1')
        self._is_fitted = True
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Compute partial correlation features for the input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input multivariate time series data to compute partial correlation features for.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            FeatureSet with original features plus computed partial correlation features.
        """
        if not self._is_fitted:
            raise RuntimeError('Transformer must be fitted before transform')
        if isinstance(data, DataBatch):
            features = data.features
            feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(features.shape[1])]
        else:
            features = data.features
            feature_names = data.feature_names
        if len(feature_names) != features.shape[1]:
            raise ValueError('Mismatch between feature names and data dimensions')
        col_indices = [feature_names.index(col) for col in self.feature_columns]
        selected_features = features[:, col_indices]
        corr_matrix = np.corrcoef(selected_features, rowvar=False)
        eps = 1e-08
        np.fill_diagonal(corr_matrix, 1.0 + eps)
        try:
            precision_matrix = np.linalg.inv(corr_matrix)
        except np.linalg.LinAlgError:
            precision_matrix = np.linalg.inv(corr_matrix + np.eye(len(corr_matrix)) * 1e-06)
        diag_sqrt = np.sqrt(np.diag(precision_matrix))
        partial_corr = -precision_matrix / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(partial_corr, 1.0)
        n_samples = selected_features.shape[0]
        dof = n_samples - 2 - self.max_conditioning_vars
        if dof <= 0:
            raise ValueError('Not enough samples for partial correlation test with given conditioning variables')
        with np.errstate(divide='ignore'):
            partial_corr_clipped = np.clip(partial_corr, -0.9999, 0.9999)
            z_scores = 0.5 * np.log((1 + partial_corr_clipped) / (1 - partial_corr_clipped))
            stderr = 1.0 / np.sqrt(dof)
            z_statistics = z_scores / stderr
            p_values = 2 * (1 - np.abs(z_statistics))
        n_feat = len(self.feature_columns)
        pc_feature_names = []
        significant_mask = np.zeros_like(partial_corr, dtype=bool)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                if p_values[i, j] < self.significance_level:
                    significant_mask[i, j] = True
                    pc_feature_names.append(f'partial_corr_{self.feature_columns[i]}_{self.feature_columns[j]}')
        pc_features = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                if significant_mask[i, j]:
                    pc_features.append(partial_corr[i, j])
        if pc_features:
            pc_features_array = np.array(pc_features).reshape(1, -1) if len(pc_features) > 0 else np.empty((features.shape[0], 0))
            pc_features_array = np.tile(pc_features_array, (features.shape[0], 1))
        else:
            pc_features_array = np.empty((features.shape[0], 0))
        combined_features = np.hstack([features, pc_features_array])
        combined_names = list(feature_names) + pc_feature_names
        return FeatureSet(features=combined_features, feature_names=combined_names)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for this transformer.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed data to invert (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            The input data unchanged.
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for partial correlation analysis.')