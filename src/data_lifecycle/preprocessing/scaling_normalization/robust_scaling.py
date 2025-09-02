from typing import Optional, Union, List, Dict, Any, Tuple
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np


# ...(code omitted)...


class AdaptiveRobustScaler(BaseTransformer):
    """
    A transformer that adaptively scales features using robust statistics with dynamic quantile adjustment.
    
    This scaler extends robust scaling by adapting to the data characteristics, potentially adjusting
    quantile ranges or scaling strategies based on data distribution properties. It's particularly
    useful when data distributions vary significantly across features or over time.
    
    The transformation adapts based on data characteristics:
        scaled = (X - adaptive_median) / adaptive_IQR
    
    where adaptive parameters are computed based on data distribution properties.
    
    Attributes
    ----------
    initial_quantile_range : tuple (q_min, q_max), default=(25.0, 75.0)
        Initial quantile range used for scaling.
    adaptivity_threshold : float, default=0.1
        Threshold for determining when to adapt scaling parameters based on data distribution changes.
    max_quantile_range : tuple (q_min, q_max), default=(10.0, 90.0)
        Maximum quantile range allowed during adaptation.
    min_quantile_range : tuple (q_min, q_max), default=(30.0, 70.0)
        Minimum quantile range allowed during adaptation.
    name : str, optional
        Name of the transformer instance.
    
    Examples
    --------
    >>> import numpy as np
    >>> from general.structures.feature_set import FeatureSet
    >>> from src.data_lifecycle.preprocessing.scaling_normalization.robust_scaling import AdaptiveRobustScaler
    >>> 
    >>> # Create sample data with varying distributions
    >>> X = np.random.exponential(scale=2.0, size=(100, 2))
    >>> feature_set = FeatureSet(features=X, feature_names=['f1', 'f2'])
    >>> 
    >>> # Initialize and apply the adaptive robust scaler
    >>> scaler = AdaptiveRobustScaler()
    >>> scaled_features = scaler.fit_transform(feature_set)
    """

    def __init__(self, initial_quantile_range: tuple=(25.0, 75.0), adaptivity_threshold: float=0.1, max_quantile_range: tuple=(10.0, 90.0), min_quantile_range: tuple=(30.0, 70.0), name: Optional[str]=None):
        super().__init__(name=name)
        self.initial_quantile_range = initial_quantile_range
        self.adaptivity_threshold = adaptivity_threshold
        self.max_quantile_range = max_quantile_range
        self.min_quantile_range = min_quantile_range
        self.adaptive_params_ = {}
        self.center_ = None
        self.scale_ = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AdaptiveRobustScaler':
        """
        Compute adaptive scaling parameters based on data characteristics.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            The data used to compute adaptive scaling parameters.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        AdaptiveRobustScaler
            Fitted scaler instance with adaptive parameters.
            
        Raises
        ------
        ValueError
            If quantile ranges are not valid tuples or outside (0, 100) range.
        """
        if not isinstance(self.initial_quantile_range, tuple) or len(self.initial_quantile_range) != 2:
            raise ValueError('initial_quantile_range must be a tuple of two values')
        if not isinstance(self.max_quantile_range, tuple) or len(self.max_quantile_range) != 2:
            raise ValueError('max_quantile_range must be a tuple of two values')
        if not isinstance(self.min_quantile_range, tuple) or len(self.min_quantile_range) != 2:
            raise ValueError('min_quantile_range must be a tuple of two values')
        (q_min, q_max) = self.initial_quantile_range
        if not 0 < q_min < 100 or not 0 < q_max < 100 or q_min >= q_max:
            raise ValueError('initial_quantile_range values must be between 0 and 100, with q_min < q_max')
        (q_min_max, q_max_max) = self.max_quantile_range
        if not 0 < q_min_max < 100 or not 0 < q_max_max < 100 or q_min_max >= q_max_max:
            raise ValueError('max_quantile_range values must be between 0 and 100, with q_min < q_max')
        (q_min_min, q_max_min) = self.min_quantile_range
        if not 0 < q_min_min < 100 or not 0 < q_max_min < 100 or q_min_min >= q_max_min:
            raise ValueError('min_quantile_range values must be between 0 and 100, with q_min < q_max')
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        (n_samples, n_features) = X.shape
        medians = np.median(X, axis=0)
        q75 = np.percentile(X, q_max, axis=0)
        q25 = np.percentile(X, q_min, axis=0)
        iqrs = q75 - q25
        iqrs = np.where(iqrs == 0, 1.0, iqrs)
        self.adaptive_params_ = {'center': medians, 'scale': iqrs, 'quantile_range': self.initial_quantile_range, 'n_features': n_features}
        self.center_ = medians
        self.scale_ = iqrs
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Scale features using adaptive robust parameters.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            The data to transform.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed data with adaptively scaled features.
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted yet.
        """
        if not self.adaptive_params_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
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
        if X.shape[1] != self.adaptive_params_['n_features']:
            raise ValueError(f"Input data has {X.shape[1]} features, but scaler was fitted on {self.adaptive_params_['n_features']} features")
        medians = self.adaptive_params_['center']
        iqrs = self.adaptive_params_['scale']
        X_scaled = (X - medians) / iqrs
        metadata['scaling_method'] = 'adaptive_robust'
        metadata['scaling_params'] = self.adaptive_params_.copy()
        return FeatureSet(features=X_scaled, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Scale back the data to the original representation.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            The scaled data to transform back.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        FeatureSet
            Original data restored from scaled representation.
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted yet.
        """
        if not self.adaptive_params_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
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
        if X.shape[1] != self.adaptive_params_['n_features']:
            raise ValueError(f"Input data has {X.shape[1]} features, but scaler was fitted on {self.adaptive_params_['n_features']} features")
        medians = self.adaptive_params_['center']
        iqrs = self.adaptive_params_['scale']
        X_original = X * iqrs + medians
        if 'scaling_method' in metadata:
            metadata.pop('scaling_method', None)
        if 'scaling_params' in metadata:
            metadata.pop('scaling_params', None)
        return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def update_adaptive_parameters(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AdaptiveRobustScaler':
        """
        Update adaptive scaling parameters based on new data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            New data used to update adaptive scaling parameters.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        AdaptiveRobustScaler
            Updated scaler instance with new adaptive parameters.
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted yet.
        """
        if not self.adaptive_params_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if X.shape[1] != self.adaptive_params_['n_features']:
            raise ValueError(f"Input data has {X.shape[1]} features, but scaler was fitted on {self.adaptive_params_['n_features']} features")
        current_center = self.adaptive_params_['center']
        current_scale = self.adaptive_params_['scale']
        current_range = self.adaptive_params_['quantile_range']
        new_medians = np.median(X, axis=0)
        q_max = current_range[1]
        q_min = current_range[0]
        new_q75 = np.percentile(X, q_max, axis=0)
        new_q25 = np.percentile(X, q_min, axis=0)
        new_iqrs = new_q75 - new_q25
        new_iqrs = np.where(new_iqrs == 0, 1.0, new_iqrs)
        median_changes = np.abs(new_medians - current_center) / (current_scale + 1e-08)
        iqr_changes = np.abs(new_iqrs - current_scale) / (current_scale + 1e-08)
        if np.any(median_changes > self.adaptivity_threshold) or np.any(iqr_changes > self.adaptivity_threshold):
            range_adjustment = 5.0
            new_q_min = max(self.min_quantile_range[0], min(current_range[0] - range_adjustment, self.max_quantile_range[0]))
            new_q_max = min(self.max_quantile_range[1], max(current_range[1] + range_adjustment, self.min_quantile_range[1]))
            if new_q_max - new_q_min < self.min_quantile_range[1] - self.min_quantile_range[0]:
                mid = (new_q_min + new_q_max) / 2
                half_min_range = (self.min_quantile_range[1] - self.min_quantile_range[0]) / 2
                new_q_min = max(self.min_quantile_range[0], mid - half_min_range)
                new_q_max = min(self.max_quantile_range[1], mid + half_min_range)
            if (new_q_min, new_q_max) != current_range:
                new_q75 = np.percentile(X, new_q_max, axis=0)
                new_q25 = np.percentile(X, new_q_min, axis=0)
                new_iqrs = new_q75 - new_q25
                new_iqrs = np.where(new_iqrs == 0, 1.0, new_iqrs)
                self.adaptive_params_['quantile_range'] = (new_q_min, new_q_max)
            self.adaptive_params_['center'] = new_medians
            self.adaptive_params_['scale'] = new_iqrs
            self.center_ = new_medians
            self.scale_ = new_iqrs
        return self


# ...(code omitted)...


class RobustScalerOutlierHandler(BaseTransformer):
    """
    A transformer that extends robust scaling with explicit outlier handling capabilities.
    
    This scaler builds upon robust scaling methodology but adds specialized methods for
    identifying, analyzing, and handling outliers that may be present in the data. It
    provides detailed control over how outliers are treated during the scaling process
    while maintaining the robustness of the underlying scaling approach.
    
    The transformation includes:
    1. Standard robust scaling (median/IQR)
    2. Outlier identification using multiple methods
    3. Configurable outlier treatment strategies
    
    Attributes
    ----------
    scaling_method : str, default='iqr'
        Robust scaling method to use ('iqr', 'mad', 'percentile').
    outlier_detection_methods : list of str, default=['iqr']
        Methods to use for outlier detection (['iqr', 'zscore', 'modified_zscore', 'isolation_forest']).
    outlier_treatment : str, default='preserve'
        How to treat outliers ('preserve', 'clip', 'separate_scaling', 'mask').
    outlier_threshold : float, default=1.5
        Threshold multiplier for IQR-based outlier detection.
    name : str, optional
        Name of the transformer instance.
    
    Examples
    --------
    >>> import numpy as np
    >>> from general.structures.feature_set import FeatureSet
    >>> from src.data_lifecycle.preprocessing.scaling_normalization.robust_scaling import RobustScalerOutlierHandler
    >>> 
    >>> # Create sample data with outliers
    >>> X = np.concatenate([np.random.normal(0, 1, (95, 2)), np.random.normal(10, 1, (5, 2))])
    >>> feature_set = FeatureSet(features=X, feature_names=['f1', 'f2'])
    >>> 
    >>> # Initialize and apply the robust scaler with outlier handling
    >>> scaler = RobustScalerOutlierHandler(
    ...     scaling_method='iqr',
    ...     outlier_detection_methods=['iqr', 'zscore'],
    ...     outlier_treatment='clip'
    ... )
    >>> scaled_features = scaler.fit_transform(feature_set)
    >>> 
    >>> # Analyze detected outliers
    >>> outlier_info = scaler.get_outlier_analysis()
    """

    def __init__(self, scaling_method: str='iqr', outlier_detection_methods: List[str]=None, outlier_treatment: str='preserve', outlier_threshold: float=1.5, name: Optional[str]=None):
        super().__init__(name=name)
        self.scaling_method = scaling_method
        self.outlier_detection_methods = outlier_detection_methods or ['iqr']
        self.outlier_treatment = outlier_treatment
        self.outlier_threshold = outlier_threshold
        self.scaling_params_ = {}
        self.outlier_info_ = {}
        valid_scaling_methods = ['iqr', 'mad', 'percentile']
        if self.scaling_method not in valid_scaling_methods:
            raise ValueError(f'scaling_method must be one of {valid_scaling_methods}')
        valid_detection_methods = ['iqr', 'zscore', 'modified_zscore', 'isolation_forest']
        for method in self.outlier_detection_methods:
            if method not in valid_detection_methods:
                raise ValueError(f'All outlier_detection_methods must be in {valid_detection_methods}')
        valid_treatments = ['preserve', 'clip', 'separate_scaling', 'mask']
        if self.outlier_treatment not in valid_treatments:
            raise ValueError(f'outlier_treatment must be one of {valid_treatments}')

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'RobustScalerOutlierHandler':
        """
        Compute robust scaling parameters and analyze outliers in the data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            The data used to compute scaling parameters and analyze outliers.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        RobustScalerOutlierHandler
            Fitted scaler instance with scaling parameters and outlier information.
            
        Raises
        ------
        ValueError
            If scaling method or outlier detection method is not supported.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_ = data.feature_names
        else:
            X = np.asarray(data)
            self.feature_names_ = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        self.n_features_ = n_features
        self.scaling_params_ = {}
        for i in range(n_features):
            feature_data = X[:, i]
            if self.scaling_method == 'iqr':
                q1 = np.percentile(feature_data, 25)
                q3 = np.percentile(feature_data, 75)
                iqr = q3 - q1
                center = np.median(feature_data)
                scale = iqr if iqr != 0 else 1.0
            elif self.scaling_method == 'mad':
                from src.data_lifecycle.mathematical_foundations.statistical_methods.descriptive_statistics import median_absolute_deviation
                center = np.median(feature_data)
                mad = median_absolute_deviation(feature_data)
                scale = mad if mad != 0 else 1.0
            elif self.scaling_method == 'percentile':
                q1 = np.percentile(feature_data, 25)
                q3 = np.percentile(feature_data, 75)
                center = np.median(feature_data)
                scale = (q3 - q1) / 2 if q3 - q1 != 0 else 1.0
            self.scaling_params_[i] = {'center': center, 'scale': scale}
        self.outlier_info_ = {}
        for i in range(n_features):
            feature_data = X[:, i]
            self.outlier_info_[i] = {}
            outlier_masks = {}
            if 'iqr' in self.outlier_detection_methods:
                q1 = np.percentile(feature_data, 25)
                q3 = np.percentile(feature_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - self.outlier_threshold * iqr
                upper_bound = q3 + self.outlier_threshold * iqr
                iqr_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
                outlier_masks['iqr'] = iqr_mask
                iqr_indices = np.where(iqr_mask)[0]
                self.outlier_info_[i]['iqr'] = {'indices': iqr_indices.tolist(), 'values': feature_data[iqr_indices].tolist()}
            if 'zscore' in self.outlier_detection_methods:
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                if std_val != 0:
                    z_scores = np.abs((feature_data - mean_val) / std_val)
                else:
                    z_scores = np.zeros_like(feature_data)
                zscore_mask = z_scores > self.outlier_threshold
                outlier_masks['zscore'] = zscore_mask
                zscore_indices = np.where(zscore_mask)[0]
                self.outlier_info_[i]['zscore'] = {'indices': zscore_indices.tolist(), 'values': feature_data[zscore_indices].tolist()}
            if 'modified_zscore' in self.outlier_detection_methods:
                median_val = np.median(feature_data)
                mad = np.median(np.abs(feature_data - median_val))
                if mad != 0:
                    modified_z_scores = 0.6745 * (feature_data - median_val) / mad
                else:
                    modified_z_scores = np.zeros_like(feature_data)
                modified_zscore_mask = np.abs(modified_z_scores) > self.outlier_threshold
                outlier_masks['modified_zscore'] = modified_zscore_mask
                modified_zscore_indices = np.where(modified_zscore_mask)[0]
                self.outlier_info_[i]['modified_zscore'] = {'indices': modified_zscore_indices.tolist(), 'values': feature_data[modified_zscore_indices].tolist()}
            if 'isolation_forest' in self.outlier_detection_methods:
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination='auto', random_state=42)
                    iso_forest.fit(feature_data.reshape(-1, 1))
                    iso_predictions = iso_forest.predict(feature_data.reshape(-1, 1))
                    isolation_mask = iso_predictions == -1
                    outlier_masks['isolation_forest'] = isolation_mask
                    isolation_indices = np.where(isolation_mask)[0]
                    self.outlier_info_[i]['isolation_forest'] = {'indices': isolation_indices.tolist(), 'values': feature_data[isolation_indices].tolist()}
                except ImportError:
                    pass
            if outlier_masks:
                combined_mask = np.zeros(len(feature_data), dtype=bool)
                for mask in outlier_masks.values():
                    combined_mask |= mask
                self.outlier_info_[i]['combined'] = combined_mask
                all_masks = list(outlier_masks.values())
                if all_masks:
                    full_agreement_mask = all_masks[0].copy()
                    for mask in all_masks[1:]:
                        full_agreement_mask &= mask
                    full_agreement_indices = np.where(full_agreement_mask)[0]
                    self.outlier_info_[i]['full_agreement'] = full_agreement_indices.tolist()
                else:
                    self.outlier_info_[i]['full_agreement'] = []
        self.is_fitted_ = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply robust scaling and outlier treatment to the data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            The data to transform.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed data with robust scaling and outlier treatment applied.
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted yet.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = np.asarray(data)
            feature_names = getattr(self, 'feature_names_', None)
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_} features')
        X_scaled = X.copy().astype(float)
        for i in range(self.n_features_):
            center = self.scaling_params_[i]['center']
            scale = self.scaling_params_[i]['scale']
            X_scaled[:, i] = (X_scaled[:, i] - center) / scale
        X_treated = self.apply_outlier_treatment_to_array(X_scaled, X)
        metadata['scaling_method'] = self.scaling_method
        metadata['outlier_treatment'] = self.outlier_treatment
        metadata['scaling_params'] = self.scaling_params_.copy()
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_treated, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return FeatureSet(features=X_treated, feature_names=feature_names, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Scale back the data to the original representation.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            The scaled data to transform back.
        **kwargs : dict
            Additional keyword arguments (ignored).
            
        Returns
        -------
        FeatureSet
            Original data restored from scaled representation.
            
        Raises
        ------
        ValueError
            If the scaler has not been fitted yet.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = np.asarray(data)
            feature_names = getattr(self, 'feature_names_', None)
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Input data has {X.shape[1]} features, but scaler was fitted on {self.n_features_} features')
        X_original = X.copy().astype(float)
        for i in range(self.n_features_):
            center = self.scaling_params_[i]['center']
            scale = self.scaling_params_[i]['scale']
            X_original[:, i] = X_original[:, i] * scale + center
        if 'scaling_method' in metadata:
            metadata.pop('scaling_method', None)
        if 'outlier_treatment' in metadata:
            metadata.pop('outlier_treatment', None)
        if 'scaling_params' in metadata:
            metadata.pop('scaling_params', None)
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_original, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return FeatureSet(features=X_original, feature_names=feature_names, metadata=metadata)

    def apply_outlier_treatment_to_array(self, X_scaled: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """
        Apply outlier treatment to scaled array.
        
        Parameters
        ----------
        X_scaled : np.ndarray
            Scaled data to apply treatment to.
        X_original : np.ndarray
            Original unscaled data (needed for some treatments).
            
        Returns
        -------
        np.ndarray
            Data with outlier treatment applied.
        """
        X_treated = X_scaled.copy()
        for i in range(self.n_features_):
            if i in self.outlier_info_ and 'combined' in self.outlier_info_[i]:
                combined_mask = self.outlier_info_[i]['combined']
            else:
                continue
            if self.outlier_treatment == 'preserve':
                pass
            elif self.outlier_treatment == 'clip':
                if self.scaling_method == 'iqr':
                    lower_bound = -self.outlier_threshold
                    upper_bound = self.outlier_threshold
                elif self.scaling_method == 'mad':
                    lower_bound = -self.outlier_threshold
                    upper_bound = self.outlier_threshold
                else:
                    lower_bound = -self.outlier_threshold
                    upper_bound = self.outlier_threshold
                X_treated = X_treated.astype(float)
                X_treated[combined_mask, i] = np.clip(X_treated[combined_mask, i], lower_bound, upper_bound)
            elif self.outlier_treatment == 'separate_scaling':
                outlier_values = X_original[combined_mask, i]
                if len(outlier_values) > 0:
                    outlier_median = np.median(outlier_values)
                    if self.scaling_method == 'iqr':
                        q1 = np.percentile(outlier_values, 25)
                        q3 = np.percentile(outlier_values, 75)
                        outlier_iqr = q3 - q1
                        outlier_scale = outlier_iqr if outlier_iqr != 0 else 1.0
                    elif self.scaling_method == 'mad':
                        from src.data_lifecycle.mathematical_foundations.statistical_methods.descriptive_statistics import median_absolute_deviation
                        outlier_mad = median_absolute_deviation(outlier_values)
                        outlier_scale = outlier_mad if outlier_mad != 0 else 1.0
                    else:
                        q1 = np.percentile(outlier_values, 25)
                        q3 = np.percentile(outlier_values, 75)
                        outlier_scale = (q3 - q1) / 2 if q3 - q1 != 0 else 1.0
                    X_treated = X_treated.astype(float)
                    X_treated[combined_mask, i] = (outlier_values - outlier_median) / outlier_scale
            elif self.outlier_treatment == 'mask':
                X_treated = X_treated.astype(float)
                X_treated[combined_mask, i] = np.nan
        return X_treated

    def apply_outlier_treatment(self, data: Union[FeatureSet, np.ndarray], treatment: Optional[str]=None) -> FeatureSet:
        """
        Apply outlier treatment to already scaled data.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            The data to apply treatment to.
        treatment : str, optional
            Treatment strategy to use. If None, uses the default strategy.
            
        Returns
        -------
        FeatureSet
            Data with outlier treatment applied.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        strategy = treatment if treatment is not None else self.outlier_treatment
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = np.asarray(data)
            feature_names = getattr(self, 'feature_names_', None)
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_treated = X.copy()
        original_strategy = self.outlier_treatment
        self.outlier_treatment = strategy
        X_treated = self.apply_outlier_treatment_to_array(X_treated, X)
        self.outlier_treatment = original_strategy
        metadata['outlier_treatment_applied'] = strategy
        if isinstance(data, FeatureSet):
            return FeatureSet(features=X_treated, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        else:
            return FeatureSet(features=X_treated, feature_names=feature_names, metadata=metadata)

    def get_outlier_analysis(self) -> Dict:
        """
        Get detailed analysis of detected outliers.
        
        Returns
        -------
        dict
            Dictionary containing outlier analysis with counts, indices, values,
            and agreement information for each feature.
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        analysis = {}
        outlier_counts = {}
        for i in range(self.n_features_):
            feature_analysis = {}
            counts = {}
            for method in self.outlier_detection_methods:
                if method in self.outlier_info_[i]:
                    counts[method] = len(self.outlier_info_[i][method]['indices'])
                    feature_analysis[method] = {'indices': self.outlier_info_[i][method]['indices'], 'values': self.outlier_info_[i][method]['values']}
            if 'combined' in self.outlier_info_[i]:
                counts['combined'] = int(np.sum(self.outlier_info_[i]['combined']))
            if 'full_agreement' in self.outlier_info_[i]:
                feature_analysis['full_agreement'] = self.outlier_info_[i]['full_agreement']
                counts['full_agreement'] = len(self.outlier_info_[i]['full_agreement'])
            feature_analysis['counts'] = counts
            analysis[i] = feature_analysis
            outlier_counts[i] = counts
        analysis['outlier_counts'] = outlier_counts
        return analysis

def robust_scale(data: Union[FeatureSet, np.ndarray], with_centering: bool=True, with_scaling: bool=True, quantile_range: tuple=(25.0, 75.0), unit_variance: bool=False, copy: bool=True) -> Union[FeatureSet, np.ndarray]:
    """
    Scale features using statistics that are robust to outliers.
    
    This function transforms features by subtracting the median and dividing by the
    interquartile range (IQR). It is particularly useful for data with outliers,
    as it uses robust statistics for centering and scaling.
    
    The transformation is given by:
        scaled = (X - median) / IQR
    
    where IQR is the interquartile range (75th quantile - 25th quantile).
    
    Parameters
    ----------
    data : FeatureSet or np.ndarray
        The data to scale. If FeatureSet, the returned object will preserve metadata.
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to unit variance (using IQR).
    quantile_range : tuple (q_min, q_max), default=(25.0, 75.0)
        Quantile range used to calculate the scale. By default, this is equal to the IQR.
    unit_variance : bool, default=False
        If True, scale data so that normally distributed features have a variance of 1.
        In general, if the difference between q_max and q_min is close to 1, the output
        will have approximately unit variance.
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        
    Returns
    -------
    FeatureSet
        Scaled data with preserved metadata if input was FeatureSet, otherwise a new FeatureSet.
        
    Examples
    --------
    >>> import numpy as np
    >>> from general.structures.feature_set import FeatureSet
    >>> from src.data_lifecycle.preprocessing.scaling_normalization.robust_scaling import robust_scale
    >>> 
    >>> # Create sample data with outliers
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [100, 5]])
    >>> feature_set = FeatureSet(features=X, feature_names=['f1', 'f2'])
    >>> 
    >>> # Apply robust scaling
    >>> scaled_features = robust_scale(feature_set)
    """
    if not isinstance(quantile_range, tuple) or len(quantile_range) != 2:
        raise ValueError('quantile_range must be a tuple of two values')
    (q_min, q_max) = quantile_range
    if not 0 <= q_min <= 100 or not 0 <= q_max <= 100 or q_min >= q_max:
        raise ValueError('quantile_range values must be between 0 and 100, and q_min < q_max')
    if isinstance(data, FeatureSet):
        X = data.features
        feature_names = data.feature_names
        feature_types = data.feature_types
        sample_ids = data.sample_ids
        metadata = data.metadata.copy() if data.metadata else {}
        quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        input_is_feature_set = True
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Input array must be 2-dimensional')
        X = data
        feature_names = None
        feature_types = None
        sample_ids = None
        metadata = {}
        quality_scores = {}
        input_is_feature_set = False
    else:
        raise ValueError('Input data must be either a FeatureSet or numpy array')
    if not copy and input_is_feature_set:
        X_result = X
    else:
        X_result = X.copy()
    if with_centering or with_scaling:
        median = np.median(X, axis=0)
        if with_scaling:
            q_lower = np.percentile(X, q_min, axis=0)
            q_upper = np.percentile(X, q_max, axis=0)
            iqr = q_upper - q_lower
            iqr = np.where(iqr == 0, 1.0, iqr)
            if unit_variance:
                scale_factor = (q_max - q_min) / 100.0
                iqr = iqr * scale_factor / 1.34896
    if with_centering:
        X_result -= median
    if with_scaling:
        if with_centering:
            X_result /= iqr
        else:
            q_lower = np.percentile(X, q_min, axis=0)
            q_upper = np.percentile(X, q_max, axis=0)
            iqr = q_upper - q_lower
            iqr = np.where(iqr == 0, 1.0, iqr)
            if unit_variance:
                scale_factor = (q_max - q_min) / 100.0
                iqr = iqr * scale_factor / 1.34896
            X_result /= iqr
    if input_is_feature_set:
        metadata['scaling_method'] = 'robust'
        metadata['robust_scaling_params'] = {'with_centering': with_centering, 'with_scaling': with_scaling, 'quantile_range': quantile_range, 'unit_variance': unit_variance}
        if not copy:
            data.features[:] = X_result
            data.metadata.update(metadata)
            data.quality_scores.update(quality_scores)
            return data
        else:
            return FeatureSet(features=X_result, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
    else:
        return X_result