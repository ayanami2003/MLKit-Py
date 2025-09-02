from typing import Optional, Union
import numpy as np
from scipy import stats
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.preprocessing.scaling_normalization.standard_scaling import ZScoreNormalizer
from src.data_lifecycle.preprocessing.scaling_normalization.minmax_scaling import MinMaxScaler
from src.data_lifecycle.preprocessing.scaling_normalization.robust_scaling import AdaptiveRobustScaler

class AdaptiveDataScaler(BaseTransformer):

    def __init__(self, strategy: str='adaptive', adaptivity_threshold: float=0.5, outlier_threshold: float=3.0, name: Optional[str]=None):
        """
        Initialize the AdaptiveDataScaler.
        
        Parameters
        ----------
        strategy : str, default='adaptive'
            Base scaling strategy to use ('adaptive' enables automatic selection)
        adaptivity_threshold : float, default=0.5
            Threshold for triggering adaptive behavior based on data statistics
        outlier_threshold : float, default=3.0
            Z-score threshold for identifying outliers during adaptive scaling
        name : Optional[str], default=None
            Name for the transformer instance
        """
        super().__init__(name=name)
        self.strategy = strategy
        self.adaptivity_threshold = adaptivity_threshold
        self.outlier_threshold = outlier_threshold
        self._scaler_params = {}
        self._fitted_scaler = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'AdaptiveDataScaler':
        """
        Analyze the input data distribution and configure scaling parameters.
        
        This method examines the data to determine optimal scaling approaches,
        considering factors like skewness, kurtosis, and outlier presence.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to analyze for scaling configuration
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        AdaptiveDataScaler
            Self instance for method chaining
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
        if n_samples == 0 or n_features == 0:
            selected_strategy = 'standard'
            self._fitted_scaler = ZScoreNormalizer()
            self._fitted_scaler.fit(data)
            self._scaler_params = {'strategy': selected_strategy, 'n_samples': n_samples, 'n_features': n_features}
            return self
        if n_samples == 1:
            selected_strategy = 'minmax'
            self._fitted_scaler = MinMaxScaler()
            self._fitted_scaler.fit(data)
            self._scaler_params = {'strategy': selected_strategy, 'n_samples': n_samples, 'n_features': n_features}
            return self
        skewness = np.full(n_features, 0.0)
        kurtosis = np.full(n_features, 0.0)
        outlier_ratios = np.full(n_features, 0.0)
        for i in range(n_features):
            feature_data = X[:, i]
            valid_mask = np.isfinite(feature_data)
            valid_data = feature_data[valid_mask]
            if len(valid_data) == 0:
                skewness[i] = 0.0
                kurtosis[i] = 0.0
                outlier_ratios[i] = 0.0
                continue
            if len(valid_data) == 1:
                skewness[i] = 0.0
                kurtosis[i] = 0.0
                outlier_ratios[i] = 0.0
                continue
            if len(valid_data) > 2:
                try:
                    skewness[i] = stats.skew(valid_data)
                    if np.isnan(skewness[i]) or np.isinf(skewness[i]):
                        skewness[i] = 0.0
                except Exception:
                    skewness[i] = 0.0
                try:
                    kurtosis[i] = stats.kurtosis(valid_data)
                    if np.isnan(kurtosis[i]) or np.isinf(kurtosis[i]):
                        kurtosis[i] = 0.0
                except Exception:
                    kurtosis[i] = 0.0
            else:
                skewness[i] = 0.0
                kurtosis[i] = 0.0
            std_val = np.std(valid_data)
            if std_val > 0 and len(valid_data) > 1:
                mean_val = np.mean(valid_data)
                z_scores = np.abs((valid_data - mean_val) / std_val)
                outlier_count = np.sum(z_scores > self.outlier_threshold)
                outlier_ratios[i] = outlier_count / len(valid_data)
            else:
                outlier_ratios[i] = 0.0
        if self.strategy == 'adaptive':
            avg_skewness = np.nanmean(np.abs(skewness))
            avg_outlier_ratio = np.nanmean(outlier_ratios)
            if np.isnan(avg_skewness):
                avg_skewness = 0.0
            if np.isnan(avg_outlier_ratio):
                avg_outlier_ratio = 0.0
            if avg_outlier_ratio <= 0.1 and avg_skewness < 0.5:
                selected_strategy = 'minmax'
            elif avg_outlier_ratio > self.adaptivity_threshold:
                selected_strategy = 'robust'
            elif avg_skewness > self.adaptivity_threshold:
                selected_strategy = 'minmax'
            else:
                selected_strategy = 'standard'
        else:
            selected_strategy = self.strategy
        if selected_strategy == 'standard':
            self._fitted_scaler = ZScoreNormalizer()
        elif selected_strategy == 'minmax':
            self._fitted_scaler = MinMaxScaler()
        elif selected_strategy == 'robust':
            self._fitted_scaler = AdaptiveRobustScaler()
        else:
            raise ValueError(f'Unknown scaling strategy: {selected_strategy}')
        self._fitted_scaler.fit(data)
        self._scaler_params = {'strategy': selected_strategy, 'skewness': skewness, 'kurtosis': kurtosis, 'outlier_ratios': outlier_ratios, 'n_samples': n_samples, 'n_features': n_features}
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply adaptive scaling to the input data based on fitted parameters.
        
        Uses the scaling strategy determined during fitting, applying robust
        techniques when outliers are detected or data is highly skewed.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Scaled feature set with preserved metadata
        """
        if self._fitted_scaler is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' first.")
        return self._fitted_scaler.transform(data)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reverse the adaptive scaling transformation to recover original data scale.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Scaled data to inverse transform
        **kwargs : dict
            Additional inverse transformation parameters
            
        Returns
        -------
        FeatureSet
            Data restored to original scale
        """
        if self._fitted_scaler is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' first.")
        return self._fitted_scaler.inverse_transform(data)