from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class TrendExtractor(BaseTransformer):

    def __init__(self, method: str='ma', window_size: Optional[int]=None, poly_degree: int=2, name: Optional[str]=None):
        """
        Initialize the TrendExtractor.
        
        Parameters
        ----------
        method : str, default='ma'
            Trend extraction method to use. Options include:
            - 'ma': Moving average
            - 'polynomial': Polynomial fitting
            - 'stl': Seasonal-Trend decomposition using Loess
        window_size : int, optional
            Window size for moving average methods. Required for 'ma' method.
        poly_degree : int, default=2
            Degree of polynomial for polynomial fitting methods.
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name)
        self.method = method
        self.window_size = window_size
        self.poly_degree = poly_degree
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'TrendExtractor':
        """
        Fit the trend extractor to the input data.
        
        This method validates the input data and prepares any necessary
        parameters for trend extraction.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing time-series or sequential observations
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        TrendExtractor
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If required parameters are missing or data format is invalid
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self._feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            X = data
            self._feature_names = None
        else:
            raise ValueError('Input data must be a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        if self.method == 'ma':
            if self.window_size is None:
                raise ValueError('window_size is required for moving average method')
            if not isinstance(self.window_size, int) or self.window_size <= 0:
                raise ValueError('window_size must be a positive integer')
            if self.window_size > X.shape[0]:
                self.window_size = X.shape[0]
        elif self.method == 'polynomial':
            if not isinstance(self.poly_degree, int) or self.poly_degree < 0:
                raise ValueError('poly_degree must be a non-negative integer')
        elif self.method == 'stl':
            if X.shape[0] < 4:
                raise ValueError('STL method requires at least 4 observations')
        else:
            raise ValueError(f'Unsupported method: {self.method}')
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        self.is_fitted_ = True
        if self.method == 'polynomial':
            self._poly_models = []
            self._poly_features = PolynomialFeatures(degree=self.poly_degree)
            X_poly = self._poly_features.fit_transform(np.arange(self.n_samples_).reshape(-1, 1))
            for i in range(self.n_features_):
                model = LinearRegression()
                model.fit(X_poly, X[:, i])
                self._poly_models.append(model)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Extract trend components from the input data.
        
        Applies the specified trend extraction method to identify and
        extract underlying trends in the data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing time-series or sequential observations
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            FeatureSet containing extracted trend components
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or data format is invalid
        """
        if not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be a FeatureSet or numpy array')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError('Input data must be 1D or 2D')
        if X.shape[0] != self.n_samples_:
            raise ValueError('Number of samples in data does not match fitted data')
        if X.shape[1] != self.n_features_:
            raise ValueError('Number of features in data does not match fitted data')
        trends = np.zeros_like(X)
        if self.method == 'ma':
            for i in range(self.n_features_):
                series = X[:, i]
                if self.window_size >= len(series):
                    trends[:, i] = np.mean(series)
                else:
                    trends[:, i] = np.convolve(series, np.ones(self.window_size) / self.window_size, mode='same')
                    half_window = self.window_size // 2
                    for j in range(half_window):
                        trends[j, i] = np.mean(series[:j + half_window + 1])
                        trends[-(j + 1), i] = np.mean(series[-(j + half_window + 1):])
        elif self.method == 'polynomial':
            X_poly = self._poly_features.transform(np.arange(self.n_samples_).reshape(-1, 1))
            for i in range(self.n_features_):
                trends[:, i] = self._poly_models[i].predict(X_poly)
        elif self.method == 'stl':
            for i in range(self.n_features_):
                series = X[:, i]
                window = min(max(5, len(series) // 10 * 2 + 1), len(series))
                if window % 2 == 0:
                    window += 1
                if window >= len(series):
                    trends[:, i] = np.mean(series)
                else:
                    trends[:, i] = np.convolve(series, np.ones(window) / window, mode='same')
                    half_window = window // 2
                    for j in range(half_window):
                        trends[j, i] = np.mean(series[:j + half_window + 1])
                        trends[-(j + 1), i] = np.mean(series[-(j + half_window + 1):])
        if self._feature_names is not None:
            trend_feature_names = [f'{name}_trend' if not name.endswith('_trend') else name for name in self._feature_names]
        else:
            trend_feature_names = [f'feature_{i}_trend' for i in range(self.n_features_)]
        return FeatureSet(features=trends, feature_names=trend_feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reconstruct original data from trend components (if supported).
        
        Attempts to reconstruct the original data from the extracted trends.
        Not all methods support inverse transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Trend components to reconstruct from
        **kwargs : dict
            Additional inverse transformation parameters
            
        Returns
        -------
        FeatureSet
            Reconstructed data if inverse transformation is supported
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported for the method used
        """
        if not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet.')
        if self.method == 'polynomial':
            if isinstance(data, FeatureSet):
                return FeatureSet(features=data.features, feature_names=data.feature_names)
            else:
                return FeatureSet(features=data)
        else:
            raise NotImplementedError(f"Inverse transform not supported for method '{self.method}'")