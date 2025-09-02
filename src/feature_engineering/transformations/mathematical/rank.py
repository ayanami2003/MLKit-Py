from typing import Union, Optional
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import pandas as pd

class RankTransformer(BaseTransformer):
    """
    Transformer that applies rank-based transformation to features.
    
    This transformer converts numerical features into their rank equivalents,
    which can help in reducing the impact of outliers and making the data
    more robust for certain machine learning algorithms. It handles ties
    according to specified methods and can optionally normalize the ranks.
    
    The transformation preserves the relative ordering of values while
    changing their scale to ranks (1, 2, 3, ...), with options for handling
    tied values and normalization.
    
    Attributes
    ----------
    method : str, default='average'
        Method used to assign ranks to tied values. Options are:
        - 'average': Average rank of tied group
        - 'min': Minimum rank of tied group
        - 'max': Maximum rank of tied group
        - 'first': Ranks assigned in order of appearance
        - 'dense': Like 'min', but ranks increase by 1 between groups
    normalize : bool, default=False
        Whether to normalize ranks to [0, 1] range
    ascending : bool, default=True
        Whether to rank in ascending order (True) or descending order (False)
    """

    def __init__(self, method: str='average', normalize: bool=False, ascending: bool=True, name: Optional[str]=None):
        """
        Initialize the RankTransformer.
        
        Parameters
        ----------
        method : str, default='average'
            Method used to assign ranks to tied values. Options are:
            - 'average': Average rank of tied group
            - 'min': Minimum rank of tied group
            - 'max': Maximum rank of tied group
            - 'first': Ranks assigned in order of appearance
            - 'dense': Like 'min', but ranks increase by 1 between groups
        normalize : bool, default=False
            Whether to normalize ranks to [0, 1] range
        ascending : bool, default=True
            Whether to rank in ascending order (True) or descending order (False)
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.normalize = normalize
        self.ascending = ascending
        valid_methods = ['average', 'min', 'max', 'first', 'dense']
        if self.method not in valid_methods:
            raise ValueError(f"Method '{self.method}' is not supported. Valid options are: {valid_methods}")

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'RankTransformer':
        """
        Fit the transformer to the input data.
        
        For rank transformation, fitting doesn't require learning any parameters,
        but we validate the input data and store relevant metadata.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to fit the transformer on. If FeatureSet, uses the features attribute.
        **kwargs : dict
            Additional parameters (ignored for this transformer)
            
        Returns
        -------
        RankTransformer
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
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Apply rank transformation to input data.
        
        Converts each feature to its rank equivalent, handling ties according
        to the specified method. Optionally normalizes ranks to [0, 1].
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform. If FeatureSet, transforms the features attribute.
        **kwargs : dict
            Additional parameters (ignored for this transformer)
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Transformed data with ranks. Type matches input type.
        """
        if not hasattr(self, 'n_features_'):
            raise ValueError("This transformer has not been fitted yet. Call 'fit' before using this method.")
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            is_feature_set = True
        else:
            X = data.copy()
            is_feature_set = False
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet with numpy array features')
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features but got {X.shape[1]}')
        ranked_data = np.empty_like(X, dtype=np.float64)
        for i in range(self.n_features_):
            series = pd.Series(X[:, i])
            if self.method == 'dense':
                ranked_series = self._dense_rank(series, ascending=self.ascending)
            else:
                ranked_series = series.rank(method=self.method, ascending=self.ascending)
            if self.normalize:
                n_unique = ranked_series.nunique()
                if n_unique > 1:
                    if self.method == 'dense':
                        max_rank = n_unique
                        min_rank = 1
                    else:
                        max_rank = ranked_series.max()
                        min_rank = ranked_series.min()
                    ranked_series = (ranked_series - min_rank) / (max_rank - min_rank)
                else:
                    ranked_series = pd.Series([0.0] * len(ranked_series), index=ranked_series.index)
            ranked_data[:, i] = ranked_series.values
        if is_feature_set:
            result = FeatureSet(features=ranked_data, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)
            return result
        else:
            return ranked_data

    def _dense_rank(self, series: pd.Series, ascending: bool=True) -> pd.Series:
        """
        Perform dense ranking on a pandas Series.
        
        Dense ranking assigns the same rank to tied values, but the next rank
        increases by 1 (no gaps).
        
        Parameters
        ----------
        series : pd.Series
            Input series to rank
        ascending : bool, default=True
            Whether to rank in ascending order
            
        Returns
        -------
        pd.Series
            Densely ranked series
        """
        unique_values = np.sort(pd.unique(series.dropna()))
        if not ascending:
            unique_values = unique_values[::-1]
        value_to_rank = {value: rank + 1 for (rank, value) in enumerate(unique_values)}
        ranked_series = series.map(value_to_rank)
        ranked_series[series.isna()] = np.nan
        return ranked_series

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> Union[FeatureSet, np.ndarray]:
        """
        Inverse transform is not supported for rank transformation.
        
        Rank transformation is not invertible since the original values
        are lost during the transformation process.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        Union[FeatureSet, np.ndarray]
            Same as input data (identity operation)
            
        Raises
        ------
        NotImplementedError
            Always raised since inverse transformation is not possible
        """
        raise NotImplementedError('Inverse transformation is not supported for rank transformation as original values are lost.')