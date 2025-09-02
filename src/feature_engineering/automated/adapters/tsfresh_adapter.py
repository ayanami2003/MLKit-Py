from typing import Optional, Dict, Any, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import pandas as pd
import numpy as np


# ...(code omitted)...


class TSFreshFeatureExtractor(BaseTransformer):
    """
    Adapter for integrating TSFresh library for automated time series feature extraction.
    
    This transformer wraps the TSFresh library to extract comprehensive time series features
    from temporal data. It supports various TSFresh configuration options and handles
    the conversion between MLKit-Py data structures and TSFresh-compatible formats.
    
    The extractor can process both univariate and multivariate time series data,
    generating hundreds of interpretable statistical features that capture temporal patterns.
    
    Attributes
    ----------
    default_fc_parameters : Optional[Dict[str, Any]]
        TSFresh feature calculation parameters. If None, uses TSFresh's default settings.
    kind_to_fc_parameters : Optional[Dict[str, Dict[str, Any]]]
        Per-kind feature calculation parameters for multivariate series.
    column_id : str
        Name of the column containing entity identifiers.
    column_time : str
        Name of the column containing time stamps.
    column_value : str
        Name of the column containing time series values.
    column_kind : Optional[str]
        Name of the column containing kind identifiers for multivariate series.
    n_jobs : int
        Number of parallel processes to use for feature extraction.
    """

    def __init__(self, default_fc_parameters: Optional[Dict[str, Any]]=None, kind_to_fc_parameters: Optional[Dict[str, Dict[str, Any]]]=None, column_id: str='id', column_time: str='time', column_value: str='value', column_kind: Optional[str]=None, n_jobs: int=1, name: Optional[str]=None):
        """
        Initialize the TSFresh adapter.
        
        Parameters
        ----------
        default_fc_parameters : Optional[Dict[str, Any]]
            TSFresh feature calculation parameters. If None, uses TSFresh's default settings.
        kind_to_fc_parameters : Optional[Dict[str, Dict[str, Any]]]
            Per-kind feature calculation parameters for multivariate series.
        column_id : str
            Name of the column containing entity identifiers.
        column_time : str
            Name of the column containing time stamps.
        column_value : str
            Name of the column containing time series values.
        column_kind : Optional[str]
            Name of the column containing kind identifiers for multivariate series.
        n_jobs : int
            Number of parallel processes to use for feature extraction.
        name : Optional[str]
            Name of the transformer component.
        """
        super().__init__(name=name)
        if not TSFRESH_AVAILABLE:
            raise ImportError("TSFresh is not installed. Please install it with 'pip install tsfresh'")
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.column_id = column_id
        self.column_time = column_time
        self.column_value = column_value
        self.column_kind = column_kind
        self.n_jobs = n_jobs
        self._fitted = False

    def fit(self, data: Union[DataBatch, FeatureSet], **kwargs) -> 'TSFreshFeatureExtractor':
        """
        Fit the transformer to the input time series data.
        
        This method validates the input data format and prepares the extractor
        for feature extraction. For TSFresh, fitting primarily involves
        validating column names and data structure.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input time series data in long format with id, time, and value columns.
        **kwargs : dict
            Additional fitting parameters (ignored for TSFresh).
            
        Returns
        -------
        TSFreshFeatureExtractor
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If required columns are missing or data format is invalid.
        """
        df = self._convert_to_dataframe(data)
        required_columns = [self.column_id, self.column_time, self.column_value]
        if self.column_kind is not None:
            required_columns.append(self.column_kind)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')
        if df.empty:
            raise ValueError('Input data is empty')
        if not pd.api.types.is_numeric_dtype(df[self.column_time]):
            raise ValueError(f"Column '{self.column_time}' must be numeric")
        if not pd.api.types.is_numeric_dtype(df[self.column_value]):
            raise ValueError(f"Column '{self.column_value}' must be numeric")
        self._fitted = True
        return self

    def transform(self, data: Union[DataBatch, FeatureSet], **kwargs) -> FeatureSet:
        """
        Extract time series features using TSFresh.
        
        This method applies TSFresh feature extraction to the input time series data,
        generating a wide set of statistical features for each time series entity.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet]
            Input time series data in long format with id, time, and value columns.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Extracted features with entity identifiers preserved.
            
        Raises
        ------
        ValueError
            If data format is incompatible or required columns are missing.
        RuntimeError
            If feature extraction fails due to TSFresh internal errors.
        """
        if not self._fitted:
            raise ValueError('Transformer must be fitted before transform. Call fit() first.')
        df = self._convert_to_dataframe(data)
        required_columns = [self.column_id, self.column_time, self.column_value]
        if self.column_kind is not None:
            required_columns.append(self.column_kind)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')
        try:
            extracted_features = extract_features(df, column_id=self.column_id, column_sort=self.column_time, column_value=self.column_value, column_kind=self.column_kind, default_fc_parameters=self.default_fc_parameters, kind_to_fc_parameters=self.kind_to_fc_parameters, n_jobs=self.n_jobs)
            extracted_features_imputed = impute(extracted_features)
            feature_names = extracted_features_imputed.columns.tolist()
            sample_ids = extracted_features_imputed.index.tolist()
            feature_set = FeatureSet(features=extracted_features_imputed.values, feature_names=feature_names, sample_ids=[str(sid) for sid in sample_ids], feature_types=['numeric'] * len(feature_names), metadata={'transformer_name': self.name or 'TSFreshFeatureExtractor', 'column_id': self.column_id, 'column_time': self.column_time, 'column_value': self.column_value, 'column_kind': self.column_kind})
            return feature_set
        except Exception as e:
            raise RuntimeError(f'TSFresh feature extraction failed: {str(e)}') from e

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        TSFresh feature extraction is not invertible.
        
        This method raises NotImplementedError as time series feature extraction
        is a irreversible transformation that loses the original temporal structure.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Feature data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Never returns - always raises exception.
            
        Raises
        ------
        NotImplementedError
            Always raised as inversion is not possible.
        """
        raise NotImplementedError('TSFresh feature extraction is not invertible. The original time series cannot be reconstructed from extracted features.')

    def _convert_to_dataframe(self, data: Union[DataBatch, FeatureSet, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert MLKit-Py data structures to pandas DataFrame compatible with TSFresh.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, pd.DataFrame]
            Input data to convert.
            
        Returns
        -------
        pd.DataFrame
            DataFrame in TSFresh-compatible format.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, DataBatch):
            if isinstance(data.data, pd.DataFrame):
                return data.data.copy()
            elif isinstance(data.data, np.ndarray):
                if data.data.ndim == 2 and data.feature_names:
                    return pd.DataFrame(data.data, columns=data.feature_names)
                else:
                    return pd.DataFrame(data.data)
            else:
                return pd.DataFrame(data.data)
        elif isinstance(data, FeatureSet):
            df = pd.DataFrame(data.features)
            if data.feature_names:
                df.columns = data.feature_names
            if data.sample_ids:
                df.index = data.sample_ids
            return df
        else:
            return pd.DataFrame(data)