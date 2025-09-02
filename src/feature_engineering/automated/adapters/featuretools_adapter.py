from typing import Optional, Dict, Any, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import numpy as np


# ...(code omitted)...


class FeatureToolsAdapter(BaseTransformer):
    """
    Adapter for integrating FeatureTools library for automated feature engineering.
    
    This class provides a standardized interface to leverage FeatureTools' automated
    feature engineering capabilities within the MLKit-Py framework. It transforms
    raw datasets into rich feature sets by automatically generating features based
    on detected relationships and entity structures.
    
    The adapter handles the conversion between MLKit-Py's internal data structures
    and FeatureTools' EntitySet format, executes feature generation, and returns
    results in a consistent FeatureSet format.
    
    Attributes
    ----------
    max_depth : int, default=2
        Maximum depth of features to generate
    trans_primitives : List[str], optional
        List of transform primitives to use
    agg_primitives : List[str], optional
        List of aggregation primitives to use
    ignore_entities : List[str], optional
        Entities to ignore during feature generation
    ignore_variables : Dict[str, List[str]], optional
        Variables to ignore per entity
    seed_features : List, optional
        Custom seed features to use
    drop_contains : List[str], optional
        Drop features that contain these strings
    drop_exact : List[str], optional
        Drop features that match these strings exactly
    where_primitives : List[str], optional
        Primitives to use with where clauses
    max_features : int, default=1000
        Maximum number of features to generate
    """

    def __init__(self, max_depth: int=2, trans_primitives: Optional[List[str]]=None, agg_primitives: Optional[List[str]]=None, ignore_entities: Optional[List[str]]=None, ignore_variables: Optional[Dict[str, List[str]]]=None, seed_features: Optional[List]=None, drop_contains: Optional[List[str]]=None, drop_exact: Optional[List[str]]=None, where_primitives: Optional[List[str]]=None, max_features: int=1000, name: Optional[str]=None):
        """
        Initialize the FeatureTools adapter.
        
        Parameters
        ----------
        max_depth : int, default=2
            Maximum depth of features to generate
        trans_primitives : List[str], optional
            List of transform primitives to use
        agg_primitives : List[str], optional
            List of aggregation primitives to use
        ignore_entities : List[str], optional
            Entities to ignore during feature generation
        ignore_variables : Dict[str, List[str]], optional
            Variables to ignore per entity
        seed_features : List, optional
            Custom seed features to use
        drop_contains : List[str], optional
            Drop features that contain these strings
        drop_exact : List[str], optional
            Drop features that match these strings exactly
        where_primitives : List[str], optional
            Primitives to use with where clauses
        max_features : int, default=1000
            Maximum number of features to generate
        name : str, optional
            Name of the transformer
        """
        super().__init__(name=name)
        if not FEATURETOOLS_AVAILABLE:
            raise ImportError("FeatureTools is not installed. Please install it with 'pip install featuretools'")
        self.max_depth = max_depth
        self.trans_primitives = trans_primitives or []
        self.agg_primitives = agg_primitives or []
        self.ignore_entities = ignore_entities or []
        self.ignore_variables = ignore_variables or {}
        self.seed_features = seed_features or []
        self.drop_contains = drop_contains or []
        self.drop_exact = drop_exact or []
        self.where_primitives = where_primitives or []
        self.max_features = max_features
        self._entityset = None
        self._feature_defs = None

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'FeatureToolsAdapter':
        """
        Fit the adapter by analyzing the data structure and preparing for feature generation.
        
        This method analyzes the input data to understand its structure and prepares
        the internal FeatureTools EntitySet for feature generation. For DataBatch inputs,
        it will attempt to infer relationships between different data entities.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to analyze and prepare for feature engineering
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        FeatureToolsAdapter
            Self instance for method chaining
        """
        self._entityset = ft.EntitySet(id='mlkit_entityset')
        if isinstance(data, FeatureSet):
            entity_df = data.features
            feature_names = data.feature_names or [f'feature_{i}' for i in range(entity_df.shape[1])]
            sample_ids = data.sample_ids or [f'sample_{i}' for i in range(entity_df.shape[0])]
            import pandas as pd
            df = pd.DataFrame(entity_df, columns=feature_names)
            df.index = sample_ids
            df.index.name = 'index'
            self._entityset = self._entityset.add_dataframe(dataframe_name='main_entity', dataframe=df, index='index', make_index=False)
        elif isinstance(data, DataBatch):
            import pandas as pd
            if isinstance(data.data, list):
                df = pd.DataFrame(data.data)
            else:
                df = pd.DataFrame(data.data)
            if data.feature_names:
                df.columns = data.feature_names
            if data.sample_ids:
                df.index = data.sample_ids
                df.index.name = 'index'
            else:
                df.index.name = 'index'
                if df.index.name is None:
                    df.reset_index(drop=True, inplace=True)
                    df.index.name = 'index'
            self._entityset = self._entityset.add_dataframe(dataframe_name='main_entity', dataframe=df, index='index', make_index=not data.sample_ids)
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Generate new features using FeatureTools based on the fitted data structure.
        
        This method applies FeatureTools' automated feature engineering to generate
        new features from the input data. It returns a FeatureSet containing both
        the original features and the newly generated ones.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to generate features from
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Enhanced feature set with newly generated features
        """
        if self._entityset is None:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        (feature_matrix, feature_defs) = ft.dfs(entityset=self._entityset, target_dataframe_name='main_entity', trans_primitives=self.trans_primitives, agg_primitives=self.agg_primitives, max_depth=self.max_depth, ignore_dataframes=self.ignore_entities, ignore_columns=self.ignore_variables, seed_features=self.seed_features, drop_contains=self.drop_contains, drop_exact=self.drop_exact, where_primitives=self.where_primitives, max_features=self.max_features, features_only=False)
        self._feature_defs = feature_defs
        features = feature_matrix.values
        feature_names = list(feature_matrix.columns)
        sample_ids = list(feature_matrix.index)
        feature_types = []
        for col in feature_matrix.columns:
            dtype = feature_matrix[col].dtype
            if dtype.kind in 'biufc':
                feature_types.append('numeric')
            elif dtype.kind in 'OSU':
                feature_types.append('categorical')
            else:
                feature_types.append('unknown')
        metadata = {}
        quality_scores = {}
        if isinstance(data, FeatureSet):
            metadata = data.metadata or {}
            quality_scores = data.quality_scores or {}
        elif isinstance(data, DataBatch):
            metadata = data.metadata or {}
        metadata['featuretools_features'] = {'count': len(feature_defs), 'max_depth': self.max_depth}
        return FeatureSet(features=features, feature_names=feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the transformation (not applicable for feature generation).
        
        Since feature generation is a one-way process that creates new information,
        this method simply returns the input data without modification.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to pass through
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original input data unchanged
        """
        if isinstance(data, FeatureSet):
            return data
        elif isinstance(data, DataBatch):
            import numpy as np
            features = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data
            return FeatureSet(features=features, feature_names=data.feature_names, sample_ids=data.sample_ids, metadata=data.metadata)
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')

    def get_feature_definitions(self) -> Optional[List]:
        """
        Get the FeatureTools feature definitions generated during transformation.
        
        Returns
        -------
        List, optional
            List of FeatureTools feature definitions
        """
        return self._feature_defs