from typing import Any, Dict, Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import Any, Dict, Optional, Union, List
import numpy as np

class DictionaryMapper(BaseTransformer):
    """
    A transformer that maps values in data using a provided dictionary.
    
    This component applies a mapping dictionary to transform values in specified columns
    or features of a dataset. It supports both forward mapping and inverse mapping when
    an inverse dictionary is provided.
    
    Attributes
    ----------
    mapping_dict : Dict[Any, Any]
        Dictionary defining the value mappings (original -> mapped)
    inverse_mapping_dict : Optional[Dict[Any, Any]]
        Dictionary defining inverse mappings (mapped -> original), if available
    columns : Optional[Union[str, list]]
        Column name(s) to apply mapping to. If None, applies to all applicable columns
    handle_unknown : str
        How to handle unknown values. Options: 'preserve' (keep as-is), 'ignore' (set to None)
    name : Optional[str]
        Name of the transformer instance
        
    Methods
    -------
    fit(data, **kwargs)
        Fit the transformer to the data (identifies columns to transform)
    transform(data, **kwargs)
        Apply the dictionary mapping to the data
    inverse_transform(data, **kwargs)
        Apply the inverse dictionary mapping if available
    """

    def __init__(self, mapping_dict: Dict[Any, Any], inverse_mapping_dict: Optional[Dict[Any, Any]]=None, columns: Optional[Union[str, list]]=None, handle_unknown: str='preserve', name: Optional[str]=None):
        """
        Initialize the DictionaryMapper transformer.
        
        Parameters
        ----------
        mapping_dict : Dict[Any, Any]
            Dictionary defining the value mappings (original -> mapped)
        inverse_mapping_dict : Optional[Dict[Any, Any]], default=None
            Dictionary defining inverse mappings (mapped -> original)
        columns : Optional[Union[str, list]], default=None
            Column name(s) to apply mapping to. If None, applies to all applicable columns
        handle_unknown : str, default='preserve'
            How to handle unknown values. Options: 'preserve', 'ignore'
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.mapping_dict = mapping_dict
        self.inverse_mapping_dict = inverse_mapping_dict
        self.columns = columns
        if handle_unknown not in ['preserve', 'ignore']:
            raise ValueError("handle_unknown must be either 'preserve' or 'ignore'")
        self.handle_unknown = handle_unknown
        self._fitted_columns: List[str] = []

    def fit(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> 'DictionaryMapper':
        """
        Fit the transformer to the input data.
        
        This method identifies which columns to transform based on the data structure
        and the specified columns parameter.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Input data to fit the transformer on
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        DictionaryMapper
            Self instance for method chaining
        """
        self._fitted_columns = []
        if isinstance(data, DataBatch):
            if data.feature_names is not None:
                if self.columns is None:
                    self._fitted_columns = data.feature_names
                elif isinstance(self.columns, str):
                    if self.columns in data.feature_names:
                        self._fitted_columns = [self.columns]
                    else:
                        raise ValueError(f"Column '{self.columns}' not found in DataBatch feature_names")
                elif isinstance(self.columns, list):
                    for col in self.columns:
                        if col not in data.feature_names:
                            raise ValueError(f"Column '{col}' not found in DataBatch feature_names")
                    self._fitted_columns = self.columns
            elif self.columns is None:
                if hasattr(data.data, 'shape') and len(data.data.shape) > 1:
                    self._fitted_columns = [str(i) for i in range(data.data.shape[1])]
                else:
                    self._fitted_columns = ['0']
            elif isinstance(self.columns, str):
                self._fitted_columns = [self.columns]
            elif isinstance(self.columns, list):
                self._fitted_columns = [str(col) for col in self.columns]
        elif isinstance(data, FeatureSet):
            if data.feature_names is not None:
                if self.columns is None:
                    self._fitted_columns = data.feature_names
                elif isinstance(self.columns, str):
                    if self.columns in data.feature_names:
                        self._fitted_columns = [self.columns]
                    else:
                        raise ValueError(f"Column '{self.columns}' not found in FeatureSet feature_names")
                elif isinstance(self.columns, list):
                    for col in self.columns:
                        if col not in data.feature_names:
                            raise ValueError(f"Column '{col}' not found in FeatureSet feature_names")
                    self._fitted_columns = self.columns
            else:
                n_features = data.features.shape[1] if len(data.features.shape) > 1 else 1
                if self.columns is None:
                    self._fitted_columns = [str(i) for i in range(n_features)]
                elif isinstance(self.columns, str):
                    self._fitted_columns = [self.columns]
                elif isinstance(self.columns, list):
                    self._fitted_columns = [str(col) for col in self.columns]
        elif self.columns is None:
            if hasattr(data, 'shape') and len(data.shape) > 1:
                self._fitted_columns = [str(i) for i in range(data.shape[1])]
            elif hasattr(data, '__len__') and len(data) > 0 and hasattr(data[0], '__len__'):
                self._fitted_columns = [str(i) for i in range(len(data[0]))]
            else:
                self._fitted_columns = ['0']
        elif isinstance(self.columns, str):
            self._fitted_columns = [self.columns]
        elif isinstance(self.columns, list):
            self._fitted_columns = [str(col) for col in self.columns]
        return self

    def transform(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> Union[DataBatch, FeatureSet, Any]:
        """
        Apply the dictionary mapping to input data.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Input data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        Union[DataBatch, FeatureSet, Any]
            Transformed data with values mapped according to mapping_dict
        """
        if not self._fitted_columns:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")

        def map_value(value):
            if value in self.mapping_dict:
                return self.mapping_dict[value]
            elif self.handle_unknown == 'preserve':
                return value
            else:
                return None
        if isinstance(data, DataBatch):
            transformed_data = self._transform_data_batch(data, map_value)
            return transformed_data
        elif isinstance(data, FeatureSet):
            transformed_data = self._transform_feature_set(data, map_value)
            return transformed_data
        else:
            return self._transform_generic_data(data, map_value)

    def inverse_transform(self, data: Union[DataBatch, FeatureSet, Any], **kwargs) -> Union[DataBatch, FeatureSet, Any]:
        """
        Apply the inverse dictionary mapping if available.
        
        Parameters
        ----------
        data : Union[DataBatch, FeatureSet, Any]
            Transformed data to invert
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        Union[DataBatch, FeatureSet, Any]
            Data with values mapped back according to inverse_mapping_dict
            
        Raises
        ------
        ValueError
            If inverse_mapping_dict is not provided
        """
        if not self._fitted_columns:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.inverse_mapping_dict is None:
            raise ValueError('inverse_mapping_dict is not provided, cannot perform inverse transformation')

        def inverse_map_value(value):
            if value in self.inverse_mapping_dict:
                return self.inverse_mapping_dict[value]
            elif self.handle_unknown == 'preserve':
                return value
            else:
                return None
        if isinstance(data, DataBatch):
            transformed_data = self._transform_data_batch(data, inverse_map_value, is_inverse=True)
            return transformed_data
        elif isinstance(data, FeatureSet):
            transformed_data = self._transform_feature_set(data, inverse_map_value, is_inverse=True)
            return transformed_data
        else:
            return self._transform_generic_data(data, inverse_map_value)

    def _transform_data_batch(self, data: DataBatch, map_func, is_inverse: bool=False) -> DataBatch:
        """Helper method to transform DataBatch objects."""
        new_data = data.data.copy() if hasattr(data.data, 'copy') else data.data
        if data.feature_names is not None and len(self._fitted_columns) > 0:
            for col_name in self._fitted_columns:
                if col_name in data.feature_names:
                    col_idx = data.feature_names.index(col_name)
                    if hasattr(new_data, 'shape') and len(new_data.shape) > 1:
                        for row_idx in range(new_data.shape[0]):
                            new_data[row_idx, col_idx] = map_func(new_data[row_idx, col_idx])
                    else:
                        new_data[col_idx] = map_func(new_data[col_idx])
        else:
            col_indices = [int(col) if col.isdigit() else 0 for col in self._fitted_columns]
            if hasattr(new_data, 'shape') and len(new_data.shape) > 1:
                for col_idx in col_indices:
                    if col_idx < new_data.shape[1]:
                        for row_idx in range(new_data.shape[0]):
                            new_data[row_idx, col_idx] = map_func(new_data[row_idx, col_idx])
            elif hasattr(new_data, '__len__') and len(new_data) > 0:
                if hasattr(new_data[0], '__len__'):
                    for col_idx in col_indices:
                        if col_idx < len(new_data[0]):
                            for row_idx in range(len(new_data)):
                                new_data[row_idx][col_idx] = map_func(new_data[row_idx][col_idx])
                else:
                    for col_idx in col_indices:
                        if col_idx < len(new_data):
                            new_data[col_idx] = map_func(new_data[col_idx])
        return DataBatch(data=new_data, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)

    def _transform_feature_set(self, data: FeatureSet, map_func, is_inverse: bool=False) -> FeatureSet:
        """Helper method to transform FeatureSet objects."""
        new_features = data.features.copy()
        if data.feature_names is not None and len(self._fitted_columns) > 0:
            for col_name in self._fitted_columns:
                if col_name in data.feature_names:
                    col_idx = data.feature_names.index(col_name)
                    new_features[:, col_idx] = np.array([map_func(val) for val in new_features[:, col_idx]])
        else:
            col_indices = [int(col) if col.isdigit() else 0 for col in self._fitted_columns]
            for col_idx in col_indices:
                if col_idx < new_features.shape[1]:
                    new_features[:, col_idx] = np.array([map_func(val) for val in new_features[:, col_idx]])
        return FeatureSet(features=new_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata, quality_scores=data.quality_scores)

    def _transform_generic_data(self, data: Any, map_func) -> Any:
        """Helper method to transform generic data structures."""
        if hasattr(data, 'copy'):
            new_data = data.copy()
        else:
            new_data = data
        col_indices = [int(col) if col.isdigit() else 0 for col in self._fitted_columns]
        if hasattr(new_data, 'shape') and len(new_data.shape) > 1:
            for col_idx in col_indices:
                if col_idx < new_data.shape[1]:
                    for row_idx in range(new_data.shape[0]):
                        new_data[row_idx, col_idx] = map_func(new_data[row_idx, col_idx])
        elif hasattr(new_data, '__len__') and len(new_data) > 0:
            if hasattr(new_data[0], '__len__'):
                for col_idx in col_indices:
                    if col_idx < len(new_data[0]):
                        for row_idx in range(len(new_data)):
                            new_data[row_idx][col_idx] = map_func(new_data[row_idx][col_idx])
            else:
                for col_idx in col_indices:
                    if col_idx < len(new_data):
                        new_data[col_idx] = map_func(new_data[col_idx])
        elif '0' in self._fitted_columns or 0 in col_indices:
            new_data = map_func(new_data)
        return new_data