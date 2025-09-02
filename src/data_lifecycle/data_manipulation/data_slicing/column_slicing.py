from typing import List, Union, Optional
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np

class ColumnSlicer(BaseTransformer):

    def __init__(self, columns: Union[List[str], List[int], None]=None, name: Optional[str]=None):
        """
        Initialize the ColumnSlicer transformer.
        
        Parameters
        ----------
        columns : Union[List[str], List[int], None], optional
            Column names (strings) or indices (integers) to select. 
            If None, all columns will be retained.
        name : Optional[str], optional
            Custom name for the transformer instance.
        """
        super().__init__(name=name)
        self.columns = columns
        self._selected_indices = []

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'ColumnSlicer':
        """
        Fit the slicer to the input data by determining column indices.
        
        This method analyzes the input data structure to map column names
        to indices, which are then stored for use in transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data containing columns to slice. Must have identifiable
            column structure (e.g., feature_names attribute).
        **kwargs : dict
            Additional parameters (reserved for future extensions).
            
        Returns
        -------
        ColumnSlicer
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If specified column names are not found in the data.
        TypeError
            If data type is unsupported or lacks column information.
        """
        if isinstance(data, FeatureSet):
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            feature_names = data.feature_names
        else:
            raise TypeError('Unsupported data type. Expected FeatureSet or DataBatch.')
        if self.columns is None:
            if feature_names is not None:
                self._selected_indices = list(range(len(feature_names)))
            elif isinstance(data, DataBatch) and isinstance(data.data, np.ndarray):
                self._selected_indices = list(range(data.data.shape[1]))
            else:
                raise TypeError('Data must have identifiable column structure when columns=None.')
            return self
        self._selected_indices = []
        if feature_names is not None:
            if all((isinstance(col, str) for col in self.columns)):
                for col in self.columns:
                    try:
                        idx = feature_names.index(col)
                        self._selected_indices.append(idx)
                    except ValueError:
                        raise ValueError(f"Column '{col}' not found in data.")
            elif all((isinstance(col, int) for col in self.columns)):
                n_features = len(feature_names)
                for idx in self.columns:
                    if idx < 0 or idx >= n_features:
                        raise IndexError(f'Column index {idx} is out of bounds for data with {n_features} columns.')
                    self._selected_indices.append(idx)
            else:
                raise TypeError('All elements in columns must be either strings or integers.')
        elif isinstance(data, DataBatch) and isinstance(data.data, np.ndarray):
            n_features = data.data.shape[1]
            if all((isinstance(col, int) for col in self.columns)):
                for idx in self.columns:
                    if idx < 0 or idx >= n_features:
                        raise IndexError(f'Column index {idx} is out of bounds for data with {n_features} columns.')
                    self._selected_indices.append(idx)
            else:
                raise TypeError('Column names provided but data has no feature names.')
        else:
            raise TypeError('Data must have identifiable column structure.')
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Apply column slicing to the input data.
        
        Selects only the specified columns from the input data, returning
        a new data structure of the same type with reduced dimensionality.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to slice by columns.
        **kwargs : dict
            Additional parameters (reserved for future extensions).
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            New data structure containing only the selected columns.
            
        Raises
        ------
        RuntimeError
            If transformer has not been fitted before transformation.
        """
        if not hasattr(self, '_selected_indices'):
            raise RuntimeError('Transformer has not been fitted yet.')
        if isinstance(data, FeatureSet):
            sliced_features = data.features[:, self._selected_indices]
            sliced_feature_names = [data.feature_names[i] for i in self._selected_indices] if data.feature_names is not None else None
            sliced_feature_types = [data.feature_types[i] for i in self._selected_indices] if data.feature_types is not None else None
            sliced_sample_ids = data.sample_ids
            sliced_metadata = data.metadata.copy() if data.metadata else {}
            sliced_quality_scores = {name: data.quality_scores[name] for name in sliced_feature_names or [] if name in (data.quality_scores or {})}
            return FeatureSet(features=sliced_features, feature_names=sliced_feature_names, feature_types=sliced_feature_types, sample_ids=sliced_sample_ids, metadata=sliced_metadata, quality_scores=sliced_quality_scores)
        elif isinstance(data, DataBatch):
            if isinstance(data.data, np.ndarray):
                sliced_data = data.data[:, self._selected_indices]
            elif isinstance(data.data, list) and all((hasattr(row, '__getitem__') for row in data.data)):
                sliced_data = [row[self._selected_indices] for row in data.data]
            else:
                raise TypeError('DataBatch.data must be a numpy array or list of subscriptable rows')
            sliced_feature_names = [data.feature_names[i] for i in self._selected_indices] if data.feature_names is not None else None
            return DataBatch(data=sliced_data, labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=sliced_feature_names, batch_id=data.batch_id)
        else:
            raise TypeError('Unsupported data type. Expected FeatureSet or DataBatch.')

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Reverse the column slicing operation (not supported).
        
        Since column slicing is a lossy operation (data is discarded),
        inverse transformation is not mathematically possible without
        storing the original data.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        Union[FeatureSet, DataBatch]
            Always raises NotImplementedError.
            
        Raises
        ------
        NotImplementedError
            Column slicing cannot be inverted due to information loss.
        """
        raise NotImplementedError('Column slicing cannot be inverted due to information loss.')