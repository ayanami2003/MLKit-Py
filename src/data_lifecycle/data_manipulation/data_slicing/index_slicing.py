from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np

class IndexSlicer(BaseTransformer):

    def __init__(self, indices: Union[List[int], slice, None]=None, name: Optional[str]=None):
        """
        Initialize the IndexSlicer.
        
        Args:
            indices (Union[List[int], slice, None]): Row indices to select.
                - List[int]: Explicit row positions to include
                - slice: Range of rows to select (e.g., slice(10, 20))
                - None: Select all rows (identity operation)
            name (Optional[str]): Optional name for the transformer.
        """
        super().__init__(name)
        self.indices = indices

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'IndexSlicer':
        """
        Configure the transformer with input data.
        
        This is a no-op for IndexSlicer as it doesn't require fitting.
        Exists for API consistency with BaseTransformer.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data to configure with.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            IndexSlicer: Self instance for method chaining.
        """
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Slice the input data according to configured indices.
        
        Applies row-based slicing to either a FeatureSet or DataBatch,
        returning a new instance containing only the selected rows.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data to slice.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            Union[FeatureSet, DataBatch]: New instance with sliced data.
            
        Raises:
            IndexError: If any index is out of bounds for the input data.
            TypeError: If indices type is not compatible with the data structure.
        """
        if self.indices is None:
            return data
        if isinstance(data, FeatureSet):
            data_length = data.features.shape[0]
        elif isinstance(data, DataBatch):
            data_length = len(data.data) if hasattr(data.data, '__len__') else 0
        else:
            raise TypeError('Unsupported data type. Expected FeatureSet or DataBatch.')
        if isinstance(self.indices, slice):
            selected_indices = list(range(data_length))[self.indices]
        elif isinstance(self.indices, list):
            if any((idx < -data_length or idx >= data_length for idx in self.indices)):
                raise IndexError('One or more indices are out of bounds')
            selected_indices = [idx if idx >= 0 else data_length + idx for idx in self.indices]
        else:
            raise TypeError('Indices must be a list of integers, a slice object, or None')
        if isinstance(data, FeatureSet):
            sliced_features = data.features[selected_indices, :]
            sliced_sample_ids = [data.sample_ids[i] for i in selected_indices] if data.sample_ids is not None else None
            sliced_metadata = data.metadata.copy() if data.metadata else {}
            return FeatureSet(features=sliced_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=sliced_sample_ids, metadata=sliced_metadata, quality_scores=data.quality_scores)
        elif isinstance(data, DataBatch):
            if isinstance(data.data, np.ndarray):
                sliced_data = data.data[selected_indices, :] if data.data.ndim > 1 else data.data[selected_indices]
            elif isinstance(data.data, list):
                sliced_data = [data.data[i] for i in selected_indices]
            else:
                raise TypeError('DataBatch.data must be a numpy array or list')
            sliced_labels = None
            if data.labels is not None:
                if isinstance(data.labels, np.ndarray):
                    sliced_labels = data.labels[selected_indices]
                elif isinstance(data.labels, list):
                    sliced_labels = [data.labels[i] for i in selected_indices]
                else:
                    raise TypeError('DataBatch.labels must be a numpy array or list')
            sliced_sample_ids = [data.sample_ids[i] for i in selected_indices] if data.sample_ids is not None else None
            return DataBatch(data=sliced_data, labels=sliced_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=sliced_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Raise NotImplementedError as inverse transform is not meaningful for index slicing.
        
        Index slicing is a destructive operation that cannot be meaningfully reversed,
        since information about the non-selected rows is permanently lost.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Data to inversely transform.
            **kwargs: Additional parameters (ignored).
            
        Raises:
            NotImplementedError: Always raised as inverse transform is unsupported.
        """
        raise NotImplementedError('Index slicing cannot be inverted as information about non-selected rows is lost.')