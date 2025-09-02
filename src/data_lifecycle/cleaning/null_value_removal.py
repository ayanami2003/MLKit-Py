from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import Union, Optional

def drop_rows_with_any_nulls(data: Union[np.ndarray, DataBatch]) -> Union[np.ndarray, DataBatch]:
    """
    Remove rows that contain any null values from the input data.

    This function scans the input data and removes any rows where at least one element is null (NaN for numeric data
    or None for object data). It supports both raw NumPy arrays and DataBatch objects, returning the filtered result
    in the same format as the input.

    Args:
        data (Union[np.ndarray, DataBatch]): Input data from which rows with nulls will be removed. If a DataBatch
                                             is provided, its metadata and structure are preserved in the output.

    Returns:
        Union[np.ndarray, DataBatch]: Filtered data with rows containing any nulls removed. The return type matches
                                      the input type.

    Raises:
        ValueError: If the input data format is unsupported or malformed.
    """
    if not isinstance(data, (np.ndarray, DataBatch)):
        raise ValueError('Input data must be either a NumPy array or a DataBatch object.')
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('Input NumPy array must be 2-dimensional.')
        if data.size == 0:
            return data.copy()
        try:
            nan_mask = np.isnan(data).any(axis=1)
        except (TypeError, ValueError):
            nan_mask = np.array([any((np.isnan(element) if isinstance(element, (int, float, np.number)) and (not isinstance(element, (bool, np.bool_))) else False for element in row)) for row in data])
        none_mask = np.array([any((element is None for element in row)) for row in data])
        null_mask = nan_mask | none_mask
        return data[~null_mask]
    else:
        if not hasattr(data, 'data') or data.data is None:
            raise ValueError("DataBatch must have a 'data' attribute.")
        batch_data = data.data
        if not isinstance(batch_data, (np.ndarray, list)):
            raise ValueError('DataBatch.data must be either a NumPy array or a list.')
        if len(batch_data) == 0:
            return DataBatch(data=batch_data.copy() if hasattr(batch_data, 'copy') else [], labels=data.labels.copy() if hasattr(data.labels, 'copy') and data.labels is not None else data.labels[:] if isinstance(data.labels, list) and data.labels is not None else data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids.copy() if hasattr(data.sample_ids, 'copy') and data.sample_ids is not None else data.sample_ids[:] if isinstance(data.sample_ids, list) and data.sample_ids is not None else data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        if isinstance(batch_data, np.ndarray):
            if batch_data.ndim != 2:
                raise ValueError('DataBatch.data NumPy array must be 2-dimensional.')
            if batch_data.size == 0:
                filtered_data = batch_data.copy()
                valid_indices = np.array([], dtype=int)
            else:
                try:
                    nan_mask = np.isnan(batch_data).any(axis=1)
                except (TypeError, ValueError):
                    nan_mask = np.array([any((np.isnan(element) if isinstance(element, (int, float, np.number)) and (not isinstance(element, (bool, np.bool_))) else False for element in row)) for row in batch_data])
                none_mask = np.array([any((element is None for element in row)) for row in batch_data])
                null_mask = nan_mask | none_mask
                valid_indices = np.where(~null_mask)[0]
                filtered_data = batch_data[~null_mask]
        else:
            valid_indices = []
            for (i, row) in enumerate(batch_data):
                has_null = False
                for element in row:
                    if element is None or (isinstance(element, (float, np.floating)) and np.isnan(element)):
                        has_null = True
                        break
                if not has_null:
                    valid_indices.append(i)
            filtered_data = [batch_data[i] for i in valid_indices]
        filtered_labels = None
        if data.labels is not None:
            if isinstance(data.labels, np.ndarray):
                filtered_labels = data.labels[valid_indices]
            else:
                filtered_labels = [data.labels[i] for i in valid_indices] if isinstance(valid_indices, list) else [data.labels[i] for i in valid_indices]
        filtered_sample_ids = None
        if data.sample_ids is not None:
            if isinstance(data.sample_ids, np.ndarray):
                filtered_sample_ids = data.sample_ids[valid_indices]
            else:
                filtered_sample_ids = [data.sample_ids[i] for i in valid_indices] if isinstance(valid_indices, list) else [data.sample_ids[i] for i in valid_indices]
        return DataBatch(data=filtered_data, labels=filtered_labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=filtered_sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)