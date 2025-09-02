from typing import List, Union, Optional
from general.structures.data_batch import DataBatch
import numpy as np

def concatenate_datasets(datasets: List[DataBatch], axis: int=0, ignore_index: bool=False, validate_schema: bool=True) -> DataBatch:
    """
    Concatenate multiple DataBatch objects into a single DataBatch along a specified axis.

    This function combines multiple datasets (represented as DataBatch instances) either
    row-wise (axis=0, default) or column-wise (axis=1). It ensures schema compatibility
    when concatenating along axis=0 and handles metadata appropriately.

    Args:
        datasets (List[DataBatch]): A list of DataBatch objects to concatenate. Must not be empty.
        axis (int, optional): The axis along which to concatenate. 0 for row-wise (default), 
                              1 for column-wise concatenation.
        ignore_index (bool, optional): If True, do not use the index from the input datasets 
                                       and reset the index in the result. Defaults to False.
        validate_schema (bool, optional): If True, validate that feature names and types match
                                          when concatenating along axis=0. Defaults to True.

    Returns:
        DataBatch: A new DataBatch containing concatenated data.

    Raises:
        ValueError: If datasets list is empty, or if schema validation fails when required.
        TypeError: If datasets contain non-DataBatch objects.
    """
    if not isinstance(datasets, list):
        raise TypeError("Input 'datasets' must be a list of DataBatch objects.")
    if not datasets:
        raise ValueError("Input 'datasets' cannot be empty.")
    for (i, dataset) in enumerate(datasets):
        if not isinstance(dataset, DataBatch):
            raise TypeError(f'Element at index {i} is not a DataBatch instance.')
    if len(datasets) == 1:
        return datasets[0]
    if axis == 0 and validate_schema:
        reference_features = datasets[0].feature_names
        if reference_features is not None:
            for (i, dataset) in enumerate(datasets[1:], 1):
                if dataset.feature_names != reference_features:
                    raise ValueError(f'Schema mismatch at index {i}: feature names do not match. Expected {reference_features}, got {dataset.feature_names}')
    data_list = [db.data for db in datasets]
    labels_list = [db.labels for db in datasets]
    sample_ids_list = [db.sample_ids for db in datasets]
    if isinstance(data_list[0], np.ndarray):
        concatenated_data = np.concatenate(data_list, axis=axis)
    elif axis == 0:
        concatenated_data = []
        for data in data_list:
            concatenated_data.extend(data)
    else:
        concatenated_data = []
        for i in range(len(data_list[0])):
            row = []
            for data in data_list:
                row.extend(data[i])
            concatenated_data.append(row)
    concatenated_labels = None
    if all((labels is not None for labels in labels_list)):
        if isinstance(labels_list[0], np.ndarray):
            concatenated_labels = np.concatenate(labels_list, axis=0 if axis == 0 else None)
        else:
            concatenated_labels = []
            for labels in labels_list:
                concatenated_labels.extend(labels)
    elif any((labels is not None for labels in labels_list)):
        raise ValueError("Some datasets have labels while others don't. All datasets must consistently have or not have labels.")
    concatenated_sample_ids = None
    if all((sample_ids is not None for sample_ids in sample_ids_list)):
        if ignore_index:
            concatenated_sample_ids = None
        elif axis == 0:
            concatenated_sample_ids = []
            for sample_ids in sample_ids_list:
                concatenated_sample_ids.extend(sample_ids)
        else:
            concatenated_sample_ids = sample_ids_list[0]
    elif any((sample_ids is not None for sample_ids in sample_ids_list)):
        raise ValueError("Some datasets have sample_ids while others don't. All datasets must consistently have or not have sample_ids.")
    feature_names = datasets[0].feature_names
    if axis == 1 and feature_names is not None:
        combined_feature_names = []
        for dataset in datasets:
            if dataset.feature_names:
                combined_feature_names.extend(dataset.feature_names)
            else:
                combined_feature_names = None
                break
        feature_names = combined_feature_names if combined_feature_names else None
    combined_metadata = {}
    for dataset in datasets:
        if dataset.metadata:
            combined_metadata.update(dataset.metadata)
    batch_id = f"concatenated_{'_'.join((str(db.batch_id) if db.batch_id else str(i) for (i, db) in enumerate(datasets)))}"
    if concatenated_sample_ids is not None and axis == 1:
        data_length = len(concatenated_data) if hasattr(concatenated_data, '__len__') else 0
        if len(concatenated_sample_ids) != data_length:
            concatenated_sample_ids = None
    return DataBatch(data=concatenated_data, labels=concatenated_labels, metadata=combined_metadata, sample_ids=concatenated_sample_ids, feature_names=feature_names, batch_id=batch_id)