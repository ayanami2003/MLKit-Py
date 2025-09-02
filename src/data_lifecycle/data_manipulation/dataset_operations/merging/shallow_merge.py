from general.structures.data_batch import DataBatch
from typing import List, Optional, Union
import numpy as np

def shallow_merge_datasets(datasets: List[DataBatch], on: Optional[Union[str, List[str]]]=None, how: str='inner', suffixes: Optional[List[str]]=None, validate_schema: bool=True) -> DataBatch:
    """
    Perform a shallow merge of multiple datasets based on common keys without deep copying underlying data.

    This function merges a list of DataBatch objects based on specified column(s), preserving references
    to the original data where possible to minimize memory usage. It supports various join types and
    allows customization of column naming conflicts through suffixes.

    Args:
        datasets (List[DataBatch]): A list of DataBatch objects to be merged. Must contain at least two datasets.
        on (Optional[Union[str, List[str]]]): Column name(s) to merge on. If None, merges on overlapping columns.
        how (str): Type of merge to perform ('inner', 'outer', 'left', 'right'). Defaults to 'inner'.
        suffixes (Optional[List[str]]): Suffixes to apply to overlapping column names in left and right datasets.
        validate_schema (bool): Whether to perform schema compatibility checks before merging. Defaults to True.

    Returns:
        DataBatch: A new DataBatch containing merged data with preserved metadata from the first dataset.

    Raises:
        ValueError: If fewer than two datasets are provided or if specified columns don't exist.
        TypeError: If datasets have incompatible schemas when validation is enabled.
    """
    if len(datasets) < 2:
        raise ValueError('At least two datasets are required for merging.')
    if suffixes is None:
        suffixes = ['_x', '_y']
    elif len(suffixes) != 2:
        raise ValueError('Suffixes must be a list of two strings.')
    if isinstance(on, str):
        on = [on]
    if on is None:
        all_feature_names = [set(ds.feature_names) for ds in datasets if ds.feature_names is not None]
        if not all_feature_names:
            raise ValueError('Cannot infer merge keys when datasets have no feature names.')
        on = list(set.intersection(*all_feature_names))
        if not on:
            raise ValueError('No common columns found to merge on.')
    for (i, ds) in enumerate(datasets):
        if ds.feature_names is None:
            raise ValueError(f'Dataset at index {i} does not have feature names.')
        missing_cols = set(on) - set(ds.feature_names)
        if missing_cols:
            raise ValueError(f'Dataset at index {i} is missing columns: {missing_cols}')
    structured_arrays = []
    for ds in datasets:
        if isinstance(ds.data, np.ndarray) and ds.data.ndim == 2:
            dtype_spec = []
            for (i, name) in enumerate(ds.feature_names):
                dtype = ds.data.dtype if ds.data.ndim == 1 else ds.data[:, i].dtype
                dtype_spec.append((name, dtype))
            structured = np.empty(ds.data.shape[0], dtype=dtype_spec)
            for (i, name) in enumerate(ds.feature_names):
                structured[name] = ds.data[:, i]
            structured_arrays.append(structured)
        else:
            dtype_spec = [(name, object) for name in ds.feature_names]
            structured = np.empty(len(ds.data), dtype=dtype_spec)
            for (i, row) in enumerate(ds.data):
                if isinstance(row, (list, np.ndarray)):
                    for (j, name) in enumerate(ds.feature_names):
                        structured[i][name] = row[j] if j < len(row) else None
                else:
                    structured[i][ds.feature_names[0]] = row
            structured_arrays.append(structured)
    if validate_schema and len(structured_arrays) > 1:
        first_array = structured_arrays[0]
        for (i, arr) in enumerate(structured_arrays[1:], 1):
            for name in arr.dtype.names:
                if name in first_array.dtype.names and name not in on:
                    first_dtype = first_array.dtype[name]
                    current_dtype = arr.dtype[name]
                    if first_dtype != current_dtype:
                        try:
                            np.promote_types(first_dtype, current_dtype)
                        except Exception:
                            raise TypeError(f"Schema mismatch in dataset {i} for column '{name}': expected {first_dtype}, got {current_dtype}")
    processed_arrays = []
    for (i, (ds, structured_arr)) in enumerate(zip(datasets, structured_arrays)):
        if i == 0:
            overlapping_with_future = set()
            for future_ds in datasets[1:]:
                overlapping_with_future.update(set(ds.feature_names) & set(future_ds.feature_names) - set(on))
            if overlapping_with_future:
                new_dtype = []
                rename_map = {}
                for name in structured_arr.dtype.names:
                    if name in overlapping_with_future:
                        new_name = f'{name}{suffixes[0]}'
                        rename_map[name] = new_name
                        new_dtype.append((new_name, structured_arr.dtype[name]))
                    else:
                        new_dtype.append((name, structured_arr.dtype[name]))
                if rename_map:
                    renamed_structured = np.empty(structured_arr.shape, dtype=new_dtype)
                    for name in structured_arr.dtype.names:
                        new_name = rename_map.get(name, name)
                        renamed_structured[new_name] = structured_arr[name]
                    processed_arrays.append(renamed_structured)
                else:
                    processed_arrays.append(structured_arr)
            else:
                processed_arrays.append(structured_arr)
        else:
            first_processed = processed_arrays[0]
            overlapping = set(ds.feature_names) & set(first_processed.dtype.names) - set(on)
            if overlapping:
                new_dtype = []
                rename_map = {}
                for name in structured_arr.dtype.names:
                    if name in overlapping:
                        new_name = f'{name}{suffixes[1]}'
                        rename_map[name] = new_name
                        new_dtype.append((new_name, structured_arr.dtype[name]))
                    else:
                        new_dtype.append((name, structured_arr.dtype[name]))
                if rename_map:
                    renamed_structured = np.empty(structured_arr.shape, dtype=new_dtype)
                    for name in structured_arr.dtype.names:
                        new_name = rename_map.get(name, name)
                        renamed_structured[new_name] = structured_arr[name]
                    processed_arrays.append(renamed_structured)
                else:
                    processed_arrays.append(structured_arr)
            else:
                processed_arrays.append(structured_arr)
    result_array = processed_arrays[0]
    for (i, (ds, structured_arr)) in enumerate(zip(datasets[1:], processed_arrays[1:]), 1):
        if how == 'inner':
            result_array = _inner_join(result_array, structured_arr, on)
        elif how == 'outer':
            result_array = _outer_join(result_array, structured_arr, on)
        elif how == 'left':
            result_array = _left_join(result_array, structured_arr, on)
        elif how == 'right':
            result_array = _right_join(result_array, structured_arr, on)
        else:
            raise ValueError(f'Unsupported merge type: {how}')
    if result_array.size == 0:
        feature_names = list(result_array.dtype.names) if result_array.dtype.names else []
        return DataBatch(data=np.empty((0, len(feature_names))) if feature_names else [], metadata=datasets[0].metadata.copy() if datasets[0].metadata else {}, feature_names=feature_names, batch_id=f'merged_{datasets[0].batch_id}' if datasets[0].batch_id else None)
    feature_names = list(result_array.dtype.names) if result_array.dtype.names else []
    ordered_features = []
    for col in on:
        if col in feature_names:
            ordered_features.append(col)
    added_columns = set(ordered_features)
    for col in datasets[0].feature_names:
        suffixed_col = f'{col}{suffixes[0]}'
        if col not in on:
            if suffixed_col in feature_names and suffixed_col not in added_columns:
                ordered_features.append(suffixed_col)
                added_columns.add(suffixed_col)
            elif col in feature_names and col not in added_columns:
                ordered_features.append(col)
                added_columns.add(col)
    for ds in datasets[1:]:
        for col in ds.feature_names:
            suffixed_col = f'{col}{suffixes[1]}'
            if col not in on and suffixed_col in feature_names and (suffixed_col not in added_columns):
                ordered_features.append(suffixed_col)
                added_columns.add(suffixed_col)
            elif col not in on and col in feature_names and (col not in added_columns):
                ordered_features.append(col)
                added_columns.add(col)
    if len(ordered_features) > 0:
        column_dtypes = [result_array.dtype[name] for name in ordered_features]
        all_numeric = all((dtype.kind in ['i', 'f', 'b', 'u', 'c'] for dtype in column_dtypes))
        if all_numeric:
            common_dtype = column_dtypes[0]
            for dtype in column_dtypes[1:]:
                common_dtype = np.promote_types(common_dtype, dtype)
            data = np.empty((len(result_array), len(ordered_features)), dtype=common_dtype)
            for (i, name) in enumerate(ordered_features):
                data[:, i] = result_array[name]
        else:
            data = np.empty((len(result_array), len(ordered_features)), dtype=object)
            for (i, name) in enumerate(ordered_features):
                data[:, i] = result_array[name]
    else:
        data = np.empty((len(result_array), 0))
    return DataBatch(data=data, metadata=datasets[0].metadata.copy() if datasets[0].metadata else {}, feature_names=ordered_features, batch_id=f'merged_{datasets[0].batch_id}' if datasets[0].batch_id else None)