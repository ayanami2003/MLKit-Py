from general.structures.data_batch import DataBatch
from typing import List, Optional, Union
import pandas as pd
import numpy as np

def left_join_datasets(left_dataset: DataBatch, right_dataset: DataBatch, on: Optional[Union[str, List[str]]]=None, left_on: Optional[Union[str, List[str]]]=None, right_on: Optional[Union[str, List[str]]]=None, suffixes: Optional[List[str]]=None, validate_schema: bool=True) -> DataBatch:
    """
    Perform a left join operation between two datasets based on specified key columns.

    This function merges two DataBatch objects using a left join strategy, where all rows 
    from the left dataset are retained, and matching rows from the right dataset are included. 
    Non-matching rows from the right dataset are excluded, and missing values are filled 
    with null equivalents.

    Args:
        left_dataset (DataBatch): The primary dataset to join from (left side).
        right_dataset (DataBatch): The secondary dataset to join with (right side).
        on (Optional[Union[str, List[str]]]): Column(s) to join on if identical in both datasets.
        left_on (Optional[Union[str, List[str]]]): Column(s) from the left dataset to join on.
        right_on (Optional[Union[str, List[str]]]): Column(s) from the right dataset to join on.
        suffixes (Optional[List[str]]): Suffixes to apply to overlapping column names 
                                         (defaults to ['_left', '_right'] if not provided).
        validate_schema (bool): Whether to perform schema compatibility checks before merging.

    Returns:
        DataBatch: A new dataset containing the result of the left join operation.

    Raises:
        ValueError: If neither `on` nor both `left_on` and `right_on` are specified.
        ValueError: If schema validation fails and `validate_schema` is True.
    """
    if on is None and (left_on is None or right_on is None):
        raise ValueError("Either 'on' must be specified or both 'left_on' and 'right_on' must be provided.")
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on' simultaneously.")
    if on is not None:
        left_keys = right_keys = on if isinstance(on, list) else [on]
    else:
        left_keys = left_on if isinstance(left_on, list) else [left_on]
        right_keys = right_on if isinstance(right_on, list) else [right_on]
    left_df = pd.DataFrame(left_dataset.data) if not isinstance(left_dataset.data, pd.DataFrame) else left_dataset.data
    right_df = pd.DataFrame(right_dataset.data) if not isinstance(right_dataset.data, pd.DataFrame) else right_dataset.data
    missing_left = set(left_keys) - set(left_df.columns)
    missing_right = set(right_keys) - set(right_df.columns)
    if missing_left:
        raise ValueError(f'Key columns {list(missing_left)} not found in left dataset.')
    if missing_right:
        raise ValueError(f'Key columns {list(missing_right)} not found in right dataset.')
    if suffixes is None:
        if validate_schema:
            suffixes = ['_left', '_right']
        else:
            suffixes = ['_x', '_y']
    elif len(suffixes) != 2:
        raise ValueError('Suffixes must be a list of two strings.')
    if validate_schema:
        for (l_key, r_key) in zip(left_keys, right_keys):
            if len(left_df) == 0 or len(right_df) == 0:
                continue
            l_dtype = left_df[l_key].dtype
            r_dtype = right_df[r_key].dtype
            try:
                pd.concat([left_df[l_key].iloc[:1], right_df[r_key].iloc[:1]], ignore_index=True)
            except Exception:
                raise ValueError(f"Key column '{l_key}' in left dataset (type: {l_dtype}) is incompatible with '{r_key}' in right dataset (type: {r_dtype}).")
    right_dtypes = {}
    for col in right_df.columns:
        if col not in right_keys:
            right_dtypes[col] = right_df[col].dtype
    merged_df = pd.merge(left=left_df, right=right_df, left_on=left_keys, right_on=right_keys, how='left', suffixes=suffixes)
    for (col, original_dtype) in right_dtypes.items():
        suffixed_col = None
        if col in merged_df.columns:
            suffixed_col = col
        else:
            for suffix in [suffixes[1], suffixes[0]]:
                candidate = f'{col}{suffix}'
                if candidate in merged_df.columns:
                    suffixed_col = candidate
                    break
        if suffixed_col and suffixed_col in merged_df.columns:
            current_dtype = merged_df[suffixed_col].dtype
            if current_dtype != original_dtype:
                try:
                    if pd.api.types.is_numeric_dtype(original_dtype):
                        merged_df[suffixed_col] = merged_df[suffixed_col].astype(original_dtype)
                    elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                        merged_df[suffixed_col] = pd.to_datetime(merged_df[suffixed_col], errors='coerce').astype(original_dtype)
                    elif pd.api.types.is_bool_dtype(original_dtype):
                        merged_df[suffixed_col] = merged_df[suffixed_col].astype('boolean')
                except (ValueError, TypeError):
                    pass
    return DataBatch(data=merged_df)