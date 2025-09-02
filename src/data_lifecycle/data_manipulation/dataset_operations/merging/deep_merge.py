import pandas as pd
from typing import List, Optional, Union, Dict, Any
from general.structures.data_batch import DataBatch

def deep_merge_datasets(datasets: List[DataBatch], on: Optional[Union[str, List[str]]]=None, how: str='inner', suffixes: Optional[List[str]]=None, recursive: bool=True, max_depth: Optional[int]=None) -> DataBatch:
    """
    Perform a deep merge of multiple datasets based on common keys, preserving nested structures.

    This function merges multiple DataBatch objects by performing recursive joins when nested
    structures are encountered. Unlike standard merges, deep merge handles hierarchical data
    by recursively merging nested dictionaries or lists within matching rows.

    Args:
        datasets (List[DataBatch]): A list of DataBatch objects to be merged. Must contain at least two datasets.
        on (Optional[Union[str, List[str]]]): Column(s) to join on. If None, uses intersection of column names.
        how (str): Type of merge to perform ('inner', 'outer', 'left', 'right'). Defaults to 'inner'.
        suffixes (Optional[List[str]]): Suffixes to apply to overlapping column names in left and right datasets.
        recursive (bool): Whether to perform recursive merging on nested structures. Defaults to True.
        max_depth (Optional[int]): Maximum recursion depth for nested structures. If None, no limit is applied.

    Returns:
        DataBatch: A new DataBatch containing the deeply merged data with preserved nested structures.

    Raises:
        ValueError: If fewer than two datasets are provided or if merge keys are invalid.
        TypeError: If datasets contain incompatible nested structures.
    """
    if len(datasets) < 2:
        raise ValueError('At least two datasets are required for merging.')
    if suffixes is None:
        suffixes = ['_x', '_y']
    if on is None:
        common_columns = set(datasets[0].data.columns)
        for db in datasets[1:]:
            common_columns &= set(db.data.columns)
        if not common_columns:
            raise ValueError('No common columns found to merge on.')
        on = list(common_columns)
    elif isinstance(on, str):
        on = [on]
    for (i, db) in enumerate(datasets):
        missing_keys = set(on) - set(db.data.columns)
        if missing_keys:
            raise ValueError(f'Dataset {i} is missing merge keys: {missing_keys}')
    result_df = datasets[0].data.copy()
    for db in datasets[1:]:
        left_cols = set(result_df.columns) - set(on)
        right_cols = set(db.data.columns) - set(on)
        overlap = left_cols & right_cols
        left_suffix = suffixes[0] if len(suffixes) > 0 else '_x'
        right_suffix = suffixes[1] if len(suffixes) > 1 else '_y'
        left_rename_map = {col: col + left_suffix for col in overlap if col + left_suffix not in result_df.columns} if left_suffix else {}
        right_rename_map = {col: col + right_suffix for col in overlap if col + right_suffix not in db.data.columns} if right_suffix else {}
        result_df_renamed = result_df.rename(columns=left_rename_map)
        right_data_renamed = db.data.rename(columns=right_rename_map)
        merged_temp = pd.merge(result_df_renamed, right_data_renamed, on=on, how=how)
        if recursive and overlap and (max_depth is None or max_depth > 0):
            merged_temp = _merge_nested_structures(merged_temp, overlap, left_suffix, right_suffix, max_depth=max_depth if max_depth is not None else float('inf'), current_depth=0)
        result_df = merged_temp
    return DataBatch(data=result_df)