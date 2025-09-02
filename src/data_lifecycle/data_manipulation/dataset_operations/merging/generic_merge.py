from general.structures.data_batch import DataBatch
from typing import List, Optional, Union
import numpy as np
import pandas as pd

def merge_datasets(datasets: List[DataBatch], on: Optional[Union[str, List[str]]]=None, how: str='inner', suffixes: Optional[List[str]]=None, validate_schema: bool=True) -> DataBatch:
    """
    Merge multiple datasets into a single DataBatch based on common keys or row alignment.

    This function performs dataset merging operations similar to SQL joins or pandas DataFrame merges.
    It supports merging on specified key columns or by row alignment when no keys are provided.
    Multiple datasets can be merged sequentially using the same join strategy.

    Args:
        datasets (List[DataBatch]): A list of DataBatch objects to merge. Must contain at least two datasets.
        on (Optional[Union[str, List[str]]]): Column name(s) to merge on. If None, merges by row alignment.
        how (str): Type of merge to perform ('inner', 'outer', 'left', 'right'). Defaults to 'inner'.
        suffixes (Optional[List[str]]): Suffixes to apply to overlapping column names in left and right datasets.
        validate_schema (bool): Whether to validate schema compatibility before merging. Defaults to True.

    Returns:
        DataBatch: A new DataBatch containing merged data with combined features and aligned samples.

    Raises:
        ValueError: If fewer than two datasets are provided or if key columns don't exist in all datasets.
        TypeError: If datasets have incompatible schemas when validate_schema is True.
    """
    if not isinstance(datasets, list):
        raise TypeError('Datasets must be provided as a list')
    if len(datasets) < 2:
        raise ValueError('At least two datasets are required for merging')
    for (i, dataset) in enumerate(datasets):
        if not isinstance(dataset, DataBatch):
            raise TypeError(f'Element at index {i} is not a DataBatch instance')
    valid_how = ['inner', 'outer', 'left', 'right']
    if how not in valid_how:
        raise ValueError(f"Invalid merge type '{how}'. Must be one of {valid_how}")
    if suffixes is None:
        suffixes = ['_x', '_y']
    elif len(suffixes) != 2:
        raise ValueError('Suffixes must be a list of two strings')
    dataframes = []
    label_column_names = []
    for (i, dataset) in enumerate(datasets):
        if isinstance(dataset.data, pd.DataFrame):
            df = dataset.data.copy()
        elif isinstance(dataset.data, np.ndarray):
            if dataset.feature_names:
                df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            else:
                df = pd.DataFrame(dataset.data, columns=[f'col_{j}' for j in range(dataset.data.shape[1] if dataset.data.ndim > 1 else 1)])
        elif dataset.feature_names:
            df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        else:
            df = pd.DataFrame(dataset.data)
        if dataset.labels is not None:
            label_col_name = 'label'
            counter = 0
            original_label_col_name = label_col_name
            while label_col_name in df.columns:
                label_col_name = f'{original_label_col_name}_{counter}'
                counter += 1
            df[label_col_name] = dataset.labels
            label_column_names.append(label_col_name)
        dataframes.append(df)
    if validate_schema and on is not None:
        on_columns = [on] if isinstance(on, str) else on
        for (i, df) in enumerate(dataframes):
            missing_cols = set(on_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f'Dataset at index {i} is missing key columns: {missing_cols}')
    if on is None:
        result_df = dataframes[0]
        for df in dataframes[1:]:
            if len(result_df) != len(df):
                raise ValueError('All datasets must have the same number of rows for row-aligned merging')
            overlapping_cols = set(result_df.columns) & set(df.columns)
            if overlapping_cols:
                left_rename = {col: f'{col}{suffixes[0]}' for col in overlapping_cols}
                right_rename = {col: f'{col}{suffixes[1]}' for col in overlapping_cols}
                result_df = result_df.rename(columns=left_rename)
                df = df.rename(columns=right_rename)
            for col in df.columns:
                result_df[col] = df[col].values
    else:
        on_columns = [on] if isinstance(on, str) else on
        result_df = dataframes[0]
        for df in dataframes[1:]:
            result_df = pd.merge(result_df, df, on=on_columns, how=how, suffixes=suffixes)
    labels = None
    label_columns_in_result = [col for col in label_column_names if col in result_df.columns]
    if label_columns_in_result:
        first_label_col = label_columns_in_result[0]
        labels = result_df[first_label_col].values
        feature_names = [col for col in result_df.columns if col not in label_column_names]
    else:
        feature_names = list(result_df.columns)
    merged_data = result_df[feature_names].values
    combined_metadata = {}
    for dataset in datasets:
        if dataset.metadata:
            combined_metadata.update(dataset.metadata)
    batch_ids = [str(dataset.batch_id) if dataset.batch_id else str(i) for (i, dataset) in enumerate(datasets)]
    new_batch_id = f"merged_{'_'.join(batch_ids)}"
    sample_ids = None
    if all((dataset.sample_ids is not None for dataset in datasets)):
        if on is None:
            sample_ids = datasets[0].sample_ids
        else:
            sample_ids = None
    return DataBatch(data=merged_data, labels=labels, metadata=combined_metadata, sample_ids=sample_ids, feature_names=feature_names, batch_id=new_batch_id)