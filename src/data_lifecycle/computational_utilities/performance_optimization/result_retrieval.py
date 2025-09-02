from general.structures.data_batch import DataBatch
from typing import Any, Callable, Optional, Union

def retrieve_result(data: Union[DataBatch, Any], retrieval_fn: Optional[Callable[[Any], Any]]=None) -> Any:
    """
    Retrieve computed results from processed data, applying an optional transformation function.

    This function is designed to extract or transform results from data processing operations,
    particularly in performance-optimized workflows. It supports both raw data and structured
    DataBatch objects, allowing flexible integration with various computational pipelines.

    Args:
        data (Union[DataBatch, Any]): The processed data from which to retrieve results.
            Can be a DataBatch object containing structured data or any other data container.
        retrieval_fn (Optional[Callable[[Any], Any]]): An optional function to apply to the
            data before returning. This allows for custom transformations or extractions.
            If None, the data is returned as-is.

    Returns:
        Any: The retrieved result, potentially transformed by retrieval_fn.

    Raises:
        TypeError: If data is None or retrieval_fn is not callable when provided.
    """
    if data is None:
        raise TypeError('Data cannot be None')
    if retrieval_fn is not None and (not callable(retrieval_fn)):
        raise TypeError('retrieval_fn must be callable when provided')
    if isinstance(data, DataBatch):
        extracted_data = data.data
    else:
        extracted_data = data
    if retrieval_fn is not None:
        return retrieval_fn(extracted_data)
    return extracted_data