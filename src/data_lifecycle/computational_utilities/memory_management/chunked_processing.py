from typing import Iterator, Any, Callable, Optional
from general.structures.data_batch import DataBatch

def process_in_chunks(data: Any, chunk_size: int, processor: Callable[[Any], Any], post_processor: Optional[Callable[[Iterator[Any]], Any]]=None) -> Any:
    """
    Process data in chunks to manage memory usage efficiently.

    This function divides the input data into smaller chunks of a specified size and applies a
    processing function to each chunk independently. It is particularly useful for handling large
    datasets that might not fit into memory all at once. An optional post-processing function can
    be applied to the results of processing each chunk.

    Args:
        data (Any): The input data to be processed. This could be a list, array, or any iterable.
        chunk_size (int): The size of each chunk. Determines how the data will be divided.
        processor (Callable[[Any], Any]): A function that takes a chunk of data as input and
                                          returns processed results.
        post_processor (Optional[Callable[[Iterator[Any]], Any]]): An optional function that
                                                                   takes an iterator of processed
                                                                   chunk results and returns a
                                                                   final combined result.

    Returns:
        Any: The final processed result. If a post_processor is provided, its output is returned.
             Otherwise, a list of results from processing each chunk is returned.

    Raises:
        ValueError: If chunk_size is not a positive integer.
        TypeError: If data is not iterable or processor is not callable.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if not callable(processor):
        raise TypeError('processor must be callable')
    try:
        iterator = iter(data)
    except TypeError:
        raise TypeError('data must be iterable')

    def chunk_generator():
        chunk = []
        for item in iterator:
            chunk.append(item)
            if len(chunk) == chunk_size:
                yield processor(chunk)
                chunk = []
        if chunk:
            yield processor(chunk)
    if post_processor is not None:
        return post_processor(chunk_generator())
    return list(chunk_generator())