from typing import Callable, Any, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from general.structures.data_batch import DataBatch

def process_in_parallel(data: Any, processor: Callable[[Any], Any], n_jobs: int=-1, chunk_size: Optional[int]=None) -> Any:
    """
    Process data in parallel using multiple worker processes.

    This function distributes data processing tasks across multiple processes to improve performance
    on CPU-intensive operations. It automatically handles data chunking and result aggregation.

    Args:
        data (Any): Input data to process. Can be any iterable or array-like structure.
        processor (Callable[[Any], Any]): Function to apply to each data chunk.
        n_jobs (int): Number of parallel jobs to run. -1 means using all processors. Defaults to -1.
        chunk_size (Optional[int]): Size of chunks to split the data into. If None, it will be 
                                    automatically determined based on data size and n_jobs.

    Returns:
        Any: Processed data in the same structure as input.

    Raises:
        ValueError: If n_jobs is 0 or less than -1.
        RuntimeError: If parallel processing fails due to internal issues.
    """
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError('n_jobs must be -1 or a positive integer')
    if n_jobs == -1:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = n_jobs
    if n_workers == 1:
        return processor(data)
    if isinstance(data, DataBatch):
        primary_data = data.data
        data_length = len(primary_data) if hasattr(primary_data, '__len__') else 0
        if data_length == 0:
            return data
        if chunk_size is None:
            chunk_size = max(1, data_length // (n_workers * 4))
        chunks = [primary_data[i:i + chunk_size] for i in range(0, data_length, chunk_size)]
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_chunk = {executor.submit(processor, chunk): i for (i, chunk) in enumerate(chunks)}
                results = [None] * len(chunks)
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        results[chunk_idx] = future.result()
                    except Exception as e:
                        raise RuntimeError(f'Parallel processing failed in chunk {chunk_idx}') from e
            if isinstance(primary_data, (list, tuple, np.ndarray)) and (not isinstance(primary_data, (str, bytes))):
                should_flatten = True
                for r in results:
                    if not isinstance(r, (list, tuple, np.ndarray)) or isinstance(r, (str, bytes)):
                        should_flatten = False
                        break
                if should_flatten:
                    flattened = []
                    for r in results:
                        if isinstance(r, (list, tuple)):
                            flattened.extend(r)
                        elif isinstance(r, np.ndarray):
                            flattened.extend(r.tolist())
                    return DataBatch(data=np.array(flattened) if isinstance(primary_data, np.ndarray) else flattened, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
            return DataBatch(data=results, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        except Exception as e:
            if not isinstance(e, RuntimeError):
                raise RuntimeError('Parallel processing failed due to internal issues') from e
            else:
                raise
    if not hasattr(data, '__getitem__') or not hasattr(data, '__len__'):
        try:
            data = list(data)
        except Exception as e:
            raise TypeError('Data must be convertible to a list for chunking') from e
    data_length = len(data)
    if data_length == 0:
        return data
    if chunk_size is None:
        chunk_size = max(1, data_length // (n_workers * 4))
    chunks = [data[i:i + chunk_size] for i in range(0, data_length, chunk_size)]
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk = {executor.submit(processor, chunk): i for (i, chunk) in enumerate(chunks)}
            results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    results[chunk_idx] = future.result()
                except Exception as e:
                    raise RuntimeError(f'Parallel processing failed in chunk {chunk_idx}') from e
        if isinstance(data, (list, tuple, np.ndarray)) and (not isinstance(data, (str, bytes))):
            should_flatten = True
            for r in results:
                if not isinstance(r, (list, tuple, np.ndarray)) or isinstance(r, (str, bytes)):
                    should_flatten = False
                    break
            if should_flatten:
                flattened = []
                for r in results:
                    if isinstance(r, (list, tuple)):
                        flattened.extend(r)
                    elif isinstance(r, np.ndarray):
                        flattened.extend(r.tolist())
                if isinstance(data, np.ndarray):
                    return np.array(flattened)
                elif isinstance(data, tuple):
                    return tuple(flattened)
                else:
                    return flattened
        return results
    except Exception as e:
        if not isinstance(e, RuntimeError):
            raise RuntimeError('Parallel processing failed due to internal issues') from e
        else:
            raise