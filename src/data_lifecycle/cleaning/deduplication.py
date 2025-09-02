from typing import Union, Optional, List, Dict, Any, Callable
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher

class DuplicateRemover(BaseTransformer):

    def __init__(self, subset: Optional[Union[str, list]]=None, keep: str='first', ignore_index: bool=False, name: Optional[str]=None):
        """
        Initialize the DuplicateRemover transformer.

        Args:
            subset: Column name(s) to consider when identifying duplicates.
                   If None, all columns are considered.
            keep: Which duplicate to keep - 'first', 'last', or False (remove all duplicates).
            ignore_index: Whether to reset the index after removing duplicates.
            name: Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.subset = subset
        self.keep = keep
        self.ignore_index = ignore_index

    def fit(self, data: DataBatch, **kwargs) -> 'DuplicateRemover':
        """
        Learn duplication patterns in the data (no-op for this implementation).

        Args:
            data: Input DataBatch to analyze for duplicates.
            **kwargs: Additional parameters (ignored).

        Returns:
            self: Returns the instance for method chaining.
        """
        pass

    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Remove duplicate records from the data.

        Args:
            data: Input DataBatch from which to remove duplicates.
            **kwargs: Additional parameters (ignored).

        Returns:
            DataBatch: New DataBatch with duplicates removed according to configuration.
        """
        if not isinstance(data.data, np.ndarray):
            data_array = np.array(data.data)
        else:
            data_array = data.data.copy()
        temp_data = data_array
        if self.subset is not None:
            if isinstance(self.subset, str):
                subset_cols = [self.subset]
            else:
                subset_cols = list(self.subset)
            if data.feature_names is not None:
                try:
                    col_indices = [data.feature_names.index(col) for col in subset_cols]
                except ValueError as e:
                    raise ValueError(f'One or more subset columns not found in feature names: {e}')
            else:
                col_indices = subset_cols
            if isinstance(temp_data, np.ndarray):
                temp_data = temp_data[:, col_indices]
            else:
                temp_data = [[row[i] for i in col_indices] for row in data.data]
        if isinstance(temp_data, np.ndarray):
            (_, unique_indices) = np.unique(temp_data, axis=0, return_index=True)
        else:
            seen = set()
            unique_indices = []
            for (i, row) in enumerate(temp_data):
                row_tuple = tuple(row) if isinstance(row, (list, np.ndarray)) else row
                if row_tuple not in seen:
                    seen.add(row_tuple)
                    unique_indices.append(i)
        unique_indices = sorted(unique_indices)
        if self.keep == 'first':
            pass
        elif self.keep == 'last':
            if isinstance(temp_data, np.ndarray):
                (_, inverse_indices, counts) = np.unique(temp_data, axis=0, return_inverse=True, return_counts=True)
                last_indices = []
                for (i, count) in enumerate(counts):
                    if count > 1:
                        positions = np.where(inverse_indices == i)[0]
                        last_indices.append(positions[-1])
                    else:
                        last_indices.append(unique_indices[i])
                unique_indices = sorted(last_indices)
            else:
                seen = {}
                for (i, row) in enumerate(temp_data):
                    row_tuple = tuple(row) if isinstance(row, (list, np.ndarray)) else row
                    seen[row_tuple] = i
                unique_indices = sorted(seen.values())
        elif self.keep is False:
            if isinstance(temp_data, np.ndarray):
                (unique_values, counts) = np.unique(temp_data, axis=0, return_counts=True)
                singular_indices = np.where(counts == 1)[0]
                if len(singular_indices) > 0:
                    (_, inverse_indices) = np.unique(temp_data, axis=0, return_inverse=True)
                    unique_indices = []
                    for idx in singular_indices:
                        position = np.where(inverse_indices == idx)[0][0]
                        unique_indices.append(position)
                else:
                    unique_indices = []
            else:
                counts = {}
                indices_map = {}
                for (i, row) in enumerate(temp_data):
                    row_tuple = tuple(row) if isinstance(row, (list, np.ndarray)) else row
                    if row_tuple in counts:
                        counts[row_tuple] += 1
                    else:
                        counts[row_tuple] = 1
                        indices_map[row_tuple] = i
                unique_indices = sorted([indices_map[key] for (key, count) in counts.items() if count == 1])
        else:
            raise ValueError("keep must be 'first', 'last', or False")
        if isinstance(data.data, np.ndarray):
            new_data = data_array[unique_indices]
        else:
            new_data = [data.data[i] for i in unique_indices]
        new_labels = None
        if data.labels is not None:
            if isinstance(data.labels, np.ndarray):
                new_labels = data.labels[unique_indices]
            else:
                new_labels = [data.labels[i] for i in unique_indices]
        new_sample_ids = None
        if data.sample_ids is not None:
            new_sample_ids = [data.sample_ids[i] for i in unique_indices]
        if self.ignore_index and new_sample_ids is not None:
            new_sample_ids = None
        new_feature_names = data.feature_names
        new_metadata = data.metadata.copy() if data.metadata else {}
        new_metadata['deduplication_info'] = {'original_size': len(data.data), 'new_size': len(new_data), 'removed_count': len(data.data) - len(new_data), 'subset': self.subset, 'keep': self.keep}
        new_batch_id = f'{data.batch_id}_deduplicated' if data.batch_id else 'deduplicated_batch'
        return DataBatch(data=new_data, labels=new_labels, metadata=new_metadata, sample_ids=new_sample_ids, feature_names=new_feature_names, batch_id=new_batch_id)

    def inverse_transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Inverse transformation is not supported for duplicate removal.

        Args:
            data: Input DataBatch (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            DataBatch: Always raises NotImplementedError.
        """
        raise NotImplementedError('Inverse transformation is not supported for duplicate removal as it is a lossy operation.')


# ...(code omitted)...


class CategoryMerger(BaseTransformer):

    def __init__(self, column: str, similarity_threshold: float=0.8, category_mapping: Optional[Dict[str, str]]=None, merge_method: str='jaccard', name: Optional[str]=None):
        """
        Initialize the CategoryMerger transformer.

        Args:
            column: Name of the categorical column to process.
            similarity_threshold: Threshold for automatic similarity detection.
            category_mapping: Manual mapping of categories to their merged values.
            merge_method: Method to use for similarity detection.
            name: Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.column = column
        self.similarity_threshold = similarity_threshold
        self.category_mapping = category_mapping or {}
        self.merge_method = merge_method

    def fit(self, data: DataBatch, **kwargs) -> 'CategoryMerger':
        """
        Learn which categories should be merged based on similarity or manual mapping.

        Args:
            data: Input DataBatch containing the categorical column to process.
            **kwargs: Additional parameters (ignored).

        Returns:
            self: Returns the instance for method chaining.
        """
        if self.column not in data.feature_names:
            raise ValueError(f"Column '{self.column}' not found in data")
        col_index = data.feature_names.index(self.column)
        if isinstance(data.data, np.ndarray):
            categories = np.unique(data.data[:, col_index])
        else:
            categories = list(set((row[col_index] for row in data.data)))
        categories = [str(cat) for cat in categories]
        if self.category_mapping:
            self._fitted_category_mapping = self.category_mapping.copy()
            return self
        self._fitted_category_mapping = {}
        if self.merge_method == 'manual':
            return self
        remaining_categories = categories.copy()
        merged_groups = {}
        while remaining_categories:
            base_category = remaining_categories.pop(0)
            merged_groups[base_category] = [base_category]
            categories_to_remove = []
            for cat in remaining_categories:
                similarity = self._compute_similarity(base_category, cat)
                if similarity >= self.similarity_threshold:
                    merged_groups[base_category].append(cat)
                    categories_to_remove.append(cat)
            for cat in categories_to_remove:
                remaining_categories.remove(cat)
        for (base_cat, group) in merged_groups.items():
            for cat in group:
                self._fitted_category_mapping[cat] = base_cat
        return self

    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Apply category merging to the specified column in the data.

        Args:
            data: Input DataBatch with categorical column to merge.
            **kwargs: Additional parameters (ignored).

        Returns:
            DataBatch: New DataBatch with merged categories in the specified column.
        """
        if not hasattr(self, '_fitted_category_mapping'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if self.column not in data.feature_names:
            raise ValueError(f"Column '{self.column}' not found in data")
        col_index = data.feature_names.index(self.column)
        if isinstance(data.data, np.ndarray):
            transformed_data = data.data.copy()
        else:
            transformed_data = [row[:] for row in data.data]
        if isinstance(transformed_data, np.ndarray):
            for i in range(len(transformed_data)):
                original_value = str(transformed_data[i, col_index])
                if original_value in self._fitted_category_mapping:
                    transformed_data[i, col_index] = self._fitted_category_mapping[original_value]
        else:
            for i in range(len(transformed_data)):
                original_value = str(transformed_data[i][col_index])
                if original_value in self._fitted_category_mapping:
                    transformed_data[i][col_index] = self._fitted_category_mapping[original_value]
        return DataBatch(data=transformed_data, feature_names=data.feature_names, target=data.target, metadata=data.metadata)

    def inverse_transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Inverse transformation is not supported for category merging.

        Args:
            data: Input DataBatch (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            DataBatch: Always raises NotImplementedError.
        """
        pass

def check_exact_duplicates(data_batch: DataBatch, subset: Optional[Union[str, List[str]]]=None) -> Dict[str, Any]:
    """
    Check for exact duplicate records in a dataset and return detailed information about them.

    This function analyzes a DataBatch to identify records that are identical across all
    columns or specified subset of columns. It provides detailed statistics about the
    duplicates found, including counts and indices.

    Args:
        data: Input DataBatch to check for duplicates.
        subset: Column name(s) to consider when identifying duplicates.
               If None, all columns are considered.

    Returns:
        dict: A dictionary containing:
            - 'has_duplicates' (bool): True if duplicates exist, False otherwise.
            - 'duplicate_count' (int): Total number of duplicated records.
            - 'unique_duplicate_groups' (int): Number of unique groups with duplicates.
            - 'duplicate_indices' (List[int]): Indices of all duplicated records.
    """
    if not hasattr(data_batch.data, '__len__') or len(data_batch.data) == 0:
        return {'has_duplicates': False, 'duplicate_count': 0, 'unique_duplicate_groups': 0, 'duplicate_indices': []}
    if len(data_batch.data) == 1:
        return {'has_duplicates': False, 'duplicate_count': 0, 'unique_duplicate_groups': 0, 'duplicate_indices': []}
    if isinstance(data_batch.data, list):
        try:
            data_array = np.array(data_batch.data, dtype=object)
        except Exception:
            data_array = np.array(data_batch.data, dtype=object)
    else:
        data_array = data_batch.data
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)
    if subset is not None:
        if data_batch.feature_names is None:
            raise ValueError('DataBatch must have feature_names when using subset parameter')
        if isinstance(subset, str):
            subset = [subset]
        missing_cols = [col for col in subset if col not in data_batch.feature_names]
        if missing_cols:
            raise ValueError(f'Subset columns {missing_cols} not found in feature_names')
        col_indices = [data_batch.feature_names.index(col) for col in subset]
        check_data = data_array[:, col_indices]
    else:
        check_data = data_array
    if check_data.ndim == 1:
        rows_as_strings = np.array([str(x) for x in check_data])
    else:
        rows_as_strings = np.array(['|'.join((str(item) for item in row)) for row in check_data])
    (unique_rows, inverse_indices, counts) = np.unique(rows_as_strings, return_inverse=True, return_counts=True)
    duplicate_mask = counts[inverse_indices] > 1
    duplicate_indices = np.where(duplicate_mask)[0].tolist()
    has_duplicates = len(duplicate_indices) > 0
    duplicate_count = int(np.sum(counts[counts > 1] - 1))
    unique_duplicate_groups = int(np.sum(counts > 1))
    return {'has_duplicates': has_duplicates, 'duplicate_count': duplicate_count, 'unique_duplicate_groups': unique_duplicate_groups, 'duplicate_indices': duplicate_indices}

def compare_records(data_batch: DataBatch, record_indices: List[int], columns: Optional[Union[str, List[str]]]=None, comparison_func: Optional[Callable]=None) -> Dict[str, Any]:
    """
    Compare specific records in a dataset using customizable comparison functions.

    This function enables detailed pairwise or multi-way comparisons between records
    to determine their similarity or differences. It supports custom comparison logic
    and can focus on specific columns of interest.

    Args:
        data: Input DataBatch containing records to compare.
        record_indices: List of indices specifying which records to compare.
        comparison_func: Custom function to compare records. If None, uses default equality.
        columns: Column name(s) to consider for comparison.
                If None, all columns are considered.

    Returns:
        dict: A dictionary containing:
            - 'comparison_matrix' (List[List[bool]]): Matrix showing pairwise comparison results.
            - 'differences' (dict): Details about differences between records for each column.
            - 'identical_fields' (List[str]): Columns where all compared records are identical.
            - 'varying_fields' (List[str]): Columns where compared records differ.
    """
    if not record_indices:
        return {'comparison_matrix': [], 'differences': {}, 'identical_fields': [], 'varying_fields': []}
    data_length = len(data_batch.data) if hasattr(data_batch.data, '__len__') else 0
    for idx in record_indices:
        if idx < 0 or idx >= data_length:
            raise IndexError(f'Record index {idx} is out of bounds for data of length {data_length}')
    if columns is None:
        if data_batch.feature_names:
            compare_columns = data_batch.feature_names
        else:
            first_row = data_batch.data[record_indices[0]] if record_indices and data_length > 0 else []
            if hasattr(first_row, '__len__'):
                compare_columns = list(range(len(first_row)))
            else:
                compare_columns = [0] if data_length > 0 else []
    elif isinstance(columns, str):
        compare_columns = [columns]
    else:
        compare_columns = columns
    if data_batch.feature_names:
        actual_columns = [col for col in compare_columns if col in data_batch.feature_names]
        col_indices = [data_batch.feature_names.index(col) for col in actual_columns]
    else:
        actual_columns = compare_columns
        col_indices = actual_columns
    n_records = len(record_indices)
    comparison_matrix = [[True for _ in range(n_records)] for _ in range(n_records)]
    differences = {col: [] for col in actual_columns}
    if comparison_func is None:
        comparison_func = lambda x, y: x == y
    if isinstance(data_batch.data, np.ndarray):
        selected_data = data_batch.data[record_indices]
        if data_batch.feature_names:
            if col_indices:
                selected_data = selected_data[:, col_indices]
        elif col_indices:
            selected_data = selected_data[:, col_indices] if selected_data.ndim > 1 else selected_data.reshape(-1, 1)
    else:
        selected_data = [data_batch.data[i] for i in record_indices]
        if col_indices and len(selected_data) > 0:
            if hasattr(selected_data[0], '__len__'):
                selected_data = [[row[i] for i in col_indices] for row in selected_data]
    for i in range(n_records):
        for j in range(i + 1, n_records):
            records_match = True
            for (col_idx, col_name) in enumerate(actual_columns):
                if isinstance(data_batch.data, np.ndarray):
                    val_i = selected_data[i, col_idx] if selected_data.ndim > 1 else selected_data[i]
                    val_j = selected_data[j, col_idx] if selected_data.ndim > 1 else selected_data[j]
                elif len(selected_data) > 0 and hasattr(selected_data[0], '__len__'):
                    val_i = selected_data[i][col_idx]
                    val_j = selected_data[j][col_idx]
                else:
                    val_i = selected_data[i]
                    val_j = selected_data[j]
                try:
                    are_equal = comparison_func(val_i, val_j)
                except Exception:
                    are_equal = False
                if not are_equal:
                    records_match = False
                    differences[col_name].append((record_indices[i], record_indices[j]))
            comparison_matrix[i][j] = records_match
            comparison_matrix[j][i] = records_match
    for i in range(n_records):
        comparison_matrix[i][i] = True
    identical_fields = []
    varying_fields = []
    for col_name in actual_columns:
        if not differences[col_name]:
            identical_fields.append(col_name)
        else:
            varying_fields.append(col_name)
    return {'comparison_matrix': comparison_matrix, 'differences': differences, 'identical_fields': identical_fields, 'varying_fields': varying_fields}

def fuzzy_duplicate_check(data_batch: DataBatch, columns: Optional[Union[str, List[str]]]=None, threshold: float=0.8, similarity_func: Optional[Callable[[str, str], float]]=None) -> Dict[str, Any]:
    """
    Identify potential duplicate records using fuzzy matching techniques.

    This function performs approximate string matching to find records that are
    similar but not necessarily identical. It's particularly useful for finding
    duplicates in text data where minor variations exist (typos, formatting differences, etc.).

    Args:
        data: Input DataBatch to check for fuzzy duplicates.
        columns: Column name(s) to consider for fuzzy matching.
                If None, all string columns are considered.
        threshold: Similarity threshold (0-1) above which records are considered duplicates.
                  Higher values require closer matches.
        similarity_func: Custom function to compute similarity between records.
                        If None, uses default string similarity metric.

    Returns:
        dict: A dictionary containing:
            - 'potential_duplicates' (List[tuple]): Pairs of indices identified as potential duplicates.
            - 'similarity_scores' (List[float]): Similarity scores for each pair.
            - 'duplicate_groups' (List[List[int]]): Groups of indices that are transitively similar.
    """
    if data_batch.data is None or len(data_batch.data) == 0:
        return {'potential_duplicates': [], 'similarity_scores': [], 'duplicate_groups': []}
    data = np.array(data_batch.data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if columns is None:
        columns_to_check = []
        for col_idx in range(data.shape[1]):
            sample_values = data[:min(10, len(data)), col_idx]
            string_count = sum((1 for val in sample_values if isinstance(val, str)))
            if string_count / len(sample_values) > 0.5:
                columns_to_check.append(col_idx)
        if not columns_to_check:
            columns_to_check = list(range(data.shape[1]))
    else:
        if isinstance(columns, str):
            columns = [columns]
        if hasattr(data_batch, 'columns') and data_batch.columns:
            col_name_to_idx = {name: idx for (idx, name) in enumerate(data_batch.columns)}
            columns_to_check = [col_name_to_idx[col] for col in columns if col in col_name_to_idx]
        else:
            columns_to_check = [col for col in columns if isinstance(col, int) and 0 <= col < data.shape[1]]
    if similarity_func is None:

        def default_similarity(a, b):
            if a is None or b is None:
                return 0.0
            (str_a, str_b) = (str(a), str(b))
            return SequenceMatcher(None, str_a, str_b).ratio()
        similarity_func = default_similarity
    potential_duplicates = []
    similarity_scores = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            total_similarity = 0.0
            valid_columns = 0
            for col_idx in columns_to_check:
                if col_idx < data.shape[1]:
                    val_i = data[i, col_idx]
                    val_j = data[j, col_idx]
                    if (val_i is None or val_i == '') and (val_j is None or val_j == ''):
                        continue
                    sim = similarity_func(val_i, val_j)
                    total_similarity += sim
                    valid_columns += 1
            if valid_columns > 0:
                avg_similarity = total_similarity / valid_columns
                if avg_similarity >= threshold:
                    potential_duplicates.append((i, j))
                    similarity_scores.append(avg_similarity)
    adj_list = defaultdict(set)
    for ((i, j), score) in zip(potential_duplicates, similarity_scores):
        adj_list[i].add(j)
        adj_list[j].add(i)
    visited = set()
    duplicate_groups = []
    for i in range(len(data)):
        if i not in visited and adj_list[i]:
            group = []
            queue = [i]
            visited.add(i)
            while queue:
                node = queue.pop(0)
                group.append(node)
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(group) > 1:
                duplicate_groups.append(sorted(group))
    return {'potential_duplicates': potential_duplicates, 'similarity_scores': similarity_scores, 'duplicate_groups': duplicate_groups}

def jaccard_similarity_deduplication(data_batch: DataBatch, columns: Optional[Union[str, List[str]]]=None, threshold: float=0.9) -> Dict[str, Any]:
    """
    Perform deduplication based on Jaccard similarity between records.

    This function computes Jaccard similarity coefficients between records based on
    set operations of their values. It's especially effective for categorical data
    or tokenized text where duplicates can be identified by comparing the overlap
    of their elements.

    Args:
        data: Input DataBatch to deduplicate using Jaccard similarity.
        columns: Column name(s) to consider for Jaccard similarity computation.
                If None, all columns are considered.
        threshold: Jaccard similarity threshold (0-1) above which records are considered duplicates.
                  Higher values require more similar records.

    Returns:
        dict: A dictionary containing:
            - 'duplicate_pairs' (List[tuple]): Pairs of indices identified as duplicates.
            - 'jaccard_scores' (List[float]): Jaccard similarity scores for each pair.
            - 'records_to_remove' (List[int]): Indices of records recommended for removal.
            - 'representative_records' (List[int]): Indices of records chosen as representatives.
    """
    if not hasattr(data_batch.data, '__len__') or len(data_batch.data) == 0:
        return {'duplicate_pairs': [], 'jaccard_scores': [], 'records_to_remove': [], 'representative_records': []}
    if len(data_batch.data) == 1:
        return {'duplicate_pairs': [], 'jaccard_scores': [], 'records_to_remove': [], 'representative_records': [0]}
    if columns is None:
        if data_batch.feature_names:
            use_columns = data_batch.feature_names
        else:
            first_row = data_batch.data[0] if len(data_batch.data) > 0 else []
            if hasattr(first_row, '__len__'):
                use_columns = list(range(len(first_row)))
            else:
                use_columns = [0]
    elif isinstance(columns, str):
        use_columns = [columns]
    else:
        use_columns = columns
    if data_batch.feature_names:
        missing_cols = [col for col in use_columns if col not in data_batch.feature_names]
        if missing_cols:
            raise ValueError(f'Columns {missing_cols} not found in feature_names')
        col_indices = [data_batch.feature_names.index(col) for col in use_columns]
    else:
        col_indices = use_columns
    if isinstance(data_batch.data, np.ndarray):
        processed_data = data_batch.data
    else:
        try:
            processed_data = np.array(data_batch.data, dtype=object)
        except Exception:
            processed_data = np.array(data_batch.data, dtype=object)
    if processed_data.ndim == 1:
        processed_data = processed_data.reshape(-1, 1)
    if col_indices:
        try:
            selected_data = processed_data[:, col_indices]
        except IndexError:
            selected_data = processed_data
    else:
        selected_data = processed_data
    row_sets = []
    for i in range(len(selected_data)):
        if selected_data.ndim == 1:
            row_value = selected_data[i]
            row_set = {row_value} if row_value is not None else set()
        else:
            row_values = selected_data[i]
            row_set = set()
            for val in row_values:
                if val is not None:
                    try:
                        row_set.add(val)
                    except TypeError:
                        row_set.add(str(val))
        row_sets.append(row_set)
    duplicate_pairs = []
    jaccard_scores = []
    n_records = len(row_sets)
    for i in range(n_records):
        for j in range(i + 1, n_records):
            set_i = row_sets[i]
            set_j = row_sets[j]
            if len(set_i) == 0 and len(set_j) == 0:
                jaccard_score = 1.0
            elif len(set_i.union(set_j)) == 0:
                jaccard_score = 1.0
            else:
                intersection_size = len(set_i.intersection(set_j))
                union_size = len(set_i.union(set_j))
                jaccard_score = intersection_size / union_size if union_size > 0 else 0.0
            if jaccard_score >= threshold:
                duplicate_pairs.append((i, j))
                jaccard_scores.append(jaccard_score)
    if not duplicate_pairs:
        representative_records = list(range(n_records))
        records_to_remove = []
    else:
        records_to_remove = set()
        representative_records = set()
        processed_records = set()
        paired_scores = list(zip(duplicate_pairs, jaccard_scores))
        paired_scores.sort(key=lambda x: x[1], reverse=True)
        for ((i, j), score) in paired_scores:
            if i not in processed_records and j not in processed_records:
                representative_records.add(i)
                records_to_remove.add(j)
                processed_records.add(i)
                processed_records.add(j)
            elif i in processed_records and j not in processed_records:
                records_to_remove.add(j)
                processed_records.add(j)
            elif j in processed_records and i not in processed_records:
                records_to_remove.add(i)
                processed_records.add(i)
        all_records = set(range(n_records))
        unprocessed = all_records - processed_records
        representative_records.update(unprocessed)
        representative_records = sorted(list(representative_records))
        records_to_remove = sorted(list(records_to_remove))
    return {'duplicate_pairs': duplicate_pairs, 'jaccard_scores': jaccard_scores, 'records_to_remove': records_to_remove, 'representative_records': representative_records}