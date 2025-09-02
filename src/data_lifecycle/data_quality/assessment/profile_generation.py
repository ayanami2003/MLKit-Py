from typing import Dict, List, Optional, Tuple, Any, Union
from general.structures.data_batch import DataBatch
import numpy as np
import pandas as pd
ANALYSIS_TYPES = {'summary_statistics', 'data_types', 'missing_values', 'distribution_analysis', 'bivariate_analysis', 'multivariate_analysis'}


# ...(code omitted)...


def generate_data_quality_report(data_batch: DataBatch, include: Optional[List[str]]=None, exclude: Optional[List[str]]=None) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report for a data batch.

    This function creates a detailed report that consolidates various data profiling results
    including summary statistics, data type information, missing value analysis, and
    distribution characteristics. The report can be customized to include or exclude
    specific analysis sections.

    Args:
        data_batch (DataBatch): The input data batch to analyze and report on.
        include (Optional[list]): List of analysis types to include in the report 
                                 (e.g., ['summary', 'types', 'missing', 'distribution']).
                                 If None, all analyses are included.
        exclude (Optional[list]): List of analysis types to exclude from the report.
                                 Takes precedence over 'include' if both are specified.

    Returns:
        Dict[str, Any]: A comprehensive data quality report containing:
                       - 'summary_statistics': Output from compute_summary_statistics
                       - 'data_types': Inferred types from infer_data_types
                       - 'missing_analysis': Information about missing values
                       - 'distribution_info': Details about data distributions
                       - 'bivariate_analysis': Key bivariate relationships
                       - 'multivariate_analysis': Selected multivariate metrics

    Raises:
        ValueError: If conflicting or invalid analysis types are specified in include/exclude.
    """
    if include is not None:
        invalid_include = set(include) - ANALYSIS_TYPES
        if invalid_include:
            raise ValueError(f"Invalid analysis types in 'include': {invalid_include}")
    if exclude is not None:
        invalid_exclude = set(exclude) - ANALYSIS_TYPES
        if invalid_exclude:
            raise ValueError(f"Invalid analysis types in 'exclude': {invalid_exclude}")
    if include is None:
        selected_analyses = ANALYSIS_TYPES.copy()
    else:
        selected_analyses = set(include)
    if exclude is not None:
        selected_analyses = selected_analyses - set(exclude)
    report = {}
    if 'summary_statistics' in selected_analyses:
        try:
            report['summary_statistics'] = compute_summary_statistics(data_batch)
        except Exception as e:
            report['summary_statistics'] = {'error': str(e)}
    if 'data_types' in selected_analyses:
        try:
            report['data_types'] = infer_data_types(data_batch)
        except Exception as e:
            report['data_types'] = {'error': str(e)}
    if 'missing_values' in selected_analyses:
        try:
            if hasattr(data_batch.data, '__len__'):
                missing_report = {}
                if isinstance(data_batch.data, np.ndarray):
                    for (i, col_data) in enumerate(data_batch.data.T if data_batch.data.ndim > 1 else [data_batch.data]):
                        col_name = data_batch.feature_names[i] if data_batch.feature_names and i < len(data_batch.feature_names) else f'column_{i}'
                        missing_count = np.sum(np.isnan(col_data)) if col_data.dtype.kind in 'fc' else np.sum(pd.isnull(col_data))
                        missing_report[col_name] = {'missing_count': int(missing_count), 'missing_percentage': float(missing_count / len(col_data) * 100) if len(col_data) > 0 else 0.0}
                elif isinstance(data_batch.data, list) and len(data_batch.data) > 0:
                    if isinstance(data_batch.data[0], dict):
                        for key in data_batch.data[0].keys():
                            col_data = [row.get(key) for row in data_batch.data]
                            missing_count = sum((1 for x in col_data if pd.isnull(x)))
                            missing_report[key] = {'missing_count': missing_count, 'missing_percentage': missing_count / len(col_data) * 100 if len(col_data) > 0 else 0.0}
                    else:
                        for (i, col_data) in enumerate(zip(*data_batch.data) if data_batch.data else []):
                            col_name = data_batch.feature_names[i] if data_batch.feature_names and i < len(data_batch.feature_names) else f'column_{i}'
                            missing_count = sum((1 for x in col_data if pd.isnull(x)))
                            missing_report[col_name] = {'missing_count': missing_count, 'missing_percentage': missing_count / len(col_data) * 100 if len(col_data) > 0 else 0.0}
                report['missing_values'] = missing_report
            else:
                report['missing_values'] = {}
        except Exception as e:
            report['missing_values'] = {'error': str(e)}
    if 'distribution_analysis' in selected_analyses:
        report['distribution_analysis'] = {'message': 'Distribution analysis not yet implemented'}
    if 'bivariate_analysis' in selected_analyses:
        try:
            report['bivariate_analysis'] = perform_bivariate_analysis(data_batch)
        except Exception as e:
            report['bivariate_analysis'] = {'error': str(e)}
    if 'multivariate_analysis' in selected_analyses:
        try:
            report['multivariate_analysis'] = perform_multivariate_analysis(data_batch)
        except Exception as e:
            report['multivariate_analysis'] = {'error': str(e)}
    return report