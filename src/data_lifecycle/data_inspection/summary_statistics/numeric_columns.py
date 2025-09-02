from typing import Dict, Any, Optional, Union
from general.structures.data_batch import DataBatch
import numpy as np
import pandas as pd

class NumericColumnSummaryStatistics:

    def __init__(self, include_quantiles: bool=True, custom_quantiles: Optional[list]=None):
        """
        Initialize the summary statistics calculator.
        
        Args:
            include_quantiles (bool): Whether to compute quantile statistics (default: True)
            custom_quantiles (Optional[list]): Custom quantiles to compute if include_quantiles is True
        """
        self.include_quantiles = include_quantiles
        self.custom_quantiles = custom_quantiles
        if self.custom_quantiles is None and self.include_quantiles:
            self.custom_quantiles = [0.25, 0.5, 0.75]
        self.statistics: Dict[str, Dict[str, float]] = {}
        self.column_names: list = []

    def compute_statistics(self, data: DataBatch) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Compute summary statistics for all numeric columns in the provided data batch.
        
        This method analyzes each numeric column in the data batch and computes:
        - Count of values
        - Mean
        - Standard deviation
        - Minimum value
        - Maximum value
        - Quartiles (Q1, Median, Q3) or custom quantiles
        - Skewness
        - Kurtosis
        
        Args:
            data (DataBatch): Input data batch containing numeric columns to analyze
            
        Returns:
            Dict[str, Dict[str, Union[float, int]]]: A dictionary mapping column names to their statistics.
                                                   Each column's statistics include keys like 'mean', 'std',
                                                   'min', 'max', 'count', and optionally quantiles.
        
        Raises:
            ValueError: If the data batch contains no numeric columns or is improperly formatted
        """
        raw_data = data.data
        feature_names = data.feature_names
        if isinstance(raw_data, pd.DataFrame):
            df = raw_data
        elif isinstance(raw_data, np.ndarray):
            if raw_data.ndim == 0:
                df = pd.DataFrame()
            elif raw_data.ndim == 1:
                if feature_names is None:
                    feature_names = ['col_0']
                elif len(feature_names) == 0:
                    feature_names = ['col_0']
                df = pd.DataFrame(raw_data.reshape(-1, 1), columns=feature_names[:1])
            else:
                if feature_names is None:
                    feature_names = [f'col_{i}' for i in range(raw_data.shape[1])]
                df = pd.DataFrame(raw_data, columns=feature_names)
        elif isinstance(raw_data, list):
            if len(raw_data) == 0:
                df = pd.DataFrame()
            elif all((isinstance(row, (list, np.ndarray)) for row in raw_data)):
                if feature_names is None:
                    feature_names = [f'col_{i}' for i in range(len(raw_data[0]) if raw_data else 0)]
                df = pd.DataFrame(raw_data, columns=feature_names)
            else:
                if feature_names is None:
                    feature_names = ['col_0']
                elif len(feature_names) == 0:
                    feature_names = ['col_0']
                df = pd.DataFrame({feature_names[0]: raw_data})
        else:
            raise ValueError('Unsupported data format in DataBatch. Expected DataFrame, numpy array or list of lists.')
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            self.column_names = []
            self.statistics = {}
            return self.statistics
        self.column_names = list(numeric_df.columns)
        self.statistics = {}
        for col in self.column_names:
            series = numeric_df[col].dropna()
            stats = {}
            stats['count'] = len(series)
            if stats['count'] == 0:
                stats.update({'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan})
                if self.include_quantiles:
                    quantile_names = [f'{int(q * 100)}%' for q in self.custom_quantiles]
                    for q_name in quantile_names:
                        stats[q_name] = np.nan
                self.statistics[col] = stats
                continue
            stats['mean'] = float(np.mean(series))
            stats['std'] = float(np.std(series, ddof=0)) if stats['count'] > 1 else 0.0
            stats['min'] = float(np.min(series))
            stats['max'] = float(np.max(series))
            if self.include_quantiles:
                quantiles = np.quantile(series, self.custom_quantiles)
                quantile_names = [f'{int(q * 100)}%' for q in self.custom_quantiles]
                for (i, q_name) in enumerate(quantile_names):
                    stats[q_name] = float(quantiles[i])
            if stats['count'] > 1:
                mean = stats['mean']
                std = stats['std']
                if std == 0:
                    stats['skewness'] = 0.0
                    stats['kurtosis'] = -3.0
                else:
                    m3 = np.mean(((series - mean) / std) ** 3)
                    skewness = m3 * (stats['count'] / ((stats['count'] - 1) * (stats['count'] - 2))) if stats['count'] > 2 else m3
                    stats['skewness'] = float(skewness)
                    m4 = np.mean(((series - mean) / std) ** 4)
                    kurtosis = (stats['count'] * (stats['count'] + 1) * m4 - 3 * (stats['count'] - 1) ** 2) / ((stats['count'] - 1) * (stats['count'] - 2) * (stats['count'] - 3)) if stats['count'] > 3 else m4 - 3
                    stats['kurtosis'] = float(kurtosis)
            else:
                stats['skewness'] = np.nan
                stats['kurtosis'] = np.nan
            self.statistics[col] = stats
        return self.statistics

    def get_column_statistics(self, column_name: str) -> Dict[str, Union[float, int]]:
        """
        Retrieve computed statistics for a specific column.
        
        Args:
            column_name (str): Name of the column to retrieve statistics for
            
        Returns:
            Dict[str, Union[float, int]]: Statistics for the specified column
            
        Raises:
            KeyError: If statistics haven't been computed yet or column doesn't exist
        """
        if not self.statistics:
            raise KeyError('No statistics have been computed yet. Call compute_statistics first.')
        if column_name not in self.statistics:
            raise KeyError(f"Column '{column_name}' not found in computed statistics.")
        return self.statistics[column_name]

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all computed statistics to a dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary containing all computed statistics and metadata
        """
        if self.include_quantiles and self.custom_quantiles is not None:
            custom_quantiles_export = self.custom_quantiles
        else:
            custom_quantiles_export = None
        return {'statistics': self.statistics, 'column_names': self.column_names, 'include_quantiles': self.include_quantiles, 'custom_quantiles': custom_quantiles_export}