from general.structures.data_batch import DataBatch
from general.base_classes.validator_base import BaseValidator
from typing import Dict, Any, Optional, List, Union
import numpy as np
import json
import warnings
import pandas as pd
from scipy import stats
import os


# ...(code omitted)...


class DataQualityReportGenerator(BaseValidator):
    """
    Generate comprehensive data quality reports for datasets.
    
    This validator analyzes input data and produces detailed quality reports
    including metrics, summaries, and potential issues. It extends BaseValidator
    to integrate with the validation framework while focusing specifically on
    report generation capabilities.
    
    Attributes:
        report_content (Dict[str, Any]): Stores the latest generated report
        include_visualizations (bool): Whether to include visualization data
        metrics_config (Dict[str, Any]): Configuration for quality metrics calculation
    """

    def __init__(self, name: Optional[str]=None, include_visualizations: bool=False, metrics_config: Optional[Dict[str, Any]]=None):
        """
        Initialize the DataQualityReportGenerator.
        
        Args:
            name: Optional name for the validator
            include_visualizations: Whether to include visualization data in reports
            metrics_config: Configuration dictionary for quality metrics calculation
        """
        super().__init__(name)
        self.include_visualizations = include_visualizations
        self.metrics_config = metrics_config
        self.report_content: Dict[str, Any] = {}

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Generate a data quality report for the input data.
        
        This method creates a comprehensive quality report and stores it in
        report_content attribute. The validation always passes unless there
        are critical errors in processing.
        
        Args:
            data: DataBatch to analyze and generate report for
            **kwargs: Additional parameters for report generation
            
        Returns:
            bool: Always True unless critical processing errors occur
        """
        try:
            self.reset_validation_state()
            self.report_content = {'dataset_info': {}, 'data_quality_metrics': {}, 'potential_issues': [], 'column_summaries': {}, 'overall_assessment': {}}
            self.report_content['dataset_info'] = {'shape': data.get_shape(), 'batch_id': data.batch_id, 'has_labels': data.is_labeled(), 'feature_names': data.feature_names, 'sample_count': len(data.data) if hasattr(data.data, '__len__') else 0}
            if isinstance(data.data, np.ndarray):
                self._analyze_numerical_data(data.data, data.feature_names)
            elif isinstance(data.data, list):
                try:
                    np_data = np.array(data.data)
                    self._analyze_numerical_data(np_data, data.feature_names)
                except:
                    self._analyze_mixed_data(data.data, data.feature_names)
            else:
                self.add_warning('Unsupported data type for detailed analysis')
                self.report_content['data_quality_metrics']['basic_info'] = {'data_type': type(data.data).__name__}
            if data.is_labeled():
                self._analyze_labels(data.labels)
            self._calculate_overall_assessment()
            self.report_content['columns'] = list(self.report_content.get('column_summaries', {}).keys())
            return True
        except Exception as e:
            self.add_error(f'Critical error during report generation: {str(e)}')
            return False

    def _analyze_numerical_data(self, data: np.ndarray, feature_names: Optional[List[str]]=None):
        """Analyze numerical data and populate quality metrics."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        (n_rows, n_cols) = data.shape
        column_names = feature_names or [f'col_{i}' for i in range(n_cols)]
        completeness_metrics = {}
        statistical_measures = {}
        distribution_characteristics = {}
        column_summaries = {}
        potential_issues = []
        for i in range(n_cols):
            col_data = data[:, i]
            col_name = column_names[i]
            valid_mask = ~np.isnan(col_data) & ~np.isinf(col_data)
            valid_data = col_data[valid_mask]
            missing_count = np.sum(~valid_mask)
            valid_count = np.sum(valid_mask)
            completeness_rate = np.sum(valid_mask) / len(col_data) if len(col_data) > 0 else 0
            completeness_metrics[col_name] = {'completeness_rate': completeness_rate, 'missing_count': missing_count, 'valid_count': valid_count}
            if valid_count > 0:
                mean_val = np.mean(valid_data)
                median_val = np.median(valid_data)
                std_val = np.std(valid_data)
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                q25 = np.percentile(valid_data, 25)
                q75 = np.percentile(valid_data, 75)
                statistical_measures[col_name] = {'mean': mean_val, 'median': median_val, 'std': std_val, 'min': min_val, 'max': max_val, 'q25': q25, 'q75': q75}
                skewness = self._calculate_skewness(valid_data)
                kurtosis = self._calculate_kurtosis(valid_data)
                distribution_characteristics[col_name] = {'skewness': skewness, 'kurtosis': kurtosis}
                if abs(skewness) > 2:
                    potential_issues.append(f'Column {col_name} has high skewness ({skewness:.2f})')
                if abs(kurtosis) > 7:
                    potential_issues.append(f'Column {col_name} has high kurtosis ({kurtosis:.2f})')
            else:
                statistical_measures[col_name] = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'q25': np.nan, 'q75': np.nan}
                distribution_characteristics[col_name] = {'skewness': np.nan, 'kurtosis': np.nan}
            column_summaries[col_name] = {'data_type': 'numerical', 'completeness_rate': completeness_rate}
        self.report_content['data_quality_metrics'] = {'completeness': completeness_metrics, 'statistical_measures': statistical_measures, 'distribution_characteristics': distribution_characteristics}
        self.report_content['column_summaries'] = column_summaries
        self.report_content['potential_issues'].extend(potential_issues)

    def _analyze_mixed_data(self, data: List[Any], feature_names: Optional[List[str]]=None):
        """Analyze mixed-type data and populate quality metrics."""
        n_rows = len(data)
        if n_rows == 0:
            return
        if isinstance(data[0], (list, tuple)):
            n_cols = len(data[0])
        else:
            n_cols = 1
            data = [[item] for item in data]
        column_names = feature_names or [f'col_{i}' for i in range(n_cols)]
        completeness_metrics = {}
        column_summaries = {}
        for i in range(n_cols):
            col_data = [row[i] if i < len(row) else None for row in data]
            col_name = column_names[i]
            valid_mask = [val is not None and val is not np.nan and (str(val).lower() not in ['nan', 'none', 'null', '']) for val in col_data]
            missing_count = np.sum([not v for v in valid_mask])
            valid_count = np.sum(valid_mask)
            completeness_rate = valid_count / len(col_data) if len(col_data) > 0 else 0
            completeness_metrics[col_name] = {'completeness_rate': completeness_rate, 'missing_count': missing_count, 'valid_count': valid_count}
            column_summaries[col_name] = {'data_type': 'mixed', 'completeness_rate': completeness_rate}
        self.report_content['data_quality_metrics']['completeness'] = completeness_metrics
        self.report_content['column_summaries'] = column_summaries

    def _analyze_labels(self, labels: Union[np.ndarray, List[Any]]):
        """Analyze label data and populate quality metrics."""
        if isinstance(labels, list):
            labels = np.array(labels)
        unique_labels = np.unique(labels)
        label_counts = {str(label): np.sum(labels == label) for label in unique_labels}
        self.report_content['label_analysis'] = {'unique_labels': len(unique_labels), 'label_distribution': label_counts}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        skewness = np.sum(((data - mean) / std) ** 3) / n
        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        n = len(data)
        kurtosis = np.sum(((data - mean) / std) ** 4) / n - 3
        return kurtosis

    def _detect_column_issues(self, data: np.ndarray, column_names: List[str]):
        """Detect potential issues in columns."""
        issues = []
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        (n_rows, n_cols) = data.shape
        for i in range(n_cols):
            col_data = data[:, i]
            col_name = column_names[i]
            valid_mask = ~np.isnan(col_data) & ~np.isinf(col_data)
            valid_data = col_data[valid_mask]
            if len(valid_data) == 0:
                issues.append(f'Column {col_name} has no valid data')
                continue
            zero_variance = np.var(valid_data) == 0
            if zero_variance:
                issues.append(f'Column {col_name} has zero variance')
            unique_ratio = len(np.unique(valid_data)) / len(valid_data)
            if unique_ratio < 0.01:
                issues.append(f'Column {col_name} has very low diversity (unique ratio: {unique_ratio:.4f})')
        self.report_content['potential_issues'].extend(issues)

    def _calculate_overall_assessment(self):
        """Calculate overall data quality assessment."""
        completeness_metrics = self.report_content.get('data_quality_metrics', {}).get('completeness', {})
        if not completeness_metrics:
            self.report_content['overall_assessment'] = {'average_completeness': 0, 'issue_count': 0, 'quality_score': 0}
            return
        completeness_rates = [metrics['completeness_rate'] for metrics in completeness_metrics.values()]
        average_completeness = np.mean(completeness_rates) if completeness_rates else 0
        issue_count = len(self.report_content.get('potential_issues', []))
        quality_score = max(0, min(1, average_completeness - 0.1 * issue_count))
        self.report_content['overall_assessment'] = {'average_completeness': average_completeness, 'issue_count': issue_count, 'quality_score': quality_score}

    def get_quality_report(self) -> Dict[str, Any]:
        """
        Retrieve the most recently generated quality report.
        
        Returns:
            Dict[str, Any]: Complete quality report or empty dict if none exists
        """
        return self.report_content.copy() if self.report_content else {}

    def export_report(self, format: str='json', destination: Optional[str]=None) -> Union[str, Dict[str, Any]]:
        """
        Export the quality report in specified format.
        
        Args:
            format: Output format ('json', 'dict', 'html', 'pdf')
            destination: Optional file path to save the report
            
        Returns:
            Union[str, Dict[str, Any]]: Formatted report content
        """
        if format.lower() == 'json':
            report_str = json.dumps(self.report_content, indent=2)
            if destination:
                with open(destination, 'w') as f:
                    f.write(report_str)
            return report_str
        elif format.lower() == 'dict':
            return self.report_content.copy()
        elif format.lower() == 'html':
            return self._generate_html_report()
        elif format.lower() == 'pdf':
            return 'PDF export not implemented'
        else:
            raise ValueError(f'Unsupported format: {format}')

    def _generate_html_report(self) -> str:
        """Generate HTML representation of the report."""
        html_content = '<html><head><title>Data Quality Report</title></head><body>'
        html_content += f'<h1>Data Quality Report - {self.name}</h1>'
        html_content += '<h2>Dataset Information</h2><ul>'
        for (key, value) in self.report_content.get('dataset_info', {}).items():
            html_content += f'<li>{key}: {value}</li>'
        html_content += '</ul>'
        html_content += '<h2>Overall Assessment</h2><ul>'
        for (key, value) in self.report_content.get('overall_assessment', {}).items():
            html_content += f'<li>{key}: {value}</li>'
        html_content += '</ul>'
        html_content += '<h2>Potential Issues</h2><ul>'
        for issue in self.report_content.get('potential_issues', []):
            html_content += f'<li>{issue}</li>'
        html_content += '</ul></body></html>'
        return html_content