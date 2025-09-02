from general.base_classes.validator_base import BaseValidator
from typing import Union, Optional, Dict, Any, List
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression

class PredictivePowerAssessor(BaseValidator):
    """
    Assesses the predictive power of individual features in a dataset.
    
    This component evaluates how well each feature can predict the target variable
    using various statistical measures and scoring methods. It supports both
    classification and regression targets.
    
    Attributes
    ----------
    scoring_method : str
        Method used to assess predictive power ('mutual_info', 'anova_f', 'chi2')
    task_type : str
        Type of prediction task ('classification' or 'regression')
    power_threshold : float
        Minimum predictive power score to consider a feature useful
    feature_scores : Dict[str, float]
        Cached scores for each feature after assessment
    """

    def __init__(self, scoring_method: str='mutual_info', task_type: str='classification', power_threshold: float=0.01, name: Optional[str]=None):
        super().__init__(name)
        self.scoring_method = scoring_method
        self.task_type = task_type
        self.power_threshold = power_threshold
        self.feature_scores: Dict[str, float] = {}

    def validate(self, data: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> bool:
        """
        Assess predictive power of features and validate against threshold.
        
        Computes predictive power scores for all features and determines if they meet
        the minimum threshold for usefulness. Results are stored in feature_scores.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to assess. If FeatureSet, uses feature names for reporting.
        y : Union[np.ndarray, list]
            Target values for computing predictive power
        **kwargs : dict
            Additional parameters for scoring methods
            
        Returns
        -------
        bool
            True if any feature meets the predictive power threshold, False otherwise
            
        Raises
        ------
        ValueError
            If data and target dimensions don't match or unsupported method specified
        """
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        if X.shape[0] != len(y):
            raise ValueError(f'Number of samples in data ({X.shape[0]}) does not match number of targets ({len(y)})')
        if self.scoring_method == 'mutual_info':
            if self.task_type == 'classification':
                scores = mutual_info_classif(X, y, **kwargs)
            else:
                scores = mutual_info_regression(X, y, **kwargs)
        elif self.scoring_method == 'anova_f':
            if self.task_type == 'classification':
                scores = f_classif(X, y, **kwargs)[0]
            else:
                scores = f_regression(X, y, **kwargs)[0]
        elif self.scoring_method == 'chi2':
            if self.task_type != 'classification':
                raise ValueError('Chi-square test is only supported for classification tasks')
            if np.any(X < 0):
                raise ValueError('Chi-square test requires non-negative feature values')
            scores = chi2(X, y, **kwargs)[0]
        else:
            raise ValueError(f'Unsupported scoring method: {self.scoring_method}')
        self.feature_scores = dict(zip(feature_names, scores))
        return any((score >= self.power_threshold for score in scores))

    def get_feature_ranking(self) -> Dict[str, float]:
        """
        Get ranked features by their predictive power scores.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names/indices to their predictive power scores,
            sorted in descending order of importance
        """
        return dict(sorted(self.feature_scores.items(), key=lambda item: item[1], reverse=True))

class MulticollinearityDiagnostic(BaseValidator):
    """
    Diagnoses multicollinearity issues among features in a dataset.
    
    This component detects highly correlated features that can cause instability
    in model training and interpretation. It computes several diagnostic metrics
    including Variance Inflation Factor (VIF) and correlation matrices.
    
    Attributes
    ----------
    vif_threshold : float
        Threshold above which features are considered multicollinear (default: 5.0)
    correlation_threshold : float
        Threshold above which pairwise correlations are flagged (default: 0.8)
    diagnostic_metrics : Dict[str, Any]
        Cached diagnostic results including VIF scores and correlation data
    collinear_groups : List[List[str]]
        Groups of features identified as multicollinear
    """

    def __init__(self, vif_threshold: float=5.0, correlation_threshold: float=0.8, name: Optional[str]=None):
        super().__init__(name)
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.diagnostic_metrics: Dict[str, Any] = {}
        self.collinear_groups: List[List[str]] = []

    def validate(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> bool:
        """
        Perform multicollinearity diagnostics on the input features.
        
        Computes VIF scores and correlation matrices to identify problematic
        multicollinearity. Results are stored in diagnostic_metrics and
        collinear_groups attributes.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to analyze for multicollinearity
        **kwargs : dict
            Additional parameters for computation methods
            
        Returns
        -------
        bool
            True if no severe multicollinearity detected, False if issues found
            
        Raises
        ------
        ValueError
            If data is not 2D or contains non-numeric features
        """
        self.reset_validation_state()
        if isinstance(data, FeatureSet):
            features = data.features
            feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(features.shape[1])]
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError('Data must be a 2D array')
            features = data
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        else:
            raise ValueError('Data must be either a FeatureSet or numpy array')
        if not np.issubdtype(features.dtype, np.number):
            raise ValueError('All features must be numeric')
        (n_samples, n_features) = features.shape
        if n_features < 2:
            raise ValueError('At least two features are required for multicollinearity analysis')
        correlation_matrix = np.corrcoef(features, rowvar=False)
        correlation_matrix = np.nan_to_num(correlation_matrix)
        vif_scores = {}
        for i in range(n_features):
            if n_features > 1:
                X_others = np.delete(features, i, axis=1)
                y_target = features[:, i]
                reg = LinearRegression()
                reg.fit(X_others, y_target)
                r_squared = reg.score(X_others, y_target)
                if r_squared < 1.0:
                    vif = 1.0 / (1.0 - r_squared)
                else:
                    vif = float('inf')
                vif_scores[feature_names[i]] = vif
            else:
                vif_scores[feature_names[i]] = 1.0
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_val = abs(correlation_matrix[i, j])
                if corr_val >= self.correlation_threshold:
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr_val))
        collinear_groups = self._group_correlated_features(high_corr_pairs, feature_names)
        self.diagnostic_metrics = {'vif_scores': vif_scores, 'correlation_matrix': correlation_matrix, 'high_correlation_pairs': high_corr_pairs, 'feature_names': feature_names}
        self.collinear_groups = collinear_groups
        has_high_vif = any((vif > self.vif_threshold for vif in vif_scores.values()))
        has_collinear_groups = len(collinear_groups) > 0
        if has_high_vif:
            self.add_warning(f'Some features have VIF scores above threshold ({self.vif_threshold})')
        if has_collinear_groups:
            self.add_warning(f'Found {len(collinear_groups)} groups of highly correlated features')
        return not (has_high_vif or has_collinear_groups)

    def _group_correlated_features(self, high_corr_pairs: List[tuple], feature_names: List[str]) -> List[List[str]]:
        """
        Group correlated features into clusters.
        
        Parameters
        ----------
        high_corr_pairs : List[tuple]
            List of tuples (feature1, feature2, correlation_value) with high correlation
        feature_names : List[str]
            List of all feature names
            
        Returns
        -------
        List[List[str]]
            List of groups, each containing names of correlated features
        """
        if not high_corr_pairs:
            return []
        adj_list = {name: set() for name in feature_names}
        for (feat1, feat2, _) in high_corr_pairs:
            adj_list[feat1].add(feat2)
            adj_list[feat2].add(feat1)
        visited = set()
        groups = []
        for feature in feature_names:
            if feature not in visited and adj_list[feature]:
                group = []
                queue = [feature]
                visited.add(feature)
                while queue:
                    current = queue.pop(0)
                    group.append(current)
                    for neighbor in adj_list[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(group) > 1:
                    groups.append(group)
        return groups

    def get_vif_scores(self) -> Dict[str, float]:
        """
        Get Variance Inflation Factor scores for all features.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names/indices to their VIF scores
        """
        if 'vif_scores' not in self.diagnostic_metrics:
            raise RuntimeError('VIF scores not computed. Run validate() first.')
        return self.diagnostic_metrics['vif_scores'].copy()

    def suggest_removals(self) -> List[str]:
        """
        Suggest features to remove to resolve multicollinearity.
        
        Based on VIF scores and correlation groups, identifies a minimal set
        of features whose removal would resolve multicollinearity issues.
        
        Returns
        -------
        List[str]
            List of feature names/indices recommended for removal
        """
        if not self.diagnostic_metrics or not self.collinear_groups:
            return []
        vif_scores = self.diagnostic_metrics.get('vif_scores', {})
        features_to_remove = set()
        for group in self.collinear_groups:
            sorted_features = sorted(group, key=lambda x: vif_scores.get(x, 0), reverse=True)
            features_to_remove.update(sorted_features[1:])
        for (feature, vif) in vif_scores.items():
            if feature not in features_to_remove and vif > self.vif_threshold:
                in_group = any((feature in group for group in self.collinear_groups))
                if not in_group:
                    features_to_remove.add(feature)
        return list(features_to_remove)