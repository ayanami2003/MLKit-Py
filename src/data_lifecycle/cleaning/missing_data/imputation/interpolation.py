from typing import Dict, List, Optional, Union, Any, Tuple
from general.base_classes.transformer_base import BaseTransformer
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


# ...(code omitted)...


class MissingnessCorrelationAnalyzer(BaseValidator):
    """
    Validator for analyzing correlations in missing data patterns.

    This validator examines pairwise correlations between missing indicators of features,
    helping to identify systematic relationships in missing data. It supports multiple
    correlation methods and can cluster features with similar missingness patterns.

    Attributes:
        correlation_method (str): Method for computing correlations ('pearson', 'spearman', 'mutual_info')
        threshold (float): Significance threshold for identifying meaningful correlations
        compute_conditional_probs (bool): Whether to compute conditional missingness probabilities
        clustering_algorithm (str): Algorithm for clustering ('kmeans', 'hierarchical')
        name (str): Name of the validator instance
    """

    def __init__(self, correlation_method: str='pearson', threshold: float=0.3, compute_conditional_probs: bool=False, clustering_algorithm: str='kmeans', name: Optional[str]=None):
        """
        Initialize the MissingnessCorrelationAnalyzer.

        Args:
            correlation_method (str): Correlation method to use ('pearson', 'spearman', 'mutual_info').
            threshold (float): Threshold for identifying significant correlations (0-1).
            compute_conditional_probs (bool): Whether to compute conditional probabilities.
            clustering_algorithm (str): Clustering algorithm ('kmeans', 'hierarchical').
            name (Optional[str]): Name of the validator instance.
        """
        super().__init__(name=name)
        if correlation_method not in ['pearson', 'spearman', 'mutual_info']:
            raise ValueError("correlation_method must be one of 'pearson', 'spearman', 'mutual_info'")
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        if clustering_algorithm not in ['kmeans', 'hierarchical']:
            raise ValueError("clustering_algorithm must be 'kmeans' or 'hierarchical'")
        self.correlation_method = correlation_method
        self.threshold = threshold
        self.compute_conditional_probs = compute_conditional_probs
        self.clustering_algorithm = clustering_algorithm
        self._validated = False
        self._correlation_matrix = None
        self._conditional_probs = None

    def validate(self, data: Union[DataBatch, FeatureSet, np.ndarray], **kwargs) -> bool:
        """
        Perform correlation analysis on input data.

        Args:
            data (Union[DataBatch, FeatureSet, np.ndarray]): Input data to analyze
            **kwargs: Additional parameters

        Returns:
            bool: Always returns True
        """
        if isinstance(data, np.ndarray):
            data_array = data
            feature_names = [f'feature_{i}' for i in range(data_array.shape[1])] if data_array.ndim > 1 else ['feature_0']
        elif isinstance(data, FeatureSet):
            data_array = data.features
            feature_names = data.feature_names or [f'feature_{i}' for i in range(data_array.shape[1])]
        elif isinstance(data, DataBatch):
            data_array = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data
            feature_names = data.feature_names or [f'feature_{i}' for i in range(data_array.shape[1])] if data_array.ndim > 1 else ['feature_0']
        else:
            raise TypeError('Unsupported data type. Expected np.ndarray, FeatureSet, or DataBatch.')
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        self._feature_names = feature_names
        missing_indicators = np.isnan(data_array).astype(int)
        if self.correlation_method in ['pearson', 'spearman']:
            import pandas as pd
            df = pd.DataFrame(missing_indicators, columns=feature_names)
            self._correlation_matrix = df.corr(method=self.correlation_method).values
        elif self.correlation_method == 'mutual_info':
            self._correlation_matrix = self._compute_mutual_info_correlation(missing_indicators)
        if self.compute_conditional_probs:
            self._conditional_probs = self._compute_conditional_probabilities(missing_indicators)
        else:
            self._conditional_probs = None
        self._validated = True
        return True

    def _compute_mutual_info_correlation(self, missing_indicators: np.ndarray) -> np.ndarray:
        """Compute mutual information correlation matrix."""
        n_features = missing_indicators.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    mi_matrix[i, j] = 1.0
                else:
                    mi_matrix[i, j] = self._mutual_information(missing_indicators[:, i], missing_indicators[:, j])
        max_mi = np.max(mi_matrix)
        if max_mi > 0:
            mi_matrix = mi_matrix / max_mi
        return mi_matrix

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two binary vectors."""
        px0 = np.mean(x == 0)
        px1 = np.mean(x == 1)
        py0 = np.mean(y == 0)
        py1 = np.mean(y == 1)
        p00 = np.mean((x == 0) & (y == 0))
        p01 = np.mean((x == 0) & (y == 1))
        p10 = np.mean((x == 1) & (y == 0))
        p11 = np.mean((x == 1) & (y == 1))
        mi = 0.0
        for (px, py, pxy) in [(px0, py0, p00), (px0, py1, p01), (px1, py0, p10), (px1, py1, p11)]:
            if pxy > 0 and px > 0 and (py > 0):
                mi += pxy * np.log(pxy / (px * py))
        return mi

    def _compute_conditional_probabilities(self, missing_indicators: np.ndarray) -> Dict[Tuple[str, str], float]:
        """Compute conditional probabilities P(missing in A | missing in B)."""
        n_features = missing_indicators.shape[1]
        cond_probs = {}
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    missing_i = missing_indicators[:, i] == 1
                    missing_j = missing_indicators[:, j] == 1
                    both_missing = np.mean(missing_i & missing_j)
                    prob_j = np.mean(missing_j)
                    if prob_j > 0:
                        cond_prob = both_missing / prob_j
                    else:
                        cond_prob = 0.0
                    feature_a = self._feature_names[i] if self._feature_names else f'feature_{i}'
                    feature_b = self._feature_names[j] if self._feature_names else f'feature_{j}'
                    cond_probs[feature_a, feature_b] = cond_prob
        return cond_probs

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Return computed correlation matrix of missing indicators.

        Returns:
            np.ndarray: Correlation matrix of missing indicators

        Raises:
            RuntimeError: If validate() has not been called yet
        """
        if not self._validated:
            raise RuntimeError('Correlation matrix not available. Call validate() first.')
        return self._correlation_matrix

    def get_significant_correlations(self) -> List[Tuple[str, str, float]]:
        """
        Extract feature pairs exceeding correlation threshold.

        Returns:
            List[Tuple[str, str, float]]: List of (feature1, feature2, correlation) tuples
        """
        if not self._validated:
            raise RuntimeError('No correlations available. Call validate() first.')
        significant = []
        n_features = self._correlation_matrix.shape[0]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = self._correlation_matrix[i, j]
                if abs(corr) >= self.threshold:
                    feature_a = self._feature_names[i] if self._feature_names else f'feature_{i}'
                    feature_b = self._feature_names[j] if self._feature_names else f'feature_{j}'
                    significant.append((feature_a, feature_b, corr))
        return significant

    def get_conditional_probabilities(self) -> Dict[Tuple[str, str], float]:
        """
        Provide conditional probability mapping when enabled.

        Returns:
            Dict[Tuple[str, str], float]: Mapping of (feature_i, feature_j) to P(missing_i | missing_j)

        Raises:
            RuntimeError: If conditional probabilities were not computed
        """
        if not self.compute_conditional_probs:
            raise RuntimeError('Conditional probabilities were not computed. Set compute_conditional_probs=True.')
        if not self._validated:
            raise RuntimeError('No conditional probabilities available. Call validate() first.')
        return self._conditional_probs

    def identify_missingness_clusters(self, n_clusters: Optional[int]=None) -> Dict[str, int]:
        """
        Group features with similar missingness patterns.

        Args:
            n_clusters (Optional[int]): Number of clusters. If None, determined automatically.

        Returns:
            Dict[str, int]: Mapping of feature names to cluster labels
        """
        if not self._validated:
            raise RuntimeError('No clustering available. Call validate() first.')
        corr_matrix = self._correlation_matrix
        distance_matrix = 1 - np.abs(corr_matrix)
        if n_clusters is None:
            n_clusters = min(max(2, corr_matrix.shape[0] // 3), 5)
        if self.clustering_algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(distance_matrix)
        else:
            try:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
            except TypeError:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            labels = clusterer.fit_predict(distance_matrix)
        feature_clusters = {}
        for (i, label) in enumerate(labels):
            feature_name = self._feature_names[i] if self._feature_names else f'feature_{i}'
            feature_clusters[feature_name] = int(label)
        return feature_clusters

class MultipleImputationConvergenceChecker(BaseValidator):
    """
    Validator for checking convergence of multiple imputation processes.

    This validator monitors the convergence of iterative multiple imputation algorithms
    by analyzing the stability of imputed values across iterations. It helps determine
    when the imputation process has stabilized and additional iterations are unlikely
    to significantly improve results.

    The checker implements several convergence diagnostics:
    - Rubin's rules for pooling estimates
    - Potential scale reduction factor (PSRF) aka R-hat statistic
    - Geweke diagnostic for chain convergence
    - Trace plot analysis for visual convergence assessment

    Attributes:
        convergence_threshold (float): Threshold for determining convergence
        diagnostic_methods (List[str]): List of convergence diagnostics to apply
        burn_in_period (int): Number of initial iterations to exclude from analysis
        name (str): Name of the validator instance
    """

    def __init__(self, convergence_threshold: float=1.05, diagnostic_methods: Optional[List[str]]=None, burn_in_period: int=10, name: Optional[str]=None):
        """
        Initialize the MultipleImputationConvergenceChecker.

        Args:
            convergence_threshold (float): Convergence threshold for diagnostic tests.
                                         For R-hat, values <1.05 typically indicate convergence.
                                         For other diagnostics, interpretation varies.
            diagnostic_methods (Optional[List[str]]): List of diagnostic methods to apply.
                                                    Options: ['rubin', 'psrf', 'geweke', 'trace']
                                                    If None, applies all available methods.
            burn_in_period (int): Number of initial iterations to exclude from convergence analysis.
                                These iterations are considered part of the burn-in period.
            name (Optional[str]): Name of the validator instance.
        """
        super().__init__(name=name)
        self.convergence_threshold = convergence_threshold
        self.diagnostic_methods = diagnostic_methods or ['rubin', 'psrf', 'geweke', 'trace']
        self.burn_in_period = burn_in_period
        self._imputation_history = []
        self._convergence_diagnostics = {}

    def validate(self, data: Union[List[DataBatch], List[FeatureSet], List[np.ndarray]], **kwargs) -> bool:
        """
        Check convergence of multiple imputation iterations.

        This method analyzes a sequence of imputed datasets to determine if the
        imputation process has converged according to specified diagnostic criteria.

        Args:
            data (Union[List[DataBatch], List[FeatureSet], List[np.ndarray]]): 
                 Sequence of imputed datasets from successive iterations
            **kwargs: Additional parameters for convergence checking

        Returns:
            bool: True if all enabled convergence diagnostics indicate convergence,
                  False otherwise
        """
        self.reset_validation_state()
        if not isinstance(data, list):
            self.add_error('Input data must be a list of imputed datasets')
            return False
        if len(data) == 0:
            self.add_error('Input data list is empty')
            return False
        self._imputation_history = []
        for item in data:
            self.add_imputation_result(item)
        if len(self._imputation_history) <= self.burn_in_period:
            self.add_error(f'Insufficient iterations for convergence analysis. Need more than burn_in_period ({self.burn_in_period}) iterations.')
            return False
        try:
            self.compute_convergence_diagnostics()
            convergence_status = self.get_convergence_status()
            return all((converged for (method, converged) in convergence_status.items() if method in self.diagnostic_methods))
        except Exception as e:
            self.add_error(f'Error computing convergence diagnostics: {str(e)}')
            return False

    def add_imputation_result(self, imputed_data: Union[DataBatch, FeatureSet, np.ndarray]) -> None:
        """
        Add a single imputation result to the history for convergence monitoring.

        This method allows incremental addition of imputation results as they become
        available during an iterative imputation process.

        Args:
            imputed_data (Union[DataBatch, FeatureSet, np.ndarray]): 
                         Imputed dataset from a single iteration
        """
        if isinstance(imputed_data, DataBatch):
            data_array = np.array(imputed_data.data)
        elif isinstance(imputed_data, FeatureSet):
            data_array = imputed_data.features
        elif isinstance(imputed_data, np.ndarray):
            data_array = imputed_data
        else:
            raise TypeError(f'Unsupported data type: {type(imputed_data)}. Supported types: DataBatch, FeatureSet, np.ndarray')
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        elif data_array.ndim != 2:
            raise ValueError(f'Data must be 1D or 2D, got {data_array.ndim}D')
        self._imputation_history.append(data_array)

    def compute_convergence_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute all configured convergence diagnostics.

        Calculates convergence metrics for all features based on the stored
        imputation history.

        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary where outer keys are
                                       diagnostic method names, inner keys are
                                       feature names, and values are diagnostic
                                       statistics
        """
        if len(self._imputation_history) <= self.burn_in_period:
            raise ValueError(f'Insufficient iterations for convergence analysis. Need more than burn_in_period ({self.burn_in_period}) iterations.')
        analysis_data = self._imputation_history[self.burn_in_period:]
        n_iterations = len(analysis_data)
        n_features = analysis_data[0].shape[1]
        self._convergence_diagnostics = {method: {} for method in self.diagnostic_methods}
        for feature_idx in range(n_features):
            feature_name = f'feature_{feature_idx}'
            feature_data = [iteration[:, feature_idx] for iteration in analysis_data]
            if 'rubin' in self.diagnostic_methods:
                rubin_stat = self._compute_rubin_diagnostic(feature_data)
                self._convergence_diagnostics['rubin'][feature_name] = rubin_stat
            if 'psrf' in self.diagnostic_methods:
                psrf_stat = self._compute_psrf_diagnostic(feature_data)
                self._convergence_diagnostics['psrf'][feature_name] = psrf_stat
            if 'geweke' in self.diagnostic_methods:
                geweke_stat = self._compute_geweke_diagnostic(feature_data)
                self._convergence_diagnostics['geweke'][feature_name] = geweke_stat
            if 'trace' in self.diagnostic_methods:
                trace_stat = self._compute_trace_diagnostic(feature_data)
                self._convergence_diagnostics['trace'][feature_name] = trace_stat
        return self._convergence_diagnostics

    def get_convergence_status(self) -> Dict[str, bool]:
        """
        Get convergence status for each diagnostic method.

        Evaluates whether each diagnostic indicates convergence based on
        the configured threshold.

        Returns:
            Dict[str, bool]: Dictionary mapping diagnostic method names to
                           convergence status (True/False)
        """
        if not self._convergence_diagnostics:
            raise RuntimeError('Convergence diagnostics not computed. Call compute_convergence_diagnostics() first.')
        convergence_status = {}
        for method in self.diagnostic_methods:
            if method not in self._convergence_diagnostics:
                continue
            method_diagnostics = self._convergence_diagnostics[method]
            if not method_diagnostics:
                convergence_status[method] = False
                continue
            if method == 'psrf':
                convergence_status[method] = all((stat < self.convergence_threshold for stat in method_diagnostics.values() if not np.isnan(stat)))
            elif method == 'geweke':
                convergence_status[method] = all((abs(stat) < self.convergence_threshold for stat in method_diagnostics.values() if not np.isnan(stat)))
            elif method == 'rubin':
                convergence_status[method] = all((abs(stat - 1.0) < self.convergence_threshold - 1.0 for stat in method_diagnostics.values() if not np.isnan(stat)))
            elif method == 'trace':
                convergence_status[method] = all((stat < self.convergence_threshold for stat in method_diagnostics.values() if not np.isnan(stat)))
            else:
                convergence_status[method] = all((stat < self.convergence_threshold for stat in method_diagnostics.values() if not np.isnan(stat)))
        return convergence_status

    def generate_convergence_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive convergence analysis report.

        Creates a detailed report including diagnostic statistics, convergence
        status, and recommendations for continuing or stopping imputation.

        Returns:
            Dict[str, Any]: Comprehensive convergence report containing:
                          - diagnostics: Detailed diagnostic results
                          - converged: Overall convergence status
                          - recommendations: Suggestions for next steps
                          - iteration_count: Number of iterations analyzed
        """
        if not self._convergence_diagnostics:
            try:
                self.compute_convergence_diagnostics()
            except Exception as e:
                return {'diagnostics': {}, 'converged': False, 'recommendations': f'Unable to compute diagnostics: {str(e)}', 'iteration_count': 0, 'burn_in_period': self.burn_in_period}
        try:
            convergence_status = self.get_convergence_status()
            overall_converged = all((status for status in convergence_status.values()))
            if overall_converged:
                recommendations = 'Convergence achieved. Imputation process can be stopped.'
            else:
                non_converged_methods = [method for (method, status) in convergence_status.items() if not status]
                recommendations = f'Convergence not yet achieved for methods: {non_converged_methods}. Consider running additional iterations.'
        except Exception as e:
            overall_converged = False
            recommendations = f'Error evaluating convergence: {str(e)}'
        return {'diagnostics': self._convergence_diagnostics, 'converged': overall_converged, 'convergence_by_method': convergence_status, 'recommendations': recommendations, 'iteration_count': len(self._imputation_history) - self.burn_in_period, 'burn_in_period': self.burn_in_period, 'convergence_threshold': self.convergence_threshold, 'methods_applied': self.diagnostic_methods}

    def _compute_rubin_diagnostic(self, feature_data: List[np.ndarray]) -> float:
        """
        Compute Rubin's rules convergence diagnostic for a feature.
        
        This computes the ratio of total variance to within-imputation variance.
        
        Args:
            feature_data: List of arrays containing values for this feature across iterations
            
        Returns:
            float: Rubin's variance ratio statistic
        """
        if len(feature_data) < 2:
            return np.nan
        within_var = np.mean([np.var(chain) for chain in feature_data])
        if within_var == 0:
            return 1.0
        chain_means = [np.mean(chain) for chain in feature_data]
        between_var = np.var(chain_means)
        total_var = within_var + (1 + 1 / len(feature_data)) * between_var
        return total_var / within_var

    def _compute_psrf_diagnostic(self, feature_data: List[np.ndarray]) -> float:
        """
        Compute Potential Scale Reduction Factor (PSRF) aka R-hat.
        
        Args:
            feature_data: List of arrays containing values for this feature across iterations
            
        Returns:
            float: PSRF statistic (values near 1.0 indicate convergence)
        """
        if len(feature_data) < 2:
            return np.nan
        n_chains = len(feature_data)
        if n_chains < 2:
            full_chain = np.concatenate(feature_data)
            mid_point = len(full_chain) // 2
            chains = [full_chain[:mid_point], full_chain[mid_point:]]
        else:
            chains = feature_data
        chain_vars = [np.var(chain, ddof=1) for chain in chains if len(chain) > 1]
        if not chain_vars or any((np.isnan(v) for v in chain_vars)):
            return np.nan
        w = np.mean(chain_vars)
        if w == 0:
            return 1.0
        chain_means = [np.mean(chain) for chain in chains]
        b = len(chain_means) * np.var(chain_means, ddof=1)
        n = np.mean([len(chain) for chain in chains])
        sigma2_hat = (n - 1) / n * w + 1 / n * b
        psrf = np.sqrt(sigma2_hat / w)
        return psrf if not np.isnan(psrf) else np.nan

    def _compute_geweke_diagnostic(self, feature_data: List[np.ndarray]) -> float:
        """
        Compute Geweke convergence diagnostic for a feature.
        
        This compares the means of the early and late portions of a chain.
        
        Args:
            feature_data: List of arrays containing values for this feature across iterations
            
        Returns:
            float: Geweke Z-score (values near 0 indicate convergence)
        """
        full_chain = np.concatenate(feature_data)
        if len(full_chain) < 10:
            return np.nan
        first_window = int(0.1 * len(full_chain))
        last_window = int(0.5 * len(full_chain))
        if first_window < 2 or last_window < 2:
            return np.nan
        early_mean = np.mean(full_chain[:first_window])
        early_var = np.var(full_chain[:first_window], ddof=1)
        late_start = len(full_chain) - last_window
        late_mean = np.mean(full_chain[late_start:])
        late_var = np.var(full_chain[late_start:], ddof=1)
        try:
            z_score = (early_mean - late_mean) / np.sqrt(early_var / first_window + late_var / last_window)
            return z_score if not np.isnan(z_score) else np.nan
        except (ZeroDivisionError, FloatingPointError):
            return np.nan

    def _compute_trace_diagnostic(self, feature_data: List[np.ndarray]) -> float:
        """
        Compute trace plot convergence diagnostic.
        
        This measures the autocorrelation in the chain as a convergence indicator.
        
        Args:
            feature_data: List of arrays containing values for this feature across iterations
            
        Returns:
            float: Autocorrelation measure (lower values indicate better convergence)
        """
        full_chain = np.concatenate(feature_data)
        if len(full_chain) < 3:
            return np.nan
        diff_chain = np.diff(full_chain)
        if np.all(diff_chain == 0):
            return 0.0
        autocorr = np.corrcoef(full_chain[:-1], full_chain[1:])[0, 1]
        return abs(autocorr) if not np.isnan(autocorr) else np.nan