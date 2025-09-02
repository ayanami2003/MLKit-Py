from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, f_regression, f_classif

class MinMaxRedundancySelector(BaseTransformer):

    def __init__(self, n_features: Optional[int]=None, method: str='MID', relevance_metric: str='mutual_info', redundancy_metric: str='pearson', discrete_features: Union[bool, List[bool]]=False, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_features = n_features
        self.method = method
        self.relevance_metric = relevance_metric
        self.redundancy_metric = redundancy_metric
        self.discrete_features = discrete_features
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, DataBatch], y: Optional[Union[list, np.ndarray]]=None, **kwargs) -> 'MinMaxRedundancySelector':
        """
        Compute relevance and redundancy scores to determine feature importance.
        
        Calculates the relevance of each feature to the target variable and the
        redundancy among features. Uses these metrics to rank features according
        to the mRMR criterion.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input features to analyze for selection. Must contain a 2D array of features.
        y : Optional[Union[list, np.ndarray]]
            Target values corresponding to the input features. Required for supervised selection.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        MinMaxRedundancySelector
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If target values are not provided for supervised selection.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            if y is None and data.metadata and ('labels' in data.metadata):
                y = data.metadata['labels']
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            if y is None:
                y = data.labels
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if y is not None and (not isinstance(y, np.ndarray)):
            y = np.array(y)
        if X.ndim != 2:
            raise ValueError('Features must be a 2D array')
        if y is None:
            raise ValueError('Target values (y) must be provided for supervised feature selection')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_features = X.shape[1]
        if self.n_features is None:
            self.n_features = max(1, n_features // 2)
        elif self.n_features > n_features:
            raise ValueError(f'n_features ({self.n_features}) cannot be larger than the number of available features ({n_features})')
        relevance_scores = self._compute_relevance_scores(X, y)
        redundancy_matrix = self._compute_redundancy_matrix(X)
        self.selected_indices_ = self._calculate_mrmr_scores(relevance_scores, redundancy_matrix)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Apply feature selection to reduce the feature set.
        
        Selects features based on precomputed mRMR scores, returning a new FeatureSet
        containing only the selected features.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input features to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed feature set with only selected features retained.
        """
        if not hasattr(self, 'selected_indices_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        selected_features = X[:, self.selected_indices_]
        selected_feature_names = None
        if feature_names:
            selected_feature_names = [feature_names[i] for i in self.selected_indices_]
        selected_feature_types = None
        if feature_types:
            selected_feature_types = [feature_types[i] for i in self.selected_indices_]
        return FeatureSet(features=selected_features, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the transformation (not supported in feature selection).
        
        As feature selection is a destructive operation that removes features,
        the inverse transformation cannot recover the original feature set.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Transformed features (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Original feature set with all features (placeholder implementation).
        """
        if isinstance(data, FeatureSet):
            return data
        elif isinstance(data, DataBatch):
            return FeatureSet(features=np.array(data.data), metadata={'labels': data.labels} if data.labels is not None else {})
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')

    def get_support(self) -> list:
        """
        Get indices of selected features.
        
        Returns
        -------
        list
            List of indices corresponding to selected features.
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet.
        """
        if not hasattr(self, 'selected_indices_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' before 'get_support'.")
        return self.selected_indices_.tolist()

    def _compute_relevance_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute relevance scores between each feature and the target.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
            
        Returns
        -------
        np.ndarray
            Relevance scores for each feature.
        """
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        if self.relevance_metric == 'mutual_info':
            unique_y = np.unique(y)
            is_classification = len(unique_y) < max(10, 0.05 * len(y))
            if is_classification:
                if isinstance(self.discrete_features, bool):
                    discrete_features = self.discrete_features
                else:
                    discrete_features_arr = np.asarray(self.discrete_features, dtype=bool)
                    if len(discrete_features_arr) == n_features:
                        discrete_features = discrete_features_arr
                    else:
                        discrete_features = np.full(n_features, False, dtype=bool)
                scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=self.random_state)
            else:
                if isinstance(self.discrete_features, bool):
                    discrete_features = self.discrete_features
                else:
                    discrete_features_arr = np.asarray(self.discrete_features, dtype=bool)
                    if len(discrete_features_arr) == n_features:
                        discrete_features = discrete_features_arr
                    else:
                        discrete_features = np.full(n_features, False, dtype=bool)
                scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=self.random_state)
        elif self.relevance_metric == 'f_score':
            unique_y = np.unique(y)
            is_classification = len(unique_y) < max(10, 0.05 * len(y))
            if is_classification:
                (scores, _) = f_classif(X, y)
            else:
                (scores, _) = f_regression(X, y)
        else:
            raise ValueError(f'Unsupported relevance metric: {self.relevance_metric}')
        return scores

    def _compute_redundancy_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute redundancy matrix among features.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
            
        Returns
        -------
        np.ndarray
            Redundancy matrix of shape (n_features, n_features).
        """
        n_features = X.shape[1]
        redundancy_matrix = np.zeros((n_features, n_features))
        if self.redundancy_metric == 'pearson':
            for i in range(n_features):
                for j in range(i, n_features):
                    if i == j:
                        redundancy_matrix[i, j] = 1.0
                    else:
                        try:
                            (corr, _) = pearsonr(X[:, i], X[:, j])
                            redundancy_val = abs(corr) if not np.isnan(corr) else 0.0
                        except:
                            redundancy_val = 0.0
                        redundancy_matrix[i, j] = redundancy_val
                        redundancy_matrix[j, i] = redundancy_val
        elif self.redundancy_metric == 'mutual_info':
            if isinstance(self.discrete_features, bool):
                discrete_features_flag = self.discrete_features
            else:
                discrete_features_flag = np.asarray(self.discrete_features, dtype=bool)
                if len(discrete_features_flag) != n_features:
                    discrete_features_flag = np.full(n_features, False, dtype=bool)
            for i in range(n_features):
                for j in range(i, n_features):
                    if i == j:
                        redundancy_matrix[i, j] = 1.0
                    else:
                        try:
                            x_i = X[:, i]
                            x_j = X[:, j]
                            discrete_flag_i = discrete_features_flag
                            discrete_flag_j = discrete_features_flag
                            if not isinstance(discrete_features_flag, bool):
                                if len(self.discrete_features) > i:
                                    discrete_flag_i = self.discrete_features[i]
                                if len(self.discrete_features) > j:
                                    discrete_flag_j = self.discrete_features[j]
                            mi = mutual_info_regression(x_i.reshape(-1, 1), x_j, discrete_features=discrete_flag_i, random_state=self.random_state)
                            redundancy_val = mi[0] if len(mi) > 0 else 0.0
                        except Exception:
                            redundancy_val = 0.0
                        redundancy_matrix[i, j] = redundancy_val
                        redundancy_matrix[j, i] = redundancy_val
        else:
            raise ValueError(f'Unsupported redundancy metric: {self.redundancy_metric}')
        return redundancy_matrix

    def _calculate_mrmr_scores(self, relevance_scores: np.ndarray, redundancy_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate mRMR scores and select top features.
        
        Parameters
        ----------
        relevance_scores : np.ndarray
            Relevance scores for each feature.
        redundancy_matrix : np.ndarray
            Redundancy matrix among features.
            
        Returns
        -------
        np.ndarray
            Indices of selected features.
        """
        n_features = len(relevance_scores)
        selected_indices = []
        remaining_indices = list(range(n_features))
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        for _ in range(1, min(self.n_features, n_features)):
            best_score = -np.inf
            best_idx = None
            for candidate_idx in remaining_indices:
                if len(selected_indices) > 0:
                    avg_redundancy = np.mean([redundancy_matrix[candidate_idx, selected_idx] for selected_idx in selected_indices])
                else:
                    avg_redundancy = 0.0
                if self.method == 'MID':
                    mrmr_score = relevance_scores[candidate_idx] - avg_redundancy
                elif self.method == 'MIQ':
                    if avg_redundancy == 0:
                        mrmr_score = np.inf if relevance_scores[candidate_idx] > 0 else 0
                    else:
                        mrmr_score = relevance_scores[candidate_idx] / avg_redundancy
                else:
                    raise ValueError(f'Unsupported method: {self.method}')
                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_idx = candidate_idx
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        return np.array(selected_indices)