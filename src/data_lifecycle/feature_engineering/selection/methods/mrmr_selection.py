from typing import Optional, Union
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class MRMRFeatureSelector(BaseTransformer):

    def __init__(self, n_features: Optional[int]=None, method: str='MID', discrete_features: bool=False, discrete_target: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.n_features = n_features
        self.method = method
        self.discrete_features = discrete_features
        self.discrete_target = discrete_target
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'MRMRFeatureSelector':
        """
        Fit the mRMR selector to the input data.
        
        Calculates relevance scores between features and target, and prepares
        internal state for feature selection.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data with features and labels. Must contain labeled data
            for supervised feature selection.
        **kwargs : dict
            Additional fitting parameters (ignored in this implementation)
            
        Returns
        -------
        MRMRFeatureSelector
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data doesn't contain labels or has inconsistent dimensions
        """
        if isinstance(data, FeatureSet):
            X = data.features
            y = kwargs.get('y')
            if y is None:
                raise ValueError("FeatureSet input must contain labels via 'y' keyword argument")
            self.feature_names_ = data.feature_names
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            y = data.labels
            if y is None:
                raise ValueError('DataBatch input must contain labels')
            self.feature_names_ = data.feature_names
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if X.ndim != 2:
            raise ValueError('Features must be a 2D array')
        if len(X) != len(y):
            raise ValueError('Number of samples in features and labels must match')
        n_features = X.shape[1]
        if self.n_features is None:
            self.n_features = min(10, n_features)
        else:
            self.n_features = min(self.n_features, n_features)
        if self.method not in ['MID', 'MIQ']:
            raise ValueError("Method must be either 'MID' or 'MIQ'")
        discrete_features_param = self.discrete_features
        if isinstance(discrete_features_param, (list, tuple, np.ndarray)):
            if len(discrete_features_param) == 0:
                discrete_features_param = False
            elif len(discrete_features_param) == n_features:
                discrete_features_param = np.asarray(discrete_features_param, dtype=bool)
                if discrete_features_param.size == 0 or not discrete_features_param.any():
                    discrete_features_param = False
            else:
                discrete_features_param = bool(np.asarray(discrete_features_param).any())
        elif isinstance(discrete_features_param, (int, np.integer)):
            discrete_features_param = bool(discrete_features_param)
        elif not isinstance(discrete_features_param, (bool, int)):
            discrete_features_param = False
        if isinstance(discrete_features_param, np.ndarray):
            if discrete_features_param.size == 0:
                discrete_features_param = False
            elif discrete_features_param.dtype == bool and (not discrete_features_param.any()):
                discrete_features_param = False
        if self.discrete_target:
            self.relevance_scores_ = mutual_info_classif(X, y, discrete_features=discrete_features_param, random_state=self.random_state)
        else:
            self.relevance_scores_ = mutual_info_regression(X, y, discrete_features=discrete_features_param, random_state=self.random_state)
        self.X_ = X
        self.y_ = y
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Apply mRMR feature selection to transform the input data.
        
        Selects features based on the fitted mRMR criteria and returns a new
        FeatureSet containing only the selected features.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data to transform
        **kwargs : dict
            Additional transformation parameters (ignored in this implementation)
            
        Returns
        -------
        FeatureSet
            Transformed data with only selected features
            
        Raises
        ------
        RuntimeError
            If transform is called before fit
        """
        if not hasattr(self, 'relevance_scores_'):
            raise RuntimeError('Transformer must be fitted before transform can be called')
        if isinstance(data, FeatureSet):
            X = data.features
            sample_ids = data.sample_ids
            metadata = data.metadata
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            sample_ids = data.sample_ids
            metadata = data.metadata
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        n_features_available = X.shape[1]
        n_features_to_select = min(self.n_features, n_features_available) if self.n_features is not None else min(10, n_features_available)
        n_features_to_select = max(0, n_features_to_select)
        if n_features_to_select == 0 or n_features_available == 0:
            selected_features = np.array([]).reshape(X.shape[0], 0)
            selected_feature_names = []
        else:
            n_total_features = self.X_.shape[1]
            redundancy_matrix = np.zeros((n_total_features, n_total_features))
            for i in range(n_total_features):
                for j in range(i + 1, n_total_features):
                    if np.std(self.X_[:, i]) > 0 and np.std(self.X_[:, j]) > 0:
                        corr = np.corrcoef(self.X_[:, i], self.X_[:, j])[0, 1]
                        redundancy = abs(corr) if not np.isnan(corr) else 0
                    else:
                        redundancy = 0
                    redundancy_matrix[i, j] = redundancy
                    redundancy_matrix[j, i] = redundancy
            selected_indices = []
            if len(self.relevance_scores_) > 0:
                selected_indices.append(np.argmax(self.relevance_scores_))
                for _ in range(1, min(n_features_to_select, n_total_features)):
                    best_score = -np.inf
                    best_idx = -1
                    for idx in range(n_total_features):
                        if idx in selected_indices:
                            continue
                        if len(selected_indices) > 0:
                            redundancy_values = []
                            for sel_idx in selected_indices:
                                if idx < redundancy_matrix.shape[0] and sel_idx < redundancy_matrix.shape[1]:
                                    redundancy_values.append(redundancy_matrix[idx, sel_idx])
                            redundancy = np.mean(redundancy_values) if redundancy_values else 0
                        else:
                            redundancy = 0
                        relevance = self.relevance_scores_[idx] if idx < len(self.relevance_scores_) else 0
                        if self.method == 'MID':
                            mrmr_score = relevance - redundancy
                        else:
                            mrmr_score = relevance / (redundancy + 1e-12)
                        if mrmr_score > best_score:
                            best_score = mrmr_score
                            best_idx = idx
                    if best_idx != -1 and best_idx not in selected_indices:
                        selected_indices.append(best_idx)
                    else:
                        break
            if selected_indices and n_features_available >= max(selected_indices) + 1:
                selected_features = X[:, selected_indices]
                selected_feature_names = [self.feature_names_[i] for i in selected_indices] if hasattr(self, 'feature_names_') and len(self.feature_names_) > max(selected_indices) else [f'feature_{i}' for i in selected_indices]
            else:
                selected_features = np.array([]).reshape(X.shape[0], 0)
                selected_feature_names = []
        return FeatureSet(features=selected_features, feature_names=selected_feature_names, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Reverse the feature selection transformation.
        
        Since feature selection is a lossy transformation (irreversible),
        this method raises a NotImplementedError.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Data to inverse transform (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            This method always raises NotImplementedError
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported
        """
        raise NotImplementedError('Inverse transformation is not supported for feature selection as it is a lossy operation')