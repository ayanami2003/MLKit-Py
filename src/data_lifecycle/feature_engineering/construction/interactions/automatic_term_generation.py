import numpy as np
from typing import Optional, List, Union, Tuple
from itertools import combinations
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor

class AutomaticInteractionTermGenerator(BaseTransformer):

    def __init__(self, max_degree: int=2, interaction_method: str='polynomial', max_features: Optional[int]=None, include_bias: bool=False, name: Optional[str]=None):
        """
        Initialize the AutomaticInteractionTermGenerator.
        
        Parameters
        ----------
        max_degree : int, default=2
            Maximum degree of interaction terms to generate
        interaction_method : str, default='polynomial'
            Method used to generate interactions ('polynomial', 'ratio', 'tree-based', 'cluster-based')
        max_features : int, optional
            Maximum number of interaction terms to generate
        include_bias : bool, default=False
            Whether to include a bias column in the output
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.max_degree = max_degree
        self.interaction_method = interaction_method
        self.max_features = max_features
        self.include_bias = include_bias
        self.feature_names_: List[str] = []

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'AutomaticInteractionTermGenerator':
        """
        Learn which interaction terms to generate based on the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to analyze for interaction term generation
        y : np.ndarray, optional
            Target values for supervised interaction generation methods
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        AutomaticInteractionTermGenerator
            Self instance for method chaining
        """
        supported_methods = ['polynomial', 'ratio', 'tree-based', 'cluster-based']
        if self.interaction_method not in supported_methods:
            raise ValueError(f"Unsupported interaction method '{self.interaction_method}'. Supported methods are: {supported_methods}")
        if isinstance(data, FeatureSet):
            X = data.features
            self.input_feature_names_ = data.feature_names
        else:
            X = data
            self.input_feature_names_ = None
        self.n_features_in_ = X.shape[1]
        if self.input_feature_names_ is None:
            self.input_feature_names_ = [f'x{i}' for i in range(self.n_features_in_)]
        if self.interaction_method == 'polynomial':
            self._generate_polynomial_terms(X)
        elif self.interaction_method == 'ratio':
            self._generate_ratio_terms(X)
        elif self.interaction_method == 'tree-based':
            if y is None:
                raise ValueError("Target variable 'y' is required for tree-based interaction method")
            self._generate_tree_based_terms(X, y)
        elif self.interaction_method == 'cluster-based':
            self._generate_cluster_based_terms(X)
        if self.max_features is not None and len(self.interaction_terms_) > self.max_features:
            self.interaction_terms_ = self.interaction_terms_[:self.max_features]
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the learned interaction terms to generate new features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed features with generated interaction terms
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        interaction_features = []
        for term in self.interaction_terms_:
            if self.interaction_method == 'polynomial':
                feature = np.ones(X.shape[0])
                for idx in term:
                    feature *= X[:, idx]
            elif self.interaction_method == 'ratio':
                (i, j) = term
                feature = X[:, i] / X[:, j]
            elif self.interaction_method in ['tree-based', 'cluster-based']:
                feature = np.ones(X.shape[0])
                for idx in term:
                    feature *= X[:, idx]
            interaction_features.append(feature)
        if interaction_features:
            interaction_matrix = np.column_stack(interaction_features)
            result_features = np.column_stack([X, interaction_matrix])
        else:
            result_features = X.copy()
        if self.include_bias:
            bias_column = np.ones((result_features.shape[0], 1))
            result_features = np.column_stack([bias_column, result_features])
        feature_names = self.get_feature_names()
        return FeatureSet(features=result_features, feature_names=feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the inverse transformation to revert to original features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed features to revert
        **kwargs : dict
            Additional inverse transformation parameters
            
        Returns
        -------
        FeatureSet
            Original features without interaction terms
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        n_original_features = self.n_features_in_
        start_col = 0
        if self.include_bias:
            start_col = 1
        original_features = X[:, start_col:start_col + n_original_features]
        if hasattr(self, 'input_feature_names_'):
            feature_names = self.input_feature_names_
        else:
            feature_names = [f'x{i}' for i in range(n_original_features)]
        return FeatureSet(features=original_features, feature_names=feature_names)

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of the generated interaction features.
        
        Parameters
        ----------
        input_features : List[str], optional
            Names of the input features
            
        Returns
        -------
        List[str]
            Names of the generated interaction features
        """
        if input_features is not None:
            feature_names = input_features
        elif hasattr(self, 'input_feature_names_') and self.input_feature_names_ is not None:
            feature_names = self.input_feature_names_
        else:
            feature_names = [f'x{i}' for i in range(self.n_features_in_)]
        result_names = []
        if self.include_bias:
            result_names.append('bias')
        result_names.extend(feature_names)
        for term in self.interaction_terms_:
            if self.interaction_method == 'polynomial':
                name = '_x_'.join([feature_names[i] for i in term])
            elif self.interaction_method == 'ratio':
                (i, j) = term
                name = f'{feature_names[i]}/{feature_names[j]}'
            elif self.interaction_method in ['tree-based', 'cluster-based']:
                name = '_x_'.join([feature_names[i] for i in term])
            result_names.append(name)
        return result_names

    def _generate_polynomial_terms(self, X: np.ndarray):
        """Generate polynomial interaction terms."""
        feature_indices = list(range(self.n_features_in_))
        self.interaction_terms_ = []
        for degree in range(2, min(self.max_degree + 1, self.n_features_in_ + 1)):
            for combo in combinations(feature_indices, degree):
                self.interaction_terms_.append(combo)

    def _generate_ratio_terms(self, X: np.ndarray):
        """Generate ratio-based interaction terms."""
        feature_indices = list(range(self.n_features_in_))
        self.interaction_terms_ = []
        processed_pairs = set()
        for i in feature_indices:
            for j in feature_indices:
                if i != j:
                    pair_key = tuple(sorted([i, j]))
                    if pair_key not in processed_pairs:
                        if not np.any(np.abs(X[:, j]) < 1e-10):
                            self.interaction_terms_.append((i, j))
                            processed_pairs.add(pair_key)

    def _generate_tree_based_terms(self, X: np.ndarray, y: np.ndarray):
        """Generate interaction terms based on decision tree paths."""
        tree = DecisionTreeRegressor(max_depth=self.max_degree, random_state=42)
        tree.fit(X, y)
        tree_structure = tree.tree_
        paths = self._extract_tree_paths(tree_structure, self.n_features_in_)
        combination_counts = {}
        for path in paths:
            features_in_path = path['features']
            for degree in range(2, min(self.max_degree + 1, len(features_in_path) + 1)):
                for combo in combinations(sorted(features_in_path), degree):
                    combination_counts[combo] = combination_counts.get(combo, 0) + 1
        self.interaction_terms_ = []
        for (combo, count) in combination_counts.items():
            if count >= 2:
                self.interaction_terms_.append(combo)

    def _generate_cluster_based_terms(self, X: np.ndarray):
        """Generate interaction terms based on cluster analysis."""
        n_clusters = min(10, max(2, X.shape[0] // 10))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        self.interaction_terms_ = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) < 2:
                continue
            cluster_features = X[cluster_mask]
            feature_indices = list(range(self.n_features_in_))
            for combo in combinations(feature_indices, min(2, self.max_degree)):
                self.interaction_terms_.append(combo)

    def _extract_tree_paths(self, tree, n_features: int) -> List[dict]:
        """Extract decision paths from a fitted tree."""
        paths = []

        def recurse(node_id: int, features_used: set, path_samples: int):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                if len(features_used) > 1:
                    paths.append({'features': list(features_used), 'weight': path_samples})
                return
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            feature_idx = tree.feature[node_id]
            if feature_idx >= 0 and feature_idx < n_features:
                if left_child != -1:
                    recurse(left_child, features_used | {feature_idx}, tree.n_node_samples[left_child])
                if right_child != -1:
                    recurse(right_child, features_used | {feature_idx}, tree.n_node_samples[right_child])
        if tree.node_count > 0:
            recurse(0, set(), tree.n_node_samples[0])
        return paths