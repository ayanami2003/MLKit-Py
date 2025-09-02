from general.base_classes.transformer_base import BaseTransformer
from typing import Optional, List, Union, Tuple
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict, Counter
from itertools import combinations


# ...(code omitted)...


class TreeBasedFeatureCombiner(BaseTransformer):

    def __init__(self, tree_model: Optional[object]=None, max_interaction_degree: int=2, min_interaction_support: int=10, feature_selection_method: str='importance', name: Optional[str]=None):
        super().__init__(name=name)
        if tree_model is None:
            self.tree_model = DecisionTreeRegressor()
        else:
            self.tree_model = tree_model
        self.max_interaction_degree = max_interaction_degree
        self.min_interaction_support = min_interaction_support
        self.feature_selection_method = feature_selection_method

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs) -> 'TreeBasedFeatureCombiner':
        """
        Fit the tree model and identify feature combinations.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to analyze for interactions.
        y : Union[np.ndarray, List]
            Target values for supervised learning.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        TreeBasedFeatureCombiner
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_in_ = data.feature_names
        else:
            X = data
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        self.tree_model.fit(X, y)
        tree = self.tree_model.tree_
        paths = self._extract_paths(tree, X.shape[1])
        combination_counts = defaultdict(int)
        for path in paths:
            features_in_path = path['features']
            weight = path['weight']
            for degree in range(2, min(self.max_interaction_degree + 1, len(features_in_path) + 1)):
                for combo in combinations(sorted(features_in_path), degree):
                    combination_counts[combo] += weight
        self.interaction_terms_ = []
        for (combo, count) in combination_counts.items():
            if count >= self.min_interaction_support:
                self.interaction_terms_.append(combo)
        if self.feature_selection_method == 'importance':
            importances = self.tree_model.feature_importances_
            self.combination_weights_ = []
            for combo in self.interaction_terms_:
                weight = np.prod([importances[i] for i in combo])
                self.combination_weights_.append(weight)
        elif self.feature_selection_method == 'frequency':
            self.combination_weights_ = []
            total_count = sum(combination_counts.values())
            for combo in self.interaction_terms_:
                count = combination_counts[combo]
                self.combination_weights_.append(count / total_count if total_count > 0 else 0)
        else:
            self.combination_weights_ = [1.0] * len(self.interaction_terms_)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the input data by generating interaction features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input features to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed feature set with interaction features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        else:
            X = data
            feature_names = None
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        if hasattr(self, 'interaction_terms_') and self.interaction_terms_:
            interaction_features = []
            for (i, combo) in enumerate(self.interaction_terms_):
                interaction_feature = np.ones(X.shape[0])
                for feature_idx in combo:
                    interaction_feature *= X[:, feature_idx]
                if hasattr(self, 'combination_weights_') and i < len(self.combination_weights_):
                    interaction_feature *= self.combination_weights_[i]
                interaction_features.append(interaction_feature)
            if interaction_features:
                interaction_features = np.column_stack(interaction_features)
                X_transformed = np.hstack([X, interaction_features])
            else:
                X_transformed = X
        else:
            X_transformed = X
        new_feature_names = None
        if feature_names is not None:
            new_feature_names = feature_names.copy()
            if hasattr(self, 'interaction_terms_') and self.interaction_terms_:
                for combo in self.interaction_terms_:
                    interaction_name = '_x_'.join([feature_names[i] for i in combo])
                    new_feature_names.append(interaction_name)
        new_feature_types = None
        if feature_types is not None:
            new_feature_types = feature_types.copy()
            if hasattr(self, 'interaction_terms_') and self.interaction_terms_:
                new_feature_types.extend(['numeric'] * len(self.interaction_terms_))
        return FeatureSet(features=X_transformed, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for this transformer as it creates new features.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data to inverse transform.
        **kwargs : dict
            Additional parameters.
            
        Returns
        -------
        FeatureSet
            Original feature set (identity operation).
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported.
        """
        raise NotImplementedError('Inverse transformation is not supported for feature generation transformers.')

    def _extract_paths(self, tree, n_features: int) -> List[dict]:
        """
        Extract decision paths from a fitted tree.
        
        Parameters
        ----------
        tree : sklearn.tree.Tree
            The fitted tree structure.
        n_features : int
            Number of features in the dataset.
            
        Returns
        -------
        List[dict]
            List of paths, each containing features used and path weight.
        """
        paths = []

        def recurse(node_id: int, features_used: set, path_samples: int):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                if len(features_used) > 1:
                    paths.append({'features': list(features_used), 'weight': path_samples})
                return
            left_child = tree.children_left[node_id]
            if left_child != -1:
                feature_idx = tree.feature[node_id]
                if feature_idx >= 0 and feature_idx < n_features:
                    recurse(left_child, features_used | {feature_idx}, tree.n_node_samples[left_child])
            right_child = tree.children_right[node_id]
            if right_child != -1:
                feature_idx = tree.feature[node_id]
                if feature_idx >= 0 and feature_idx < n_features:
                    recurse(right_child, features_used | {feature_idx}, tree.n_node_samples[right_child])
        if tree.node_count > 0:
            recurse(0, set(), tree.n_node_samples[0])
        return paths