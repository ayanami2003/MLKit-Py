from general.base_classes.transformer_base import BaseTransformer
from typing import Optional, List, Union, Callable, Tuple
from general.structures.feature_set import FeatureSet
import numpy as np
from itertools import combinations
from collections import defaultdict
import warnings
from sklearn.metrics import accuracy_score, mean_squared_error

class BayesianInteractionSearcher:

    def __init__(self, max_degree: int=2, n_candidates: int=10, n_iterations: int=20, scoring_func: Optional[Callable]=None, interaction_threshold: float=0.01, random_state: Optional[int]=None):
        self.max_degree = max_degree
        self.n_candidates = n_candidates
        self.n_iterations = n_iterations
        self.scoring_func = scoring_func
        self.interaction_threshold = interaction_threshold
        self.random_state = random_state
        self.selected_interactions: List[Tuple[int, ...]] = []

    def _generate_interaction_features(self, X: np.ndarray, interactions: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Generate interaction features by multiplying the specified feature columns.
        
        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            interactions (List[Tuple[int, ...]]): List of tuples containing feature indices to interact.
            
        Returns:
            np.ndarray: Array of interaction features.
        """
        if not interactions:
            return np.empty((X.shape[0], 0))
        interaction_features = []
        for interaction in interactions:
            if len(interaction) == 0:
                continue
            interaction_feature = np.ones(X.shape[0])
            for idx in interaction:
                interaction_feature *= X[:, idx]
            interaction_features.append(interaction_feature)
        if interaction_features:
            return np.column_stack(interaction_features)
        else:
            return np.empty((X.shape[0], 0))

    def search(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Run Bayesian optimization to find optimal feature interactions.
        
        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            
        Returns:
            List[Tuple[int, ...]]: Selected interactions as tuples of feature indices.
            
        Raises:
            ValueError: If scoring function is not provided or inputs are invalid.
        """
        if self.scoring_func is None:
            raise ValueError('scoring_func must be provided')
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError('X and y must be numpy arrays')
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples')
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError('X and y must not contain NaN values')
        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError('X and y must not contain infinite values')
        n_features = X.shape[1]
        if n_features == 1:
            self.selected_interactions = []
            return []
        if self.max_degree > n_features:
            raise ValueError('max_degree cannot be greater than the number of features')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        all_interactions = []
        for degree in range(2, min(self.max_degree + 1, n_features + 1)):
            all_interactions.extend(list(combinations(range(n_features), degree)))
        if not all_interactions:
            self.selected_interactions = []
            return []
        evaluated_interactions = {}
        n_initial_candidates = min(self.n_candidates, len(all_interactions))
        initial_candidates = np.random.choice(len(all_interactions), size=n_initial_candidates, replace=False)
        for idx in initial_candidates:
            interaction = all_interactions[idx]
            if interaction not in evaluated_interactions:
                X_interact = self._generate_interaction_features(X, [interaction])
                score = self.scoring_func(X_interact, y)
                evaluated_interactions[interaction] = score
        for _ in range(self.n_iterations):
            remaining_interactions = [inter for inter in all_interactions if inter not in evaluated_interactions]
            if not remaining_interactions:
                break
            batch_size = min(self.n_candidates, len(remaining_interactions))
            candidate_indices = np.random.choice(len(remaining_interactions), size=batch_size, replace=False)
            candidates = [remaining_interactions[i] for i in candidate_indices]
            for interaction in candidates:
                if interaction not in evaluated_interactions:
                    X_interact = self._generate_interaction_features(X, [interaction])
                    score = self.scoring_func(X_interact, y)
                    evaluated_interactions[interaction] = score
        selected_interactions = [interaction for (interaction, score) in evaluated_interactions.items() if score >= self.interaction_threshold]
        self.selected_interactions = selected_interactions
        return selected_interactions

class BayesianInteractionOptimizer(BaseTransformer):

    def __init__(self, max_degree: int=2, n_candidates: int=10, n_iterations: int=20, scoring_func: Optional[callable]=None, interaction_threshold: float=0.01, random_state: Optional[int]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.max_degree = max_degree
        self.n_candidates = n_candidates
        self.n_iterations = n_iterations
        self.scoring_func = scoring_func
        self.interaction_threshold = interaction_threshold
        self.random_state = random_state

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'BayesianInteractionOptimizer':
        """
        Fit the Bayesian optimizer by searching for optimal feature interactions.
        
        This method runs Bayesian optimization to identify the best subset of feature interactions.
        It requires target values `y` to evaluate the quality of interactions via the scoring function.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input features to construct interactions from.
                If FeatureSet, uses its `.features` attribute.
            y (Optional[np.ndarray]): Target values for scoring candidate interactions.
            **kwargs: Additional parameters passed to the scoring function.
            
        Returns:
            BayesianInteractionOptimizer: Returns self for method chaining.
            
        Raises:
            ValueError: If `y` is not provided or if scoring function is undefined.
        """
        if y is None:
            raise ValueError("Target values 'y' must be provided for fitting")
        if self.scoring_func is None:
            raise ValueError('Scoring function must be provided')
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_in_ = data.feature_names
        else:
            X = data
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        searcher = BayesianInteractionSearcher(max_degree=self.max_degree, n_candidates=self.n_candidates, n_iterations=self.n_iterations, scoring_func=self.scoring_func, interaction_threshold=self.interaction_threshold, random_state=self.random_state)
        selected_interactions = searcher.search(X, y)
        self.selected_interactions_ = selected_interactions if selected_interactions is not None else []
        if self.feature_names_in_ is not None:
            self.interaction_feature_names_ = []
            for interaction in self.selected_interactions_:
                names = [self.feature_names_in_[i] for i in interaction]
                self.interaction_feature_names_.append(':'.join(names))
        else:
            self.interaction_feature_names_ = None
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the selected feature interactions to the input data.
        
        Constructs new features based on the interactions identified during fitting. Only interactions
        meeting the importance threshold are included in the output FeatureSet.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input features to apply interactions to.
            **kwargs: Additional transformation parameters (ignored).
            
        Returns:
            FeatureSet: Transformed feature set with added interaction features.
        """
        if not hasattr(self, 'selected_interactions_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
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
        if self.selected_interactions_:
            interaction_features = []
            for interaction in self.selected_interactions_:
                interaction_feature = np.ones(X.shape[0])
                for idx in interaction:
                    interaction_feature *= X[:, idx]
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
            if hasattr(self, 'interaction_feature_names_') and self.interaction_feature_names_:
                new_feature_names.extend(self.interaction_feature_names_)
        new_feature_types = None
        if feature_types is not None:
            new_feature_types = feature_types.copy()
            if self.selected_interactions_:
                new_feature_types.extend(['numeric'] * len(self.selected_interactions_))
        return FeatureSet(features=X_transformed, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Remove interaction features and return original feature space.
        
        This operation removes all constructed interaction features, returning the FeatureSet
        to its original state before transformation.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Transformed feature set with interactions.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Original feature set without interaction features.
        """
        if not hasattr(self, 'selected_interactions_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
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
        if self.selected_interactions_ and len(self.selected_interactions_) > 0:
            n_original_features = X.shape[1] - len(self.selected_interactions_)
            X_original = X[:, :n_original_features]
        else:
            X_original = X
        original_feature_names = None
        original_feature_types = None
        if feature_names is not None:
            if self.selected_interactions_ and len(self.selected_interactions_) > 0:
                original_feature_names = feature_names[:len(feature_names) - len(self.selected_interactions_)]
            else:
                original_feature_names = feature_names
        if feature_types is not None:
            if self.selected_interactions_ and len(self.selected_interactions_) > 0:
                original_feature_types = feature_types[:len(feature_types) - len(self.selected_interactions_)]
            else:
                original_feature_types = feature_types
        return FeatureSet(features=X_original, feature_names=original_feature_names, feature_types=original_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def get_selected_interactions(self) -> List[tuple]:
        """
        Retrieve the list of selected feature interactions.
        
        Returns:
            List[tuple]: Tuples representing selected interactions (e.g., (0, 1), (2, 3, 4)).
        """
        if not hasattr(self, 'selected_interactions_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        return self.selected_interactions_ if self.selected_interactions_ is not None else []

class BayesianInteractionTransformer(BaseTransformer):
    """
    Transformer that applies previously selected feature interactions.
    
    This class takes a list of feature interactions (typically obtained from
    BayesianInteractionSearcher) and constructs new features based on them.
    It supports both fitting (to store interaction metadata) and transforming
    data by applying the interactions.
    
    Attributes:
        selected_interactions (List[Tuple[int, ...]]): Interactions to apply.
        interaction_names (List[str]): Names for constructed interaction features.
    """

    def __init__(self, selected_interactions: Optional[List[Tuple[int, ...]]]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.selected_interactions = selected_interactions or []
        self.interaction_names: List[str] = []

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'BayesianInteractionTransformer':
        """
        Fit the transformer by storing metadata about interactions.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input features to validate interactions against.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            BayesianInteractionTransformer: Returns self for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        n_features = X.shape[1]
        for interaction in self.selected_interactions:
            if not isinstance(interaction, tuple):
                raise TypeError('Each interaction must be a tuple of integers')
            if not all((isinstance(idx, int) for idx in interaction)):
                raise TypeError('All indices in interactions must be integers')
            if not all((0 <= idx < n_features for idx in interaction)):
                raise ValueError(f'All indices in interactions must be between 0 and {n_features - 1}')
        self.interaction_names = []
        for interaction in self.selected_interactions:
            if feature_names is not None:
                names = [feature_names[idx] for idx in interaction]
                interaction_name = '_x_'.join(names)
            else:
                indices = [str(idx) for idx in interaction]
                interaction_name = 'feat_' + '_x_feat_'.join(indices)
            self.interaction_names.append(interaction_name)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply the selected feature interactions to the input data.
        
        Constructs new features by multiplying the values of features involved in each interaction.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Input features to apply interactions to.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Transformed feature set with added interaction features.
        """
        if not hasattr(self, 'interaction_names'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
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
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        n_features = X.shape[1]
        for interaction in self.selected_interactions:
            if not all((0 <= idx < n_features for idx in interaction)):
                raise ValueError('Feature indices in interactions are incompatible with input data dimensions')
        if self.selected_interactions:
            interaction_features = []
            for interaction in self.selected_interactions:
                if len(interaction) == 0:
                    continue
                interaction_feature = np.ones(X.shape[0])
                for idx in interaction:
                    interaction_feature *= X[:, idx]
                interaction_features.append(interaction_feature)
            if interaction_features:
                interaction_matrix = np.column_stack(interaction_features)
                X_transformed = np.column_stack([X, interaction_matrix])
            else:
                X_transformed = X
        else:
            X_transformed = X
        if feature_names is not None:
            new_feature_names = feature_names + self.interaction_names
        else:
            original_names = [f'feat_{i}' for i in range(n_features)]
            new_feature_names = original_names + self.interaction_names
        if feature_types is not None:
            new_feature_types = feature_types + ['numeric'] * len(self.interaction_names)
        else:
            new_feature_types = None
        metadata['original_n_features'] = n_features
        metadata['interaction_features_added'] = len(self.interaction_names)
        return FeatureSet(features=X_transformed, feature_names=new_feature_names, feature_types=new_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Remove interaction features and return original feature space.
        
        This operation removes all constructed interaction features, returning the FeatureSet
        to its original state before transformation.
        
        Args:
            data (Union[FeatureSet, np.ndarray]): Transformed feature set with interactions.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            FeatureSet: Original feature set without interaction features.
        """
        if not hasattr(self, 'interaction_names'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' first.")
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
        if not isinstance(X, np.ndarray):
            raise TypeError('Input data must be a numpy array or FeatureSet')
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional')
        original_n_features = metadata.get('original_n_features')
        if original_n_features is None:
            original_n_features = X.shape[1] - len(self.interaction_names)
        if X.shape[1] < original_n_features:
            raise ValueError('Input data has fewer features than expected original feature set')
        X_original = X[:, :original_n_features]
        if feature_names is not None:
            original_feature_names = feature_names[:original_n_features]
        else:
            original_feature_names = None
        if feature_types is not None:
            original_feature_types = feature_types[:original_n_features]
        else:
            original_feature_types = None
        if 'original_n_features' in metadata:
            del metadata['original_n_features']
        if 'interaction_features_added' in metadata:
            del metadata['interaction_features_added']
        return FeatureSet(features=X_original, feature_names=original_feature_names, feature_types=original_feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

def apply_bayesian_optimization_for_feature_interactions(data: Union[FeatureSet, np.ndarray], y: np.ndarray, max_degree: int=2, n_candidates: int=10, n_iterations: int=20, scoring_func: Optional[Callable]=None, interaction_threshold: float=0.01, random_state: Optional[int]=None) -> FeatureSet:
    """
    Apply Bayesian optimization to automatically construct feature interactions.
    
    This function combines searching for and applying optimal feature interactions
    into a single operation. It first uses Bayesian optimization to identify
    promising interactions based on a scoring function, then constructs new
    features from those interactions and appends them to the input data.
    
    Args:
        data (Union[FeatureSet, np.ndarray]): Input features to construct interactions from.
            If FeatureSet, uses its `.features` attribute.
        y (np.ndarray): Target values used to score candidate interactions.
        max_degree (int): Maximum degree of interactions to consider (default: 2).
        n_candidates (int): Number of candidate interactions to evaluate per iteration (default: 10).
        n_iterations (int): Number of Bayesian optimization iterations (default: 20).
        scoring_func (Optional[Callable]): Function to score feature sets; takes (X, y) and returns float.
            If None, a default scoring function will be used.
        interaction_threshold (float): Minimum importance threshold for including an interaction (default: 0.01).
        random_state (Optional[int]): Random seed for reproducibility (default: None).
        
    Returns:
        FeatureSet: Original features augmented with newly constructed interaction features.
        
    Raises:
        ValueError: If `y` is not provided or if scoring function is undefined.
    """
    if scoring_func is None:
        unique_vals = np.unique(y)
        if len(unique_vals) <= 20 and all((isinstance(val, (int, np.integer)) for val in unique_vals)):
            from sklearn.metrics import accuracy_score

            def scoring_func(X_interact, y_true):
                if X_interact.shape[1] == 0:
                    return 0.0
                pred = X_interact[:, 0]
                median_val = np.median(pred)
                pred_classes = (pred > median_val).astype(int)
                if len(np.unique(pred_classes)) < 2:
                    return 0.0
                return accuracy_score(y_true, pred_classes)
        else:
            from sklearn.metrics import mean_squared_error

            def scoring_func(X_interact, y_true):
                if X_interact.shape[1] == 0:
                    return -np.inf
                pred = X_interact[:, 0]
                return -mean_squared_error(y_true, pred)
    if isinstance(data, FeatureSet):
        X = data.features
        input_feature_names = data.feature_names
    else:
        X = data
        input_feature_names = None
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError('X and y must be numpy arrays')
    if X.ndim != 2:
        raise ValueError('X must be a 2D array')
    if y.ndim != 1:
        raise ValueError('y must be a 1D array')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples')
    if input_feature_names is None:
        input_feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        feature_set_data = FeatureSet(features=X, feature_names=input_feature_names)
    else:
        feature_set_data = data if isinstance(data, FeatureSet) else FeatureSet(features=X, feature_names=input_feature_names)
    optimizer = BayesianInteractionOptimizer(max_degree=max_degree, n_candidates=n_candidates, n_iterations=n_iterations, scoring_func=scoring_func, interaction_threshold=interaction_threshold, random_state=random_state)
    optimizer.fit(feature_set_data, y)
    result = optimizer.transform(feature_set_data)
    return result