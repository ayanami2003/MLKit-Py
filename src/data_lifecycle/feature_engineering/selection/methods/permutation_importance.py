from typing import Optional, Union, Dict, Any
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from general.structures.model_artifact import ModelArtifact
import numpy as np
from sklearn.metrics import accuracy_score, r2_score


class PermutationFeatureImportanceCalculator(BaseTransformer):
    """
    Calculate feature importance using permutation importance method.
    
    Permutation importance measures the increase in model prediction error after permuting the values of a feature.
    This approach provides a model-agnostic way to assess the contribution of each feature to the model's performance.
    
    This class implements two related features:
    1. "permutation feature importance" - computes importance scores via permutation testing
    2. "calculate feature importance" - generic interface for computing feature importance scores
    
    The transformer fits on a model and validation data, then computes importance scores that can be used
    for feature selection or interpretation.
    
    Attributes
    ----------
    scoring_function : callable, optional
        Function to compute the model score (default: accuracy for classification, R2 for regression)
    n_repeats : int, default=5
        Number of times to permute each feature
    random_state : int, optional
        Random seed for reproducibility
    importance_scores_ : dict
        Dictionary mapping feature names/indices to their importance scores
    baseline_score_ : float
        Baseline model score before permutations
    """

    def __init__(self, scoring_function: Optional[callable]=None, n_repeats: int=5, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the PermutationFeatureImportanceCalculator.
        
        Parameters
        ----------
        scoring_function : callable, optional
            Function that takes (y_true, y_pred) and returns a scalar score.
            If None, uses accuracy for classification or R2 for regression.
        n_repeats : int, default=5
            Number of times to permute each feature to compute importance.
        random_state : int, optional
            Random seed for reproducible permutations.
        name : str, optional
            Name identifier for the transformer instance.
        """
        super().__init__(name=name)
        self.scoring_function = scoring_function
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.importance_scores_ = {}
        self.baseline_score_ = 0.0

    def fit(self, data: Union[FeatureSet, DataBatch], model: Any, **kwargs) -> 'PermutationFeatureImportanceCalculator':
        """
        Fit the calculator by computing permutation importance scores.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data with features and labels for evaluating model performance.
        model : Any
            Trained model with predict method to evaluate feature importance on.
        **kwargs : dict
            Additional parameters (e.g., 'scoring_params' for scoring function).
            
        Returns
        -------
        PermutationFeatureImportanceCalculator
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If data does not contain labels or if model lacks predict method.
        """
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")
        if isinstance(data, FeatureSet):
            if data.target is None:
                raise ValueError('Data must contain labels for permutation importance calculation')
            X = data.features
            y = data.target
            feature_names = data.feature_names
        elif isinstance(data, DataBatch):
            if data.labels is None:
                raise ValueError('Data must contain labels for permutation importance calculation')
            X = data.data
            y = data.labels
            feature_names = data.feature_names if data.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        else:
            raise ValueError('Data must be either a FeatureSet or DataBatch')
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if self.scoring_function is None:
            unique_values = np.unique(y)
            if len(unique_values) <= min(20, len(y) // 10):
                from sklearn.metrics import accuracy_score
                self.scoring_function = accuracy_score
            else:
                from sklearn.metrics import r2_score
                self.scoring_function = r2_score
        y_pred = model.predict(X)
        self.baseline_score_ = self.scoring_function(y, y_pred)
        rng = np.random.default_rng(self.random_state)
        self.importance_scores_ = {}
        n_features = X.shape[1]
        for i in range(n_features):
            score_drops = []
            for _ in range(self.n_repeats):
                X_permuted = X.copy()
                permuted_indices = rng.permutation(len(X_permuted))
                X_permuted[:, i] = X_permuted[permuted_indices, i]
                y_pred_permuted = model.predict(X_permuted)
                permuted_score = self.scoring_function(y, y_pred_permuted)
                score_drop = self.baseline_score_ - permuted_score
                score_drops.append(score_drop)
            feature_name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            self.importance_scores_[feature_name] = np.mean(score_drops)
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Transform data by returning feature importance scores.
        
        This method doesn't modify the input data but returns a FeatureSet containing
        feature importance scores that can be used for feature selection.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data (used for feature metadata, not modified).
        **kwargs : dict
            Additional parameters (unused).
            
        Returns
        -------
        FeatureSet
            FeatureSet containing feature importance scores as features.
        """
        return FeatureSet(features=np.array(list(self.importance_scores_.values())).reshape(1, -1), target=None, feature_names=list(self.importance_scores_.keys()), name=f'{self.name}_importance_scores')

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not applicable for feature importance calculation.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch]
            Input data (ignored).
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Returns the input data unchanged.
        """
        return data

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the computed feature importance scores.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to their importance scores.
            Higher scores indicate more important features.
        """
        return self.importance_scores_.copy()