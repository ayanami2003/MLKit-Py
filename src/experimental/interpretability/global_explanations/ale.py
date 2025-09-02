import numpy as np
from typing import Union, Optional, List
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet

def compute_accumulated_local_effects(model: BaseModel, features: FeatureSet, feature_indices: Union[int, List[int]], grid_size: int=10, include_interaction: bool=False, center_at_zero: bool=True, random_state: Optional[int]=None) -> dict:
    """
    Compute Accumulated Local Effects (ALE) for specified features to explain model predictions.

    This function calculates the ALE plots for numerical features in a model to understand
    how changes in feature values influence the model's predictions. ALE provides a global
    explanation that is centered around the prediction differences rather than the predictions
    themselves, making it less biased than partial dependence plots for correlated features.

    Args:
        model (BaseModel): Trained model implementing the BaseModel interface with a predict method.
        features (FeatureSet): FeatureSet containing the data used for computing ALE plots.
                               Should include feature names and types information.
        feature_indices (Union[int, List[int]]): Index or list of indices of features for which
                                                 to compute ALE plots.
        grid_size (int): Number of points in the grid for computing ALE. Defaults to 10.
        include_interaction (bool): Whether to compute second-order ALE for feature interactions.
                                    Defaults to False.
        center_at_zero (bool): Whether to center the ALE plot at zero. Defaults to True.
        random_state (Optional[int]): Random seed for reproducibility. Defaults to None.

    Returns:
        dict: Dictionary containing:
              - 'ale_effects': Array of ALE values for each feature
              - 'feature_grids': Grid points used for each feature
              - 'feature_names': Names of the features analyzed
              - 'interaction_effects': (if requested) Interaction ALE values

    Raises:
        ValueError: If feature indices are out of bounds or if features are not numerical.
        TypeError: If model does not implement required prediction interface.
    """
    if random_state is not None:
        np.random.seed(random_state)
    if not hasattr(model, 'predict'):
        raise TypeError("Model must implement a 'predict' method")
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]
    n_features = len(features.feature_names)
    for idx in feature_indices:
        if idx < 0 or idx >= n_features:
            raise ValueError(f'Feature index {idx} is out of bounds for features with {n_features} columns')
    for idx in feature_indices:
        if features.feature_types[idx] != 'numeric':
            raise ValueError(f"Feature '{features.feature_names[idx]}' at index {idx} is not numerical")
    X = features.features
    ale_effects = []
    feature_grids = []
    feature_names = []
    for idx in feature_indices:
        feature_name = features.feature_names[idx]
        feature_names.append(feature_name)
        feature_values = X[:, idx]
        (feature_min, feature_max) = (np.min(feature_values), np.max(feature_values))
        grid_points = np.linspace(feature_min, feature_max, grid_size)
        feature_grids.append(grid_points)
        ale_vals = _compute_ale_single_feature(model, X, idx, grid_points)
        if center_at_zero:
            ale_vals = ale_vals - np.mean(ale_vals)
        ale_effects.append(ale_vals)
    result = {'ale_effects': ale_effects, 'feature_grids': feature_grids, 'feature_names': feature_names}
    if include_interaction and len(feature_indices) >= 2:
        interaction_effects = _compute_ale_interactions(model, X, feature_indices, grid_size, center_at_zero)
        result['interaction_effects'] = interaction_effects
    return result