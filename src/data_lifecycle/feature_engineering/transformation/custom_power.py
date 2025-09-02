from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np


# ...(code omitted)...


def apply_custom_power_transformation(data: FeatureSet, lambdas: Union[float, List[float]], feature_names: Optional[List[str]]=None, shift_factor: float=0.0, inplace: bool=False) -> FeatureSet:
    """
    Apply a custom power transformation to selected features in a FeatureSet.
    
    This function transforms features using a power function of the form
    sign(x) * |x + shift_factor|^lambda for each specified feature and lambda.
    It provides a flexible way to apply power transformations beyond standard
    Box-Cox or Yeo-Johnson methods.
    
    Parameters
    ----------
    data : FeatureSet
        Input data to transform.
    lambdas : Optional[Union[float, List[float]]], default=1.0
        Power parameter(s) for transformation. Can be a single value or
        a list matching the number of features to transform.
    feature_names : Optional[List[str]], default=None
        Names of features to transform. If None, transforms all features.
    shift_factor : float, default=0.0
        Constant to add to features before transformation to ensure
        proper domain for power operation.
    inplace : bool, default=False
        Whether to modify the input data in-place or return a new FeatureSet.
        
    Returns
    -------
    FeatureSet
        Transformed data with custom power transformation applied.
        
    Raises
    ------
    ValueError
        If the number of lambdas doesn't match the number of features to transform,
        or if feature names don't exist in the FeatureSet.
    """
    if not isinstance(data, FeatureSet):
        raise TypeError('data must be a FeatureSet instance')
    if feature_names is None:
        feature_indices = list(range(data.features.shape[1]))
        selected_features = data.features.copy() if not inplace else data.features
    else:
        if data.feature_names is None:
            raise ValueError('Feature names must be available in the FeatureSet when specifying feature_names')
        missing_features = set(feature_names) - set(data.feature_names)
        if missing_features:
            raise ValueError(f'Specified feature names not found in FeatureSet: {missing_features}')
        feature_indices = [data.feature_names.index(name) for name in feature_names]
        selected_features = data.features[:, feature_indices].copy() if not inplace else data.features[:, feature_indices]
    n_features_to_transform = len(feature_indices)
    if isinstance(lambdas, (int, float)):
        lambda_values = [float(lambdas)] * n_features_to_transform
    elif isinstance(lambdas, list):
        if len(lambdas) != n_features_to_transform:
            raise ValueError(f'Length of lambdas list ({len(lambdas)}) must match number of features to transform ({n_features_to_transform})')
        lambda_values = lambdas
    else:
        raise TypeError('lambdas must be a float or list of floats')
    for (i, lambda_val) in enumerate(lambda_values):
        feature_col = selected_features[:, i]
        shifted_features = feature_col + shift_factor
        transformed_col = np.sign(shifted_features) * np.power(np.abs(shifted_features), lambda_val)
        selected_features[:, i] = transformed_col
    if inplace:
        if feature_names is None:
            data.features[:] = selected_features
        else:
            for (idx, original_idx) in enumerate(feature_indices):
                data.features[:, original_idx] = selected_features[:, idx]
        result = data
    else:
        new_features = data.features.copy()
        if feature_names is None:
            new_features = selected_features
        else:
            for (idx, original_idx) in enumerate(feature_indices):
                new_features[:, original_idx] = selected_features[:, idx]
        result = FeatureSet(features=new_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else None, quality_scores=data.quality_scores.copy() if data.quality_scores else None)
    return result