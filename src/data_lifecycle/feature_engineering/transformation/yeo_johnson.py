from typing import Optional, Dict, Any, List
from general.structures.feature_set import FeatureSet
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from scipy import stats


# ...(code omitted)...


def apply_yeo_johnson_transformation(data: FeatureSet, feature_names: Optional[List[str]]=None, optimize_method: str='mle', handle_constant: str='ignore', inplace: bool=False) -> FeatureSet:
    """
    Apply Yeo-Johnson transformation to specified features in a dataset.
    
    This function performs the Yeo-Johnson power transformation on selected features
    to make their distributions more Gaussian-like. It automatically estimates the
    optimal transformation parameters for each feature.
    
    Parameters
    ----------
    data : FeatureSet
        Input feature set containing the data to transform. Must contain a 2D numpy array
        of features with shape (n_samples, n_features).
    feature_names : Optional[List[str]]
        List of feature names to apply the transformation to. If None, applies to all features.
    inplace : bool, default=False
        If True, modifies the input data directly. If False, creates a new FeatureSet.
    optimize_method : str, default='mle'
        Method to use for optimizing lambda parameters ('mle' or 'pearson')
    handle_constant : str, default='ignore'
        How to handle constant features: 'ignore', 'raise', or 'zero'
        
    Returns
    -------
    FeatureSet
        Feature set with Yeo-Johnson transformed features. If inplace=True, returns the
        modified input; otherwise returns a new FeatureSet instance.
        
    Raises
    ------
    ValueError
        If feature_names contains names not present in the input data, or if
        handle_constant='raise' and constant features are detected
    TypeError
        If input data is not a FeatureSet instance
    """
    if not isinstance(data, FeatureSet):
        raise TypeError('data must be a FeatureSet instance')
    if data.features.size == 0:
        raise ValueError('Input data is empty')
    if feature_names is None:
        feature_indices = list(range(data.features.shape[1]))
        transform_feature_names = data.feature_names if data.feature_names else [f'feature_{i}' for i in range(data.features.shape[1])]
    else:
        if data.feature_names is None:
            raise ValueError('feature_names provided but input data has no feature names')
        feature_indices = []
        for name in feature_names:
            try:
                idx = data.feature_names.index(name)
                feature_indices.append(idx)
            except ValueError:
                raise ValueError(f"Feature '{name}' not found in data")
        transform_feature_names = feature_names
    if inplace:
        result_features = data.features
        result_metadata = data.metadata.copy() if data.metadata is not None else {}
    else:
        result_features = data.features.copy()
        result_metadata = data.metadata.copy() if data.metadata is not None else {}
    lambda_dict = {}
    for (idx, feature_name) in zip(feature_indices, transform_feature_names):
        feature_column = data.features[:, idx].copy()
        if np.all(feature_column == feature_column[0]):
            if handle_constant == 'raise':
                raise ValueError(f'Feature {feature_name} is constant')
            elif handle_constant == 'zero':
                result_features[:, idx] = 0.0
                lambda_dict[feature_name] = 0.0
                continue
            elif handle_constant == 'ignore':
                lambda_dict[feature_name] = 0.0
                continue
        if optimize_method == 'mle':
            try:
                lambda_param = stats.yeojohnson_normmax(feature_column)
            except Exception:
                lambda_param = 1.0
        elif optimize_method == 'pearson':
            try:
                (_, lambda_param) = stats.yeojohnson(feature_column)
            except Exception:
                lambda_param = 1.0
        else:
            raise ValueError("optimize_method must be 'mle' or 'pearson'")
        lambda_dict[feature_name] = lambda_param
        try:
            transformed_column = stats.yeojohnson(feature_column, lmbda=lambda_param)
            result_features[:, idx] = transformed_column
        except Exception:
            pass
    if lambda_dict:
        result_metadata['yeo_johnson_lambdas'] = lambda_dict
    if inplace:
        data.features[:] = result_features
        data.metadata = result_metadata
        return data
    else:
        return FeatureSet(features=result_features, feature_names=data.feature_names, feature_types=data.feature_types, sample_ids=data.sample_ids, metadata=result_metadata, quality_scores=data.quality_scores)