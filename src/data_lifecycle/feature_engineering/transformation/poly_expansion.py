from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from itertools import combinations_with_replacement, combinations
from copy import deepcopy


# ...(code omitted)...


def apply_polynomial_expansion(feature_set: FeatureSet, degree: int=2, interaction_only: bool=False, include_bias: bool=True, feature_names: Optional[List[str]]=None, inplace: bool=False) -> FeatureSet:
    """
    Apply polynomial expansion to generate higher-order features.
    
    Creates new features by computing polynomial combinations of the input features,
    including powers and cross-products up to the specified degree.
    
    Parameters
    ----------
    data : FeatureSet
        Input feature set to transform.
    degree : int, default=2
        The degree of the polynomial features. For example, if degree=2, features
        will be squared and cross-multiplied.
    interaction_only : bool, default=False
        If True, only interaction features are produced (no pure polynomial terms
        like x1^2, x2^2, etc.).
    include_bias : bool, default=True
        If True, include a bias column (intercept term) of ones in the output.
    feature_names : Optional[List[str]], default=None
        Names for the input features. If provided, new feature names will be generated.
    inplace : bool, default=False
        If True, perform the transformation in-place on the input data.
        
    Returns
    -------
    FeatureSet
        Transformed feature set with polynomial features.
    """
    from collections import Counter
    X = feature_set.features
    (n_samples, n_features) = X.shape
    if feature_names is None:
        if feature_set.feature_names is not None:
            feature_names = feature_set.feature_names
        else:
            feature_names = [f'x{i + 1}' for i in range(n_features)]
    feature_combinations = []
    combination_names = []
    if include_bias:
        feature_combinations.append(())
        combination_names.append('1')
    for d in range(1, degree + 1):
        if interaction_only:
            combos = combinations(range(n_features), d)
        else:
            combos = combinations_with_replacement(range(n_features), d)
        for combo in combos:
            if interaction_only and len(set(combo)) == 1 and (d > 1):
                continue
            feature_combinations.append(combo)
            if len(combo) == 0:
                name = '1'
            elif len(combo) == 1:
                if d == 1:
                    name = feature_names[combo[0]]
                else:
                    name = f'{feature_names[combo[0]]}^{d}'
            else:
                counts = Counter(combo)
                terms = []
                for (idx, count) in sorted(counts.items()):
                    if count == 1:
                        terms.append(feature_names[idx])
                    else:
                        terms.append(f'{feature_names[idx]}^{count}')
                name = '*'.join(terms)
            combination_names.append(name)
    expanded_features = []
    for combo in feature_combinations:
        if len(combo) == 0:
            expanded_features.append(np.ones(n_samples))
        else:
            feature_product = np.ones(n_samples)
            for idx in combo:
                feature_product *= X[:, idx]
            expanded_features.append(feature_product)
    if expanded_features:
        expanded_data = np.column_stack(expanded_features)
    else:
        expanded_data = np.empty((n_samples, 0))
    if inplace:
        result = feature_set
        result.features = expanded_data
        result.feature_names = combination_names
        return result
    else:
        result = deepcopy(feature_set)
        result.features = expanded_data
        result.feature_names = combination_names
        return result