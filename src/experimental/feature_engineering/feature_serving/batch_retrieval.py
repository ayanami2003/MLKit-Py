from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union
import numpy as np
_DEFAULT_FEATURE_STORE = {}

def retrieve_batch_features(batch_ids: List[str], feature_store: Optional[object]=None, feature_names: Optional[List[str]]=None, include_metadata: bool=True) -> Union[FeatureSet, List[FeatureSet]]:
    """
    Retrieve precomputed features for a batch of identifiers from a feature store.

    This function fetches feature data for a specified list of batch identifiers,
    returning either a single FeatureSet (if all features are retrieved together)
    or a list of FeatureSets (if features are retrieved separately per batch item).
    It supports selective feature retrieval and optional metadata inclusion.

    Args:
        batch_ids (List[str]): A list of unique identifiers for which to retrieve features.
        feature_store (Optional[object]): An optional feature store object to retrieve features from.
                                          If not provided, uses the system's default feature store.
        feature_names (Optional[List[str]]): A list of specific feature names to retrieve.
                                             If None, retrieves all available features.
        include_metadata (bool): Whether to include metadata with the retrieved features.
                                 Defaults to True.

    Returns:
        Union[FeatureSet, List[FeatureSet]]: Retrieved features in a structured format.
                                             Returns a single FeatureSet if features are
                                             combined, or a list of FeatureSets if retrieved
                                             individually.

    Raises:
        ValueError: If batch_ids is empty or contains invalid identifiers.
        KeyError: If specified feature names are not found in the feature store.
        RuntimeError: If feature retrieval fails due to system or connectivity issues.
    """
    if not batch_ids:
        raise ValueError('batch_ids cannot be empty')
    if not all((isinstance(id_, str) for id_ in batch_ids)):
        raise ValueError('All batch_ids must be strings')
    if feature_store is None:
        feature_store = _get_default_feature_store()
    try:
        return _retrieve_combined_features(batch_ids, feature_store, feature_names, include_metadata)
    except NotImplementedError:
        return _retrieve_individual_features(batch_ids, feature_store, feature_names, include_metadata)