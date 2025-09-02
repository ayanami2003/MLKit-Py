from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional
import numpy as np

class TemporalFeatureExtractor(BaseTransformer):

    def __init__(self, datetime_columns: List[str], include_cyclical: bool=True, name: Optional[str]=None):
        """
        Initialize the TemporalFeatureExtractor.

        Args:
            datetime_columns (List[str]): List of column names that contain datetime information.
            include_cyclical (bool): If True, adds cyclical transformations for periodic features.
            name (Optional[str]): Name of the transformer instance.
        """
        super().__init__(name)
        self.datetime_columns = datetime_columns
        self.include_cyclical = include_cyclical
        self.feature_names: List[str] = []

    def fit(self, data: FeatureSet, **kwargs) -> 'TemporalFeatureExtractor':
        """
        Fit the transformer to the input data.

        This method identifies which temporal features can be extracted
        from the specified datetime columns.

        Args:
            data (FeatureSet): Input feature set containing datetime columns.
            **kwargs: Additional fitting parameters.

        Returns:
            TemporalFeatureExtractor: Returns self for method chaining.
        """
        if data.feature_names is None:
            raise ValueError('FeatureSet must have feature names specified')
        missing_columns = set(self.datetime_columns) - set(data.feature_names)
        if missing_columns:
            raise ValueError(f'Datetime columns not found in data: {missing_columns}')
        self.original_feature_names = data.feature_names.copy()
        self.feature_names = []
        for col in self.datetime_columns:
            self.feature_names.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_hour', f'{col}_minute', f'{col}_second', f'{col}_dayofweek', f'{col}_dayofyear', f'{col}_weekofyear', f'{col}_quarter'])
            if self.include_cyclical:
                cyclical_features = [f'{col}_hour_sin', f'{col}_hour_cos', f'{col}_dayofweek_sin', f'{col}_dayofweek_cos', f'{col}_month_sin', f'{col}_month_cos', f'{col}_day_sin', f'{col}_day_cos']
                self.feature_names.extend(cyclical_features)
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Extract temporal features from the datetime columns.

        This method creates new features based on the datetime columns,
        such as year, month, day, hour, minute, second, day of week,
        day of year, week of year, and quarter.

        Args:
            data (FeatureSet): Input feature set containing datetime columns.
            **kwargs: Additional transformation parameters.

        Returns:
            FeatureSet: New feature set with added temporal features.
        """
        pass

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Remove temporal features and return original feature set.

        This method removes the temporal features that were added during transformation,
        returning the feature set to its original state.

        Args:
            data (FeatureSet): Feature set with temporal features.
            **kwargs: Additional inverse transformation parameters.

        Returns:
            FeatureSet: Original feature set without temporal features.
        """
        pass