from typing import Optional, List, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
import pandas as pd

class CyclicalDatetimeEncoder(BaseTransformer):

    def __init__(self, datetime_components: Optional[List[str]]=None, feature_names: Optional[List[str]]=None, drop_original: bool=True, name: Optional[str]=None):
        super().__init__(name=name)
        self.datetime_components = datetime_components or ['hour', 'weekday', 'month']
        self.feature_names = feature_names
        self.drop_original = drop_original
        self.periods_ = {'hour': 24, 'weekday': 7, 'month': 12, 'day': 31, 'minute': 60, 'second': 60, 'weekofyear': 52}

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'CyclicalDatetimeEncoder':
        """
        Fit the encoder to the input data.
        
        Validates that specified datetime features exist and prepares internal mappings
        for cyclical transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing datetime features to encode
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        CyclicalDatetimeEncoder
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If specified features are not found in the input data
        """
        feature_names_param = kwargs.get('feature_names', None)
        if isinstance(data, FeatureSet):
            if data.feature_names is None:
                raise ValueError('FeatureSet must have feature names specified')
            feature_names = data.feature_names
        elif isinstance(data, np.ndarray):
            if self.feature_names is not None:
                feature_names = self.feature_names
            elif feature_names_param is not None:
                feature_names = feature_names_param
            else:
                raise ValueError('Feature names must be provided either in data or as parameter')
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        self._fit_feature_names = feature_names
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(feature_names)
            if missing_features:
                raise ValueError(f'Specified datetime features not found in data: {missing_features}')
        else:
            self.feature_names = [name for name in feature_names if name in self.datetime_components or any((comp in name for comp in self.datetime_components))]
        self._feature_indices = {name: i for (i, name) in enumerate(feature_names)}
        self.feature_names_ = []
        for name in feature_names:
            if self.feature_names and name in self.feature_names:
                for component in self.datetime_components:
                    self.feature_names_.extend([f'sin_{name}_{component}', f'cos_{name}_{component}'])
                if not self.drop_original:
                    self.feature_names_.append(name)
            else:
                self.feature_names_.append(name)
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Apply cyclical encoding to datetime features.
        
        Transforms specified datetime components into sine and cosine representations.
        For each component, two new features are created: sin_<component> and cos_<component>.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data with datetime features to encode
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Transformed data with cyclical datetime features
            
        Raises
        ------
        ValueError
            If transformer has not been fitted or if required features are missing
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = getattr(self, '_fit_feature_names', None)
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        if feature_names is not None and len(feature_names) != X.shape[1]:
            raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({X.shape[1]})")
        transformed_features = []
        transformed_feature_names = []
        names_to_use = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        for name in names_to_use:
            if self.feature_names and name in self.feature_names:
                col_idx = self._feature_indices[name]
                datetime_col = X[:, col_idx]
                if not np.issubdtype(datetime_col.dtype, np.datetime64):
                    datetime_col = pd.to_datetime(datetime_col)
                for component in self.datetime_components:
                    if component == 'year':
                        values = datetime_col.year
                    elif component == 'month':
                        values = datetime_col.month
                    elif component == 'day':
                        values = datetime_col.day
                    elif component == 'hour':
                        values = datetime_col.hour
                    elif component == 'minute':
                        values = datetime_col.minute
                    elif component == 'second':
                        values = datetime_col.second
                    elif component == 'weekday':
                        values = datetime_col.weekday()
                    elif component == 'weekofyear':
                        values = datetime_col.isocalendar().week
                    else:
                        raise ValueError(f'Unsupported datetime component: {component}')
                    period = self.periods_[component]
                    sin_values = np.sin(2 * np.pi * values / period)
                    cos_values = np.cos(2 * np.pi * values / period)
                    transformed_features.append(sin_values)
                    transformed_features.append(cos_values)
                    transformed_feature_names.extend([f'sin_{name}_{component}', f'cos_{name}_{component}'])
                if not self.drop_original:
                    transformed_features.append(X[:, col_idx])
                    transformed_feature_names.append(name)
            else:
                if name in self._feature_indices:
                    col_idx = self._feature_indices[name]
                else:
                    col_idx = names_to_use.index(name)
                transformed_features.append(X[:, col_idx])
                transformed_feature_names.append(name)
        X_transformed = np.column_stack(transformed_features)
        metadata['cyclical_encoding_applied'] = True
        metadata['original_feature_count'] = X.shape[1]
        metadata['transformed_feature_count'] = X_transformed.shape[1]
        return FeatureSet(features=X_transformed, feature_names=transformed_feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Reverse the cyclical encoding transformation.
        
        Attempts to reconstruct approximate original datetime components from
        sine and cosine representations using arc tangent functions.
        
        Note: Reconstruction may not be exact due to information loss in the
        cyclical transformation and potential floating-point inaccuracies.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Data with cyclical features to reverse transform
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Data with reconstructed datetime components
            
        Raises
        ------
        ValueError
            If required cyclical features are not found in input data
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            feature_types = data.feature_types
            sample_ids = data.sample_ids
            metadata = data.metadata.copy() if data.metadata else {}
            quality_scores = data.quality_scores.copy() if data.quality_scores else {}
        elif isinstance(data, np.ndarray):
            X = data
            feature_names = getattr(self, 'feature_names_', None)
            if feature_names is None:
                feature_names = getattr(self, '_fit_feature_names', None)
            feature_types = None
            sample_ids = None
            metadata = {}
            quality_scores = {}
        else:
            raise ValueError('Input data must be either a FeatureSet or numpy array')
        if X.ndim != 2:
            raise ValueError('Input data must be a 2D array')
        feature_name_to_index = {name: i for (i, name) in enumerate(feature_names)}
        reconstructed_features = []
        reconstructed_feature_names = []
        used_indices = set()
        for (i, name) in enumerate(feature_names):
            is_cyclical = False
            for orig_feature in self.feature_names or []:
                for component in self.datetime_components:
                    if name in [f'sin_{orig_feature}_{component}', f'cos_{orig_feature}_{component}']:
                        is_cyclical = True
                        break
                if is_cyclical:
                    break
            is_kept_original = not self.drop_original and self.feature_names and (name in self.feature_names)
            if not is_cyclical and (not is_kept_original):
                reconstructed_features.append(X[:, i])
                reconstructed_feature_names.append(name)
                used_indices.add(i)
            elif is_kept_original:
                reconstructed_features.append(X[:, i])
                reconstructed_feature_names.append(name)
                used_indices.add(i)
        if self.feature_names:
            for orig_feature in self.feature_names:
                component_features_added = {}
                for component in self.datetime_components:
                    sin_name = f'sin_{orig_feature}_{component}'
                    cos_name = f'cos_{orig_feature}_{component}'
                    if sin_name in feature_name_to_index and cos_name in feature_name_to_index:
                        sin_idx = feature_name_to_index[sin_name]
                        cos_idx = feature_name_to_index[cos_name]
                        sin_vals = X[:, sin_idx]
                        cos_vals = X[:, cos_idx]
                        period = self.periods_[component]
                        angles = np.arctan2(sin_vals, cos_vals)
                        reconstructed_vals = angles / (2 * np.pi) * period
                        reconstructed_vals = (reconstructed_vals + period) % period
                        if component in ['month', 'day']:
                            reconstructed_vals = np.round(reconstructed_vals) + 1
                        else:
                            reconstructed_vals = np.round(reconstructed_vals)
                        component_features_added[component] = reconstructed_vals
                        used_indices.update([sin_idx, cos_idx])
                for component in self.datetime_components:
                    if component in component_features_added:
                        reconstructed_features.append(component_features_added[component])
                        reconstructed_feature_names.append(f'{orig_feature}_{component}')
                if not self.drop_original and orig_feature in feature_name_to_index:
                    idx = feature_name_to_index[orig_feature]
                    if idx not in used_indices:
                        reconstructed_features.append(X[:, idx])
                        reconstructed_feature_names.append(orig_feature)
                        used_indices.add(idx)
        if reconstructed_features:
            X_reconstructed = np.column_stack(reconstructed_features)
        else:
            X_reconstructed = np.empty((X.shape[0], 0))
        metadata['inverse_transform_applied'] = True
        return FeatureSet(features=X_reconstructed, feature_names=reconstructed_feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def get_feature_names(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get names of features produced by this transformer.
        
        Parameters
        ----------
        input_features : List[str], optional
            Names of input features (used to determine output names if not already fitted)
            
        Returns
        -------
        List[str]
            Names of all output features including original non-datetime features
            and newly created cyclical features
        """
        if hasattr(self, 'feature_names_'):
            return self.feature_names_.copy()
        if input_features is not None:
            feature_names = input_features
        elif hasattr(self, '_fit_feature_names') and self._fit_feature_names is not None:
            feature_names = self._fit_feature_names
        elif hasattr(self, 'feature_names') and self.feature_names is not None:
            feature_names = self.feature_names
        else:
            output_names = []
            for component in self.datetime_components:
                output_names.extend([f'sin_feature_{component}', f'cos_feature_{component}'])
            return output_names
        output_names = []
        for name in feature_names:
            is_datetime_feature = hasattr(self, 'feature_names') and self.feature_names is not None and (name in self.feature_names) or (not hasattr(self, 'feature_names') or self.feature_names is None)
            if is_datetime_feature and (hasattr(self, 'feature_names') and self.feature_names is not None and (name in self.feature_names)):
                for component in self.datetime_components:
                    output_names.extend([f'sin_{name}_{component}', f'cos_{name}_{component}'])
                if not self.drop_original:
                    output_names.append(name)
            elif not (hasattr(self, 'feature_names') and self.feature_names is not None and (name in self.feature_names)):
                output_names.append(name)
            elif hasattr(self, 'feature_names') and self.feature_names is not None and (name in self.feature_names):
                if not self.drop_original:
                    output_names.append(name)
        return output_names