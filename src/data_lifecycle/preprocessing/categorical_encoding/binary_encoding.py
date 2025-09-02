from typing import List, Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class BinaryEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', handle_missing: str='error', name: Optional[str]=None):
        super().__init__(name=name)
        if handle_unknown not in ['error', 'ignore']:
            raise ValueError("handle_unknown must be either 'error' or 'ignore'")
        if handle_missing not in ['error', 'return_nan']:
            raise ValueError("handle_missing must be either 'error' or 'return_nan'")
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'BinaryEncoder':
        """
        Fit the binary encoder to the input data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode.
        **kwargs : dict
            Additional fitting parameters.
            
        Returns
        -------
        BinaryEncoder
            Fitted encoder instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_in_ = data.feature_names
        else:
            X = data
            self.feature_names_in_ = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_features_in_ = X.shape[1]
        if self.feature_names_in_ is not None and len(self.feature_names_in_) != self.n_features_in_:
            raise ValueError('Number of feature names must match number of features')
        self.categories_ = []
        for i in range(self.n_features_in_):
            column = X[:, i]
            non_null_values = []
            for val in column:
                if val is not None and (not (isinstance(val, float) and np.isnan(val))):
                    non_null_values.append(str(val))
            unique_vals = sorted(list(set(non_null_values)))
            self.categories_.append(unique_vals)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using binary encoding.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional transformation parameters.
            
        Returns
        -------
        FeatureSet
            Transformed data with binary-encoded categorical features.
        """
        if not hasattr(self, 'categories_'):
            raise ValueError("This BinaryEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but BinaryEncoder is expecting {self.n_features_in_} features as input')
        transformed_columns = []
        for i in range(self.n_features_in_):
            column = X[:, i]
            categories = self.categories_[i]
            if len(categories) == 0:
                n_bits = 0
            elif len(categories) == 1:
                n_bits = 1
            else:
                n_bits = int(np.ceil(np.log2(len(categories))))
            if n_bits == 0:
                transformed_col = np.zeros((len(column), 0))
            else:
                category_to_binary = {}
                for (idx, category) in enumerate(categories):
                    binary_repr = format(idx, f'0{n_bits}b')
                    category_to_binary[category] = [int(bit) for bit in binary_repr]
                if self.handle_missing == 'return_nan':
                    transformed_col = np.full((len(column), n_bits), np.nan, dtype=float)
                else:
                    transformed_col = np.zeros((len(column), n_bits), dtype=float)
                for (j, val) in enumerate(column):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        if self.handle_missing == 'error':
                            raise ValueError("Missing values encountered during transform. To allow missing values, set handle_missing='return_nan'")
                    else:
                        val_str = str(val)
                        if val_str not in category_to_binary:
                            if self.handle_unknown == 'error':
                                raise ValueError(f"Unknown category '{val_str}' encountered during transform. To ignore unknown categories, set handle_unknown='ignore'")
                            else:
                                transformed_col[j] = np.zeros(n_bits)
                        else:
                            transformed_col[j] = category_to_binary[val_str]
            transformed_columns.append(transformed_col)
        if transformed_columns:
            transformed_features = np.hstack(transformed_columns)
        else:
            transformed_features = np.array([]).reshape(len(X), 0)
        output_feature_names = self.get_feature_names_out(feature_names)
        return FeatureSet(features=transformed_features, feature_names=output_feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert binary-encoded features back to original categorical values.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Binary-encoded data to convert back.
        **kwargs : dict
            Additional inverse transformation parameters.
            
        Returns
        -------
        FeatureSet
            Data with categorical features restored to original values.
        """
        if not hasattr(self, 'categories_'):
            raise ValueError("This BinaryEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        reconstructed_columns = []
        feature_idx = 0
        for i in range(self.n_features_in_):
            categories = self.categories_[i]
            if len(categories) == 0:
                n_bits = 0
            elif len(categories) == 1:
                n_bits = 1
            else:
                n_bits = int(np.ceil(np.log2(len(categories))))
            if n_bits == 0:
                reconstructed_col = np.full(len(X), None, dtype=object)
            else:
                feature_bits = X[:, feature_idx:feature_idx + n_bits]
                feature_idx += n_bits
                reconstructed_col = np.empty(len(feature_bits), dtype=object)
                binary_to_category = {}
                for (idx, category) in enumerate(categories):
                    binary_repr = format(idx, f'0{n_bits}b')
                    binary_to_category[binary_repr] = category
                for (j, bits) in enumerate(feature_bits):
                    if np.isnan(bits).any():
                        reconstructed_col[j] = np.nan
                        continue
                    bit_string = ''.join((str(int(bit)) for bit in bits))
                    if bit_string in binary_to_category:
                        reconstructed_col[j] = binary_to_category[bit_string]
                    else:
                        reconstructed_col[j] = None
            reconstructed_columns.append(reconstructed_col)
        if reconstructed_columns:
            reconstructed_features = np.column_stack(reconstructed_columns)
        else:
            reconstructed_features = np.array([]).reshape(len(X), 0)
        return FeatureSet(features=reconstructed_features, feature_names=self.feature_names_in_)

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for binary-encoded features.
        
        Parameters
        ----------
        input_features : Optional[List[str]], default=None
            Input feature names. If None, uses names from fit.
            
        Returns
        -------
        List[str]
            Output feature names for binary-encoded features.
        """
        if not hasattr(self, 'categories_'):
            raise ValueError("This BinaryEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if input_features is not None:
            if len(input_features) != self.n_features_in_:
                raise ValueError(f'input_features should have length {self.n_features_in_} (the number of features passed to fit)')
            feature_names = input_features
        else:
            feature_names = self.feature_names_in_
        output_names = []
        feature_names_list = feature_names or [f'feature_{i}' for i in range(self.n_features_in_)]
        for (i, (feature_name, categories)) in enumerate(zip(feature_names_list, self.categories_)):
            if len(categories) == 0:
                n_bits = 0
            elif len(categories) == 1:
                n_bits = 1
            else:
                n_bits = int(np.ceil(np.log2(len(categories))))
            for bit_position in range(n_bits):
                output_names.append(f'{feature_name}_{bit_position}')
        return output_names