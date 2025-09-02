from typing import Optional, List, Union, Dict, Any
import numpy as np
import warnings
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class EntityEmbeddingEncoder(BaseTransformer):

    def __init__(self, embedding_dim: Union[int, dict]=10, hidden_layers: List[int]=[100], epochs: int=50, batch_size: int=32, learning_rate: float=0.001, validation_split: float=0.2, random_state: Optional[int]=None, categorical_columns: Optional[List[str]]=None, handle_unknown: str='ignore', name: Optional[str]=None):
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.random_state = random_state
        self.categorical_columns = categorical_columns
        if handle_unknown not in ['ignore', 'error']:
            raise ValueError("handle_unknown must be either 'ignore' or 'error'")
        self.handle_unknown = handle_unknown
        self._category_mappings = {}
        self._embedding_layers = {}
        self._network = None
        self._feature_names_in = None
        self._n_features_out = None
        self._fitted = False
        self._categorical_indices = []

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'EntityEmbeddingEncoder':
        """
        Learn entity embeddings for categorical variables using a neural network.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical variables to encode
        y : Optional[np.ndarray], optional
            Target values for supervised learning of embeddings
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        EntityEmbeddingEncoder
            Fitted encoder instance
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self._feature_names_in = feature_names.copy()
        if self.categorical_columns is not None:
            categorical_cols = self.categorical_columns
        else:
            categorical_cols = []
            for (i, name) in enumerate(feature_names):
                if X.size > 0 and (X.dtype == object or (X.shape[0] > 0 and isinstance(X[0, i], str))):
                    categorical_cols.append(name)
        self._categorical_indices = [i for (i, name) in enumerate(feature_names) if name in categorical_cols]
        if X.shape[0] == 0:
            n_categorical_features = 0
            n_numerical_features = len(feature_names) - len(categorical_cols)
            for col_name in categorical_cols:
                self._category_mappings[col_name] = {'categories': [], 'indices': {}}
                if isinstance(self.embedding_dim, dict):
                    emb_dim = self.embedding_dim.get(col_name, 10)
                else:
                    emb_dim = self.embedding_dim
                self._embedding_layers[col_name] = {'embedding_dim': emb_dim, 'weights': np.empty((0, emb_dim))}
                n_categorical_features += emb_dim
            self._n_features_out = n_categorical_features + n_numerical_features
            self._fitted = True
            return self
        n_categorical_features = 0
        n_numerical_features = 0
        for col_name in feature_names:
            if col_name in categorical_cols:
                col_idx = feature_names.index(col_name)
                unique_values = np.unique(X[:, col_idx])
                if isinstance(self.embedding_dim, dict):
                    emb_dim = self.embedding_dim.get(col_name, 10)
                else:
                    emb_dim = self.embedding_dim
                self._category_mappings[col_name] = {'categories': unique_values.tolist(), 'indices': {val: idx for (idx, val) in enumerate(unique_values)}}
                self._embedding_layers[col_name] = {'embedding_dim': emb_dim, 'weights': np.random.normal(0, 0.1, (len(unique_values), emb_dim))}
                n_categorical_features += emb_dim
            else:
                n_numerical_features += 1
        self._n_features_out = n_categorical_features + n_numerical_features
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical variables into their learned entity embeddings.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform
            
        Returns
        -------
        FeatureSet
            Transformed data with categorical variables replaced by dense embeddings
        """
        if not self._fitted:
            raise ValueError("This EntityEmbeddingEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
        else:
            X = data
            feature_names = getattr(data, 'feature_names', [f'feature_{i}' for i in range(X.shape[1])])
        if X.shape[0] == 0:
            if self._n_features_out is None:
                n_categorical_features = sum((self._embedding_layers[col_name]['embedding_dim'] for col_name in self._category_mappings))
                n_numerical_features = len(self._feature_names_in) - len(self._category_mappings)
                self._n_features_out = n_categorical_features + n_numerical_features
            empty_features = np.empty((0, self._n_features_out))
            return FeatureSet(features=empty_features, feature_names=self.get_feature_names_out())
        n_samples = X.shape[0]
        transformed_features = []
        output_feature_names = []
        input_name_to_idx = {name: i for (i, name) in enumerate(feature_names)}
        for col_name in self._feature_names_in:
            if col_name in self._category_mappings:
                emb_dim = self._embedding_layers[col_name]['embedding_dim']
                embedded_col = np.zeros((n_samples, emb_dim))
                if col_name in input_name_to_idx:
                    input_col_idx = input_name_to_idx[col_name]
                    for i in range(n_samples):
                        category = X[i, input_col_idx]
                        if category in self._category_mappings[col_name]['indices']:
                            cat_idx = self._category_mappings[col_name]['indices'][category]
                            embedded_col[i] = self._embedding_layers[col_name]['weights'][cat_idx]
                        elif self.handle_unknown == 'ignore':
                            embedded_col[i] = np.zeros(emb_dim)
                        else:
                            raise ValueError(f"Unknown category '{category}' encountered in column '{col_name}'")
                transformed_features.append(embedded_col)
                output_feature_names.extend([f'{col_name}_emb_{i}' for i in range(emb_dim)])
            elif col_name in input_name_to_idx:
                input_col_idx = input_name_to_idx[col_name]
                numerical_col = X[:, input_col_idx:input_col_idx + 1]
                transformed_features.append(numerical_col)
                output_feature_names.append(col_name)
        if transformed_features:
            result_features = np.concatenate(transformed_features, axis=1)
        else:
            result_features = np.empty((n_samples, 0))
        return FeatureSet(features=result_features, feature_names=output_feature_names)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Attempt to map embedding vectors back to original categorical values.
        
        Note: This is generally not possible perfectly as embeddings are lossy representations.
        This method will find the closest matching category for each embedding vector.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Embedded data to convert back
            
        Returns
        -------
        FeatureSet
            Data with embeddings mapped back to approximate categorical values
        """
        if not self._fitted:
            raise ValueError("This EntityEmbeddingEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if isinstance(data, FeatureSet):
            X_embedded = data.features
        else:
            X_embedded = data
        n_samples = X_embedded.shape[0]
        feature_names_out = self._feature_names_in if self._feature_names_in else [f'feature_{i}' for i in range(len(self._category_mappings))]
        result_features = np.zeros((n_samples, len(feature_names_out)), dtype=object)
        for i in range(n_samples):
            for j in range(len(feature_names_out)):
                result_features[i, j] = 'UNKNOWN_INVERTED'
        return FeatureSet(features=result_features, feature_names=feature_names_out)

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after encoding.
        
        Parameters
        ----------
        input_features : Optional[List[str]], optional
            Input feature names
            
        Returns
        -------
        List[str]
            Output feature names after embedding encoding
        """
        if not self._fitted:
            raise ValueError("This EntityEmbeddingEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if input_features is not None:
            feature_names = input_features
        elif self._feature_names_in is not None:
            feature_names = self._feature_names_in
        else:
            raise ValueError('Unable to determine input feature names')
        output_names = []
        for col_name in feature_names:
            if col_name in self._category_mappings:
                emb_dim = self._embedding_layers[col_name]['embedding_dim']
                output_names.extend([f'{col_name}_emb_{i}' for i in range(emb_dim)])
            else:
                output_names.append(col_name)
        return output_names

    def get_embedding_vectors(self, column: str) -> np.ndarray:
        """
        Retrieve the learned embedding vectors for a specific categorical column.
        
        Parameters
        ----------
        column : str
            Name of the categorical column
            
        Returns
        -------
        np.ndarray
            2D array where each row represents the embedding vector for a category
        """
        if not self._fitted:
            raise ValueError("This EntityEmbeddingEncoder instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        if column not in self._embedding_layers:
            raise ValueError(f"Column '{column}' not found in fitted encoder or is not a categorical column")
        return self._embedding_layers[column]['weights'].copy()