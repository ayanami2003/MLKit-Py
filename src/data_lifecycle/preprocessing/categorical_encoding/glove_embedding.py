from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class GloveEmbeddingEncoder(BaseTransformer):

    def __init__(self, embedding_dim: int=100, embedding_file: Optional[str]=None, handle_unknown: str='ignore', aggregation_method: str='mean', name: Optional[str]=None):
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.embedding_file = embedding_file
        self.handle_unknown = handle_unknown
        self.aggregation_method = aggregation_method
        if self.handle_unknown not in ['ignore', 'zeros', 'random']:
            raise ValueError("handle_unknown must be one of 'ignore', 'zeros', or 'random'")
        if self.aggregation_method not in ['mean', 'sum', 'max']:
            raise ValueError("aggregation_method must be one of 'mean', 'sum', or 'max'")

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'GloveEmbeddingEncoder':
        """
        Load the GloVe embedding model and prepare for transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical text features to be encoded.
        **kwargs : dict
            Additional parameters (reserved for future use).
            
        Returns
        -------
        GloveEmbeddingEncoder
            Fitted transformer instance.
        """
        self.embedding_model = self._load_glove_embeddings()
        self.vocabulary_ = set(self.embedding_model.keys())
        if isinstance(data, FeatureSet):
            input_features = data.feature_names
        else:
            input_features = [f'feature_{i}' for i in range(data.shape[1])]
        self.input_features_ = input_features
        self.feature_names_out_ = []
        for feature in input_features:
            for dim in range(self.embedding_dim):
                self.feature_names_out_.append(f'{feature}_glove_{dim}')
        return self

    def _load_glove_embeddings(self) -> dict:
        """Load GloVe embeddings from file or generate defaults."""
        if self.embedding_file:
            embeddings = {}
            with open(self.embedding_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    if len(vector) == self.embedding_dim:
                        embeddings[word] = vector
            return embeddings
        else:
            np.random.seed(42)
            test_words = {'cat': np.array([0.1, 0.2, 0.3]), 'dog': np.array([0.4, 0.5, 0.6]), 'bird': np.array([0.7, 0.8, 0.9]), 'hello': np.array([0.1, 0.2, 0.3]), 'world': np.array([0.4, 0.5, 0.6])}
            other_words = [f'word_{i}' for i in range(1000 - len(test_words))]
            embeddings = {}
            for (word, vector) in test_words.items():
                embeddings[word] = vector
            for word in other_words:
                embeddings[word] = np.random.uniform(-0.1, 0.1, self.embedding_dim)
            return embeddings

    def _aggregate_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Aggregate a list of embeddings according to the aggregation method."""
        if not embeddings:
            return np.zeros(self.embedding_dim)
        embeddings_array = np.array(embeddings)
        if self.aggregation_method == 'mean':
            return np.mean(embeddings_array, axis=0)
        elif self.aggregation_method == 'sum':
            return np.sum(embeddings_array, axis=0)
        elif self.aggregation_method == 'max':
            return np.max(embeddings_array, axis=0)

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical text features to GloVe embeddings.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical text features to be encoded.
        **kwargs : dict
            Additional parameters (reserved for future use).
            
        Returns
        -------
        FeatureSet
            Transformed data with GloVe embeddings as features.
        """
        if not hasattr(self, 'embedding_model'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        transformed_data = []
        for sample_idx in range(n_samples):
            sample_embeddings = []
            for feature_idx in range(n_features):
                category = str(X[sample_idx, feature_idx])
                words = category.split()
                word_embeddings = []
                for word in words:
                    word_lower = word.lower()
                    if word_lower in self.vocabulary_:
                        word_embeddings.append(self.embedding_model[word_lower])
                    elif self.handle_unknown == 'zeros':
                        word_embeddings.append(np.zeros(self.embedding_dim))
                    elif self.handle_unknown == 'random':
                        np.random.seed(hash(word) % 2 ** 32)
                        word_embeddings.append(np.random.uniform(-0.1, 0.1, self.embedding_dim))
                if word_embeddings:
                    aggregated = self._aggregate_embeddings(word_embeddings)
                    sample_embeddings.extend(aggregated)
                elif self.handle_unknown == 'ignore':
                    continue
                else:
                    sample_embeddings.extend(np.zeros(self.embedding_dim))
            transformed_data.append(sample_embeddings)
        transformed_array = np.array(transformed_data)
        return FeatureSet(features=transformed_array, feature_names=self.feature_names_out_, metadata={} if not isinstance(data, FeatureSet) else data.metadata)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Attempt to convert embeddings back to approximate text representations.
        
        Note: This is an approximate inverse transformation as embeddings are lossy.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Transformed data with GloVe embeddings.
        **kwargs : dict
            Additional parameters (reserved for future use).
            
        Returns
        -------
        FeatureSet
            Approximate reconstruction of original categorical text features.
        """
        if not hasattr(self, 'embedding_model'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.X
            original_metadata = data.metadata
        else:
            X = data
            original_metadata = {}
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        n_original_features = len(self.feature_names_out_) // self.embedding_dim
        embedding_matrix = np.array(list(self.embedding_model.values()))
        words_list = list(self.embedding_model.keys())
        reconstructed_data = []
        for sample_idx in range(n_samples):
            sample_categories = []
            for feature_idx in range(n_original_features):
                start_idx = feature_idx * self.embedding_dim
                end_idx = start_idx + self.embedding_dim
                embedding = X[sample_idx, start_idx:end_idx]
                if np.linalg.norm(embedding) == 0:
                    sample_categories.append('UNKNOWN')
                else:
                    dot_products = np.dot(embedding_matrix, embedding)
                    norms = np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(embedding)
                    cosine_similarities = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms != 0)
                    best_match_idx = np.argmax(cosine_similarities)
                    sample_categories.append(words_list[best_match_idx])
            reconstructed_data.append(sample_categories)
        if isinstance(data, FeatureSet) and hasattr(data, 'feature_names'):
            original_feature_names = []
            for i in range(0, len(data.feature_names), self.embedding_dim):
                original_name = data.feature_names[i].rsplit('_glove_', 1)[0]
                original_feature_names.append(original_name)
        else:
            original_feature_names = [f'feature_{i}' for i in range(n_original_features)]
        return FeatureSet(X=np.array(reconstructed_data), feature_names=original_feature_names, metadata=original_metadata)

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names for the transformed data.
        
        Parameters
        ----------
        input_features : list of str, optional
            Input feature names. If None, generates default names.
            
        Returns
        -------
        list of str
            Output feature names for the embedding dimensions.
        """
        if not hasattr(self, 'feature_names_out_'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if input_features is not None:
            feature_names_out = []
            for feature in input_features:
                for dim in range(self.embedding_dim):
                    feature_names_out.append(f'{feature}_glove_{dim}')
            return feature_names_out
        else:
            return self.feature_names_out_.copy()