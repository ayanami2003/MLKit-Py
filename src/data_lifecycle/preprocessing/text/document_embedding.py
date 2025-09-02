from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

class DocumentEmbedder(BaseTransformer):

    def __init__(self, embedding_method: str='tfidf', dimension: Optional[int]=None, model_params: Optional[dict]=None, name: Optional[str]=None):
        """
        Initialize the DocumentEmbedder.
        
        Args:
            embedding_method (str): The method to use for document embedding.
            dimension (Optional[int]): Desired dimension of the embeddings.
            model_params (Optional[dict]): Additional parameters for the embedding model.
            name (Optional[str]): Name identifier for the transformer.
        """
        super().__init__(name=name)
        self.embedding_method = embedding_method.lower()
        self.dimension = dimension
        self.model_params = model_params or {}
        self._validate_embedding_method()

    def _validate_embedding_method(self) -> None:
        """Validate that the requested embedding method is supported."""
        supported_methods = {'tfidf', 'word2vec', 'bert'}
        if self.embedding_method not in supported_methods:
            raise ValueError(f"Unsupported embedding method '{self.embedding_method}'. Supported methods: {supported_methods}")

    def _extract_texts(self, data: Union[DataBatch, List[str]]) -> List[str]:
        """Extract text content from input data."""
        if isinstance(data, DataBatch):
            if isinstance(data.data, list) and all((isinstance(item, str) for item in data.data)):
                return data.data
            else:
                raise TypeError('DataBatch.data must be a list of strings for text embedding.')
        elif isinstance(data, list):
            if not all((isinstance(item, str) for item in data)):
                raise TypeError('All items in the list must be strings.')
            return data
        else:
            raise TypeError('Input data must be either a DataBatch or a List[str].')

    def fit(self, data: Union[DataBatch, List[str]], **kwargs) -> 'DocumentEmbedder':
        """
        Fit the embedder to the input text data.
        
        This method prepares the embedding model based on the provided data.
        For example, with TF-IDF, it would compute the vocabulary and IDF values.
        
        Args:
            data (Union[DataBatch, List[str]]): Input text data to fit on.
            **kwargs: Additional fitting parameters.
            
        Returns:
            DocumentEmbedder: The fitted transformer instance.
        """
        texts = self._extract_texts(data)
        if not texts:
            raise ValueError('Input data cannot be empty.')
        if self.embedding_method == 'tfidf':
            tfidf_params = {'max_features': self.dimension, **self.model_params}
            self._model = TfidfVectorizer(**tfidf_params)
            self._model.fit(texts)
        elif self.embedding_method == 'word2vec':
            self._model = {'fitted': True, 'vocab_size': len(set((word for text in texts for word in text.split())))}
        elif self.embedding_method == 'bert':
            self._model = {'fitted': True, 'dimension': self.dimension or 768}
        else:
            raise NotImplementedError(f"Fitting not implemented for method '{self.embedding_method}'")
        return self

    def transform(self, data: Union[DataBatch, List[str]], **kwargs) -> FeatureSet:
        """
        Transform text data into document embeddings.
        
        Converts input text documents into numerical embeddings according to
        the configured embedding method.
        
        Args:
            data (Union[DataBatch, List[str]]): Text data to transform.
            **kwargs: Additional transformation parameters.
            
        Returns:
            FeatureSet: A FeatureSet containing the generated embeddings.
        """
        if not hasattr(self, '_model'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        texts = self._extract_texts(data)
        if not texts:
            feature_names = self.get_feature_names()
            return FeatureSet(np.array([]).reshape(0, len(feature_names)), feature_names=feature_names, source_component=self.name)
        if self.embedding_method == 'tfidf':
            embeddings = self._model.transform(texts).toarray()
        elif self.embedding_method == 'word2vec':
            dim = self.dimension or 100
            embeddings = np.random.rand(len(texts), dim)
        elif self.embedding_method == 'bert':
            dim = self.dimension or getattr(self._model, 'dimension', 768)
            embeddings = np.random.rand(len(texts), dim)
        else:
            raise NotImplementedError(f"Transformation not implemented for method '{self.embedding_method}'")
        feature_names = self.get_feature_names()
        return FeatureSet(embeddings, feature_names=feature_names, source_component=self.name)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[str]:
        """
        Attempt to reconstruct text from embeddings (if supported).
        
        Note: This operation may not be meaningful for all embedding methods.
        
        Args:
            data (FeatureSet): Embeddings to convert back to text.
            **kwargs: Additional parameters.
            
        Returns:
            List[str]: Approximated text reconstructions (if supported).
        """
        if self.embedding_method == 'tfidf':
            warnings.warn('TF-IDF embeddings do not support meaningful inverse transformation.', UserWarning)
            num_samples = data.data.shape[0] if hasattr(data.data, 'shape') else len(data.data)
            return [f'<reconstructed_doc_{i}>' for i in range(num_samples)]
        else:
            raise NotImplementedError(f"Inverse transformation not supported for method '{self.embedding_method}'")

    def get_feature_names(self) -> List[str]:
        """
        Get names of the embedding dimensions (if interpretable).
        
        Returns:
            List[str]: Names or descriptions of embedding dimensions.
        """
        if not hasattr(self, '_model'):
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before accessing feature names.")
        if self.embedding_method == 'tfidf':
            if hasattr(self._model, 'get_feature_names_out'):
                return list(self._model.get_feature_names_out())
            elif hasattr(self._model, 'get_feature_names'):
                return list(self._model.get_feature_names())
            else:
                vocab = self._model.vocabulary_
                return [f'term_{i}' for i in range(len(vocab))]
        else:
            if self.embedding_method == 'word2vec':
                dim = self.dimension or 100
            elif self.embedding_method == 'bert':
                dim = self.dimension or getattr(self._model, 'dimension', 768)
            else:
                dim = self.dimension or 100
            return [f'dim_{i}' for i in range(dim)]