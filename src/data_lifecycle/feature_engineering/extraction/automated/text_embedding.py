from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union
import numpy as np
import warnings
SentenceTransformer = None

class TextEmbeddingExtractor(BaseTransformer):

    def __init__(self, embedding_method: str='sentence-transformers', model_name: Optional[str]=None, max_length: Optional[int]=None, pooling_strategy: str='mean', name: Optional[str]=None):
        """
        Initialize the TextEmbeddingExtractor.
        
        Parameters
        ----------
        embedding_method : str, default="sentence-transformers"
            The embedding technique to use ('sentence-transformers', 'tfidf', etc.)
        model_name : str, optional
            Specific model identifier (e.g., 'all-MiniLM-L6-v2' for sentence-transformers)
        max_length : int, optional
            Maximum sequence length for tokenization
        pooling_strategy : str, default="mean"
            Strategy for pooling token embeddings ('mean', 'max', 'cls')
        name : str, optional
            Name for the transformer instance
        """
        super().__init__(name=name)
        self.embedding_method = embedding_method
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy

    def fit(self, data: Union[FeatureSet, List[str]], **kwargs) -> 'TextEmbeddingExtractor':
        """
        Fit the text embedding extractor to the input data.
        
        For some embedding methods, this may involve loading pre-trained models
        or computing corpus-level statistics.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Input text data as a FeatureSet or list of strings
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        TextEmbeddingExtractor
            Self instance for method chaining
        """
        supported_methods = ['sentence-transformers']
        if self.embedding_method not in supported_methods:
            raise ValueError(f"Unsupported embedding method '{self.embedding_method}'. Supported methods: {supported_methods}")
        supported_pooling = ['mean', 'max', 'cls']
        if self.pooling_strategy not in supported_pooling:
            raise ValueError(f"Unsupported pooling strategy '{self.pooling_strategy}'. Supported strategies: {supported_pooling}")
        if self.embedding_method == 'sentence-transformers':
            global SentenceTransformer
            if SentenceTransformer is None:
                try:
                    from sentence_transformers import SentenceTransformer as ST
                    SentenceTransformer = ST
                except ImportError:
                    raise ImportError("sentence-transformers is required for 'sentence-transformers' embedding method. Please install it with: pip install sentence-transformers")
            if self.model_name is None:
                self.model_name = 'all-MiniLM-L6-v2'
            self._model = SentenceTransformer(self.model_name)
            if self.max_length is None:
                self.max_length = getattr(self._model, 'max_seq_length', 512)
        return self

    def transform(self, data: Union[FeatureSet, List[str]], **kwargs) -> FeatureSet:
        """
        Transform text data into embedding vectors.
        
        Converts input text documents into dense vector representations using
        the configured embedding method.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Input text data as a FeatureSet or list of strings
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            FeatureSet containing the extracted embeddings with metadata
        """
        if not hasattr(self, '_model'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit()' first.")
        if isinstance(data, FeatureSet):
            if hasattr(data, 'features') and isinstance(data.features, list):
                texts = data.features
            else:
                raise ValueError('FeatureSet must contain text data in a list format')
        elif isinstance(data, list):
            texts = data
        else:
            raise TypeError('Input data must be either a FeatureSet or a list of strings')
        if not all((isinstance(text, str) for text in texts)):
            raise ValueError('All items in the input data must be strings')
        if self.embedding_method == 'sentence-transformers':
            encode_kwargs = {'convert_to_numpy': True, 'show_progress_bar': False}
            if self.max_length is not None:
                encode_kwargs['max_length'] = self.max_length
                encode_kwargs['truncate'] = True
            embeddings = self._model.encode(texts, **encode_kwargs)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            feature_names = [f'embedding_dim_{i}' for i in range(embeddings.shape[1])]
            return FeatureSet(features=embeddings.tolist(), feature_names=feature_names, metadata={'embedding_method': self.embedding_method, 'model_name': self.model_name, 'pooling_strategy': self.pooling_strategy, 'transformer_name': self.name, 'max_length': self.max_length})
        else:
            raise ValueError(f"Unsupported embedding method '{self.embedding_method}'")

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[str]:
        """
        Attempt to reconstruct text from embeddings (if supported).
        
        Note: This is not generally possible for most embedding methods as
        they are lossy representations. This method may raise NotImplementedError
        for methods that don't support inversion.
        
        Parameters
        ----------
        data : FeatureSet
            FeatureSet containing embedding vectors
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        List[str]
            Approximated/reconstructed text (method dependent)
            
        Raises
        ------
        NotImplementedError
            If the embedding method doesn't support inversion
        """
        raise NotImplementedError('Text reconstruction from embeddings is not supported for sentence-transformers method.')

    def get_feature_names(self) -> List[str]:
        """
        Get names for the embedding dimensions.
        
        Returns
        -------
        List[str]
            Names for embedding dimensions (typically dimension indices)
        """
        if not hasattr(self, '_model'):
            raise ValueError("Transformer has not been fitted yet. Call 'fit()' first.")
        if self.embedding_method == 'sentence-transformers':
            sample_text = 'sample text'
            sample_embedding = self._model.encode([sample_text], convert_to_numpy=True)
            embedding_dim = sample_embedding.shape[1]
            return [f'embedding_dim_{i}' for i in range(embedding_dim)]
        else:
            raise ValueError(f"Unsupported embedding method '{self.embedding_method}'")