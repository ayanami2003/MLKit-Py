from general.structures.feature_set import FeatureSet
from general.base_classes.transformer_base import BaseTransformer
from typing import List, Optional, Union
import numpy as np
import warnings

class SemanticEmbeddingExtractor(BaseTransformer):
    """
    Extract semantic embeddings from text data using pre-trained models.
    
    This transformer converts text documents into dense vector representations
    that capture semantic meaning. It supports various pre-trained embedding
    models and can handle both single strings and collections of texts.
    
    The extractor maintains consistency in embedding dimensions and provides
    methods to access embedding metadata including model information and
    vocabulary coverage.
    
    Attributes
    ----------
    model_name : str
        Name of the pre-trained model to use for embedding extraction
    embedding_dim : int
        Dimension of the output embeddings
    max_length : Optional[int]
        Maximum sequence length for tokenization (if applicable)
    """

    def __init__(self, model_name: str='sentence-transformers/all-MiniLM-L6-v2', embedding_dim: int=384, max_length: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the semantic embedding extractor.
        
        Parameters
        ----------
        model_name : str, default="sentence-transformers/all-MiniLM-L6-v2"
            Name or path of the pre-trained model to use
        embedding_dim : int, default=384
            Expected dimension of output embeddings
        max_length : Optional[int], default=None
            Maximum sequence length for tokenization
        name : Optional[str], default=None
            Name for this transformer instance
        """
        super().__init__(name=name)
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self._model = None
        self._statistics = {}
        self._sentence_transformers_available = self._check_sentence_transformers_availability()

    def _check_sentence_transformers_availability(self) -> bool:
        """Check if sentence-transformers library is available."""
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False

    def fit(self, data: Union[List[str], FeatureSet], **kwargs) -> 'SemanticEmbeddingExtractor':
        """
        Load the pre-trained model and prepare for embedding extraction.
        
        This method initializes the embedding model and validates compatibility
        with the specified embedding dimensions. For text data, it may also
        build vocabulary statistics.
        
        Parameters
        ----------
        data : Union[List[str], FeatureSet]
            Training data containing text documents to initialize with
        **kwargs : dict
            Additional parameters for model loading
            
        Returns
        -------
        SemanticEmbeddingExtractor
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If model loading fails or embedding dimensions mismatch
        ImportError
            If sentence-transformers package is not installed
        """
        if not self._sentence_transformers_available:
            raise ImportError("The 'sentence-transformers' package is required for SemanticEmbeddingExtractor. Please install it using: pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise ValueError(f"Failed to load model '{self.model_name}': {str(e)}")
        test_text = 'Test sentence for dimension validation.'
        try:
            test_embedding = self._model.encode([test_text])[0]
        except Exception as e:
            raise ValueError(f'Failed to generate test embedding: {str(e)}')
        if test_embedding.shape[0] != self.embedding_dim:
            raise ValueError(f'Model embedding dimension ({test_embedding.shape[0]}) does not match expected dimension ({self.embedding_dim})')
        vocab_size = None
        max_seq_length = None
        try:
            if hasattr(self._model, 'tokenizer'):
                tokenizer = self._model.tokenizer
                if hasattr(tokenizer, 'vocab_size'):
                    vocab_size = tokenizer.vocab_size
                elif hasattr(tokenizer, 'get_vocab'):
                    vocab_size = len(tokenizer.get_vocab())
                elif hasattr(tokenizer, '__len__'):
                    vocab_size = len(tokenizer)
            if hasattr(self._model, 'max_seq_length'):
                max_seq_length = self._model.max_seq_length
            elif hasattr(self._model, 'tokenizer') and hasattr(self._model.tokenizer, 'model_max_length'):
                max_seq_length = self._model.tokenizer.model_max_length
        except Exception:
            pass
        self._statistics = {'model_name': self.model_name, 'embedding_dimension': self.embedding_dim, 'vocabulary_size': vocab_size, 'max_sequence_length': max_seq_length}
        return self

    def transform(self, data: Union[List[str], FeatureSet], **kwargs) -> FeatureSet:
        """
        Extract semantic embeddings from text data.
        
        Converts input text documents into dense vector representations using
        the loaded pre-trained model. Handles both single strings and batches
        of texts efficiently.
        
        Parameters
        ----------
        data : Union[List[str], FeatureSet]
            Text data to convert to embeddings
        **kwargs : dict
            Additional parameters for embedding extraction
            
        Returns
        -------
        FeatureSet
            Feature set containing extracted embeddings with metadata
            
        Raises
        ------
        RuntimeError
            If model is not properly initialized or inference fails
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Please call 'fit()' first.")
        if isinstance(data, FeatureSet):
            texts = data.get_feature_names()
        elif isinstance(data, list):
            texts = data
        else:
            raise TypeError('Data must be either a list of strings or a FeatureSet')
        if not texts:
            raise ValueError('Input data is empty')
        if not all((isinstance(text, str) for text in texts)):
            raise TypeError('All elements in data must be strings')
        try:
            embeddings = self._model.encode(texts)
        except Exception as e:
            raise RuntimeError(f'Failed to generate embeddings: {str(e)}')
        feature_names = [f'embedding_{i}' for i in range(self.embedding_dim)]
        metadata = {'model_name': self.model_name, 'embedding_dimension': self.embedding_dim, 'source': 'semantic_embeddings', 'statistics': self._statistics}
        return FeatureSet(data=embeddings, feature_names=feature_names, metadata=metadata)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[str]:
        """
        Attempt to reconstruct text from embeddings (if supported).
        
        Some models support approximate text reconstruction from embeddings.
        This method provides that functionality when available.
        
        Parameters
        ----------
        data : FeatureSet
            Embeddings to convert back to text
        **kwargs : dict
            Additional parameters for reconstruction
            
        Returns
        -------
        List[str]
            Reconstructed text documents (may be approximations)
            
        Raises
        ------
        NotImplementedError
            If inverse transformation is not supported by the model
        """
        raise NotImplementedError('Inverse transformation is not supported for semantic embedding models. These models are designed for encoding text to embeddings, not decoding embeddings back to text.')

    def get_embedding_statistics(self) -> dict:
        """
        Get statistics about the embedding model and vocabulary.
        
        Returns information about model parameters, vocabulary coverage,
        and embedding quality metrics when available.
        
        Returns
        -------
        dict
            Dictionary containing embedding model statistics
        """
        return self._statistics.copy()