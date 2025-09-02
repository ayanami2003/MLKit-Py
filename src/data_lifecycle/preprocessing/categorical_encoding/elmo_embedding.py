from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class ELMoEmbeddingEncoder(BaseTransformer):

    def __init__(self, embedding_dim: int=1024, aggregation_method: str='mean', handle_unknown: str='ignore', max_length: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the ELMo embedding encoder.
        
        Parameters
        ----------
        embedding_dim : int, default=1024
            Dimension of ELMo embeddings (typically 1024 for ELMo original)
        aggregation_method : str, default='mean'
            Method to aggregate token embeddings ('mean', 'max', 'sum')
        handle_unknown : str, default='ignore'
            How to handle unknown categories during transform
        max_length : Optional[int], default=None
            Maximum sequence length for processing (None for no limit)
        name : Optional[str], default=None
            Name for the transformer instance
        """
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method
        self.handle_unknown = handle_unknown
        self.max_length = max_length
        self._fitted = False
        self._elmo_model = None
        self._vocab = set()

    def _load_elmo_model(self):
        """Lazy loading of ELMo model."""
        if self._elmo_model is None:
            try:
                import tensorflow_hub as hub
                self._elmo_model = hub.load('https://tfhub.dev/google/elmo/3')
            except ImportError:
                raise ImportError("ELMo encoding requires tensorflow_hub. Please install it with 'pip install tensorflow-hub'")
            except Exception as e:
                raise RuntimeError(f'Failed to load ELMo model: {e}')

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate ELMo embeddings for a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings to embed
        
        Returns
        -------
        np.ndarray
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        self._load_elmo_model()
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                import tensorflow as tf
                text_tensor = tf.constant(batch_texts)
                elmo_results = self._elmo_model.signatures['default'](text_tensor)
                word_emb = elmo_results['word_emb']
                if hasattr(word_emb, 'shape') and word_emb.shape is not None:
                    if hasattr(word_emb.shape, 'as_list'):
                        shape = word_emb.shape.as_list()
                    else:
                        shape = list(word_emb.shape)
                else:
                    shape = [len(batch_texts), 10, self.embedding_dim]
                while len(shape) < 2:
                    word_emb = tf.expand_dims(word_emb, axis=0)
                    shape = [1] + shape
                if len(shape) >= 3:
                    if self.aggregation_method == 'mean':
                        batch_embeddings = tf.reduce_mean(word_emb, axis=1)
                    elif self.aggregation_method == 'max':
                        batch_embeddings = tf.reduce_max(word_emb, axis=1)
                    elif self.aggregation_method == 'sum':
                        batch_embeddings = tf.reduce_sum(word_emb, axis=1)
                    else:
                        raise ValueError(f'Unsupported aggregation method: {self.aggregation_method}')
                elif len(shape) == 2:
                    expanded_word_emb = tf.expand_dims(word_emb, axis=1)
                    if self.aggregation_method == 'mean':
                        batch_embeddings = tf.reduce_mean(expanded_word_emb, axis=1)
                    elif self.aggregation_method == 'max':
                        batch_embeddings = tf.reduce_max(expanded_word_emb, axis=1)
                    elif self.aggregation_method == 'sum':
                        batch_embeddings = tf.reduce_sum(expanded_word_emb, axis=1)
                    else:
                        raise ValueError(f'Unsupported aggregation method: {self.aggregation_method}')
                else:
                    raise ValueError(f'Unexpected word embedding shape after processing: {shape}')
                embeddings.append(batch_embeddings.numpy())
            except Exception as e:
                raise RuntimeError(f'Error generating ELMo embeddings: {e}')
        return np.vstack(embeddings)

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'ELMoEmbeddingEncoder':
        """
        Fit the ELMo encoder on the input text data.
        
        This method prepares the vocabulary and initializes the ELMo model.
        Since ELMo is pre-trained, fitting mainly involves analyzing the input
        data structure and preparing for transformation.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input text data as FeatureSet or array of strings
        **kwargs : dict
            Additional fitting parameters (reserved for future use)
            
        Returns
        -------
        ELMoEmbeddingEncoder
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If data format is unsupported
        """
        if isinstance(data, FeatureSet):
            if data.features.dtype.kind not in ['U', 'S']:
                raise ValueError('FeatureSet features must contain string data for ELMo encoding')
            texts = data.features.flatten().tolist()
        elif isinstance(data, np.ndarray):
            if data.dtype.kind not in ['U', 'S']:
                raise ValueError('Array must contain string data for ELMo encoding')
            texts = data.flatten().tolist()
        else:
            raise ValueError('Input data must be either FeatureSet or numpy array')
        self._vocab = set(texts)
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform text categories into ELMo embeddings.
        
        Converts input text data into dense vector representations using
        ELMo embeddings. Each text category is mapped to a fixed-size vector.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input text data to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            FeatureSet containing ELMo embeddings as features
            
        Raises
        ------
        ValueError
            If encoder hasn't been fitted or data format is incorrect
        NotImplementedError
            If handling unknown categories is set to 'raise' and unknowns are found
        """
        if not self._fitted:
            raise ValueError('ELMoEmbeddingEncoder must be fitted before transform')
        if isinstance(data, FeatureSet):
            if data.features.dtype.kind not in ['U', 'S']:
                raise ValueError('FeatureSet features must contain string data for ELMo encoding')
            texts = data.features.flatten().tolist()
            original_shape = data.features.shape
            feature_names = data.feature_names
            sample_ids = data.sample_ids
        elif isinstance(data, np.ndarray):
            if data.dtype.kind not in ['U', 'S']:
                raise ValueError('Array must contain string data for ELMo encoding')
            texts = data.flatten().tolist()
            original_shape = data.shape
            feature_names = None
            sample_ids = None
        else:
            raise ValueError('Input data must be either FeatureSet or numpy array')
        if self.handle_unknown == 'raise':
            unknowns = set(texts) - self._vocab
            if unknowns:
                raise NotImplementedError(f"Unknown categories encountered: {unknowns}. handle_unknown='raise' is set.")
        embeddings = self._get_embeddings(texts)
        n_samples = original_shape[0] if len(original_shape) > 0 else 1
        if len(original_shape) > 1:
            n_features = original_shape[1]
            embeddings = embeddings.reshape(n_samples, n_features * self.embedding_dim)
            new_feature_names = []
            original_feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(n_features)]
            for fname in original_feature_names:
                for j in range(self.embedding_dim):
                    new_feature_names.append(f'{fname}_elmo_{j}')
        else:
            new_feature_names = [f'elmo_{j}' for j in range(self.embedding_dim)]
        result = FeatureSet(features=embeddings, feature_names=new_feature_names, feature_types=['numeric'] * embeddings.shape[1], sample_ids=sample_ids, metadata={'encoder_type': 'ELMoEmbeddingEncoder', 'embedding_dim': self.embedding_dim, 'aggregation_method': self.aggregation_method})
        return result

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for ELMo embeddings.
        
        ELMo embeddings are dense vector representations that cannot be
        meaningfully converted back to original text categories.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Embedding data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Never returns - always raises NotImplementedError
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported
        """
        raise NotImplementedError('Inverse transformation is not supported for ELMo embeddings. ELMo embeddings are dense vector representations that cannot be meaningfully converted back to original text categories.')