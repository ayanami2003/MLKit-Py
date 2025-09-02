from typing import List, Optional, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
import re
import warnings
from collections import defaultdict, Counter
import random

class FastTextEmbedder(BaseTransformer):

    def __init__(self, vector_size: int=100, window: int=5, min_count: int=1, sg: int=1, epochs: int=5, name: Optional[str]=None):
        """
        Initialize the FastTextEmbedder.
        
        Parameters
        ----------
        vector_size : int, default=100
            Dimension of the word vectors
        window : int, default=5
            Context window size
        min_count : int, default=1
            Minimum count of words to consider
        sg : int, default=1
            Training algorithm: 1 for skip-gram, 0 for CBOW
        epochs : int, default=5
            Number of training epochs
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.model = None
        self.word_vectors = {}
        self.subword_vectors = {}
        self.vocab = {}
        self.word_freq = Counter()
        self.min_n = 3
        self.max_n = 6

    def _get_subwords(self, word: str) -> List[str]:
        """
        Get character n-grams for a word (including < and > markers).
        
        Parameters
        ----------
        word : str
            Input word
            
        Returns
        -------
        List[str]
            List of subword units
        """
        if not word:
            return ['<>']
        word = '<' + word + '>'
        subwords = []
        subwords.append(word)
        for i in range(len(word)):
            for j in range(i + self.min_n, min(len(word) + 1, i + self.max_n + 1)):
                if j - i >= self.min_n and j - i <= self.max_n:
                    subwords.append(word[i:j])
        seen = set()
        unique_subwords = []
        for subword in subwords:
            if subword not in seen:
                seen.add(subword)
                unique_subwords.append(subword)
        return unique_subwords

    def _initialize_vectors(self, vocab_keys: List[str]):
        """
        Initialize word and subword vectors.
        
        Parameters
        ----------
        vocab_keys : List[str]
            Vocabulary keys to initialize vectors for
        """
        for word in vocab_keys:
            if word not in self.word_vectors:
                self.word_vectors[word] = np.random.uniform(-0.1, 0.1, self.vector_size)
            subwords = self._get_subwords(word)
            for subword in subwords:
                if subword not in self.subword_vectors:
                    self.subword_vectors[subword] = np.random.uniform(-0.1, 0.1, self.vector_size)

    def _get_word_vector(self, word: str) -> np.ndarray:
        """
        Get vector representation for a word (combining word and subword vectors).
        
        Parameters
        ----------
        word : str
            Input word
            
        Returns
        -------
        np.ndarray
            Combined word vector
        """
        vector = self.word_vectors.get(word, np.zeros(self.vector_size))
        subwords = self._get_subwords(word)
        subword_vec = np.zeros(self.vector_size)
        count = 0
        for subword in subwords:
            if subword in self.subword_vectors:
                subword_vec += self.subword_vectors[subword]
                count += 1
        if count > 0:
            vector = (vector + subword_vec) / (count + 1)
        return vector

    def _train_skipgram(self, sentences: List[List[str]]):
        """
        Train using skip-gram algorithm.
        
        Parameters
        ----------
        sentences : List[List[str]]
            Tokenized sentences for training
        """
        vocab_list = list(self.vocab.keys())
        word_to_idx = {word: idx for (idx, word) in enumerate(vocab_list)}
        for epoch in range(self.epochs):
            for sentence in sentences:
                words = [word for word in sentence if word in self.vocab]
                for (pos, word) in enumerate(words):
                    if word not in word_to_idx:
                        continue
                    start = max(0, pos - self.window)
                    end = min(len(words), pos + self.window + 1)
                    for context_pos in range(start, end):
                        if context_pos == pos:
                            continue
                        context_word = words[context_pos]
                        if context_word not in word_to_idx:
                            continue
                        word_vec = self._get_word_vector(word)
                        context_vec = self._get_word_vector(context_word)
                        lr = 0.025
                        grad = word_vec - context_vec
                        word_vec -= lr * grad
                        if word in self.word_vectors:
                            self.word_vectors[word] = word_vec

    def _train_cbow(self, sentences: List[List[str]]):
        """
        Train using CBOW algorithm.
        
        Parameters
        ----------
        sentences : List[List[str]]
            Tokenized sentences for training
        """
        vocab_list = list(self.vocab.keys())
        word_to_idx = {word: idx for (idx, word) in enumerate(vocab_list)}
        for epoch in range(self.epochs):
            for sentence in sentences:
                words = [word for word in sentence if word in self.vocab]
                for (pos, word) in enumerate(words):
                    if word not in word_to_idx:
                        continue
                    start = max(0, pos - self.window)
                    end = min(len(words), pos + self.window + 1)
                    context_words = []
                    for context_pos in range(start, end):
                        if context_pos == pos:
                            continue
                        context_word = words[context_pos]
                        if context_word in word_to_idx:
                            context_words.append(context_word)
                    if not context_words:
                        continue
                    context_vec = np.zeros(self.vector_size)
                    for context_word in context_words:
                        context_vec += self._get_word_vector(context_word)
                    context_vec /= len(context_words)
                    word_vec = self._get_word_vector(word)
                    lr = 0.025
                    grad = context_vec - word_vec
                    word_vec -= lr * grad
                    if word in self.word_vectors:
                        self.word_vectors[word] = word_vec

    def fit(self, data: Union[List[str], DataBatch], **kwargs) -> 'FastTextEmbedder':
        """
        Fit the FastText model on the input text data.
        
        This method trains the FastText model on the provided corpus of text documents.
        After fitting, the model can be used to transform new text data into embeddings.
        
        Parameters
        ----------
        data : Union[List[str], DataBatch]
            Training text data as a list of strings or DataBatch containing text
        **kwargs : dict
            Additional parameters for fitting (e.g., workers for parallel processing)
            
        Returns
        -------
        FastTextEmbedder
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data format is not supported
        """
        if isinstance(data, DataBatch):
            texts = data.data
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a list of strings or a DataBatch')
        if not all((isinstance(text, str) for text in texts)):
            raise ValueError('All elements in the input data must be strings')
        tokenized_texts = []
        for text in texts:
            words = text.split()
            tokenized_texts.append(words)
            self.word_freq.update(words)
        vocab = {word for (word, count) in self.word_freq.items() if count >= self.min_count}
        self._initialize_vectors(list(vocab))
        self.vocab = {word: i for (i, word) in enumerate(vocab)}
        if self.sg == 1:
            self._train_skipgram(tokenized_texts)
        else:
            self._train_cbow(tokenized_texts)
        self.model = 'fitted'
        return self

    def transform(self, data: Union[List[str], DataBatch], **kwargs) -> np.ndarray:
        """
        Transform text data into FastText embeddings.
        
        Converts input text documents into their corresponding embedding representations
        using the trained FastText model. Each document is represented as the average
        of its word embeddings.
        
        Parameters
        ----------
        data : Union[List[str], DataBatch]
            Text data to transform into embeddings
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        np.ndarray
            Array of embeddings with shape (n_documents, vector_size)
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        ValueError
            If the input data format is not supported
        """
        if self.model is None:
            raise RuntimeError("FastTextEmbedder has not been fitted yet. Call 'fit' first.")
        if isinstance(data, DataBatch):
            texts = data.data
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a list of strings or a DataBatch')
        if not all((isinstance(text, str) for text in texts)):
            raise ValueError('All elements in the input data must be strings')
        embeddings = []
        for text in texts:
            words = text.split()
            if not words:
                embeddings.append(np.zeros(self.vector_size))
                continue
            word_vectors = []
            for word in words:
                word_vectors.append(self._get_word_vector(word))
            if word_vectors:
                doc_embedding = np.mean(word_vectors, axis=0)
            else:
                doc_embedding = np.zeros(self.vector_size)
            embeddings.append(doc_embedding)
        return np.array(embeddings)

    def inverse_transform(self, data: np.ndarray, **kwargs) -> List[str]:
        """
        Approximate reconstruction of text from embeddings (not implemented for FastText).
        
        FastText embeddings are not directly invertible to reconstruct the original text.
        This method raises NotImplementedError as the operation is not supported.
        
        Parameters
        ----------
        data : np.ndarray
            Embedding data to inverse transform
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        List[str]
            This method always raises NotImplementedError
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not supported
        """
        raise NotImplementedError('FastText embeddings are not invertible to reconstruct original text')

    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a specific word.
        
        Retrieves the FastText embedding vector for a given word from the trained model.
        Can generate embeddings for out-of-vocabulary words using subword information.
        
        Parameters
        ----------
        word : str
            Word to get the embedding for
            
        Returns
        -------
        np.ndarray
            Embedding vector for the word
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet
        KeyError
            If the word cannot be embedded even with subword information
        """
        if self.model is None:
            raise RuntimeError("FastTextEmbedder has not been fitted yet. Call 'fit' first.")
        vector = self._get_word_vector(word)
        if np.all(vector == 0):
            raise KeyError(f"Word '{word}' cannot be embedded even with subword information")
        return vector