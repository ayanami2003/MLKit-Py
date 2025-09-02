from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
import numpy as np
from typing import List, Optional, Union, Dict, Any
import warnings

class DynamicTopicModeler(BaseTransformer):

    def __init__(self, n_topics: int=10, n_time_slices: int=10, alpha: float=0.1, beta: float=0.01, gamma: float=0.1, max_iter: int=100, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the DynamicTopicModeler.
        
        Args:
            n_topics: Number of topics to extract from the corpus.
            n_time_slices: Number of time slices to divide the data into.
            alpha: Document-topic density hyperparameter.
            beta: Topic-word density hyperparameter.
            gamma: Temporal dynamics hyperparameter.
            max_iter: Maximum number of iterations for the algorithm.
            random_state: Random seed for reproducibility.
            name: Optional name for the transformer instance.
        """
        super().__init__(name=name)
        self.n_topics = n_topics
        self.n_time_slices = n_time_slices
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        self._vocab = None
        self._vocab_size = None
        self._topic_word_distributions = None
        self._document_topic_distributions = None
        self._time_slice_boundaries = None
        self._fitted = False

    def _preprocess_data(self, data: Union[List[List[str]], DataBatch]) -> List[List[str]]:
        """
        Preprocess input data to extract tokenized documents.
        
        Args:
            data: Input text data as a list of documents or DataBatch.
            
        Returns:
            List[List[str]]: List of tokenized documents.
        """
        if isinstance(data, DataBatch):
            docs = data.data
        else:
            docs = data
        if not isinstance(docs, list) or not all((isinstance(doc, list) for doc in docs)):
            raise ValueError('Input data must be a list of tokenized documents (list of lists of strings)')
        return docs

    def _create_vocabulary(self, documents: List[List[str]]) -> List[str]:
        """
        Create vocabulary from documents.
        
        Args:
            documents: List of tokenized documents.
            
        Returns:
            List[str]: Sorted list of unique words in the vocabulary.
        """
        vocab_set = set()
        for doc in documents:
            vocab_set.update(doc)
        return sorted(list(vocab_set))

    def _documents_to_indices(self, documents: List[List[str]]) -> List[List[int]]:
        """
        Convert documents from words to vocabulary indices.
        
        Args:
            documents: List of tokenized documents.
            
        Returns:
            List[List[int]]: Documents represented as lists of vocabulary indices.
        """
        if self._vocab is None or self._vocab_size is None:
            raise ValueError('Vocabulary not initialized. Call fit() first.')
        word_to_idx = {word: idx for (idx, word) in enumerate(self._vocab)}
        indexed_docs = []
        for doc in documents:
            indexed_doc = [word_to_idx.get(word, -1) for word in doc]
            indexed_doc = [idx for idx in indexed_doc if idx != -1]
            indexed_docs.append(indexed_doc)
        return indexed_docs

    def _initialize_parameters(self, n_docs_per_slice: List[int]):
        """
        Initialize model parameters.
        
        Args:
            n_docs_per_slice: Number of documents in each time slice.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self._vocab_size is None:
            raise ValueError('Vocabulary size not initialized. Call fit() first.')
        self._topic_word_distributions = np.random.gamma(1.0, 1.0, (self.n_time_slices, self.n_topics, self._vocab_size))
        self._topic_word_distributions = np.apply_along_axis(lambda x: x / np.sum(x), 2, self._topic_word_distributions)
        self._document_topic_distributions = []
        for (t, n_docs) in enumerate(n_docs_per_slice):
            if n_docs > 0:
                doc_topic_dist = np.random.gamma(1.0, 1.0, (n_docs, self.n_topics))
                doc_topic_dist = np.apply_along_axis(lambda x: x / np.sum(x), 1, doc_topic_dist)
            else:
                doc_topic_dist = np.empty((0, self.n_topics))
            self._document_topic_distributions.append(doc_topic_dist)

    def fit(self, data: Union[List[List[str]], DataBatch], **kwargs) -> 'DynamicTopicModeler':
        """
        Fit the dynamic topic model to the input text data.
        
        Args:
            data: Input text data as a list of documents (each document is a list of words)
                  or a DataBatch containing text data.
            **kwargs: Additional parameters for fitting.
            
        Returns:
            Self instance for method chaining.
        """
        documents = self._preprocess_data(data)
        if len(documents) == 0:
            raise ValueError('Cannot fit model with empty data')
        self._vocab = self._create_vocabulary(documents)
        self._vocab_size = len(self._vocab)
        total_docs = len(documents)
        slice_size = total_docs // self.n_time_slices
        remainder = total_docs % self.n_time_slices
        self._time_slice_boundaries = []
        start_idx = 0
        n_docs_per_slice = []
        for i in range(self.n_time_slices):
            end_idx = start_idx + slice_size + (1 if i < remainder else 0)
            self._time_slice_boundaries.append((start_idx, end_idx))
            n_docs_per_slice.append(end_idx - start_idx)
            start_idx = end_idx
        indexed_docs = self._documents_to_indices(documents)
        self._initialize_parameters(n_docs_per_slice)
        for iteration in range(self.max_iter):
            for t in range(self.n_time_slices):
                (start_idx, end_idx) = self._time_slice_boundaries[t]
                slice_docs = indexed_docs[start_idx:end_idx]
                for (d, doc) in enumerate(slice_docs):
                    doc_topic_dist = self._document_topic_distributions[t][d]
                    topic_probs = np.zeros(self.n_topics)
                    for topic in range(self.n_topics):
                        prob = doc_topic_dist[topic]
                        for word_idx in doc:
                            prob *= self._topic_word_distributions[t][topic][word_idx]
                        topic_probs[topic] = prob
                    if np.sum(topic_probs) > 0:
                        topic_probs /= np.sum(topic_probs)
                        self._document_topic_distributions[t][d] = topic_probs
            for t in range(self.n_time_slices):
                topic_word_counts = np.zeros((self.n_topics, self._vocab_size))
                (start_idx, end_idx) = self._time_slice_boundaries[t]
                slice_docs = indexed_docs[start_idx:end_idx]
                for (d, doc) in enumerate(slice_docs):
                    doc_topic_dist = self._document_topic_distributions[t][d]
                    for topic in range(self.n_topics):
                        weight = doc_topic_dist[topic]
                        for word_idx in doc:
                            topic_word_counts[topic][word_idx] += weight
                topic_word_counts += self.beta
                for topic in range(self.n_topics):
                    total = np.sum(topic_word_counts[topic])
                    if total > 0:
                        self._topic_word_distributions[t][topic] = topic_word_counts[topic] / total
        self._fitted = True
        return self

    def transform(self, data: Union[List[List[str]], DataBatch], **kwargs) -> np.ndarray:
        """
        Transform the input data into topic distributions over time.
        
        Args:
            data: Input text data to transform.
            **kwargs: Additional parameters for transformation.
            
        Returns:
            np.ndarray: A 3D array of shape (n_time_slices, n_documents, n_topics)
                       representing topic distributions for documents in each time slice.
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before transform can be called')
        documents = self._preprocess_data(data)
        indexed_docs = self._documents_to_indices(documents)
        result = np.concatenate(self._document_topic_distributions, axis=0)
        n_docs_per_slice = result.shape[0] // self.n_time_slices
        result = result.reshape((self.n_time_slices, n_docs_per_slice, self.n_topics))
        return result

    def fit_transform(self, data: Union[List[List[str]], DataBatch], **kwargs) -> np.ndarray:
        """
        Fit the model and transform the data in one step.
        
        Args:
            data: Input text data as a list of documents.
            **kwargs: Additional parameters.
            
        Returns:
            np.ndarray: Topic distributions for the input data.
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    def inverse_transform(self, data: np.ndarray, **kwargs) -> List[List[str]]:
        """
        Convert topic distributions back to approximate word representations.
        
        Args:
            data: Topic distribution data to convert back.
            **kwargs: Additional parameters.
            
        Returns:
            List[List[str]]: Approximate word representations for each topic.
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before inverse_transform can be called')
        if data.ndim != 3 or data.shape[2] != self.n_topics:
            raise ValueError(f'Expected data shape (n_time_slices, n_documents, {self.n_topics}), got {data.shape}')
        result = []
        n_time_slices = min(data.shape[0], self.n_time_slices)
        for t in range(n_time_slices):
            time_slice_docs = []
            n_docs = data.shape[1]
            for d in range(n_docs):
                topic_probs = data[t, d, :]
                dominant_topic = np.argmax(topic_probs)
                word_probs = self._topic_word_distributions[t][dominant_topic]
                top_word_indices = np.argsort(word_probs)[::-1][:10]
                top_words = [self._vocab[idx] for idx in top_word_indices]
                time_slice_docs.append(top_words)
            result.append(time_slice_docs)
        flattened_result = []
        for time_slice in result:
            flattened_result.extend(time_slice)
        return flattened_result

    def get_topic_terms(self, time_slice: int, topic_idx: int, top_k: int=10) -> List[tuple]:
        """
        Get the top terms for a specific topic in a specific time slice.
        
        Args:
            time_slice: Index of the time slice.
            topic_idx: Index of the topic.
            top_k: Number of top terms to return.
            
        Returns:
            List[tuple]: List of (word, probability) tuples for the top terms.
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before get_topic_terms can be called')
        if time_slice < 0 or time_slice >= self.n_time_slices:
            raise IndexError(f'time_slice must be between 0 and {self.n_time_slices - 1}')
        if topic_idx < 0 or topic_idx >= self.n_topics:
            raise IndexError(f'topic_idx must be between 0 and {self.n_topics - 1}')
        if top_k <= 0:
            raise ValueError('top_k must be positive')
        word_probs = self._topic_word_distributions[time_slice][topic_idx]
        top_indices = np.argsort(word_probs)[::-1][:top_k]
        return [(self._vocab[idx], word_probs[idx]) for idx in top_indices]

    def get_topic_evolution(self, topic_idx: int) -> Dict[str, Any]:
        """
        Get the evolution of a specific topic across all time slices.
        
        Args:
            topic_idx: Index of the topic to trace.
            
        Returns:
            Dict[str, Any]: Dictionary containing topic evolution information.
        """
        if not self._fitted:
            raise RuntimeError('Model must be fitted before get_topic_evolution can be called')
        if topic_idx < 0 or topic_idx >= self.n_topics:
            raise IndexError(f'topic_idx must be between 0 and {self.n_topics - 1}')
        evolution_data = {'topic_idx': topic_idx, 'time_slices': [], 'term_trajectories': {}}
        for word in self._vocab:
            evolution_data['term_trajectories'][word] = []
        for t in range(self.n_time_slices):
            evolution_data['time_slices'].append(t)
            word_probs = self._topic_word_distributions[t][topic_idx]
            for (i, word) in enumerate(self._vocab):
                evolution_data['term_trajectories'][word].append(word_probs[i])
        return evolution_data