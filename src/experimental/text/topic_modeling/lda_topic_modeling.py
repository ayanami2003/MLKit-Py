from typing import List, Optional, Union
import numpy as np
from scipy.special import digamma
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch

class LDATopicModeler(BaseTransformer):

    def __init__(self, n_topics: int=10, max_iter: int=100, alpha: Union[float, np.ndarray]=0.1, beta: Union[float, np.ndarray]=0.01, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the LDA topic modeler.
        
        Parameters
        ----------
        n_topics : int, default=10
            Number of topics to extract from the corpus
        max_iter : int, default=100
            Maximum number of iterations for the variational Bayes algorithm
        alpha : float or np.ndarray, default=0.1
            Document-topic Dirichlet prior parameter. If float, assumes symmetric prior.
        beta : float or np.ndarray, default=0.01
            Topic-word Dirichlet prior parameter. If float, assumes symmetric prior.
        random_state : int or None, default=None
            Random seed for reproducibility
        name : str or None, default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

    def fit(self, data: Union[List[List[str]], DataBatch], **kwargs) -> 'LDATopicModeler':
        """
        Fit the LDA topic model to the input text data.
        
        This method learns the topic-word distributions and document-topic distributions
        from the provided corpus using variational inference.
        
        Parameters
        ----------
        data : Union[List[List[str]], DataBatch]
            Training data consisting of tokenized documents. Each document should be
            represented as a list of tokens (words). If DataBatch is provided, it should
            contain tokenized text data.
        **kwargs : dict
            Additional parameters for fitting (not currently used)
            
        Returns
        -------
        LDATopicModeler
            Self instance for method chaining
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if isinstance(data, DataBatch):
            docs = data.data
        else:
            docs = data
        vocab = set()
        for doc in docs:
            vocab.update(doc)
        self.vocab_ = sorted(list(vocab))
        self.vocab_size_ = len(self.vocab_)
        self.word_to_idx_ = {word: idx for (idx, word) in enumerate(self.vocab_)}
        self.doc_term_matrices_ = []
        for doc in docs:
            term_freq = np.zeros(self.vocab_size_)
            for word in doc:
                if word in self.word_to_idx_:
                    term_freq[self.word_to_idx_[word]] += 1
            self.doc_term_matrices_.append(term_freq)
        self.doc_term_matrices_ = np.array(self.doc_term_matrices_)
        self.n_docs_ = len(self.doc_term_matrices_)
        self._initialize_parameters()
        for iteration in range(self.max_iter):
            self._update_variational_parameters()
        self._update_model_parameters()
        return self

    def transform(self, data: Union[List[List[str]], DataBatch], **kwargs) -> np.ndarray:
        """
        Transform documents into topic distributions.
        
        This method infers the topic distributions for new documents based on the
        learned topic model.
        
        Parameters
        ----------
        data : Union[List[List[str]], DataBatch]
            Documents to transform, represented as lists of tokens. If DataBatch is
            provided, it should contain tokenized text data.
        **kwargs : dict
            Additional parameters for transformation (not currently used)
            
        Returns
        -------
        np.ndarray
            Document-topic matrix of shape (n_documents, n_topics) where each row
            represents the topic distribution for a document
        """
        if isinstance(data, DataBatch):
            docs = data.data
        else:
            docs = data
        doc_term_matrices = []
        for doc in docs:
            term_freq = np.zeros(self.vocab_size_)
            for word in doc:
                if word in self.word_to_idx_:
                    term_freq[self.word_to_idx_[word]] += 1
            doc_term_matrices.append(term_freq)
        doc_term_matrices = np.array(doc_term_matrices)
        n_docs = len(doc_term_matrices)
        doc_topic_vars = np.random.gamma(100.0, 1.0 / 100.0, (n_docs, self.n_topics))
        doc_topic_vars = np.where(doc_topic_vars > 0, doc_topic_vars, 1e-10)
        for _ in range(50):
            doc_topic_vars = self._update_doc_topic_vars(doc_term_matrices, doc_topic_vars)
        doc_topic_dist = doc_topic_vars / np.sum(doc_topic_vars, axis=1, keepdims=True)
        return doc_topic_dist

    def fit_transform(self, data: Union[List[List[str]], DataBatch], **kwargs) -> np.ndarray:
        """
        Fit the model to the data and transform the data in one step.
        
        This is a convenience method that combines fitting and transformation.
        
        Parameters
        ----------
        data : Union[List[List[str]], DataBatch]
            Training data consisting of tokenized documents
        **kwargs : dict
            Additional parameters for fitting and transformation
            
        Returns
        -------
        np.ndarray
            Document-topic matrix of shape (n_documents, n_topics)
        """
        return super().fit_transform(data, **kwargs)

    def inverse_transform(self, data: np.ndarray, **kwargs) -> List[List[str]]:
        """
        Approximate reconstruction of documents from topic distributions.
        
        This method attempts to reconstruct the most probable words for documents
        based on their topic distributions.
        
        Parameters
        ----------
        data : np.ndarray
            Document-topic matrix of shape (n_documents, n_topics)
        **kwargs : dict
            Additional parameters for inverse transformation (not currently used)
            
        Returns
        -------
        List[List[str]]
            Reconstructed documents as lists of words
        """
        reconstructed_docs = []
        for doc_topic_dist in data:
            reconstructed_doc = []
            top_topics = np.argsort(doc_topic_dist)[-5:]
            for topic_idx in top_topics:
                topic_word_dist = self.topic_word_dist_[topic_idx]
                top_words_indices = np.argsort(topic_word_dist)[-10:]
                for word_idx in top_words_indices:
                    reconstructed_doc.append(self.vocab_[word_idx])
            reconstructed_docs.append(reconstructed_doc)
        return reconstructed_docs

    def get_topic_words(self, topic_idx: int, top_k: int=10) -> List[str]:
        """
        Get the most probable words for a specific topic.
        
        Parameters
        ----------
        topic_idx : int
            Index of the topic to retrieve words for
        top_k : int, default=10
            Number of top words to return
            
        Returns
        -------
        List[str]
            List of the top-k most probable words for the specified topic
        """
        if not hasattr(self, 'topic_word_dist_'):
            raise ValueError('Model has not been fitted yet.')
        if topic_idx < 0 or topic_idx >= self.n_topics:
            raise ValueError(f'topic_idx must be between 0 and {self.n_topics - 1}')
        topic_word_dist = self.topic_word_dist_[topic_idx]
        top_word_indices = np.argsort(topic_word_dist)[-top_k:][::-1]
        return [self.vocab_[idx] for idx in top_word_indices]

    def get_document_topics(self, doc_idx: int, top_k: int=5) -> List[tuple]:
        """
        Get the most probable topics for a specific document.
        
        Parameters
        ----------
        doc_idx : int
            Index of the document to analyze
        top_k : int, default=5
            Number of top topics to return
            
        Returns
        -------
        List[tuple]
            List of (topic_idx, probability) tuples for the top-k topics
        """
        if not hasattr(self, 'doc_topic_dist_'):
            raise ValueError('Model has not been fitted yet.')
        if doc_idx < 0 or doc_idx >= self.n_docs_:
            raise ValueError(f'doc_idx must be between 0 and {self.n_docs_ - 1}')
        doc_topic_dist = self.doc_topic_dist_[doc_idx]
        top_topic_indices = np.argsort(doc_topic_dist)[-top_k:][::-1]
        return [(idx, doc_topic_dist[idx]) for idx in top_topic_indices]

    def _initialize_parameters(self):
        """Initialize model parameters and variational parameters."""
        self.topic_word_vars_ = np.random.gamma(100.0, 1.0 / 100.0, (self.n_topics, self.vocab_size_))
        self.topic_word_vars_ = np.where(self.topic_word_vars_ > 0, self.topic_word_vars_, 1e-10)
        self.doc_topic_vars_ = np.random.gamma(100.0, 1.0 / 100.0, (self.n_docs_, self.n_topics))
        self.doc_topic_vars_ = np.where(self.doc_topic_vars_ > 0, self.doc_topic_vars_, 1e-10)
        if np.isscalar(self.alpha):
            self.alpha_ = np.full(self.n_topics, self.alpha)
        else:
            self.alpha_ = np.array(self.alpha)
        if np.isscalar(self.beta):
            self.beta_ = np.full(self.vocab_size_, self.beta)
        else:
            self.beta_ = np.array(self.beta)

    def _update_variational_parameters(self):
        """Update variational parameters using coordinate ascent."""
        self.doc_topic_vars_ = self._update_doc_topic_vars(self.doc_term_matrices_, self.doc_topic_vars_)
        self.topic_word_vars_ = self._update_topic_word_vars(self.doc_term_matrices_, self.doc_topic_vars_)

    def _update_doc_topic_vars(self, doc_term_matrices, doc_topic_vars):
        """Update document-topic variational parameters."""
        exp_log_topic_word = np.zeros((self.n_topics, self.vocab_size_))
        for k in range(self.n_topics):
            exp_log_topic_word[k] = digamma(self.topic_word_vars_[k]) - digamma(np.sum(self.topic_word_vars_[k]))
        for d in range(len(doc_term_matrices)):
            exp_log_doc_topic = digamma(doc_topic_vars[d]) - digamma(np.sum(doc_topic_vars[d]))
            for k in range(self.n_topics):
                doc_topic_vars[d, k] = self.alpha_[k] + np.sum(doc_term_matrices[d] * exp_log_topic_word[k])
            doc_topic_vars[d] = np.where(doc_topic_vars[d] > 0, doc_topic_vars[d], 1e-10)
        return doc_topic_vars

    def _update_topic_word_vars(self, doc_term_matrices, doc_topic_vars):
        """Update topic-word variational parameters."""
        topic_word_vars = np.zeros((self.n_topics, self.vocab_size_))
        for d in range(len(doc_term_matrices)):
            resp = np.outer(doc_topic_vars[d], doc_term_matrices[d])
            resp = resp / np.sum(resp, axis=0, keepdims=True)
            resp = np.nan_to_num(resp)
            for k in range(self.n_topics):
                topic_word_vars[k] += resp[k]
        topic_word_vars += self.beta_
        return topic_word_vars

    def _update_model_parameters(self):
        """Update model parameters based on variational parameters."""
        topic_word_sum = np.sum(self.topic_word_vars_, axis=1, keepdims=True)
        topic_word_sum = np.where(topic_word_sum == 0, 1e-10, topic_word_sum)
        self.topic_word_dist_ = self.topic_word_vars_ / topic_word_sum
        doc_topic_sum = np.sum(self.doc_topic_vars_, axis=1, keepdims=True)
        doc_topic_sum = np.where(doc_topic_sum == 0, 1e-10, doc_topic_sum)
        self.doc_topic_dist_ = self.doc_topic_vars_ / doc_topic_sum