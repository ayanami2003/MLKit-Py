from typing import List, Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from collections import Counter
import re
import math

class NGramFeatureExtractor(BaseTransformer):
    """
    Extract n-gram features from text data for use in machine learning models.
    
    This transformer converts text documents into n-gram feature representations,
    supporting unigrams, bigrams, trigrams, and higher-order n-grams. It handles
    text preprocessing, tokenization, and creates a vocabulary of n-gram features
    with corresponding frequency counts or TF-IDF weights.
    
    The extractor supports various n-gram ranges, handling of stop words, and
    different weighting schemes. It outputs a FeatureSet with the extracted
    n-gram features suitable for downstream modeling tasks.
    
    Attributes
    ----------
    n_range : tuple[int, int]
        The range of n-gram sizes to extract (min_n, max_n)
    max_features : Optional[int]
        Maximum number of features to keep, ordered by frequency
    stop_words : Optional[List[str]]
        List of stop words to exclude from n-grams
    weighting_scheme : str
        Weighting scheme for features ('count', 'tfidf', 'binary')
    lowercase : bool
        Whether to convert text to lowercase before processing
    """

    def __init__(self, n_range: tuple[int, int]=(1, 2), max_features: Optional[int]=None, stop_words: Optional[List[str]]=None, weighting_scheme: str='count', lowercase: bool=True, name: Optional[str]=None):
        """
        Initialize the N-gram feature extractor.
        
        Parameters
        ----------
        n_range : tuple[int, int], default=(1, 2)
            The range of n-gram sizes to extract. For example, (1, 2) extracts
            unigrams and bigrams.
        max_features : Optional[int], default=None
            Maximum number of features to keep, ordered by frequency. If None,
            all features are kept.
        stop_words : Optional[List[str]], default=None
            List of stop words to exclude from n-grams. If None, no stop words
            are removed.
        weighting_scheme : str, default='count'
            Weighting scheme for features. Options are:
            - 'count': Raw count of n-grams
            - 'tfidf': Term Frequency-Inverse Document Frequency
            - 'binary': Binary occurrence (0 or 1)
        lowercase : bool, default=True
            Whether to convert text to lowercase before processing.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.n_range = n_range
        self.max_features = max_features
        self.stop_words = set(stop_words) if stop_words else set()
        self.weighting_scheme = weighting_scheme
        self.lowercase = lowercase
        self.vocabulary_ = None
        self.feature_names_ = None
        self.idf_ = None
        self._fitted = False

    def _validate_inputs(self, data: Union[FeatureSet, List[str], np.ndarray]) -> List[str]:
        """
        Validate and extract text data from various input formats.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str], np.ndarray]
            Input data to validate
            
        Returns
        -------
        List[str]
            List of text documents
            
        Raises
        ------
        ValueError
            If input format is not supported
        """
        if isinstance(data, FeatureSet):
            if data.features.dtype.kind in ['U', 'S'] or isinstance(data.features[0][0], str):
                return [doc for doc in data.features.flatten()]
            else:
                raise ValueError('FeatureSet must contain string data')
        elif isinstance(data, list):
            if all((isinstance(item, str) for item in data)):
                return data
            else:
                raise ValueError('All items in list must be strings')
        elif isinstance(data, np.ndarray):
            if data.dtype.kind in ['U', 'S']:
                return data.tolist()
            else:
                raise ValueError('NumPy array must contain string data')
        else:
            raise ValueError('Unsupported input format. Expected FeatureSet, List[str], or np.ndarray')

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess a text document by lowercasing and tokenizing.
        
        Parameters
        ----------
        text : str
            Input text document
            
        Returns
        -------
        List[str]
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        tokens = re.findall('\\b\\w+\\b', text)
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Extract n-grams from a list of tokens.
        
        Parameters
        ----------
        tokens : List[str]
            List of tokens
        n : int
            Size of n-grams to extract
            
        Returns
        -------
        List[str]
            List of n-grams represented as space-separated strings
        """
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """
        Build vocabulary of n-grams from documents.
        
        Parameters
        ----------
        documents : List[str]
            List of text documents
            
        Returns
        -------
        Dict[str, int]
            Dictionary mapping n-grams to indices
        """
        ngram_counts = Counter()
        for doc in documents:
            tokens = self._preprocess_text(doc)
            for n in range(self.n_range[0], self.n_range[1] + 1):
                ngrams = self._extract_ngrams(tokens, n)
                ngram_counts.update(ngrams)
        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: (-x[1], x[0]))
        if self.max_features:
            sorted_ngrams = sorted_ngrams[:self.max_features]
        vocabulary = {ngram: idx for (idx, (ngram, _)) in enumerate(sorted_ngrams)}
        return vocabulary

    def fit(self, data: Union[FeatureSet, List[str], np.ndarray], **kwargs) -> 'NGramFeatureExtractor':
        """
        Fit the n-gram extractor to the input text data.
        
        This method builds the vocabulary of n-grams from the input data and
        prepares the transformer for feature extraction. It analyzes the frequency
        of n-grams across documents and selects the top features according to
        the specified parameters.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str], np.ndarray]
            Input text data to fit the extractor. Can be:
            - A FeatureSet with text data in its features attribute
            - A list of strings where each string is a document
            - A numpy array of strings
        **kwargs : dict
            Additional parameters for fitting (not used in this implementation)
            
        Returns
        -------
        NGramFeatureExtractor
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data format is not supported
        """
        documents = self._validate_inputs(data)
        self.vocabulary_ = self._build_vocabulary(documents)
        self.feature_names_ = list(self.vocabulary_.keys())
        if self.weighting_scheme == 'tfidf':
            doc_count = len(documents)
            self.idf_ = np.zeros(len(self.vocabulary_))
            df = np.zeros(len(self.vocabulary_))
            for doc in documents:
                tokens = self._preprocess_text(doc)
                doc_ngrams = set()
                for n in range(self.n_range[0], self.n_range[1] + 1):
                    ngrams = self._extract_ngrams(tokens, n)
                    doc_ngrams.update(ngrams)
                for ngram in doc_ngrams:
                    if ngram in self.vocabulary_:
                        df[self.vocabulary_[ngram]] += 1
            self.idf_ = np.log(doc_count / (df + 1)) + 1
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, List[str], np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform text data into n-gram features.
        
        This method applies the fitted n-gram extractor to convert text documents
        into numerical feature representations. The output is a FeatureSet with
        a feature matrix where each row represents a document and each column
        represents an n-gram feature.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str], np.ndarray]
            Input text data to transform. Must be in the same format as used
            for fitting.
        **kwargs : dict
            Additional parameters for transformation (not used in this implementation)
            
        Returns
        -------
        FeatureSet
            FeatureSet containing the extracted n-gram features
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted or input data format is incorrect
        """
        if not self._fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        documents = self._validate_inputs(data)
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        feature_matrix = np.zeros((n_docs, n_features))
        for (doc_idx, doc) in enumerate(documents):
            tokens = self._preprocess_text(doc)
            doc_ngrams = Counter()
            for n in range(self.n_range[0], self.n_range[1] + 1):
                ngrams = self._extract_ngrams(tokens, n)
                doc_ngrams.update(ngrams)
            for (ngram, count) in doc_ngrams.items():
                if ngram in self.vocabulary_:
                    feature_idx = self.vocabulary_[ngram]
                    if self.weighting_scheme == 'count':
                        feature_matrix[doc_idx, feature_idx] = count
                    elif self.weighting_scheme == 'binary':
                        feature_matrix[doc_idx, feature_idx] = 1
                    elif self.weighting_scheme == 'tfidf':
                        tf = count / len(tokens) if len(tokens) > 0 else 0
                        idf = self.idf_[feature_idx]
                        feature_matrix[doc_idx, feature_idx] = tf * idf
        return FeatureSet(features=feature_matrix, feature_names=self.feature_names_, feature_types=['numeric'] * n_features, metadata={'weighting_scheme': self.weighting_scheme})

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> List[List[str]]:
        """
        Convert n-gram feature vectors back to lists of n-grams.
        
        This method takes feature vectors and returns the corresponding lists
        of n-grams that have non-zero values in each vector. This is useful
        for interpreting model predictions or understanding which n-grams
        contribute to specific documents.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Feature vectors to convert back to n-grams. Can be a FeatureSet
            or a numpy array.
        **kwargs : dict
            Additional parameters for inverse transformation (not used in this implementation)
            
        Returns
        -------
        List[List[str]]
            List of lists, where each inner list contains the n-grams with
            non-zero values for the corresponding document
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted
        """
        if not self._fitted:
            raise ValueError("Transformer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            feature_matrix = data.features
        elif isinstance(data, np.ndarray):
            feature_matrix = data
        else:
            raise ValueError('Unsupported input format. Expected FeatureSet or np.ndarray')
        result = []
        for row in feature_matrix:
            ngrams = [self.feature_names_[i] for (i, val) in enumerate(row) if val != 0]
            result.append(ngrams)
        return result

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get the names of the extracted n-gram features.
        
        Returns
        -------
        Optional[List[str]]
            List of feature names (n-grams) in the order they appear in the
            feature matrix, or None if the transformer has not been fitted
        """
        return self.feature_names_ if self._fitted else None