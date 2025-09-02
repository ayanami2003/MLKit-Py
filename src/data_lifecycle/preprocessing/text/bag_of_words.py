from typing import List, Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union, Dict, Tuple
import re
import numpy as np
from collections import Counter

class BagOfWordsVectorizer(BaseTransformer):
    """
    Transform text data into bag-of-words representation.
    
    This transformer converts a collection of text documents into numerical
    feature vectors using the bag-of-words technique. Each document is represented
    as a vector where each element corresponds to the frequency (or presence)
    of a word from the vocabulary.
    
    The transformer supports various configurations including vocabulary size limits,
    n-gram ranges, document frequency thresholds, and custom vocabularies.
    
    Attributes
    ----------
    max_features : Optional[int]
        Maximum number of features (words) to include in the vocabulary.
        If None, all features are used.
    ngram_range : tuple
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that
        min_n <= n <= max_n will be used.
    min_df : Union[int, float]
        Minimum document frequency for a word to be included in the vocabulary.
        If float, interpreted as a fraction of documents. If int, interpreted
        as absolute count.
    max_df : Union[int, float]
        Maximum document frequency for a word to be included in the vocabulary.
        If float, interpreted as a fraction of documents. If int, interpreted
        as absolute count.
    stop_words : Optional[List[str]]
        List of stop words to exclude from the vocabulary.
    vocabulary : Optional[List[str]]
        Custom vocabulary to use. If provided, overrides vocabulary learning.
    binary : bool
        If True, all non-zero counts are set to 1. Useful for discrete
        probabilistic models that model binary events rather than integer counts.
    lowercase : bool
        Convert all characters to lowercase before tokenizing.
    
    Examples
    --------
    >>> from src.data_lifecycle.preprocessing.text.bag_of_words import BagOfWordsVectorizer
    >>> texts = ["this is a sample", "this is another example"]
    >>> vectorizer = BagOfWordsVectorizer(max_features=100, ngram_range=(1, 2))
    >>> feature_set = vectorizer.fit_transform(texts)
    >>> print(feature_set.features.shape)
    """

    def __init__(self, max_features: Optional[int]=None, ngram_range: tuple=(1, 1), min_df: Union[int, float]=1, max_df: Union[int, float]=1.0, stop_words: Optional[List[str]]=None, vocabulary: Optional[List[str]]=None, binary: bool=False, lowercase: bool=True, name: Optional[str]=None):
        """
        Initialize the BagOfWordsVectorizer.
        
        Parameters
        ----------
        max_features : Optional[int], default=None
            Maximum number of features to include in the vocabulary.
        ngram_range : tuple, default=(1, 1)
            The lower and upper boundary of the range of n-values for different
            word n-grams to be extracted.
        min_df : Union[int, float], default=1
            Minimum document frequency for words to be included in vocabulary.
        max_df : Union[int, float], default=1.0
            Maximum document frequency for words to be included in vocabulary.
        stop_words : Optional[List[str]], default=None
            List of stop words to exclude from vocabulary.
        vocabulary : Optional[List[str]], default=None
            Custom vocabulary to use instead of learning from data.
        binary : bool, default=False
            If True, all non-zero counts are set to 1.
        lowercase : bool, default=True
            Convert all characters to lowercase before tokenizing.
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words or []
        self.vocabulary = vocabulary
        self.binary = binary
        self.lowercase = lowercase
        self._vocabulary_dict: Optional[Dict[str, int]] = None
        self._feature_names: Optional[List[str]] = None

    def fit(self, data: Union[FeatureSet, List[str]], **kwargs) -> 'BagOfWordsVectorizer':
        """
        Learn the vocabulary dictionary from training data.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Training data. Can be a FeatureSet with text data or a list of strings.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        BagOfWordsVectorizer
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            texts = data.data
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        if not all((isinstance(text, str) for text in texts)):
            raise ValueError('All elements in the input data must be strings')
        if self.vocabulary is not None:
            unique_vocabulary = []
            seen = set()
            for term in self.vocabulary:
                if term not in seen:
                    unique_vocabulary.append(term)
                    seen.add(term)
            self._feature_names = unique_vocabulary
            self._vocabulary_dict = {term: idx for (idx, term) in enumerate(unique_vocabulary)}
            return self
        processed_texts = []
        for text in texts:
            processed_text = text.lower() if self.lowercase else text
            processed_texts.append(processed_text)
        all_ngrams = []
        doc_ngram_sets = []
        (min_n, max_n) = self.ngram_range
        for text in processed_texts:
            tokens = re.findall('\\b\\w+\\b', text)
            tokens = [token for token in tokens if token not in self.stop_words]
            text_ngrams = set()
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[i:i + n])
                    text_ngrams.add(ngram)
            all_ngrams.extend(list(text_ngrams))
            doc_ngram_sets.append(text_ngrams)
        ngram_counts = Counter(all_ngrams)
        total_docs = len(processed_texts)
        min_df_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * total_docs)
        max_df_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * total_docs)
        filtered_ngrams = {}
        for ngram in ngram_counts.keys():
            doc_freq = sum((1 for doc_ngrams in doc_ngram_sets if ngram in doc_ngrams))
            if min_df_count <= doc_freq <= max_df_count:
                filtered_ngrams[ngram] = ngram_counts[ngram]
        sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x: (-x[1], x[0]))
        if self.max_features is not None:
            sorted_ngrams = sorted_ngrams[:self.max_features]
        self._feature_names = [ngram for (ngram, _) in sorted_ngrams]
        self._vocabulary_dict = {ngram: idx for (idx, (ngram, _)) in enumerate(sorted_ngrams)}
        return self

    def transform(self, data: Union[FeatureSet, List[str]], **kwargs) -> FeatureSet:
        """
        Transform documents to document-term matrix.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Input data to transform. Can be a FeatureSet with text data or a list of strings.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed data as a FeatureSet with bag-of-words features.
        """
        if self._vocabulary_dict is None:
            raise ValueError("BagOfWordsVectorizer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            texts = data.data
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        if not all((isinstance(text, str) for text in texts)):
            raise ValueError('All elements in the input data must be strings')
        processed_texts = []
        for text in texts:
            processed_text = text.lower() if self.lowercase else text
            processed_texts.append(processed_text)
        n_docs = len(processed_texts)
        n_features = len(self._vocabulary_dict)
        doc_term_matrix = np.zeros((n_docs, n_features), dtype=np.int32)
        (min_n, max_n) = self.ngram_range
        for (doc_idx, text) in enumerate(processed_texts):
            tokens = re.findall('\\b\\w+\\b', text)
            tokens = [token for token in tokens if token not in self.stop_words]
            doc_ngrams = []
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[i:i + n])
                    doc_ngrams.append(ngram)
            ngram_counts = Counter(doc_ngrams)
            for (ngram, count) in ngram_counts.items():
                if ngram in self._vocabulary_dict:
                    feature_idx = self._vocabulary_dict[ngram]
                    value = 1 if self.binary else count
                    doc_term_matrix[doc_idx, feature_idx] = value
        feature_names = self.get_feature_names()
        return FeatureSet(features=doc_term_matrix, feature_names=feature_names)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[List[str]]:
        """
        Transform document-term matrix back to list of words.
        
        Parameters
        ----------
        data : FeatureSet
            Document-term matrix to transform back.
        **kwargs : dict
            Additional parameters for inverse transformation.
            
        Returns
        -------
        List[List[str]]
            List of lists of words for each document.
        """
        if self._vocabulary_dict is None:
            raise ValueError("BagOfWordsVectorizer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet')
        reverse_vocab = {idx: term for (term, idx) in self._vocabulary_dict.items()}
        result = []
        matrix = data.features
        for doc_row in matrix:
            doc_terms = []
            nonzero_indices = np.nonzero(doc_row)[0] if not self.binary else np.where(doc_row > 0)[0]
            for idx in nonzero_indices:
                if idx in reverse_vocab:
                    term = reverse_vocab[idx]
                    count = doc_row[idx] if not self.binary else 1
                    doc_terms.extend([term] * count)
            result.append(doc_terms)
        return result

    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary).
        
        Returns
        -------
        List[str]
            List of feature names in the order they appear in the feature matrix.
        """
        if self._feature_names is None:
            raise ValueError("BagOfWordsVectorizer has not been fitted yet. Call 'fit' first.")
        return self._feature_names.copy()