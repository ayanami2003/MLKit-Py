from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
from collections import Counter
import re
from scipy.sparse import csr_matrix

class TFIDFVectorizer(BaseTransformer):

    def __init__(self, max_features: Optional[int]=None, ngram_range: tuple=(1, 1), min_df: Union[int, float]=1, max_df: Union[int, float]=1.0, stop_words: Optional[List[str]]=None, vocabulary: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the TF-IDF vectorizer.
        
        Parameters
        ----------
        max_features : Optional[int], default=None
            Maximum number of features to consider (vocabulary size).
        ngram_range : tuple, default=(1, 1)
            The n-gram range (min_n, max_n) to consider.
        min_df : int or float, default=1
            Minimum document frequency for terms. If int, absolute count.
            If float, proportion of documents.
        max_df : int or float, default=1.0
            Maximum document frequency for terms. If int, absolute count.
            If float, proportion of documents.
        stop_words : Optional[List[str]], default=None
            List of stop words to exclude from vocabulary.
        vocabulary : Optional[List[str]], default=None
            Predefined vocabulary. If provided, disables vocabulary learning.
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
        self._idf_values = None
        self._vocab_indices = None

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase and tokenizing.
        
        Parameters
        ----------
        text : str
            Input text to preprocess.
            
        Returns
        -------
        List[str]
            List of tokens.
        """
        if not text:
            return []
        text = text.lower()
        tokens = re.findall('\\b\\w+\\b', text)
        return tokens

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """
        Extract n-grams from tokens based on ngram_range.
        
        Parameters
        ----------
        tokens : List[str]
            List of tokens.
            
        Returns
        -------
        List[str]
            List of n-grams.
        """
        ngrams = []
        (min_n, max_n) = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = '_'.join(tokens[i:i + n])
                ngrams.append(ngram)
        return ngrams

    def _compute_idf_values(self, texts: List[str]) -> None:
        """
        Compute IDF values for vocabulary terms.
        
        Parameters
        ----------
        texts : List[str]
            List of text documents.
        """
        if self.vocabulary is not None:
            vocab_terms = self.vocabulary
        else:
            vocab_terms = list(self._vocab_indices.keys())
        n_docs = len(texts)
        idf_values = {}
        doc_freq = Counter()
        for text in texts:
            terms = self._preprocess_text(text)
            ngrams = self._extract_ngrams(terms)
            if self.stop_words:
                ngrams = [ng for ng in ngrams if ng not in self.stop_words]
            unique_terms = set(ngrams) & set(vocab_terms)
            for term in unique_terms:
                doc_freq[term] += 1
        for term in vocab_terms:
            df = doc_freq.get(term, 0)
            n_docs_total = len(texts)
            min_df_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs_total)
            max_df_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs_total)
            if self.vocabulary is not None or (df >= min_df_count and df <= max_df_count):
                idf = np.log(n_docs_total / (df + 1)) + 1
                idf_values[term] = idf
            else:
                idf_values[term] = 0.0
        self._idf_values = idf_values

    def fit(self, data: Union[FeatureSet, List[str]], **kwargs) -> 'TFIDFVectorizer':
        """
        Learn vocabulary and IDF values from training data.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Training text data as a FeatureSet or list of strings.
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        TFIDFVectorizer
            Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            if data.features.ndim == 1:
                texts = data.features.tolist()
            else:
                texts = data.features.flatten().tolist()
        else:
            texts = data
        if self.vocabulary is not None:
            self._vocab_indices = {term: i for (i, term) in enumerate(self.vocabulary)}
            self._compute_idf_values(texts)
            return self
        doc_frequencies = Counter()
        all_ngrams = []
        for text in texts:
            terms = self._preprocess_text(text)
            ngrams = self._extract_ngrams(terms)
            if self.stop_words:
                ngrams = [ng for ng in ngrams if ng not in self.stop_words]
            all_ngrams.extend(ngrams)
            for term in set(ngrams):
                doc_frequencies[term] += 1
        n_docs = len(texts)
        min_df_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_df_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        valid_terms = []
        for (term, freq) in doc_frequencies.items():
            if freq >= min_df_count and freq <= max_df_count:
                valid_terms.append(term)
        valid_terms.sort()
        if self.max_features and len(valid_terms) > self.max_features:
            term_counts = Counter(all_ngrams)
            valid_terms = sorted(valid_terms, key=lambda t: term_counts[t], reverse=True)[:self.max_features]
            valid_terms.sort()
        self._vocab_indices = {term: i for (i, term) in enumerate(valid_terms)}
        self._compute_idf_values(texts)
        return self

    def transform(self, data: Union[FeatureSet, List[str]], **kwargs) -> FeatureSet:
        """
        Transform text data into TF-IDF vectors.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Text data to transform as a FeatureSet or list of strings.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            TF-IDF vector representations with feature names.
        """
        if isinstance(data, FeatureSet):
            if data.features.ndim == 1:
                texts = data.features.tolist()
            else:
                texts = data.features.flatten().tolist()
        else:
            texts = data
        n_docs = len(texts)
        n_features = len(self._vocab_indices) if self._vocab_indices else 0
        row_indices = []
        col_indices = []
        tfidf_values = []
        for (doc_idx, text) in enumerate(texts):
            terms = self._preprocess_text(text)
            ngrams = self._extract_ngrams(terms)
            if self.stop_words:
                ngrams = [ng for ng in ngrams if ng not in self.stop_words]
            term_counts = Counter(ngrams)
            doc_length = sum(term_counts.values())
            for (term, count) in term_counts.items():
                if term in (self._vocab_indices or {}):
                    term_idx = self._vocab_indices[term]
                    tf = count / doc_length if doc_length > 0 else 0
                    idf = self._idf_values.get(term, 0) if self._idf_values else 0
                    tfidf = tf * idf
                    if tfidf > 0:
                        row_indices.append(doc_idx)
                        col_indices.append(term_idx)
                        tfidf_values.append(tfidf)
        tfidf_matrix = csr_matrix((tfidf_values, (row_indices, col_indices)), shape=(n_docs, n_features))
        tfidf_dense = tfidf_matrix.toarray()
        feature_names = self.get_feature_names()
        return FeatureSet(features=tfidf_dense, feature_names=feature_names)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[List[str]]:
        """
        Convert TF-IDF vectors back to terms.
        
        Parameters
        ----------
        data : FeatureSet
            TF-IDF vectors to convert back.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        List[List[str]]
            Terms corresponding to non-zero entries in the vectors.
        """
        if not hasattr(self, '_vocab_indices') or self._vocab_indices is None:
            raise ValueError('Vectorizer not fitted. Call fit() before inverse_transform().')
        index_to_term = {idx: term for (term, idx) in self._vocab_indices.items()}
        if hasattr(data.features, 'toarray'):
            matrix = data.features.toarray()
        else:
            matrix = data.features
        result = []
        for row in matrix:
            nonzero_indices = np.nonzero(row)[0]
            terms = [index_to_term[idx] for idx in nonzero_indices]
            result.append(terms)
        return result

    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary terms).
        
        Returns
        -------
        List[str]
            Vocabulary terms in order of feature indices.
        """
        if self._vocab_indices is None:
            return []
        sorted_items = sorted(self._vocab_indices.items(), key=lambda x: x[1])
        return [term for (term, _) in sorted_items]