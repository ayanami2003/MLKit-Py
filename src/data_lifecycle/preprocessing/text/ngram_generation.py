from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Union, Dict, Optional


# ...(code omitted)...


def generate_bigrams(text_data: Union[List[str], FeatureSet]) -> FeatureSet:
    """
    Generate bigram features from text data.
    
    This function extracts consecutive pairs of words (bigrams) from input text
    and returns them as a FeatureSet with binary occurrence indicators.
    
    Args:
        text_data (Union[List[str], FeatureSet]): Input text data as a list of strings
            or a FeatureSet containing text features.
            
    Returns:
        FeatureSet: A FeatureSet containing bigram features where each column represents
            a unique bigram and each row indicates its presence (1) or absence (0) in 
            the corresponding document.
            
    Examples:
        >>> texts = ["the quick brown fox", "quick brown dog"]
        >>> bigram_features = generate_bigrams(texts)
        >>> print(bigram_features.features.shape)
        # Will show the shape of the bigram feature matrix
        
    Notes:
        - Handles both single strings and collections of strings
        - Automatically tokenizes input text by whitespace
        - Returns binary occurrence features rather than counts
        - Preserves original sample ordering
    """
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    if isinstance(text_data, FeatureSet):
        if text_data.features.ndim == 1:
            texts = text_data.features.tolist()
        else:
            texts = text_data.features[:, 0].tolist()
        sample_ids = text_data.sample_ids
    elif isinstance(text_data, list):
        texts = text_data
        sample_ids = None
    else:
        raise TypeError('text_data must be either a List[str] or FeatureSet')
    if not texts:
        return FeatureSet(features=np.array([]).reshape(0, 0), feature_names=[], sample_ids=[])
    texts = [str(text) for text in texts]
    vectorizer = CountVectorizer(ngram_range=(2, 2), binary=True, token_pattern='\\b\\w+\\b')
    try:
        bigram_features = vectorizer.fit_transform(texts)
    except ValueError:
        return FeatureSet(features=np.array([]).reshape(len(texts), 0), feature_names=[], sample_ids=sample_ids)
    feature_names = vectorizer.get_feature_names_out().tolist()
    if hasattr(bigram_features, 'toarray'):
        bigram_features = bigram_features.toarray()
    return FeatureSet(features=bigram_features, feature_names=feature_names, sample_ids=sample_ids)

def extract_character_ngrams(text_data: Union[List[str], FeatureSet], n: int=3, min_frequency: int=1, max_features: Optional[int]=None) -> FeatureSet:
    """
    Extract character-level n-gram features from text data.
    
    This function generates character n-grams (subsequences of n characters) from 
    input text and returns them as a FeatureSet with frequency counts.
    
    Args:
        text_data (Union[List[str], FeatureSet]): Input text data as a list of strings
            or a FeatureSet containing text features.
        n (int): Length of character n-grams to extract. Defaults to 3 (trigrams).
        max_features (Optional[int]): Maximum number of character n-grams to retain,
            ranked by frequency. If None, all n-grams are retained.
        min_frequency (int): Minimum frequency threshold for including n-grams.
            
    Returns:
        FeatureSet: A FeatureSet containing character n-gram features where each 
            column represents a unique character n-gram and each cell contains the 
            count of occurrences in the corresponding document.
            
    Examples:
        >>> texts = ["hello", "world"]
        >>> char_ngrams = extract_character_ngrams(texts, n=2)
        >>> print(char_ngrams.features.shape)
        
    Notes:
        - Handles both single strings and collections of strings
        - Preserves original sample ordering
        - Overlapping character n-grams are extracted from each text
        - Case sensitive by default
    """
    from collections import Counter
    import numpy as np
    if isinstance(text_data, FeatureSet):
        if text_data.features.dtype.kind in ['U', 'S']:
            texts = text_data.features.flatten().tolist()
        else:
            texts = text_data.features[:, 0].tolist()
        original_metadata = text_data.metadata.copy() if text_data.metadata else {}
        original_sample_ids = text_data.sample_ids
        original_quality_scores = text_data.quality_scores
    else:
        texts = text_data
        original_metadata = {}
        original_sample_ids = None
        original_quality_scores = None
    doc_ngrams = []
    global_ngram_counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            text = str(text)
        text_ngrams = []
        if len(text) >= n:
            for i in range(len(text) - n + 1):
                ngram = text[i:i + n]
                text_ngrams.append(ngram)
        doc_ngrams.append(Counter(text_ngrams))
        global_ngram_counter.update(text_ngrams)
    filtered_ngrams = {ngram: count for (ngram, count) in global_ngram_counter.items() if count >= min_frequency}
    sorted_ngrams = sorted(filtered_ngrams.items(), key=lambda x: (-x[1], x[0]))
    if max_features is not None:
        sorted_ngrams = sorted_ngrams[:max_features]
    feature_names = [ngram for (ngram, _) in sorted_ngrams]
    n_docs = len(texts)
    n_features = len(feature_names)
    if n_features == 0:
        feature_matrix = np.zeros((n_docs, 0))
    else:
        feature_matrix = np.zeros((n_docs, n_features))
        ngram_to_idx = {ngram: idx for (idx, ngram) in enumerate(feature_names)}
        for (doc_idx, doc_ngram_counter) in enumerate(doc_ngrams):
            for (ngram, count) in doc_ngram_counter.items():
                if ngram in ngram_to_idx:
                    feature_idx = ngram_to_idx[ngram]
                    feature_matrix[doc_idx, feature_idx] = count
    metadata = original_metadata.copy()
    if 'source' not in metadata:
        metadata['source'] = 'character_ngrams'
    return FeatureSet(features=feature_matrix, feature_names=feature_names, feature_types=['numeric'] * n_features if n_features > 0 else [], sample_ids=original_sample_ids, metadata=metadata, quality_scores=original_quality_scores)

def count_trigrams(text_data: Union[List[str], FeatureSet]) -> FeatureSet:
    """
    Count word trigram occurrences in text data.
    
    This function extracts consecutive triplets of words (trigrams) from input text
    and returns their frequency counts as a FeatureSet.
    
    Args:
        text_data (Union[List[str], FeatureSet]): Input text data as a list of strings
            or a FeatureSet containing text features.
            
    Returns:
        FeatureSet: A FeatureSet containing trigram frequency features where each 
            column represents a unique trigram and each cell contains the count of 
            occurrences in the corresponding document.
            
    Examples:
        >>> texts = ["the quick brown fox jumps", "quick brown fox runs"]
        >>> trigram_counts = count_trigrams(texts)
        >>> print(trigram_counts.features.shape)
        
    Notes:
        - Automatically tokenizes input text by whitespace
        - Handles both single strings and collections of strings
        - Preserves original sample ordering
        - Returns frequency counts rather than binary indicators
    """
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    if isinstance(text_data, FeatureSet):
        if text_data.features.ndim == 1:
            texts = text_data.features.tolist()
        else:
            texts = text_data.features[:, 0].tolist()
        sample_ids = text_data.sample_ids
    elif isinstance(text_data, list):
        texts = text_data
        sample_ids = None
    else:
        raise TypeError('text_data must be either a List[str] or FeatureSet')
    if not texts:
        if sample_ids is None:
            sample_ids = []
        return FeatureSet(features=np.array([]).reshape(0, 0), feature_names=[], sample_ids=sample_ids)
    texts = [str(text) for text in texts]
    vectorizer = CountVectorizer(ngram_range=(3, 3), binary=False, token_pattern='\\b\\w+\\b')
    try:
        trigram_features = vectorizer.fit_transform(texts)
    except ValueError:
        if sample_ids is None:
            sample_ids = [f'sample_{i}' for i in range(len(texts))]
        return FeatureSet(features=np.array([]).reshape(len(texts), 0), feature_names=[], sample_ids=sample_ids)
    feature_names = vectorizer.get_feature_names_out().tolist()
    if hasattr(trigram_features, 'toarray'):
        trigram_features = trigram_features.toarray()
    if sample_ids is None:
        sample_ids = [f'sample_{i}' for i in range(len(texts))]
    return FeatureSet(features=trigram_features, feature_names=feature_names, sample_ids=sample_ids)