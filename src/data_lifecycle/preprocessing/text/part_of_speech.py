from typing import List, Union, Optional
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Union, Optional, Dict, Any
import numpy as np
from collections import Counter

class POSTaggingExtractor(BaseTransformer):
    """
    Extract Part-of-Speech (POS) tagging features from text data.
    
    This transformer applies POS tagging to textual input and generates features
    based on the grammatical roles of words. It can be used to enrich text data
    with syntactic information for downstream NLP tasks.
    
    The transformer supports various tagging schemes and can output different
    representations of the POS features (e.g., tag sequences, tag frequency counts).
    
    Attributes
    ----------
    tagset : str, optional
        The POS tagset to use (e.g., 'universal', 'penntreebank')
    feature_type : str, default 'sequence'
        Type of features to extract: 'sequence', 'counts', or 'presence'
    name : str, optional
        Name of the transformer instance
        
    Methods
    -------
    fit() : Learns any necessary tagging models or parameters
    transform() : Applies POS tagging and extracts features
    inverse_transform() : Not implemented for this transformer
    """

    def __init__(self, tagset: Optional[str]=None, feature_type: str='sequence', name: Optional[str]=None):
        """
        Initialize the POS tagging extractor.
        
        Parameters
        ----------
        tagset : str, optional
            The POS tagset to use (e.g., 'universal', 'penntreebank'). 
            If None, uses the default tagset of the underlying POS tagger.
        feature_type : str, default 'sequence'
            Type of features to extract:
            - 'sequence': Full sequence of POS tags
            - 'counts': Frequency count of each tag type
            - 'presence': Binary indicators for tag presence
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.tagset = tagset
        self.feature_type = feature_type
        self._fitted = False
        self._tag_vocabulary = None
        try:
            import nltk
            self._nltk_available = True
        except ImportError:
            self._nltk_available = False
            raise ImportError("NLTK is required for POSTaggingExtractor. Please install it with 'pip install nltk'")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)

    def fit(self, data: Union[FeatureSet, List[str]], **kwargs) -> 'POSTaggingExtractor':
        """
        Fit the POS tagger (if needed).
        
        For most POS taggers, this is a no-op since they're pre-trained.
        However, some implementations might require fitting to vocabulary.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Training data consisting of text samples
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        POSTaggingExtractor
            Self instance for method chaining
        """
        if isinstance(data, FeatureSet):
            texts = data.data if isinstance(data.data, list) else data.data.tolist()
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        all_tags = set()
        for text in texts:
            if not isinstance(text, str):
                raise ValueError('All elements in the input data must be strings')
            tags = self._pos_tag_text(text)
            all_tags.update((tag for (_, tag) in tags))
        self._tag_vocabulary = {tag: idx for (idx, tag) in enumerate(sorted(all_tags))}
        self._fitted = True
        return self

    def _pos_tag_text(self, text: str) -> List[tuple]:
        """
        Apply POS tagging to a single text string.
        
        Parameters
        ----------
        text : str
            Input text to tag
            
        Returns
        -------
        List[tuple]
            List of (word, tag) tuples
        """
        import nltk
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        if self.tagset:
            tagged = nltk.pos_tag(tokens, tagset=self.tagset)
        else:
            tagged = nltk.pos_tag(tokens)
        return tagged

    def _extract_sequence_features(self, texts: List[str]) -> FeatureSet:
        """
        Extract POS tag sequences as features.
        
        Parameters
        ----------
        texts : List[str]
            List of input texts
            
        Returns
        -------
        FeatureSet
            FeatureSet with sequence features
        """
        sequences = []
        max_length = 0
        for text in texts:
            tags = self._pos_tag_text(text)
            tag_sequence = [tag for (_, tag) in tags]
            sequences.append(tag_sequence)
            max_length = max(max_length, len(tag_sequence))
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + ['<PAD>'] * (max_length - len(seq))
            padded_sequences.append(padded_seq)
        if self._tag_vocabulary is not None:
            numerical_sequences = []
            for seq in padded_sequences:
                num_seq = [self._tag_vocabulary.get(tag, -1) for tag in seq]
                numerical_sequences.append(num_seq)
            feature_matrix = np.array(numerical_sequences)
            feature_names = [f'pos_{i}' for i in range(max_length)]
        else:
            feature_matrix = np.array(padded_sequences, dtype=object)
            feature_names = [f'pos_{i}' for i in range(max_length)]
        return FeatureSet(features=feature_matrix, feature_names=feature_names, metadata={'feature_type': 'sequence'})

    def _extract_count_features(self, texts: List[str]) -> FeatureSet:
        """
        Extract POS tag frequency counts as features.
        
        Parameters
        ----------
        texts : List[str]
            List of input texts
            
        Returns
        -------
        FeatureSet
            FeatureSet with count features
        """
        if self._tag_vocabulary is not None:
            tag_list = list(self._tag_vocabulary.keys())
        else:
            all_tags = set()
            for text in texts:
                tags = self._pos_tag_text(text)
                all_tags.update((tag for (_, tag) in tags))
            tag_list = list(all_tags)
            self._tag_vocabulary = {tag: idx for (idx, tag) in enumerate(tag_list)}
        feature_matrix = np.zeros((len(texts), len(tag_list)))
        for (i, text) in enumerate(texts):
            tags = self._pos_tag_text(text)
            tag_counter = Counter((tag for (_, tag) in tags))
            for (tag, count) in tag_counter.items():
                if tag in self._tag_vocabulary:
                    j = self._tag_vocabulary[tag]
                    feature_matrix[i, j] = count
        return FeatureSet(features=feature_matrix, feature_names=list(tag_list), metadata={'feature_type': 'counts'})

    def _extract_presence_features(self, texts: List[str]) -> FeatureSet:
        """
        Extract POS tag presence indicators as features.
        
        Parameters
        ----------
        texts : List[str]
            List of input texts
            
        Returns
        -------
        FeatureSet
            FeatureSet with presence features
        """
        if self._tag_vocabulary is not None:
            tag_list = list(self._tag_vocabulary.keys())
        else:
            all_tags = set()
            for text in texts:
                tags = self._pos_tag_text(text)
                all_tags.update((tag for (_, tag) in tags))
            tag_list = list(all_tags)
            self._tag_vocabulary = {tag: idx for (idx, tag) in enumerate(tag_list)}
        feature_matrix = np.zeros((len(texts), len(tag_list)))
        for (i, text) in enumerate(texts):
            tags = self._pos_tag_text(text)
            unique_tags = set((tag for (_, tag) in tags))
            for tag in unique_tags:
                if tag in self._tag_vocabulary:
                    j = self._tag_vocabulary[tag]
                    feature_matrix[i, j] = 1
        return FeatureSet(features=feature_matrix, feature_names=list(tag_list), metadata={'feature_type': 'presence'})

    def transform(self, data: Union[FeatureSet, List[str]], **kwargs) -> FeatureSet:
        """
        Apply POS tagging and extract features.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Input text data to process
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            FeatureSet containing extracted POS tagging features
        """
        if not self._fitted:
            raise ValueError("POSTaggingExtractor has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            texts = data.data if isinstance(data.data, list) else data.data.tolist()
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        for text in texts:
            if not isinstance(text, str):
                raise ValueError('All elements in the input data must be strings')
        if self.feature_type == 'sequence':
            return self._extract_sequence_features(texts)
        elif self.feature_type == 'counts':
            return self._extract_count_features(texts)
        elif self.feature_type == 'presence':
            return self._extract_presence_features(texts)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}. Supported types are 'sequence', 'counts', and 'presence'.")

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[List[str]]:
        """
        Convert POS features back to tag sequences (if possible).
        
        Parameters
        ----------
        data : FeatureSet
            POS features to convert back
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        List[List[str]]
            List of POS tag sequences
            
        Raises
        ------
        NotImplementedError
            Always raised as reconstruction is non-trivial
        """
        raise NotImplementedError('Inverse transformation is not implemented for POSTaggingExtractor since reconstructing original text from aggregated features is non-trivial.')