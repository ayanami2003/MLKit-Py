from typing import List, Union, Optional
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import re
from collections import Counter
from typing import List, Union, Optional, Dict, Tuple
import numpy as np

class TextTokenizer(BaseTransformer):
    """
    A transformer class for tokenizing text data into constituent tokens (words, subwords, etc.).
    
    This class provides methods to fit a tokenizer on a corpus and transform text data into
    token sequences. It supports various tokenization strategies and can handle both
    single strings and lists of strings.
    
    Attributes
    ----------
    token_pattern : str, optional
        Regular expression pattern used for tokenization (default: r'\\b\\w+\\b')
    lowercase : bool, optional
        Whether to convert text to lowercase before tokenization (default: True)
    max_features : int, optional
        Maximum number of tokens to keep, based on frequency (None for no limit)
    min_freq : int, optional
        Minimum frequency of tokens to consider (default: 1)
    """

    def __init__(self, token_pattern: str='\\b\\w+\\b', lowercase: bool=True, max_features: Optional[int]=None, min_freq: int=1, name: Optional[str]=None):
        """
        Initialize the TextTokenizer.
        
        Parameters
        ----------
        token_pattern : str, optional
            Regular expression pattern used for tokenization (default: r'\\b\\w+\\b')
        lowercase : bool, optional
            Whether to convert text to lowercase before tokenization (default: True)
        max_features : int, optional
            Maximum number of tokens to keep, based on frequency (None for no limit)
        min_freq : int, optional
            Minimum frequency of tokens to consider (default: 1)
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.token_pattern = token_pattern
        self.lowercase = lowercase
        self.max_features = max_features
        self.min_freq = min_freq
        self.vocabulary_ = None
        self.tokenizer_ = None

    def fit(self, data: Union[FeatureSet, List[str]], **kwargs) -> 'TextTokenizer':
        """
        Fit the tokenizer on the input text data.
        
        This method learns the vocabulary from the input data based on the specified
        tokenization parameters.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Input text data to fit the tokenizer on. Can be a FeatureSet containing
            text data or a list of strings.
        **kwargs : dict
            Additional parameters for fitting (ignored in this implementation)
            
        Returns
        -------
        TextTokenizer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the input data format is not supported
        """
        if isinstance(data, FeatureSet):
            texts = data.data
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        pattern = re.compile(self.token_pattern)
        all_tokens = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError('All elements in the input data must be strings')
            processed_text = text.lower() if self.lowercase else text
            tokens = pattern.findall(processed_text)
            all_tokens.extend(tokens)
        token_counts = Counter(all_tokens)
        filtered_tokens = {token: count for (token, count) in token_counts.items() if count >= self.min_freq}
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1], x[0]))
        if self.max_features is not None:
            sorted_tokens = sorted_tokens[:self.max_features]
        self.vocabulary_ = {token: idx for (idx, (token, _)) in enumerate(sorted_tokens)}
        self.reverse_vocabulary_ = {idx: token for (token, idx) in self.vocabulary_.items()}
        self.tokenizer_ = pattern
        return self

    def transform(self, data: Union[FeatureSet, List[str]], **kwargs) -> FeatureSet:
        """
        Transform text data into token sequences.
        
        This method applies the learned tokenization to convert text data into
        sequences of tokens.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str]]
            Input text data to tokenize. Can be a FeatureSet containing text data
            or a list of strings.
        **kwargs : dict
            Additional parameters for transformation (ignored in this implementation)
            
        Returns
        -------
        FeatureSet
            FeatureSet containing the tokenized data as a 2D array where each row
            represents a sequence of token indices
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if self.vocabulary_ is None:
            raise ValueError("TextTokenizer has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            texts = data.data
        elif isinstance(data, list):
            texts = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        token_sequences = []
        for text in texts:
            if not isinstance(text, str):
                raise ValueError('All elements in the input data must be strings')
            processed_text = text.lower() if self.lowercase else text
            tokens = self.tokenizer_.findall(processed_text)
            indices = [self.vocabulary_[token] for token in tokens if token in self.vocabulary_]
            token_sequences.append(indices)
        max_len = max((len(seq) for seq in token_sequences)) if token_sequences else 0
        padded_sequences = []
        for seq in token_sequences:
            padded_seq = seq + [0] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        result_array = np.array(padded_sequences)
        return FeatureSet(data=result_array, feature_names=[f'token_{i}' for i in range(result_array.shape[1])])

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[List[str]]:
        """
        Convert token indices back to text tokens.
        
        This method maps token indices back to their original string representations.
        
        Parameters
        ----------
        data : FeatureSet
            FeatureSet containing tokenized data as token indices
        **kwargs : dict
            Additional parameters for inverse transformation (ignored in this implementation)
            
        Returns
        -------
        List[List[str]]
            List of lists where each inner list contains the tokens for a document
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if self.vocabulary_ is None:
            raise ValueError("TextTokenizer has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet')
        if isinstance(data.data, np.ndarray):
            sequences = data.data.tolist()
        else:
            sequences = data.data
        result = []
        for sequence in sequences:
            if isinstance(sequence, (int, np.integer)):
                sequence = [sequence]
            elif isinstance(sequence, (list, np.ndarray)):
                if len(sequence) > 0 and isinstance(sequence[0], (list, np.ndarray)):
                    sequence = sequence[0]
            tokens = []
            for idx in sequence:
                if idx in self.reverse_vocabulary_:
                    tokens.append(self.reverse_vocabulary_[idx])
            result.append(tokens)
        return result

    def get_vocabulary(self) -> List[str]:
        """
        Get the learned vocabulary.
        
        Returns
        -------
        List[str]
            List of tokens in the vocabulary ordered by index
            
        Raises
        ------
        ValueError
            If the transformer has not been fitted yet
        """
        if self.vocabulary_ is None:
            raise ValueError("TextTokenizer has not been fitted yet. Call 'fit' first.")
        vocab_list = [''] * len(self.vocabulary_)
        for (token, idx) in self.vocabulary_.items():
            vocab_list[idx] = token
        return vocab_list