from typing import List, Optional, Union
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from collections import Counter
import numpy as np

class WordEncoder(BaseTransformer):

    def __init__(self, encoding_type: str='ordinal', max_features: Optional[int]=None, min_frequency: int=1, handle_unknown: str='ignore', default_token: Optional[str]=None, name: Optional[str]=None, n_components: int=50, max_length: Optional[int]=None):
        """
        Initialize the WordEncoder transformer.
        
        Parameters
        ----------
        encoding_type : str, default='ordinal'
            Type of encoding to use. Options are:
            - 'ordinal': Integer encoding based on vocabulary order
            - 'frequency': Frequency-based encoding
            - 'binary': Binary occurrence encoding
            - 'hash': Hash-based encoding
        max_features : int, optional
            Maximum size of vocabulary. If None, unlimited.
        min_frequency : int, default=1
            Minimum frequency for words to be included in vocabulary
        handle_unknown : str, default='ignore'
            Strategy for handling unknown words. Options:
            - 'ignore': Skip unknown words
            - 'use_default': Map to default token
        default_token : str, optional
            Token to use for unknown words when handle_unknown='use_default'
        name : str, optional
            Name of the transformer instance
        n_components : int, default=50
            Number of dimensions for hash encoding
        max_length : int, optional
            Maximum sequence length for ordinal encoding. If None, inferred during fit.
        """
        super().__init__(name=name)
        if encoding_type not in ['ordinal', 'frequency', 'binary', 'hash']:
            raise ValueError("encoding_type must be one of 'ordinal', 'frequency', 'binary', 'hash'")
        if handle_unknown not in ['ignore', 'use_default']:
            raise ValueError("handle_unknown must be one of 'ignore', 'use_default'")
        if handle_unknown == 'use_default' and default_token is None:
            raise ValueError("default_token must be provided when handle_unknown='use_default'")
        if max_features is not None and max_features <= 0:
            raise ValueError('max_features must be positive or None')
        if min_frequency <= 0:
            raise ValueError('min_frequency must be positive')
        if encoding_type == 'hash' and n_components <= 0:
            raise ValueError('n_components must be positive for hash encoding')
        if max_length is not None and max_length <= 0:
            raise ValueError('max_length must be positive or None')
        self.encoding_type = encoding_type
        self.max_features = max_features
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self.default_token = default_token
        self.n_components = n_components
        self.max_length = max_length
        self._vocabulary = {}
        self._reverse_vocabulary = {}
        self._token_frequencies = {}
        self._fitted = False
        self._ordinal_max_len = 0

    def fit(self, data: Union[FeatureSet, List[List[str]]], **kwargs) -> 'WordEncoder':
        """
        Learn the vocabulary and encoding mappings from the input text data.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[List[str]]]
            Input text data as list of token sequences or FeatureSet containing text
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        WordEncoder
            Fitted transformer instance
        """
        if isinstance(data, FeatureSet):
            token_sequences = data.data
        elif isinstance(data, list):
            token_sequences = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of token sequences')
        if not all((isinstance(seq, (list, np.ndarray)) for seq in token_sequences)):
            raise ValueError('Each element in data must be a list or array of tokens')
        all_tokens = []
        for sequence in token_sequences:
            all_tokens.extend(sequence)
        token_counts = Counter(all_tokens)
        filtered_tokens = {token: count for (token, count) in token_counts.items() if count >= self.min_frequency}
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1], x[0]))
        if self.max_features is not None:
            sorted_tokens = sorted_tokens[:self.max_features]
        self._vocabulary = {token: idx for (idx, (token, _)) in enumerate(sorted_tokens)}
        self._reverse_vocabulary = {idx: token for (token, idx) in self._vocabulary.items()}
        self._token_frequencies = dict(sorted_tokens)
        if self.handle_unknown == 'use_default' and self.default_token is not None:
            if self.default_token not in self._vocabulary:
                default_idx = len(self._vocabulary)
                self._vocabulary[self.default_token] = default_idx
                self._reverse_vocabulary[default_idx] = self.default_token
                self._token_frequencies[self.default_token] = 0
        if self.encoding_type == 'ordinal':
            sequence_lengths = [len(seq) for seq in token_sequences]
            max_observed_length = max(sequence_lengths) if sequence_lengths else 0
            if self.max_length is not None:
                self._ordinal_max_len = min(max_observed_length, self.max_length)
            else:
                self._ordinal_max_len = max_observed_length
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, List[List[str]]], **kwargs) -> FeatureSet:
        """
        Transform text data into numerical encodings based on fitted vocabulary.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[List[str]]]
            Input text data to transform
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            Encoded numerical representations of input text
        """
        if not self._fitted:
            raise ValueError("WordEncoder has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            token_sequences = data.data
        elif isinstance(data, list):
            token_sequences = data
        else:
            raise ValueError('Input data must be either a FeatureSet or a list of token sequences')
        if not all((isinstance(seq, (list, np.ndarray)) for seq in token_sequences)):
            raise ValueError('Each element in data must be a list or array of tokens')
        if self.encoding_type == 'hash':
            features = np.zeros((len(token_sequences), self.n_components))
            feature_names = [f'hash_{i}' for i in range(self.n_components)]
            for (seq_idx, sequence) in enumerate(token_sequences):
                for token in sequence:
                    if token in self._vocabulary or self.handle_unknown != 'ignore':
                        token_to_hash = token if token in self._vocabulary else self.default_token
                        if token_to_hash is not None:
                            hash_val = hash(token_to_hash)
                            position = hash_val % self.n_components
                            sign = 1 if hash_val % 2 == 0 else -1
                            features[seq_idx, position] += sign
        else:
            encoded_sequences = []
            for sequence in token_sequences:
                encoded_sequence = []
                for token in sequence:
                    if token in self._vocabulary:
                        encoded_sequence.append(self._vocabulary[token])
                    elif self.handle_unknown == 'use_default':
                        encoded_sequence.append(self._vocabulary[self.default_token])
                encoded_sequences.append(encoded_sequence)
            if self.encoding_type == 'ordinal':
                max_len = self._ordinal_max_len
                padded_sequences = []
                for seq in encoded_sequences:
                    truncated_seq = seq[:max_len] if max_len > 0 else []
                    padded_seq = truncated_seq + [0] * (max_len - len(truncated_seq))
                    padded_sequences.append(padded_seq)
                if len(padded_sequences) == 0 or max_len == 0:
                    features = np.zeros((len(token_sequences), max_len))
                else:
                    features = np.array(padded_sequences)
                feature_names = [f'token_{i}' for i in range(max_len)]
            elif self.encoding_type == 'binary':
                features = np.zeros((len(encoded_sequences), len(self._vocabulary)))
                for (i, seq) in enumerate(encoded_sequences):
                    for token_idx in seq:
                        features[i, token_idx] = 1
                feature_names = self.get_vocabulary()
            elif self.encoding_type == 'frequency':
                features = np.zeros((len(encoded_sequences), len(self._vocabulary)))
                for (i, seq) in enumerate(encoded_sequences):
                    for token_idx in seq:
                        features[i, token_idx] += 1
                feature_names = self.get_vocabulary()
        metadata = {'encoding_type': self.encoding_type, 'vocabulary_size': len(self._vocabulary), 'handle_unknown': self.handle_unknown}
        if self.default_token:
            metadata['default_token'] = self.default_token
        if self.encoding_type == 'hash':
            metadata['n_components'] = self.n_components
        return FeatureSet(features=features, feature_names=feature_names, metadata=metadata)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[List[str]]:
        """
        Convert numerical encodings back to original text tokens.
        
        Parameters
        ----------
        data : FeatureSet
            Numerical encoded data to convert back
        **kwargs : dict
            Additional parameters for inverse transformation
            
        Returns
        -------
        List[List[str]]
            Original text tokens reconstructed from encodings
        """
        if not self._fitted:
            raise ValueError("WordEncoder has not been fitted yet. Call 'fit' first.")
        if not isinstance(data, FeatureSet):
            raise ValueError('Input data must be a FeatureSet')
        if self.encoding_type == 'hash':
            raise NotImplementedError('Hash encoding does not support inverse transformation.')
        features = data.features
        if self.encoding_type in ['ordinal']:
            result = []
            for sequence in features:
                if self.encoding_type == 'ordinal':
                    sequence = [idx for idx in sequence if idx != 0 or self._reverse_vocabulary.get(0)]
                tokens = []
                for idx in sequence:
                    if idx in self._reverse_vocabulary:
                        tokens.append(self._reverse_vocabulary[idx])
                result.append(tokens)
            return result
        elif self.encoding_type in ['binary', 'frequency']:
            result = []
            for row in features:
                tokens = []
                for (idx, value) in enumerate(row):
                    if value > 0 and idx in self._reverse_vocabulary:
                        count = int(value) if self.encoding_type == 'frequency' else 1
                        tokens.extend([self._reverse_vocabulary[idx]] * count)
                result.append(tokens)
            return result
        return [[] for _ in range(features.shape[0])]

    def get_vocabulary(self) -> List[str]:
        """
        Get the learned vocabulary.
        
        Returns
        -------
        List[str]
            List of vocabulary terms in order of their indices
        """
        if not self._fitted:
            raise ValueError("WordEncoder has not been fitted yet. Call 'fit' first.")
        vocab_list = [''] * len(self._vocabulary)
        for (token, idx) in self._vocabulary.items():
            vocab_list[idx] = token
        return vocab_list

    def get_feature_names(self) -> List[str]:
        """
        Get names for the encoded features.
        
        Returns
        -------
        List[str]
            Names for each encoded feature dimension
        """
        if not self._fitted:
            raise ValueError("WordEncoder has not been fitted yet. Call 'fit' first.")
        if self.encoding_type == 'hash':
            return [f'hash_{i}' for i in range(self.n_components)]
        elif self.encoding_type == 'ordinal':
            max_len = self._ordinal_max_len
            return [f'token_{i}' for i in range(max_len)]
        elif self.encoding_type in ['binary', 'frequency']:
            return self.get_vocabulary()
        return []