from general.structures.feature_set import FeatureSet
from general.base_classes.transformer_base import BaseTransformer
from typing import List, Optional, Union, Dict
import numpy as np
import re
from collections import Counter
import math





class TextFeatureGenerator(BaseTransformer):
    """
    Automated text feature extraction transformer for generating numerical representations from text data.
    
    This transformer applies various automated text processing techniques to convert raw text
    into numerical features suitable for machine learning models. It supports multiple
    extraction methods including TF-IDF, bag-of-words, n-grams, and embedding-based approaches.
    
    Attributes
    ----------
    method : str
        The text feature extraction method to use ('tfidf', 'bow', 'ngram', 'embeddings')
    max_features : Optional[int]
        Maximum number of features to generate (vocabulary size)
    ngram_range : tuple
        Range of n-gram sizes (min_n, max_n) for n-gram based methods
    embedding_model : Optional[str]
        Name or path to pre-trained embedding model for embedding-based extraction
    
    Methods
    -------
    fit() : Fits the transformer to training data
    transform() : Applies text feature extraction to input data
    inverse_transform() : Not supported for text feature extraction
    get_feature_names() : Returns names of generated features
    """

    def __init__(self, method: str='tfidf', max_features: Optional[int]=10000, ngram_range: tuple=(1, 2), embedding_model: Optional[str]=None, name: Optional[str]=None):
        """
        Initialize the TextFeatureGenerator.
        
        Parameters
        ----------
        method : str, default='tfidf'
            Text feature extraction method to use. Options:
            - 'tfidf': Term Frequency-Inverse Document Frequency
            - 'bow': Bag of Words
            - 'ngram': N-gram features
            - 'embeddings': Pre-trained embedding models
        max_features : int or None, default=10000
            Maximum number of features (vocabulary size) to retain
        ngram_range : tuple, default=(1, 2)
            Minimum and maximum n-gram sizes for n-gram methods
        embedding_model : str or None, default=None
            Path or identifier for pre-trained embedding model
        name : str or None, default=None
            Optional name for the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.embedding_model = embedding_model
        self._vocabulary = None
        self._idf_values = None
        self._fitted = False

    def fit(self, data: Union[FeatureSet, List[str], np.ndarray], **kwargs) -> 'TextFeatureGenerator':
        """
        Fit the text feature generator to the input text data.
        
        This method learns vocabulary, IDF values, or loads embedding models
        depending on the selected extraction method.
        
        Parameters
        ----------
        data : FeatureSet, List[str], or np.ndarray
            Input text data to fit on. Can be:
            - FeatureSet with text data in its features
            - List of text strings
            - NumPy array of text strings
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        TextFeatureGenerator
            Fitted transformer instance
            
        Raises
        ------
        ValueError
            If the method is not supported or data format is invalid
        """
        pass

    def transform(self, data: Union[FeatureSet, List[str], np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform input text data into numerical features.
        
        Applies the configured text feature extraction method to convert
        raw text into numerical representations.
        
        Parameters
        ----------
        data : FeatureSet, List[str], or np.ndarray
            Input text data to transform. Same format options as fit().
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            FeatureSet containing extracted numerical features with metadata
            
        Raises
        ------
        RuntimeError
            If transformer has not been fitted yet
        ValueError
            If data format is incompatible with fitted vocabulary
        """
        pass

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> List[str]:
        """
        Attempt to reconstruct text from feature vectors (not supported).
        
        Text feature extraction is generally not invertible due to information loss
        during the numerical encoding process.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Numerical feature data to invert
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        List[str]
            Empty list as inversion is not supported
            
        Warning
        -------
        Always raises NotImplementedError as text feature extraction is not invertible.
        """
        pass

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get names of the extracted features.
        
        Returns feature names based on learned vocabulary for count-based methods
        or generic names for embedding-based methods.
        
        Returns
        -------
        List[str] or None
            List of feature names or None if not fitted
        """
        pass