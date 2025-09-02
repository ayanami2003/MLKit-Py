from general.structures.data_batch import DataBatch
from general.base_classes.transformer_base import BaseTransformer
from typing import Union, List
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer
import re
_ALGORITHM_MAP = {'porter': PorterStemmer, 'lancaster': LancasterStemmer}
_SUPPORTED_LANGUAGES = {'english'}
_ALGORITHM_MAP = {'porter': PorterStemmer, 'lancaster': LancasterStemmer}
_SUPPORTED_LANGUAGES = {'english'}

class StemmerTransformer(BaseTransformer):
    """
    A transformer that applies stemming to text data.
    
    This transformer reduces words to their root forms using rule-based
    suffix stripping algorithms. Stemming is a fast but approximate
    technique that may produce non-real words as stems.
    
    Attributes
    ----------
    algorithm : str
        The stemming algorithm to use (default: 'porter')
    language : str
        Language code for the stemming algorithm (default: 'english')
    name : str
        Name of the transformer instance
    
    Methods
    -------
    fit(data, **kwargs)
        Fit the transformer to the input data (no-op for stemming)
    transform(data, **kwargs)
        Apply stemming to the input text data
    inverse_transform(data, **kwargs)
        Not supported for stemming operations
    """
    _ALGORITHM_MAP = {'porter': PorterStemmer, 'lancaster': LancasterStemmer}
    _SUPPORTED_LANGUAGES = {'english'}

    def __init__(self, algorithm: str='porter', language: str='english', name: str=None):
        """
        Initialize the StemmerTransformer.
        
        Parameters
        ----------
        algorithm : str, optional
            The stemming algorithm to use ('porter', 'lancaster', etc.), by default 'porter'
        language : str, optional
            Language code for stemming, by default 'english'
        name : str, optional
            Name for the transformer instance, by default None
            
        Raises
        ------
        ValueError
            If algorithm or language is not supported
        """
        super().__init__(name=name)
        if algorithm not in self._ALGORITHM_MAP:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported algorithms: {list(self._ALGORITHM_MAP.keys())}")
        if language not in self._SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported languages: {list(self._SUPPORTED_LANGUAGES)}")
        self.algorithm = algorithm
        self.language = language
        self._stemmer = self._ALGORITHM_MAP[algorithm]()

    def fit(self, data: Union[DataBatch, List[str]], **kwargs) -> 'StemmerTransformer':
        """
        Fit the transformer to the input data.
        
        For stemming, fitting is not required as no model needs to be learned.
        This method simply returns self.
        
        Parameters
        ----------
        data : Union[DataBatch, List[str]]
            Input text data to fit on. If DataBatch, expects text data in the
            primary data field.
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        StemmerTransformer
            Self instance for method chaining
        """
        return self

    def transform(self, data: Union[DataBatch, List[str]], **kwargs) -> Union[DataBatch, List[str]]:
        """
        Apply stemming to the input text data.
        
        Parameters
        ----------
        data : Union[DataBatch, List[str]]
            Input text data to transform. Can be a list of strings or DataBatch
            containing text data.
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        Union[DataBatch, List[str]]
            Text data with words reduced to their stems using the specified
            stemming algorithm. Returns the same type as input.
        """
        if isinstance(data, DataBatch):
            texts = data.data
            is_batch = True
        elif isinstance(data, list):
            texts = data
            is_batch = False
        else:
            raise ValueError('Input data must be either a DataBatch or a list of strings')
        if not all((isinstance(text, str) for text in texts)):
            raise ValueError('All elements in the input data must be strings')
        stemmed_texts = []
        for text in texts:
            if not text.strip():
                stemmed_texts.append(text)
                continue
            words = re.findall('\\b\\w+\\b', text)
            stemmed_words = [self._stemmer.stem(word) for word in words]
            stemmed_text = text
            for i in range(len(words) - 1, -1, -1):
                original = words[i]
                stemmed = stemmed_words[i]
                if original != stemmed:
                    stemmed_text = stemmed_text.replace(original, stemmed, 1)
            stemmed_texts.append(stemmed_text)
        if is_batch:
            return DataBatch(data=stemmed_texts, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        else:
            return stemmed_texts

    def inverse_transform(self, data: Union[DataBatch, List[str]], **kwargs) -> Union[DataBatch, List[str]]:
        """
        Inverse transformation is not supported for text stemming.
        
        Parameters
        ----------
        data : Union[DataBatch, List[str]]
            Transformed data (not used)
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        Union[DataBatch, List[str]]
            Same as input data (identity operation)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not possible for
            stemming operations
        """
        raise NotImplementedError('Inverse transformation is not supported for stemming operations')


# ...(code omitted)...


class SnowballStemmerTransformer(BaseTransformer):
    """
    A transformer that applies the Snowball stemming algorithm to text data.
    
    This transformer implements the Snowball stemming algorithm, which is a
    standardized framework for stemming that supports multiple languages
    with language-specific rules. It's particularly effective for English
    and many European languages.
    
    Attributes
    ----------
    language : str
        Language code for the Snowball stemming algorithm (default: 'english')
    name : str
        Name of the transformer instance
    
    Methods
    -------
    fit(data, **kwargs)
        Fit the transformer to the input data (no-op for stemming)
    transform(data, **kwargs)
        Apply Snowball stemming to the input text data
    inverse_transform(data, **kwargs)
        Not supported for stemming operations
    """

    def __init__(self, language: str='english', name: str=None):
        """
        Initialize the SnowballStemmerTransformer.
        
        Parameters
        ----------
        language : str, optional
            Language code for Snowball stemming, by default 'english'
        name : str, optional
            Name for the transformer instance, by default None
            
        Raises
        ------
        ValueError
            If language is not supported by the Snowball stemmer
        """
        super().__init__(name=name)
        try:
            from nltk.stem import SnowballStemmer
            if language not in SnowballStemmer.languages:
                raise ValueError(f"Unsupported language '{language}'. Supported languages: {list(SnowballStemmer.languages)}")
        except ImportError:
            raise ImportError("NLTK is required for SnowballStemmerTransformer. Please install it with 'pip install nltk'")
        self.language = language
        self._stemmer = SnowballStemmer(language)

    def fit(self, data: Union[DataBatch, List[str]], **kwargs) -> 'SnowballStemmerTransformer':
        """
        Fit the transformer to the input data.
        
        For Snowball stemming, fitting is not required as no model
        needs to be learned. This method simply returns self.
        
        Parameters
        ----------
        data : Union[DataBatch, List[str]]
            Input text data to fit on. If DataBatch, expects text data in the
            primary data field.
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        SnowballStemmerTransformer
            Self instance for method chaining
        """
        return self

    def transform(self, data: Union[DataBatch, List[str]], **kwargs) -> Union[DataBatch, List[str]]:
        """
        Apply Snowball stemming to the input text data.
        
        Parameters
        ----------
        data : Union[DataBatch, List[str]]
            Input text data to transform. Can be a list of strings or DataBatch
            containing text data.
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        Union[DataBatch, List[str]]
            Text data with words reduced to their stems using the Snowball
            stemming algorithm. Returns the same type as input.
        """
        if isinstance(data, DataBatch):
            texts = data.data
            is_batch = True
        elif isinstance(data, list):
            texts = data
            is_batch = False
        else:
            raise ValueError('Input data must be either a DataBatch or a list of strings')
        if not all((isinstance(text, str) for text in texts)):
            raise ValueError('All elements in the input data must be strings')
        stemmed_texts = []
        for text in texts:
            if not text.strip():
                stemmed_texts.append(text)
                continue
            words = re.findall('\\b\\w+\\b', text)
            stemmed_words = [self._stemmer.stem(word) for word in words]
            stemmed_text = text
            for i in range(len(words) - 1, -1, -1):
                original = words[i]
                stemmed = stemmed_words[i]
                if original != stemmed:
                    stemmed_text = stemmed_text.replace(original, stemmed, 1)
            stemmed_texts.append(stemmed_text)
        if is_batch:
            return DataBatch(data=stemmed_texts, labels=data.labels, metadata=data.metadata, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
        else:
            return stemmed_texts

    def inverse_transform(self, data: Union[DataBatch, List[str]], **kwargs) -> Union[DataBatch, List[str]]:
        """
        Inverse transformation is not supported for text stemming.
        
        Parameters
        ----------
        data : Union[DataBatch, List[str]]
            Transformed data (not used)
        **kwargs : dict
            Additional parameters (not used)
            
        Returns
        -------
        Union[DataBatch, List[str]]
            Same as input data (identity operation)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not possible for
            Snowball stemming operations
        """
        raise NotImplementedError('Inverse transformation is not supported for Snowball stemming operations')