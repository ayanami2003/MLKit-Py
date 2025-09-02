from typing import Union, List, Optional, Any
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import numpy as np


# ...(code omitted)...


class SentimentAnalyzer(BaseTransformer):
    """
    A transformer that computes sentiment scores for text data.
    
    This component analyzes textual content and assigns sentiment scores,
    typically ranging from negative to positive sentiment strength. It can
    be used as part of text preprocessing pipelines to enrich text data
    with sentiment information.
    
    The analyzer supports various sentiment analysis approaches and can
    output different types of sentiment representations (polarity scores,
    categorical labels, etc.) depending on configuration.
    
    Attributes
    ----------
    method : str, optional
        The sentiment analysis method to use (e.g., 'vader', 'textblob')
    return_type : str, optional
        Type of sentiment output ('score', 'label', or 'full')
    
    Examples
    --------
    >>> analyzer = SentimentAnalyzer(method='vader')
    >>> text_data = ["I love this product!", "This is terrible."]
    >>> result = analyzer.transform(text_data)
    >>> print(result.features)
    [[0.7], [-0.6]]
    """

    def __init__(self, method: str='vader', return_type: str='score', name: Optional[str]=None):
        """
        Initialize the SentimentAnalyzer.
        
        Parameters
        ----------
        method : str, default='vader'
            The sentiment analysis method to use. Supported methods include
            'vader' for VADER sentiment analyzer and 'textblob' for TextBlob approach.
        return_type : str, default='score'
            Type of sentiment output to return:
            - 'score': Numerical sentiment score (e.g., -1 to 1)
            - 'label': Categorical label (e.g., 'positive', 'negative', 'neutral')
            - 'full': Complete sentiment analysis results
        name : str, optional
            Name for the transformer instance
        """
        super().__init__(name)
        self.method = method
        self.return_type = return_type
        self._validate_parameters()
        self._fitted = False

    def _validate_parameters(self):
        """Validate initialization parameters."""
        supported_methods = ['vader', 'textblob']
        if self.method not in supported_methods:
            raise ValueError(f"Unsupported method '{self.method}'. Supported methods are: {supported_methods}")
        supported_return_types = ['score', 'label', 'full']
        if self.return_type not in supported_return_types:
            raise ValueError(f"Unsupported return_type '{self.return_type}'. Supported types are: {supported_return_types}")

    def _extract_text_data(self, data: Union[FeatureSet, List[str], DataBatch]) -> List[str]:
        """
        Extract text data from various input formats.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str], DataBatch]
            Input data in various formats
            
        Returns
        -------
        List[str]
            Extracted text data as list of strings
            
        Raises
        ------
        ValueError
            If input data format is not supported or text data cannot be extracted
        """
        if isinstance(data, list):
            if all((isinstance(item, str) for item in data)):
                return data
            else:
                raise ValueError('All elements in the list must be strings')
        elif isinstance(data, FeatureSet):
            if data.features.ndim == 1:
                if all((isinstance(item, str) for item in data.features)):
                    return list(data.features)
                else:
                    raise ValueError('FeatureSet features must contain strings')
            elif data.features.ndim == 2 and data.features.shape[1] == 1:
                flattened = data.features.flatten()
                if all((isinstance(item, str) for item in flattened)):
                    return list(flattened)
                else:
                    raise ValueError('FeatureSet features must contain strings')
            else:
                raise ValueError('FeatureSet must contain text data in 1D array or 2D array with single column')
        elif isinstance(data, DataBatch):
            if isinstance(data.data, list):
                if all((isinstance(item, str) for item in data.data)):
                    return data.data
                else:
                    raise ValueError('DataBatch data must contain strings')
            else:
                raise ValueError('DataBatch data must be a list of strings')
        else:
            raise ValueError(f'Unsupported data type: {type(data)}. Supported types are: List[str], FeatureSet, DataBatch')

    def _get_sentiment_analyzer(self):
        """
        Initialize and return the appropriate sentiment analyzer.
        
        Returns
        -------
        Callable
            Function that computes sentiment for a text string
            
        Raises
        ------
        ImportError
            If required dependencies are not installed
        """
        if self.method == 'vader':
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                if self.return_type == 'score':
                    return lambda text: analyzer.polarity_scores(text)['compound']
                elif self.return_type == 'label':

                    def classify_sentiment(text):
                        score = analyzer.polarity_scores(text)['compound']
                        if score >= 0.05:
                            return 'positive'
                        elif score <= -0.05:
                            return 'negative'
                        else:
                            return 'neutral'
                    return classify_sentiment
                else:
                    return analyzer.polarity_scores
            except ImportError:
                raise ImportError("vaderSentiment is required for 'vader' method. Install it with: pip install vaderSentiment")
        elif self.method == 'textblob':
            try:
                from textblob import TextBlob
                if self.return_type == 'score':
                    return lambda text: TextBlob(text).sentiment.polarity
                elif self.return_type == 'label':

                    def classify_sentiment(text):
                        polarity = TextBlob(text).sentiment.polarity
                        if polarity > 0:
                            return 'positive'
                        elif polarity < 0:
                            return 'negative'
                        else:
                            return 'neutral'
                    return classify_sentiment
                else:
                    return lambda text: TextBlob(text).sentiment
            except ImportError:
                raise ImportError("textblob is required for 'textblob' method. Install it with: pip install textblob")

    def fit(self, data: Union[FeatureSet, List[str], DataBatch], **kwargs) -> 'SentimentAnalyzer':
        """
        Fit the sentiment analyzer to the input data.
        
        For rule-based sentiment analyzers, this method typically
        performs no actual fitting but validates input data format.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str], DataBatch]
            Input text data to fit on. Can be:
            - List of strings
            - FeatureSet with text data
            - DataBatch containing text data
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        SentimentAnalyzer
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If input data format is not supported
        """
        self._extract_text_data(data)
        self._fitted = True
        return self

    def transform(self, data: Union[FeatureSet, List[str], DataBatch], **kwargs) -> FeatureSet:
        """
        Compute sentiment scores for the input text data.
        
        Parameters
        ----------
        data : Union[FeatureSet, List[str], DataBatch]
            Input text data to analyze. Can be:
            - List of strings
            - FeatureSet with text data
            - DataBatch containing text data
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        FeatureSet
            FeatureSet containing sentiment scores with:
            - features: Array of sentiment scores
            - feature_names: ['sentiment_score'] or similar
            - metadata: Information about the analysis
            
        Raises
        ------
        ValueError
            If input data format is not supported or if analyzer is not fitted
        """
        if not self._fitted:
            raise ValueError('SentimentAnalyzer must be fitted before transform. Call fit() first.')
        text_data = self._extract_text_data(data)
        analyzer_func = self._get_sentiment_analyzer()
        sentiment_results = []
        for text in text_data:
            try:
                result = analyzer_func(text)
                sentiment_results.append(result)
            except Exception as e:
                raise RuntimeError(f"Error computing sentiment for text '{text[:50]}...': {str(e)}")
        if self.return_type == 'score':
            features = np.array(sentiment_results).reshape(-1, 1)
            feature_names = ['sentiment_score']
        elif self.return_type == 'label':
            unique_labels = sorted(list(set(sentiment_results)))
            label_to_num = {label: i for (i, label) in enumerate(unique_labels)}
            numerical_results = [label_to_num[label] for label in sentiment_results]
            features = np.array(numerical_results).reshape(-1, 1)
            feature_names = ['sentiment_label']
        else:
            features = np.array(sentiment_results).reshape(-1, 1)
            feature_names = ['sentiment_full_result']
        metadata = {'sentiment_method': self.method, 'sentiment_return_type': self.return_type, 'sentiment_analyzer_fitted': self._fitted}
        return FeatureSet(features=features, feature_names=feature_names, feature_types=['numerical'] if self.return_type in ['score', 'label'] else ['object'], metadata=metadata)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[str]:
        """
        Inverse transformation is not supported for sentiment analysis.
        
        Parameters
        ----------
        data : FeatureSet
            FeatureSet containing sentiment scores
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        List[str]
            Empty list as inverse transformation is not applicable
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for sentiment scores
        """
        raise NotImplementedError('Inverse transformation is not supported for SentimentAnalyzer as sentiment scores cannot be converted back to original text.')