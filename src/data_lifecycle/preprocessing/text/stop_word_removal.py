from typing import List, Union, Optional
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
import numpy as np

class StopWordRemover(BaseTransformer):

    def __init__(self, stop_words: Optional[List[str]]=None, case_sensitive: bool=False, name: Optional[str]=None):
        """
        Initialize the StopWordRemover.
        
        Args:
            stop_words: Custom list of stop words to remove. If None, uses a default English stop word list.
            case_sensitive: If True, performs case-sensitive stop word matching.
            name: Optional name for the transformer.
        """
        super().__init__(name=name)
        self.stop_words = stop_words or self._get_default_stop_words()
        self.case_sensitive = case_sensitive

    def _get_default_stop_words(self) -> List[str]:
        """Get the default list of English stop words."""
        return ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']

    def fit(self, data: Union[FeatureSet, List[str]], **kwargs) -> 'StopWordRemover':
        """
        Fit the transformer to the input data.
        
        For stop word removal, fitting doesn't require learning from data,
        so this method primarily validates the input.
        
        Args:
            data: Input text data as a FeatureSet or list of strings.
            **kwargs: Additional parameters (unused).
            
        Returns:
            StopWordRemover: Returns self for method chaining.
        """
        if not isinstance(data, (FeatureSet, list)):
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        if isinstance(data, FeatureSet):
            if data.features.ndim != 1 and data.features.ndim != 2:
                raise ValueError('FeatureSet must contain 1D or 2D text data')
            if data.features.ndim == 1:
                texts = data.features
            else:
                texts = data.features.flatten()
            for item in texts:
                if not isinstance(item, str):
                    raise ValueError('All elements in the input data must be strings')
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, str):
                    raise ValueError('All elements in the input data must be strings')
        return self

    def transform(self, data: Union[FeatureSet, List[str]], **kwargs) -> FeatureSet:
        """
        Remove stop words from the input text data.
        
        Args:
            data: Input text data as a FeatureSet or list of strings.
            **kwargs: Additional parameters (unused).
            
        Returns:
            FeatureSet: Text data with stop words removed.
        """
        if not isinstance(data, (FeatureSet, list)):
            raise ValueError('Input data must be either a FeatureSet or a list of strings')
        if isinstance(data, FeatureSet):
            if data.features.ndim == 1:
                texts = data.features.tolist()
            elif data.features.ndim == 2 and data.features.shape[1] == 1:
                texts = data.features.flatten().tolist()
            else:
                raise ValueError('FeatureSet must contain 1D text data or 2D with single column')
        else:
            texts = data
        for item in texts:
            if not isinstance(item, str):
                raise ValueError('All elements in the input data must be strings')
        if self.case_sensitive:
            stop_words_set = set(self.stop_words)
        else:
            stop_words_set = {word.lower() for word in self.stop_words}
        processed_texts = []
        for text in texts:
            words = text.split()
            if self.case_sensitive:
                filtered_words = [word for word in words if word not in stop_words_set]
            else:
                filtered_words = [word for word in words if word.lower() not in stop_words_set]
            processed_texts.append(' '.join(filtered_words))
        return FeatureSet(features=np.array(processed_texts).reshape(-1, 1), feature_names=['cleaned_text'])

    def inverse_transform(self, data: FeatureSet, **kwargs) -> List[str]:
        """
        Inverse transformation is not supported for stop word removal.
        
        Args:
            data: Transformed data (unused).
            **kwargs: Additional parameters (unused).
            
        Returns:
            List[str]: This will raise an exception as inverse transform is not supported.
            
        Raises:
            NotImplementedError: Always raised as inverse transformation is not possible.
        """
        raise NotImplementedError('Inverse transformation is not supported for stop word removal.')