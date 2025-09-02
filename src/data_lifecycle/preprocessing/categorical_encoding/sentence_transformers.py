from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union
import numpy as np

class SentenceTransformerExtractor(BaseTransformer):

    def __init__(self, model_name: str, max_length: Optional[int]=None, batch_size: int=32, normalize_embeddings: bool=True, device: Optional[str]=None, name: Optional[str]=None):
        super().__init__(name=name)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.model = None
        self.fitted = False

    def fit(self, data: Union[FeatureSet, List[str], np.ndarray], **kwargs) -> 'SentenceTransformerExtractor':
        """
        Load the sentence transformer model.
        
        Args:
            data: Input text data to fit on. Can be a FeatureSet containing text columns,
                  a list of strings, or a numpy array of strings.
            **kwargs: Additional fitting parameters.
            
        Returns:
            SentenceTransformerExtractor: Returns self for method chaining.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("The 'sentence-transformers' package is required for SentenceTransformerExtractor. Please install it using 'pip install sentence-transformers'.")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError('model_name must be a non-empty string')
        if self.device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            raise ValueError(f"Failed to load model '{self.model_name}': {str(e)}")
        if self.max_length is None:
            try:
                self.max_length = self.model.get_max_seq_length()
            except:
                self.max_length = 512
        self.fitted = True
        return self

    def transform(self, data: Union[FeatureSet, List[str], np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform text data into sentence embeddings.
        
        Args:
            data: Input text data to transform. Can be a FeatureSet containing text columns,
                  a list of strings, or a numpy array of strings.
            **kwargs: Additional transformation parameters.
            
        Returns:
            FeatureSet: FeatureSet containing the extracted embeddings as features.
            
        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self.fitted:
            raise ValueError("Transformer has not been fitted. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            if data.features.ndim == 1:
                texts = data.features.tolist()
            else:
                texts = [' '.join(map(str, row)) for row in data.features]
        elif isinstance(data, list):
            texts = [str(item) for item in data]
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                texts = [str(item) for item in data]
            else:
                texts = [' '.join(map(str, row)) for row in data]
        else:
            raise TypeError('Input data must be a FeatureSet, list of strings, or numpy array of strings.')
        if not texts:
            raise ValueError('Input data is empty.')
        try:
            from src.data_lifecycle.computational_utilities.memory_management.chunked_processing import process_in_chunks
            embeddings_list = process_in_chunks(data=texts, chunk_size=self.batch_size, processor=lambda chunk: self.model.encode(sentences=chunk, batch_size=len(chunk), show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings))
            embeddings = np.vstack(embeddings_list)
        except Exception as e:
            raise RuntimeError(f'Error during embedding generation: {str(e)}')
        n_samples = len(texts)
        n_features = embeddings.shape[1]
        feature_names = [f'sentence_embedding_{i}' for i in range(n_features)]
        sample_ids = data.sample_ids if isinstance(data, FeatureSet) and data.sample_ids else [f'sample_{i}' for i in range(n_samples)]
        metadata = {'model_name': self.model_name, 'device': self.device, 'max_length': self.max_length, 'normalize_embeddings': self.normalize_embeddings, 'source': 'SentenceTransformerExtractor'}
        return FeatureSet(features=embeddings, feature_names=feature_names, feature_types=['numeric'] * n_features, sample_ids=sample_ids, metadata=metadata)

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for sentence transformers.
        
        Args:
            data: FeatureSet containing embeddings to inverse transform.
            **kwargs: Additional parameters.
            
        Returns:
            FeatureSet: The original FeatureSet unchanged.
            
        Raises:
            NotImplementedError: Always raised as inverse transformation is not supported.
        """
        pass

class CategoricalSentenceEncoder(BaseTransformer):
    """
    Encode categorical variables into dense vector representations using sentence transformers.
    
    This encoder transforms categorical text values into semantically meaningful embeddings,
    which can better capture relationships between categories than traditional encoding methods.
    It is especially effective for high-cardinality categorical features with textual descriptions.
    
    The encoder treats input as discrete categories and learns fixed embeddings for each unique value.
    During transform, it maps categories to their corresponding pre-computed embeddings.
    
    Args:
        model_name (str): Name of the pre-trained sentence transformer model to use.
        categorical_column (str): Name of the column containing categorical values to encode.
        encoded_column_prefix (str): Prefix for naming the new encoded embedding columns. 
                                     Defaults to 'category_embedding_'.
        max_length (Optional[int]): Maximum sequence length for tokenization.
        batch_size (int): Batch size for computing embeddings. Defaults to 32.
        normalize_embeddings (bool): Whether to normalize the output embeddings. Defaults to True.
        device (Optional[str]): Device to run the model on ('cpu' or 'cuda').
        handle_unknown (str): How to handle unknown categories during transform. 
                              Either 'error' or 'ignore'. Defaults to 'error'.
        name (Optional[str]): Name for the encoder instance.
        
    Attributes:
        model: Loaded sentence transformer model.
        categories_ (List[str]): Unique categories seen during fitting.
        embeddings_ (np.ndarray): Embeddings for each category.
        fitted (bool): Whether the encoder has been fitted.
        
    Example:
        >>> encoder = CategoricalSentenceEncoder('all-MiniLM-L6-v2', 
        ...                                      categorical_column='product_category')
        >>> encoded = encoder.fit_transform(categorical_feature_set)
    """

    def __init__(self, model_name: str, categorical_column: str, encoded_column_prefix: str='category_embedding_', max_length: Optional[int]=None, batch_size: int=32, normalize_embeddings: bool=True, device: Optional[str]=None, handle_unknown: str='error', name: Optional[str]=None):
        super().__init__(name=name)
        self.model_name = model_name
        self.categorical_column = categorical_column
        self.encoded_column_prefix = encoded_column_prefix
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.handle_unknown = handle_unknown
        self.model = None
        self.categories_ = []
        self.embeddings_ = None
        self.fitted = False

    def fit(self, data: FeatureSet, **kwargs) -> 'CategoricalSentenceEncoder':
        """
        Fit the encoder by computing embeddings for unique categorical values.
        
        Args:
            data: Input FeatureSet containing the categorical column to fit on.
            **kwargs: Additional fitting parameters.
            
        Returns:
            CategoricalSentenceEncoder: Returns self for method chaining.
            
        Raises:
            ValueError: If the specified categorical column is not found in the FeatureSet.
        """
        if self.categorical_column not in data.data.columns:
            raise ValueError(f"Column '{self.categorical_column}' not found in the FeatureSet.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError('sentence-transformers is required for CategoricalSentenceEncoder but not installed.')
        self.model = SentenceTransformer(self.model_name)
        if self.device:
            self.model = self.model.to(self.device)
        self.categories_ = list(data.data[self.categorical_column].dropna().unique())
        embeddings_list = []
        for i in range(0, len(self.categories_), self.batch_size):
            batch = self.categories_[i:i + self.batch_size]
            if self.max_length:
                batch = [text[:self.max_length] for text in batch]
            embeddings = self.model.encode(batch, normalize_embeddings=self.normalize_embeddings)
            embeddings_list.append(embeddings)
        if embeddings_list:
            self.embeddings_ = np.vstack(embeddings_list)
        else:
            self.embeddings_ = np.array([]).reshape(0, self.model.get_sentence_embedding_dimension())
        self.fitted = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform categorical values into embeddings by mapping categories to their learned embeddings.
        
        Args:
            data: Input FeatureSet containing the categorical column to transform.
            **kwargs: Additional transformation parameters.
            
        Returns:
            FeatureSet: FeatureSet with the categorical column replaced by embedding features.
            
        Raises:
            ValueError: If the encoder has not been fitted or encounters unknown categories
                        when handle_unknown is set to 'error'.
        """
        if not self.fitted:
            raise ValueError("This CategoricalSentenceEncoder instance is not fitted yet. Call 'fit' before using this transformer.")
        if self.categorical_column not in data.data.columns:
            raise ValueError(f"Column '{self.categorical_column}' not found in the FeatureSet.")
        category_to_embedding = dict(zip(self.categories_, self.embeddings_))
        embedding_dim = self.embeddings_.shape[1] if self.embeddings_.size > 0 else 0
        unknown_categories = set()
        embedding_columns = [f'{self.encoded_column_prefix}{i}' for i in range(embedding_dim)]
        result_data = data.data.copy()
        for col in embedding_columns:
            result_data[col] = np.nan
        for (idx, category) in result_data[self.categorical_column].items():
            if category in category_to_embedding:
                embedding = category_to_embedding[category]
                for (i, col) in enumerate(embedding_columns):
                    result_data.at[idx, col] = embedding[i]
            elif category is not None:
                unknown_categories.add(category)
                if self.handle_unknown == 'error':
                    pass
        if self.handle_unknown == 'error' and unknown_categories:
            raise ValueError(f'Found unknown categories: {unknown_categories}')
        result_data = result_data.drop(columns=[self.categorical_column])
        result_feature_set = FeatureSet(data=result_data, target=data.target.copy() if data.target is not None else None, feature_names=[col for col in result_data.columns if col != (data.target.name if data.target is not None else None)])
        return result_feature_set

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Map embeddings back to original categorical values using nearest neighbor lookup.
        
        Finds the closest embedding in the learned embedding space and maps back to the 
        corresponding category label.
        
        Args:
            data: FeatureSet containing embeddings to inverse transform.
            **kwargs: Additional parameters.
            
        Returns:
            FeatureSet: FeatureSet containing the reconstructed categorical values.
            
        Raises:
            ValueError: If the encoder has not been fitted.
        """
        if not self.fitted:
            raise ValueError("This CategoricalSentenceEncoder instance is not fitted yet. Call 'fit' before using this transformer.")
        embedding_dim = self.embeddings_.shape[1] if self.embeddings_.size > 0 else 0
        embedding_columns = [f'{self.encoded_column_prefix}{i}' for i in range(embedding_dim)]
        missing_cols = [col for col in embedding_columns if col not in data.data.columns]
        if missing_cols:
            raise ValueError(f'Missing embedding columns in the input FeatureSet: {missing_cols}')
        result_data = data.data.copy()
        result_data[self.categorical_column] = None
        for idx in result_data.index:
            embedding = result_data.loc[idx, embedding_columns].values
            if np.isnan(embedding).any():
                continue
            distances = np.linalg.norm(self.embeddings_ - embedding, axis=1)
            closest_idx = np.argmin(distances)
            result_data.at[idx, self.categorical_column] = self.categories_[closest_idx]
        result_data = result_data.drop(columns=embedding_columns)
        result_feature_set = FeatureSet(data=result_data, target=data.target.copy() if data.target is not None else None, feature_names=[col for col in result_data.columns if col != (data.target.name if data.target is not None else None)])
        return result_feature_set