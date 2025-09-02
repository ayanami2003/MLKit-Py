from general.structures.feature_set import FeatureSet
from general.base_classes.transformer_base import BaseTransformer
from typing import List, Optional, Union
import numpy as np
import difflib

class SimilarityEncoder(BaseTransformer):

    def __init__(self, similarity_metric: str='levenshtein', handle_unknown: str='ignore', name: Optional[str]=None):
        """
        Initialize the SimilarityEncoder.
        
        Parameters
        ----------
        similarity_metric : str, default="levenshtein"
            The similarity metric to use for encoding. Supported metrics include
            "levenshtein", "jaro", "jaro_winkler", and "cosine".
        handle_unknown : str, default="ignore"
            How to handle unknown categories during transform. Options are
            "ignore" (return zeros) or "error" (raise an exception).
        name : Optional[str], default=None
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.similarity_metric = similarity_metric
        self.handle_unknown = handle_unknown
        self.categories: List[List[str]] = []
        self.similarity_matrices: List[np.ndarray] = []

    def _compute_similarity(self, str1: str, str2: str) -> float:
        """
        Compute similarity between two strings using the specified metric.
        
        Parameters
        ----------
        str1 : str
            First string for comparison.
        str2 : str
            Second string for comparison.
            
        Returns
        -------
        float
            Similarity score between 0 and 1.
        """
        if self.similarity_metric == 'levenshtein':
            return difflib.SequenceMatcher(None, str1, str2).ratio()
        elif self.similarity_metric == 'jaro':
            return difflib.SequenceMatcher(None, str1, str2).real_quick_ratio()
        elif self.similarity_metric == 'jaro_winkler':
            return difflib.SequenceMatcher(None, str1, str2).quick_ratio()
        else:
            raise ValueError(f'Unsupported similarity metric: {self.similarity_metric}')

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'SimilarityEncoder':
        """
        Fit the similarity encoder to the input data.
        
        This method computes the similarity matrices for each categorical feature
        based on the provided data.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode.
        **kwargs : dict
            Additional parameters for fitting.
            
        Returns
        -------
        SimilarityEncoder
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
        else:
            X = data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        self.categories = []
        self.similarity_matrices = []
        for i in range(n_features):
            unique_vals = np.unique(X[:, i])
            categories = [str(val) for val in unique_vals if not (isinstance(val, float) and np.isnan(val))]
            categories = sorted(categories)
            self.categories.append(categories)
            n_categories = len(categories)
            similarity_matrix = np.zeros((n_categories, n_categories))
            for j in range(n_categories):
                for k in range(n_categories):
                    if j == k:
                        similarity_matrix[j, k] = 1.0
                    else:
                        similarity_matrix[j, k] = self._compute_similarity(categories[j], categories[k])
            self.similarity_matrices.append(similarity_matrix)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the input data using the fitted similarity encoder.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation.
            
        Returns
        -------
        FeatureSet
            Transformed data with similarity-encoded features.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if n_features != len(self.categories):
            raise ValueError(f'Number of features in transform data ({n_features}) does not match fitted data ({len(self.categories)})')
        X_str = np.array([[str(val) if not (isinstance(val, float) and np.isnan(val)) else 'nan' for val in row] for row in X])
        transformed_features_list = []
        transformed_feature_names = []
        for feature_idx in range(n_features):
            categories = self.categories[feature_idx]
            similarity_matrix = self.similarity_matrices[feature_idx]
            n_categories = len(categories)
            if feature_names and feature_idx < len(feature_names):
                base_name = feature_names[feature_idx]
            else:
                base_name = f'feature_{feature_idx}'
            feature_vectors = []
            for sample_idx in range(n_samples):
                category_val = X_str[sample_idx, feature_idx]
                if category_val in categories:
                    category_idx = categories.index(category_val)
                    similarity_vector = similarity_matrix[category_idx]
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{category_val}' encountered in feature {feature_idx}")
                else:
                    similarity_vector = np.zeros(n_categories)
                feature_vectors.append(similarity_vector)
            if feature_vectors:
                feature_vectors = np.array(feature_vectors)
                transformed_features_list.append(feature_vectors)
            for (cat_idx, category) in enumerate(categories):
                transformed_feature_names.append(f'{base_name}_{category}')
        if transformed_features_list:
            X_transformed = np.hstack(transformed_features_list)
        else:
            X_transformed = np.empty((n_samples, 0))
        feature_types = ['numeric'] * len(transformed_feature_names)
        return FeatureSet(features=X_transformed, feature_names=transformed_feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert similarity-encoded data back to original categorical values.
        
        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Similarity-encoded data to convert back.
        **kwargs : dict
            Additional parameters for inverse transformation.
            
        Returns
        -------
        FeatureSet
            Data with categorical values restored.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        reconstructed_features = []
        reconstructed_feature_names = []
        feature_idx = 0
        original_feature_idx = 0
        for (categories, similarity_matrix) in zip(self.categories, self.similarity_matrices):
            n_categories = len(categories)
            if feature_idx + n_categories <= n_features:
                feature_block = X[:, feature_idx:feature_idx + n_categories]
                sample_categories = []
                for sample_idx in range(n_samples):
                    similarity_vector = feature_block[sample_idx]
                    if np.all(similarity_vector == 0):
                        sample_categories.append('unknown')
                    else:
                        closest_idx = np.argmax(similarity_vector)
                        sample_categories.append(categories[closest_idx])
                reconstructed_features.append(sample_categories)
                reconstructed_feature_names.append(f'original_feature_{original_feature_idx}')
                feature_idx += n_categories
                original_feature_idx += 1
            else:
                break
        if reconstructed_features:
            X_reconstructed = np.array(reconstructed_features).T
        else:
            X_reconstructed = np.empty((n_samples, 0))
        feature_types = ['categorical'] * len(reconstructed_feature_names)
        return FeatureSet(features=X_reconstructed, feature_names=reconstructed_feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

class StringKernelSimilarityEncoder(BaseTransformer):

    def __init__(self, kernel_type: str='spectrum', handle_unknown: str='ignore', name: Optional[str]=None, **kernel_params):
        """
        Initialize the StringKernelSimilarityEncoder.
        
        Parameters
        ----------
        kernel_type : str, default="spectrum"
            The type of string kernel to use. Supported types include
            "spectrum", "mismatch", and "substring".
        handle_unknown : str, default="ignore"
            How to handle unknown categories during transform. Options are
            "ignore" (return zeros) or "error" (raise ValueError).
        name : Optional[str], default=None
            Name of the transformer instance.
        **kernel_params : dict
            Additional parameters for the string kernel computation.
            For spectrum kernel: {"k": int} where k is the subsequence length.
            For mismatch kernel: {"k": int, "m": int} where m is the max mismatches.
            For substring kernel: {"l": int} where l is the substring length.
            
        Raises
        ------
        ValueError
            If handle_unknown is set to "error" and unknown categories are encountered during transform.
        """
        super().__init__(name=name)
        if kernel_type not in ['spectrum', 'mismatch', 'substring']:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}. Supported types: 'spectrum', 'mismatch', 'substring'")
        if handle_unknown not in ['ignore', 'error']:
            raise ValueError(f"Unsupported handle_unknown: {handle_unknown}. Options: 'ignore', 'error'")
        self.kernel_type = kernel_type
        self.handle_unknown = handle_unknown
        self.kernel_params = kernel_params
        self.categories: List[List[str]] = []
        self.kernel_matrices: List[np.ndarray] = []
        self.feature_names_: List[str] = []

    def _compute_string_kernel(self, str1: str, str2: str) -> float:
        """
        Compute string kernel similarity between two strings.
        
        Parameters
        ----------
        str1 : str
            First string for comparison.
        str2 : str
            Second string for comparison.
            
        Returns
        -------
        float
            Kernel similarity score.
        """
        if self.kernel_type == 'spectrum':
            k = self.kernel_params.get('k', 2)
            return self._spectrum_kernel(str1, str2, k)
        elif self.kernel_type == 'mismatch':
            k = self.kernel_params.get('k', 2)
            m = self.kernel_params.get('m', 1)
            return self._mismatch_kernel(str1, str2, k, m)
        elif self.kernel_type == 'substring':
            l = self.kernel_params.get('l', 2)
            return self._substring_kernel(str1, str2, l)
        else:
            raise ValueError(f'Unsupported kernel_type: {self.kernel_type}')

    def _spectrum_kernel(self, str1: str, str2: str, k: int) -> float:
        """
        Compute spectrum string kernel (k-mer matching).
        
        Parameters
        ----------
        str1 : str
            First string.
        str2 : str
            Second string.
        k : int
            Length of substrings to consider.
            
        Returns
        -------
        float
            Spectrum kernel similarity.
        """
        if len(str1) < k or len(str2) < k:
            return 0.0
        if k <= 0:
            return 0.0

        def get_kmers(s, k):
            return [s[i:i + k] for i in range(len(s) - k + 1)] if len(s) >= k else []
        kmers1 = get_kmers(str1, k)
        kmers2 = get_kmers(str2, k)
        if not kmers1 or not kmers2:
            return 0.0
        from collections import Counter
        count1 = Counter(kmers1)
        count2 = Counter(kmers2)
        result = 0.0
        for kmer in set(count1.keys()) | set(count2.keys()):
            result += count1.get(kmer, 0) * count2.get(kmer, 0)
        if result == 0.0:
            return 0.0
        norm1 = sum((v * v for v in count1.values())) ** 0.5
        norm2 = sum((v * v for v in count2.values())) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return result / (norm1 * norm2)

    def _mismatch_kernel(self, str1: str, str2: str, k: int, m: int) -> float:
        """
        Compute mismatch string kernel.
        
        Parameters
        ----------
        str1 : str
            First string.
        str2 : str
            Second string.
        k : int
            Length of substrings to consider.
        m : int
            Maximum number of mismatches allowed.
            
        Returns
        -------
        float
            Mismatch kernel similarity.
        """
        if len(str1) < k or len(str2) < k:
            return 0.0
        if k <= 0 or m < 0:
            return 0.0

        def get_kmers(s, k):
            return [s[i:i + k] for i in range(len(s) - k + 1)] if len(s) >= k else []

        def count_mismatches(s1, s2):
            return sum((c1 != c2 for (c1, c2) in zip(s1, s2)))
        kmers1 = get_kmers(str1, k)
        kmers2 = get_kmers(str2, k)
        if not kmers1 or not kmers2:
            return 0.0
        result = 0
        for kmer1 in kmers1:
            for kmer2 in kmers2:
                if count_mismatches(kmer1, kmer2) <= m:
                    result += 1
        if result == 0:
            return 0.0
        norm = (len(kmers1) * len(kmers2)) ** 0.5
        return result / norm if norm != 0 else 0.0

    def _substring_kernel(self, str1: str, str2: str, l: int) -> float:
        """
        Compute substring kernel (all substrings up to length l).
        
        Parameters
        ----------
        str1 : str
            First string.
        str2 : str
            Second string.
        l : int
            Maximum substring length to consider.
            
        Returns
        -------
        float
            Substring kernel similarity.
        """

        def get_all_substrings(s, max_len):
            substrings = []
            for length in range(1, min(len(s), max_len) + 1):
                for i in range(len(s) - length + 1):
                    substrings.append(s[i:i + length])
            return substrings
        substrings1 = get_all_substrings(str1, l)
        substrings2 = get_all_substrings(str2, l)
        if not substrings1 or not substrings2:
            return 0.0
        from collections import Counter
        count1 = Counter(substrings1)
        count2 = Counter(substrings2)
        result = 0.0
        for substring in set(count1.keys()) | set(count2.keys()):
            result += count1.get(substring, 0) * count2.get(substring, 0)
        norm1 = sum((v * v for v in count1.values())) ** 0.5
        norm2 = sum((v * v for v in count2.values())) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return result / (norm1 * norm2)

    def fit(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> 'StringKernelSimilarityEncoder':
        """
        Fit the string kernel similarity encoder to the input data.

        This method computes the string kernel matrices for each categorical feature
        based on the provided data.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data containing categorical features to encode.
        **kwargs : dict
            Additional parameters for fitting.

        Returns
        -------
        StringKernelSimilarityEncoder
            Self instance for method chaining.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            self.feature_names_ = data.feature_names if data.feature_names else []
        else:
            X = data
            self.feature_names_ = []
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]
        self.categories = []
        self.kernel_matrices = []
        for i in range(n_features):
            unique_vals = np.unique(X[:, i])
            categories = [str(val) for val in unique_vals if not (isinstance(val, float) and np.isnan(val))]
            categories = sorted(categories)
            self.categories.append(categories)
            n_categories = len(categories)
            kernel_matrix = np.zeros((n_categories, n_categories))
            for j in range(n_categories):
                for k in range(n_categories):
                    if j == k:
                        kernel_matrix[j, k] = 1.0
                    else:
                        kernel_matrix[j, k] = self._compute_string_kernel(categories[j], categories[k])
            self.kernel_matrices.append(kernel_matrix)
        return self

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform the input data using the fitted string kernel similarity encoder.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            Input data to transform.
        **kwargs : dict
            Additional parameters for transformation.

        Returns
        -------
        FeatureSet
            Transformed data with string kernel similarity-encoded features.
            
        Raises
        ------
        ValueError
            If handle_unknown is set to "error" and unknown categories are encountered.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        if n_features != len(self.categories):
            raise ValueError(f'Number of features in transform data ({n_features}) does not match fitted data ({len(self.categories)})')
        X_str = np.array([[str(val) if not (isinstance(val, float) and np.isnan(val)) else 'nan' for val in row] for row in X])
        transformed_features_list = []
        transformed_feature_names = []
        for feature_idx in range(n_features):
            categories = self.categories[feature_idx]
            kernel_matrix = self.kernel_matrices[feature_idx]
            n_categories = len(categories)
            if feature_names and feature_idx < len(feature_names):
                base_name = feature_names[feature_idx]
            else:
                base_name = f'feature_{feature_idx}'
            feature_vectors = []
            for sample_idx in range(n_samples):
                category_val = X_str[sample_idx, feature_idx]
                if category_val in categories:
                    category_idx = categories.index(category_val)
                    kernel_vector = kernel_matrix[category_idx]
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{category_val}' encountered in feature {feature_idx}")
                else:
                    kernel_vector = np.zeros(n_categories)
                feature_vectors.append(kernel_vector)
            if feature_vectors:
                feature_vectors = np.array(feature_vectors)
                transformed_features_list.append(feature_vectors)
            for (cat_idx, category) in enumerate(categories):
                transformed_feature_names.append(f'{base_name}_{category}')
        if transformed_features_list:
            X_transformed = np.hstack(transformed_features_list)
        else:
            X_transformed = np.empty((n_samples, 0))
        feature_types = ['numeric'] * len(transformed_feature_names)
        return FeatureSet(features=X_transformed, feature_names=transformed_feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert string kernel similarity-encoded data back to original categorical values.

        Note that this operation may not perfectly reconstruct the original
        categories due to the lossy nature of similarity encoding. The reconstruction
        is performed by finding the category with the highest kernel similarity score.

        Parameters
        ----------
        data : Union[FeatureSet, np.ndarray]
            String kernel similarity-encoded data to convert back.
        **kwargs : dict
            Additional parameters for inverse transformation.

        Returns
        -------
        FeatureSet
            Data with approximate categorical values restored.
        """
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names
            sample_ids = data.sample_ids
            metadata = data.metadata
            quality_scores = data.quality_scores
        else:
            X = data
            feature_names = None
            sample_ids = None
            metadata = None
            quality_scores = None
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        (n_samples, n_features) = X.shape
        reconstructed_features = []
        reconstructed_feature_names = []
        feature_idx = 0
        original_feature_idx = 0
        for (categories, kernel_matrix) in zip(self.categories, self.kernel_matrices):
            n_categories = len(categories)
            if feature_idx + n_categories <= n_features:
                feature_block = X[:, feature_idx:feature_idx + n_categories]
                sample_categories = []
                for sample_idx in range(n_samples):
                    kernel_vector = feature_block[sample_idx]
                    if np.all(kernel_vector == 0):
                        sample_categories.append('unknown')
                    else:
                        similarities = np.dot(kernel_matrix, kernel_vector)
                        closest_idx = np.argmax(similarities)
                        sample_categories.append(categories[closest_idx])
                reconstructed_features.append(sample_categories)
                if original_feature_idx < len(self.feature_names_):
                    reconstructed_feature_names.append(self.feature_names_[original_feature_idx])
                else:
                    reconstructed_feature_names.append(f'feature_{original_feature_idx}')
                feature_idx += n_categories
                original_feature_idx += 1
            else:
                break
        if reconstructed_features:
            X_reconstructed = np.array(reconstructed_features).T
        else:
            X_reconstructed = np.empty((n_samples, 0))
        feature_types = ['categorical'] * len(reconstructed_feature_names)
        return FeatureSet(features=X_reconstructed, feature_names=reconstructed_feature_names, feature_types=feature_types, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)