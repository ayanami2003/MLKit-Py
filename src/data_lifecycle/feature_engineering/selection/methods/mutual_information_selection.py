from typing import Optional, Union
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch

class MutualInformationSelector(BaseTransformer):

    def __init__(self, n_features: Optional[int]=None, method: str='continuous', threshold: float=0.0, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the MutualInformationSelector.
        
        Args:
            n_features (Optional[int]): Number of top features to select based on MI scores.
                If None, features with MI score >= threshold will be selected.
            method (str): Estimation method for mutual information ('continuous' or 'discrete').
                'continuous' uses kernel density estimation, while 'discrete' uses contingency tables.
            threshold (float): Minimum mutual information score for feature selection when
                n_features is None. Features with scores below this value are discarded.
            random_state (Optional[int]): Controls randomness in sampling-based MI computations.
            name (Optional[str]): Name identifier for the transformer instance.
        """
        super().__init__(name=name)
        if method not in ['continuous', 'discrete']:
            raise ValueError(f"Method '{method}' is not supported. Supported methods: 'continuous', 'discrete'")
        self.n_features = n_features
        self.method = method
        self.threshold = threshold
        self.random_state = random_state
        self._mi_scores = None
        self._selected_indices = None

    def fit(self, data: Union[FeatureSet, DataBatch], **kwargs) -> 'MutualInformationSelector':
        """
        Compute mutual information scores between features and target labels.
        
        This method calculates the mutual information between each feature in the input data
        and the corresponding target labels. The computed scores are stored internally and
        used during the transform phase to select relevant features.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data containing features and labels.
                Must have a labeled structure where labels are accessible via data.labels.
            **kwargs: Additional keyword arguments (ignored).
                
        Returns:
            MutualInformationSelector: Returns self for method chaining.
            
        Raises:
            ValueError: If data does not contain labels or if method is not supported.
        """
        if isinstance(data, FeatureSet):
            if data.metadata is None or 'labels' not in data.metadata:
                raise ValueError('FeatureSet must contain labels in metadata for supervised feature selection')
            y = data.metadata['labels']
            X = data.features
        elif isinstance(data, DataBatch):
            if not data.is_labeled():
                raise ValueError('DataBatch must contain labels for supervised feature selection')
            y = data.labels
            X = np.array(data.data)
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        if self.method == 'continuous':
            self._mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        else:
            n_features = X.shape[1]
            self._mi_scores = np.zeros(n_features)
            for i in range(n_features):
                self._mi_scores[i] = mutual_info_score(X[:, i], y)
        if self.n_features is not None:
            if self.n_features <= 0:
                self._selected_indices = np.array([], dtype=int)
            elif self.n_features > len(self._mi_scores):
                self._selected_indices = np.argsort(self._mi_scores)[::-1]
            else:
                self._selected_indices = np.argsort(self._mi_scores)[-self.n_features:][::-1]
        else:
            self._selected_indices = np.where(self._mi_scores >= self.threshold)[0]
            if len(self._selected_indices) == 0 and len(self._mi_scores) > 0:
                self._selected_indices = np.array([np.argmax(self._mi_scores)])
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> FeatureSet:
        """
        Select features based on previously computed mutual information scores.
        
        This method applies feature selection by retaining only those features whose
        mutual information scores meet the selection criteria (either top-k or threshold).
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data to transform. Should contain
                the same features as used during fitting.
            **kwargs: Additional keyword arguments (ignored).
                
        Returns:
            FeatureSet: New FeatureSet containing only the selected features.
            
        Raises:
            RuntimeError: If called before fitting the transformer.
        """
        if not self._is_fitted:
            raise RuntimeError('Transformer must be fitted before calling transform()')
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.metadata.get('feature_names') if data.metadata else None
        elif isinstance(data, DataBatch):
            X = np.array(data.data)
            feature_names = data.feature_names
        else:
            raise TypeError('Input data must be either FeatureSet or DataBatch')
        X_selected = X[:, self._selected_indices]
        if feature_names:
            selected_feature_names = [feature_names[i] for i in self._selected_indices]
        else:
            selected_feature_names = None
        if isinstance(data, FeatureSet):
            metadata = data.metadata.copy() if data.metadata else {}
            if selected_feature_names:
                metadata['feature_names'] = selected_feature_names
            return FeatureSet(features=X_selected, metadata=metadata)
        else:
            return FeatureSet(features=X_selected, metadata={'feature_names': selected_feature_names} if selected_feature_names else {})

    def inverse_transform(self, data: Union[FeatureSet, DataBatch], **kwargs) -> Union[FeatureSet, DataBatch]:
        """
        Return original feature set (identity operation for feature selection).
        
        Since feature selection is a lossy transformation (information is discarded),
        the inverse transform simply returns the input data without modification.
        
        Args:
            data (Union[FeatureSet, DataBatch]): Input data to pass through.
            **kwargs: Additional keyword arguments (ignored).
                
        Returns:
            Union[FeatureSet, DataBatch]: Unmodified input data.
        """
        return data