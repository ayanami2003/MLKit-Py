from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.utils.validation import check_X_y

class SelectKBestFeatures(BaseTransformer):

    def __init__(self, k: int=10, score_func: str='f_classif', name: Optional[str]=None):
        """
        Initialize the SelectKBestFeatures transformer.
        
        Parameters
        ----------
        k : int, default=10
            Number of top features to select
        score_func : str, default='f_classif'
            Scoring function to use. Options:
            - 'f_classif': ANOVA F-test for classification
            - 'chi2': Chi-squared test for classification
            - 'f_regression': F-test for regression
        name : str, optional
            Name of the transformer instance
        """
        super().__init__(name)
        self.k = k
        self.score_func = score_func
        self.feature_scores_: Optional[np.ndarray] = None
        self.selected_features_: Optional[np.ndarray] = None

    def fit(self, data: FeatureSet, y: Union[np.ndarray, List], **kwargs) -> 'SelectKBestFeatures':
        """
        Fit the transformer to the data by computing feature scores.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to evaluate
        y : array-like
            Target values
        **kwargs : dict
            Additional fitting parameters
            
        Returns
        -------
        SelectKBestFeatures
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If k is larger than the number of features
        """
        X = data.features
        (X, y) = check_X_y(X, y)
        n_features = X.shape[1]
        if self.k > n_features:
            raise ValueError(f'k ({self.k}) cannot be larger than number of features ({n_features})')
        if self.score_func == 'chi2':
            if np.any(X < 0):
                raise ValueError('Chi2 method requires non-negative features')
            (scores, _) = chi2(X, y)
        elif self.score_func == 'f_classif':
            (scores, _) = f_classif(X, y)
        elif self.score_func == 'f_regression':
            (scores, _) = f_regression(X, y)
        else:
            raise ValueError(f'Unsupported score_func: {self.score_func}')
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_scores_ = scores
        indices = np.argsort(scores)[::-1][:self.k]
        self.selected_features_ = np.zeros(n_features, dtype=bool)
        self.selected_features_[indices] = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Transform the data by selecting the top K features.
        
        Parameters
        ----------
        data : FeatureSet
            Input features to transform
        **kwargs : dict
            Additional transformation parameters
            
        Returns
        -------
        FeatureSet
            Transformed feature set with only selected features
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if self.selected_features_ is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'transform'.")
        selected_features = data.features[:, self.selected_features_]
        selected_feature_names = None
        if data.feature_names:
            selected_feature_names = [name for (i, name) in enumerate(data.feature_names) if self.selected_features_[i]]
        selected_feature_types = None
        if data.feature_types:
            selected_feature_types = [ftype for (i, ftype) in enumerate(data.feature_types) if self.selected_features_[i]]
        transformed_data = FeatureSet(features=selected_features, feature_names=selected_feature_names, feature_types=selected_feature_types, sample_ids=data.sample_ids, metadata=data.metadata.copy() if data.metadata else {})
        return transformed_data

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transform is not supported for feature selection.
        
        Parameters
        ----------
        data : FeatureSet
            Transformed feature set
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        FeatureSet
            Original feature set (without removed features)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for feature selection
        """
        raise NotImplementedError('Inverse transform is not supported for feature selection.')

    def get_support(self) -> np.ndarray:
        """
        Get the boolean mask of selected features.
        
        Returns
        -------
        np.ndarray
            Boolean array indicating which features are selected
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if self.selected_features_ is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'get_support'.")
        return self.selected_features_

    def get_scores(self) -> np.ndarray:
        """
        Get the scores of all features.
        
        Returns
        -------
        np.ndarray
            Array of feature scores
            
        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet
        """
        if self.feature_scores_ is None:
            raise RuntimeError("Transformer has not been fitted yet. Call 'fit' before 'get_scores'.")
        return self.feature_scores_