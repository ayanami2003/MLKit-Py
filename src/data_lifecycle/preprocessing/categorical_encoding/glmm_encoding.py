from typing import Optional, List, Union
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet

class GLMMEncoder(BaseTransformer):

    def __init__(self, handle_unknown: str='error', handle_missing: str='error', random_state: Optional[int]=None, cv_folds: int=5, smoothing: float=1.0, name: Optional[str]=None):
        """
        Initialize the GLMM encoder.
        
        Parameters
        ----------
        handle_unknown : str, default='error'
            How to handle unknown categories in transform.
        handle_missing : str, default='error'
            How to handle missing values.
        random_state : int, optional
            Random seed for reproducibility.
        cv_folds : int, default=5
            Number of cross-validation folds.
        smoothing : float, default=1.0
            Smoothing parameter for empirical Bayes estimation.
        name : str, optional
            Name of the transformer instance.
        """
        super().__init__(name=name)
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.smoothing = smoothing
        self.category_effects_ = {}
        self._fitted = False
        self._input_feature_names = None
        self._output_feature_names = None

    def fit(self, data: Union[FeatureSet, np.ndarray], y: Optional[np.ndarray]=None, **kwargs) -> 'GLMMEncoder':
        """
        Fit the GLMM encoder to the training data.
        
        This method estimates the random effects for each category by fitting
        a generalized linear mixed model with the target variable.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data containing categorical features to encode.
            If FeatureSet, expects features attribute to contain categorical data.
            If np.ndarray, assumes all columns are categorical.
        y : np.ndarray, optional
            Target values used to estimate category effects.
            Required for supervised encoding.
        **kwargs : dict
            Additional fitting parameters (ignored).
            
        Returns
        -------
        GLMMEncoder
            Self instance for method chaining.
            
        Raises
        ------
        ValueError
            If y is None or if data dimensions don't match.
        """
        if y is None:
            raise ValueError("Target variable 'y' is required for GLMM encoding.")
        if isinstance(data, FeatureSet):
            X = data.features
            if hasattr(data, 'feature_names') and data.feature_names is not None and (len(data.feature_names) == X.shape[1]):
                self._input_feature_names = data.feature_names.copy()
            else:
                self._input_feature_names = [f'x{i}' for i in range(X.shape[1])]
        else:
            X = data
            self._input_feature_names = [f'x{i}' for i in range(X.shape[1])]
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional.')
        if len(X) != len(y):
            raise ValueError('Data and target lengths must match.')
        if self.random_state is not None:
            np.random.seed(self.random_state)
        (n_samples, n_features) = X.shape
        self.category_effects_ = {}
        for col_idx in range(n_features):
            feature_col = X[:, col_idx]
            feature_name = self._input_feature_names[col_idx] if self._input_feature_names else f'x{col_idx}'
            if self.handle_missing == 'error' and (np.any(feature_col == None) or np.any([x is None or (isinstance(x, str) and x.lower() == 'nan') or (isinstance(x, float) and np.isnan(x)) for x in feature_col])):
                raise ValueError(f"Missing values found in feature '{feature_name}' and handle_missing='error'")
            mask_missing = np.array([x is None or (isinstance(x, str) and x.lower() == 'nan') or (isinstance(x, float) and np.isnan(x)) for x in feature_col])
            if np.any(mask_missing):
                feature_col_processed = feature_col.copy().astype(object)
                feature_col_processed[mask_missing] = '__MISSING__'
            else:
                feature_col_processed = feature_col
            categories = np.unique(feature_col_processed)
            category_effects = {}
            for category in categories:
                if category == '__MISSING__':
                    continue
                category_mask = feature_col_processed == category
                if np.sum(category_mask) < 2:
                    category_effects[category] = 0.0
                    continue
                y_category = y[category_mask]
                effect = self._estimate_category_effect(y_category, y)
                category_effects[category] = effect
            if np.any(mask_missing):
                y_missing = y[mask_missing]
                if len(y_missing) > 0:
                    missing_effect = self._estimate_category_effect(y_missing, y)
                else:
                    missing_effect = 0.0
                category_effects['__MISSING__'] = missing_effect
            if n_features == 1:
                self.category_effects_ = category_effects
            else:
                self.category_effects_[feature_name] = category_effects
        self._fitted = True
        self._output_feature_names = self._input_feature_names.copy() if self._input_feature_names else [f'x{i}' for i in range(n_features)]
        return self

    def _estimate_category_effect(self, y_category: np.ndarray, y_all: np.ndarray) -> float:
        """
        Estimate the effect of a category using empirical Bayes approach.
        
        Parameters
        ----------
        y_category : np.ndarray
            Target values for samples belonging to this category
        y_all : np.ndarray
            All target values
            
        Returns
        -------
        float
            Estimated effect for the category
        """
        if len(y_category) == 0:
            return 0.0
        global_mean = np.mean(y_all)
        category_mean = np.mean(y_category)
        global_var = np.var(y_all) if len(y_all) > 1 else 0
        category_var = np.var(y_category) if len(y_category) > 1 else 0
        if global_var <= 0:
            return category_mean - global_mean
        weight = len(y_category) / (len(y_category) + self.smoothing)
        effect = weight * (category_mean - global_mean)
        return effect

    def transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Transform categorical features using fitted GLMM encodings.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Input data containing categorical features to transform.
        **kwargs : dict
            Additional transformation parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Transformed data with GLMM encodings replacing categorical features.
            
        Raises
        ------
        ValueError
            If transformer is not fitted or unknown categories encountered with handle_unknown='error'.
        """
        if not self._fitted:
            raise ValueError("GLMMEncoder has not been fitted yet. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            feature_names = data.feature_names if data.feature_names else self._input_feature_names
        else:
            X = data
            feature_names = self._input_feature_names
        if X.ndim != 2:
            raise ValueError('Input data must be 2-dimensional.')
        if feature_names and X.shape[1] != len(feature_names):
            raise ValueError(f'Expected {len(feature_names)} features, got {X.shape[1]}')
        (n_samples, n_features) = X.shape
        transformed_features = np.zeros((n_samples, n_features), dtype=float)
        for col_idx in range(n_features):
            feature_col = X[:, col_idx]
            feature_name = feature_names[col_idx] if feature_names else f'x{col_idx}'
            if n_features == 1:
                encodings = self.category_effects_
            else:
                if feature_name not in self.category_effects_:
                    raise ValueError(f"No encodings found for feature '{feature_name}'. Was this feature present during fitting?")
                encodings = self.category_effects_[feature_name]
            mask_missing = np.array([x is None or (isinstance(x, str) and x.lower() == 'nan') or (isinstance(x, float) and np.isnan(x)) for x in feature_col])
            if np.any(mask_missing):
                if self.handle_missing == 'error':
                    raise ValueError(f"Missing values found in feature '{feature_name}' and handle_missing='error'")
                else:
                    feature_col_processed = feature_col.copy().astype(object)
                    feature_col_processed[mask_missing] = '__MISSING__'
            else:
                feature_col_processed = feature_col
            for (i, category) in enumerate(feature_col_processed):
                if category == '__MISSING__':
                    if '__MISSING__' in encodings:
                        transformed_features[i, col_idx] = encodings['__MISSING__']
                    else:
                        transformed_features[i, col_idx] = 0.0
                elif category in encodings:
                    transformed_features[i, col_idx] = encodings[category]
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{category}' encountered in feature '{feature_name}' and handle_unknown='error'")
                else:
                    transformed_features[i, col_idx] = 0.0
        return FeatureSet(features=transformed_features, feature_names=feature_names if feature_names else [f'x{i}' for i in range(n_features)])

    def inverse_transform(self, data: Union[FeatureSet, np.ndarray], **kwargs) -> FeatureSet:
        """
        Convert GLMM encodings back to original categorical values (not implemented).
        
        As GLMM encoding is a many-to-one mapping (multiple categories to numerical values),
        exact inverse transformation is generally not possible.
        
        Parameters
        ----------
        data : FeatureSet or np.ndarray
            Encoded data to convert back.
        **kwargs : dict
            Additional parameters (ignored).
            
        Returns
        -------
        FeatureSet
            Data with approximate categorical reconstructions.
            
        Raises
        ------
        NotImplementedError
            Always raised as exact inverse is not possible.
        """
        raise NotImplementedError('Exact inverse transformation is not possible for GLMM encoding since multiple categories can map to the same numerical value.')

    def get_feature_names_out(self, input_features: Optional[List[str]]=None) -> List[str]:
        """
        Get output feature names after encoding.
        
        Parameters
        ----------
        input_features : list of str, optional
            Input feature names. If None, generates default names.
            
        Returns
        -------
        list of str
            Output feature names after GLMM encoding.
        """
        if input_features is not None:
            return input_features
        elif self._output_feature_names is not None:
            return self._output_feature_names
        elif self._input_feature_names is not None:
            return self._input_feature_names
        else:
            n_features = 1 if not self.category_effects_ or isinstance(list(self.category_effects_.values())[0], dict) else 1
            return [f'x{i}' for i in range(n_features)]