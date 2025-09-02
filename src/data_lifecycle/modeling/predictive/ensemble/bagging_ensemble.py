from typing import Optional, List, Any
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
import numpy as np
from sklearn.base import clone
from sklearn.utils import resample

class BaggingEnsembleModel(BaseModel):

    def __init__(self, base_estimator: BaseModel, n_estimators: int=10, max_samples: float=1.0, max_features: float=1.0, bootstrap: bool=True, bootstrap_features: bool=False, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the BaggingEnsembleModel.
        
        Args:
            base_estimator (BaseModel): The base estimator to be fitted on random subsets.
            n_estimators (int): The number of base estimators in the ensemble. Defaults to 10.
            max_samples (float): The fraction of samples to draw for each estimator. Defaults to 1.0.
            max_features (float): The fraction of features to draw for each estimator. Defaults to 1.0.
            bootstrap (bool): Whether to sample with replacement. Defaults to True.
            bootstrap_features (bool): Whether to sample features with replacement. Defaults to False.
            random_state (Optional[int]): Random state for reproducibility. Defaults to None.
            name (Optional[str]): Name of the model. Defaults to class name.
        """
        super().__init__(name)
        if not 0.0 < max_samples <= 1.0:
            raise ValueError('max_samples must be in (0, 1]')
        if not 0.0 < max_features <= 1.0:
            raise ValueError('max_features must be in (0, 1]')
        if n_estimators <= 0:
            raise ValueError('n_estimators must be positive')
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.estimators_: List[BaseModel] = []

    def fit(self, X: FeatureSet, y: DataBatch, **kwargs) -> 'BaggingEnsembleModel':
        """
        Fit the bagging ensemble model according to the given training data.
        
        This method trains multiple instances of the base estimator on different bootstrap
        samples of the training data.
        
        Args:
            X (FeatureSet): The training input samples.
            y (DataBatch): The target values.
            **kwargs: Additional fitting parameters.
            
        Returns:
            BaggingEnsembleModel: Fitted ensemble model.
        """
        self.estimators_ = []
        rng = np.random.RandomState(self.random_state)
        if hasattr(X, 'features'):
            X_array = X.features
        elif hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        if hasattr(y, 'labels') and y.labels is not None:
            y_array = y.labels
        elif hasattr(y, 'values') and y.values is not None:
            y_array = y.values
        else:
            y_array = np.array(y)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        elif X_array.ndim == 0:
            raise ValueError('X cannot be scalar')
        (n_samples, n_features) = X_array.shape
        n_samples_bootstrap = max(1, int(self.max_samples * n_samples))
        n_features_bootstrap = max(1, int(self.max_features * n_features))
        if self.bootstrap_features:
            feature_indices_list = [resample(np.arange(n_features), n_samples=n_features_bootstrap, replace=True, random_state=rng.randint(0, 2 ** 32 - 1)) for _ in range(self.n_estimators)]
        else:
            feature_indices_list = [np.arange(n_features) for _ in range(self.n_estimators)]
        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = resample(np.arange(n_samples), n_samples=n_samples_bootstrap, replace=True, random_state=rng.randint(0, 2 ** 32 - 1))
            else:
                indices = np.arange(n_samples)[:n_samples_bootstrap]
            X_subset = X_array[np.ix_(indices, feature_indices_list[i])]
            if y_array.ndim == 0:
                y_subset = y_array
            elif y_array.ndim == 1:
                y_subset = y_array[indices]
            else:
                y_subset = y_array[indices]
            estimator = clone(self.base_estimator)
            estimator.fit(X_subset, y_subset, **kwargs)
            self.estimators_.append(estimator)
        return self

    def predict(self, X: FeatureSet, **kwargs) -> Any:
        """
        Predict regression target for X using the bagging ensemble.
        
        The predicted value is computed as the aggregated predictions of all base estimators.
        
        Args:
            X (FeatureSet): The input samples.
            **kwargs: Additional prediction parameters.
            
        Returns:
            Any: The predicted values.
        """
        if not self.estimators_:
            raise ValueError("Model has not been fitted yet. Call 'fit' before using this method.")
        if hasattr(X, 'features'):
            X_array = X.features
        elif hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        elif X_array.ndim == 0:
            raise ValueError('X cannot be scalar')
        predictions = []
        for estimator in self.estimators_:
            pred = estimator.predict(X_array, **kwargs)
            predictions.append(pred)
        predictions = np.array(predictions)
        if predictions.ndim == 1:
            return predictions
        elif predictions.ndim == 2:
            if predictions.shape[0] == 1:
                return predictions[0]
            else:
                try:
                    float_predictions = predictions.astype(float)
                    return np.mean(float_predictions, axis=0)
                except (ValueError, TypeError):
                    from scipy.stats import mode
                    mode_result = mode(predictions, axis=0, keepdims=True)
                    return mode_result.mode[0] if hasattr(mode_result, 'mode') else mode_result[0][0]
        else:
            return np.mean(predictions, axis=0)

    def score(self, X: FeatureSet, y: DataBatch, **kwargs) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
        ((y_true - y_true.mean()) ** 2).sum().
        
        Args:
            X (FeatureSet): Test samples.
            y (DataBatch): True values for X.
            **kwargs: Additional scoring parameters.
            
        Returns:
            float: R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X, **kwargs)
        y_true = y.values if hasattr(y, 'values') else np.array(y)
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - ss_res / ss_tot