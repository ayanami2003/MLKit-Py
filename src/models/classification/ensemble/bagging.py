from typing import Optional, Union, List, Any
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from joblib import Parallel, delayed

class BaggingClassifier(BaseModel):
    """
    A bagging classifier that fits base classifiers on random subsets of the training data.

    This ensemble method improves the stability and accuracy of machine learning algorithms by 
    combining predictions from multiple models trained on different subsets of the data. It reduces 
    overfitting by averaging out biases and lowers variance through bootstrap sampling.

    The classifier supports various base estimators and allows customization of the ensemble size,
    sampling strategy, and other hyperparameters.

    Attributes:
        base_estimator: The base classifier to fit on each bootstrap sample.
        n_estimators: Number of base estimators in the ensemble.
        max_samples: Number of samples to draw for each base estimator.
        max_features: Number of features to draw for each base estimator.
        bootstrap: Whether to bootstrap samples for each estimator.
        bootstrap_features: Whether to bootstrap features for each estimator.
        oob_score: Whether to compute out-of-bag score.
        warm_start: Whether to reuse solution from previous calls.
        n_jobs: Number of jobs to run in parallel.
        random_state: Random seed for reproducibility.
    """

    def __init__(self, base_estimator: Optional[BaseModel]=None, n_estimators: int=10, max_samples: Union[int, float]=1.0, max_features: Union[int, float]=1.0, bootstrap: bool=True, bootstrap_features: bool=False, oob_score: bool=False, warm_start: bool=False, n_jobs: Optional[int]=None, random_state: Optional[int]=None, **kwargs: Any):
        """
        Initialize the BaggingClassifier.

        Args:
            base_estimator: The base classifier to fit on each bootstrap sample.
                           If None, then a default classifier is used.
            n_estimators: Number of base estimators in the ensemble.
                         Must be a positive integer.
            max_samples: Number of samples to draw for each base estimator.
                        If float, interpreted as fraction of total samples.
                        Must be > 0.
            max_features: Number of features to draw for each base estimator.
                         If float, interpreted as fraction of total features.
                         Must be > 0.
            bootstrap: Whether to bootstrap samples for each estimator.
                      If False, whole dataset is used for each estimator.
            bootstrap_features: Whether to bootstrap features for each estimator.
                               If False, all features are used for each estimator.
            oob_score: Whether to compute out-of-bag score.
                      Requires bootstrap=True.
            warm_start: When set to True, reuse the solution from previous calls.
                       Otherwise, erase previous solution.
            n_jobs: Number of jobs to run in parallel.
                   None means 1, -1 means using all processors.
            random_state: Random seed for reproducibility.
                         If None, uses random seed.
            **kwargs: Additional keyword arguments passed to base estimator.
        """
        super().__init__(name='BaggingClassifier')
        if n_estimators <= 0:
            raise ValueError('n_estimators must be a positive integer')
        if isinstance(max_samples, float) and (not 0 < max_samples <= 1):
            raise ValueError('max_samples must be in (0, 1] when float')
        if isinstance(max_samples, int) and max_samples <= 0:
            raise ValueError('max_samples must be positive when int')
        if isinstance(max_features, float) and (not 0 < max_features <= 1):
            raise ValueError('max_features must be in (0, 1] when float')
        if isinstance(max_features, int) and max_features <= 0:
            raise ValueError('max_features must be positive when int')
        if oob_score and (not bootstrap):
            raise ValueError('oob_score requires bootstrap=True')
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        self._estimators: List[BaseModel] = []
        self._oob_score: Optional[float] = None
        self._features_indices: List[np.ndarray] = []
        self._samples_indices: List[np.ndarray] = []
        self._n_features: Optional[int] = None
        self._n_samples: Optional[int] = None
        self._classes: Optional[np.ndarray] = None

    def _validate_data(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, List]]=None) -> tuple:
        """Validate and extract data from FeatureSet or numpy array."""
        if isinstance(X, FeatureSet):
            X_array = X.features
        elif isinstance(X, np.ndarray):
            X_array = X
        else:
            raise TypeError('X must be either FeatureSet or numpy array')
        if y is not None:
            y_array = np.asarray(y)
            if X_array.shape[0] != len(y_array):
                raise ValueError('X and y must have the same number of samples')
            return (X_array, y_array)
        return (X_array, None)

    def _get_subsample(self, rng: np.random.RandomState, n_samples: int, max_samples: Union[int, float]) -> np.ndarray:
        """Get indices for a subsample."""
        if isinstance(max_samples, float):
            n_samples_to_pick = int(max_samples * n_samples)
        else:
            n_samples_to_pick = min(max_samples, n_samples)
        if self.bootstrap:
            indices = rng.randint(0, n_samples, n_samples_to_pick)
        else:
            indices = rng.permutation(n_samples)[:n_samples_to_pick]
        return indices

    def _get_subfeatures(self, rng: np.random.RandomState, n_features: int, max_features: Union[int, float]) -> np.ndarray:
        """Get indices for a subset of features."""
        if isinstance(max_features, float):
            n_features_to_pick = int(max_features * n_features)
        else:
            n_features_to_pick = min(max_features, n_features)
        if self.bootstrap_features:
            indices = rng.randint(0, n_features, n_features_to_pick)
        else:
            indices = rng.permutation(n_features)[:n_features_to_pick]
        return indices

    def _fit_single_estimator(self, estimator: BaseModel, X: np.ndarray, y: np.ndarray, sample_indices: np.ndarray, feature_indices: np.ndarray) -> BaseModel:
        """Fit a single estimator on a subset of data."""
        X_subset = X[sample_indices][:, feature_indices]
        y_subset = y[sample_indices]
        if hasattr(estimator, 'clone'):
            est = estimator.clone()
        else:
            est = type(estimator)(**self.kwargs)
        return est.fit(X_subset, y_subset)

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Optional[Union[np.ndarray, List]]=None, **kwargs: Any) -> 'BaggingClassifier':
        """
        Fit the bagging classifier according to the training data.

        This method trains multiple instances of the base estimator on different
        bootstrap samples of the training data.

        Args:
            X: Training features. Can be a FeatureSet containing features and metadata,
               or a numpy array of shape (n_samples, n_features).
            y: Target values of shape (n_samples,) or (n_samples, n_outputs).
               For classification, should contain class labels.
            **kwargs: Additional fitting parameters.

        Returns:
            BaggingClassifier: Fitted estimator.

        Raises:
            ValueError: If training data is invalid or incompatible.
        """
        (X_array, y_array) = self._validate_data(X, y)
        if y_array is None:
            raise ValueError('y cannot be None for supervised learning')
        rng = np.random.RandomState(self.random_state)
        (self._n_samples, self._n_features) = X_array.shape
        self._classes = np.unique(y_array)
        if not self.warm_start:
            self._estimators = []
            self._features_indices = []
            self._samples_indices = []
        n_already_trained = len(self._estimators)
        n_to_fit = self.n_estimators - n_already_trained
        if n_to_fit <= 0:
            return self
        if self.base_estimator is None:
            from src.models.classification.bayesian.naive_bayes import NaiveBayesClassifier
            base_est = NaiveBayesClassifier()
        else:
            base_est = self.base_estimator
        new_samples_indices = []
        new_features_indices = []
        for _ in range(n_to_fit):
            sample_idx = self._get_subsample(rng, self._n_samples, self.max_samples)
            feature_idx = self._get_subfeatures(rng, self._n_features, self.max_features)
            new_samples_indices.append(sample_idx)
            new_features_indices.append(feature_idx)
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        if n_jobs == -1:
            n_jobs = None
        fitted_estimators = Parallel(n_jobs=n_jobs)((delayed(self._fit_single_estimator)(base_est, X_array, y_array, sample_idx, feature_idx) for (sample_idx, feature_idx) in zip(new_samples_indices, new_features_indices)))
        self._estimators.extend(fitted_estimators)
        self._samples_indices.extend(new_samples_indices)
        self._features_indices.extend(new_features_indices)
        if self.oob_score and self.bootstrap:
            self._compute_oob_score(X_array, y_array)
        return self

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute out-of-bag score."""
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples, len(self._classes)))
        oob_counts = np.zeros(n_samples)
        for (i, (estimator, sample_indices)) in enumerate(zip(self._estimators, self._samples_indices)):
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[sample_indices] = False
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) > 0:
                X_oob = X[oob_idx][:, self._features_indices[i]]
                try:
                    pred_proba = estimator.predict_proba(X_oob)
                    oob_predictions[oob_idx] += pred_proba
                    oob_counts[oob_idx] += 1
                except AttributeError:
                    continue
        valid_oob_mask = oob_counts > 0
        if np.sum(valid_oob_mask) > 0:
            oob_predictions[valid_oob_mask] /= oob_counts[valid_oob_mask, np.newaxis]
            oob_pred_classes = self._classes[np.argmax(oob_predictions[valid_oob_mask], axis=1)]
            self._oob_score = np.mean(oob_pred_classes == y[valid_oob_mask])
        else:
            self._oob_score = 0.0

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Predict class labels for samples in X.

        This method aggregates predictions from all base estimators to make
        final predictions. For classification, it returns the class with
        the highest aggregated score.

        Args:
            X: Samples to predict. Can be a FeatureSet or numpy array of
               shape (n_samples, n_features).
            **kwargs: Additional prediction parameters.

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._estimators:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        (X_array, _) = self._validate_data(X)
        predictions = []
        for (i, estimator) in enumerate(self._estimators):
            X_subset = X_array[:, self._features_indices[i]]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        predictions = np.column_stack(predictions)
        predicted_classes = []
        for i in range(predictions.shape[0]):
            (unique, counts) = np.unique(predictions[i], return_counts=True)
            predicted_classes.append(unique[np.argmax(counts)])
        return np.array(predicted_classes)

    def predict_proba(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Computes the probability of each class for each sample by averaging
        the probabilities predicted by each base estimator.

        Args:
            X: Samples to predict. Can be a FeatureSet or numpy array of
               shape (n_samples, n_features).
            **kwargs: Additional prediction parameters.

        Returns:
            np.ndarray: Predicted class probabilities of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._estimators:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        (X_array, _) = self._validate_data(X)
        if not all((hasattr(est, 'predict_proba') for est in self._estimators)):
            raise RuntimeError('All base estimators must support predict_proba for this method')
        probas = np.zeros((X_array.shape[0], len(self._classes)))
        for (i, estimator) in enumerate(self._estimators):
            X_subset = X_array[:, self._features_indices[i]]
            est_probas = estimator.predict_proba(X_subset)
            if hasattr(estimator, 'classes_'):
                est_classes = estimator.classes_
            else:
                est_classes = self._classes
            for (j, cls) in enumerate(self._classes):
                if cls in est_classes:
                    idx = np.where(est_classes == cls)[0][0]
                    probas[:, j] += est_probas[:, idx]
        probas /= len(self._estimators)
        return probas

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List], **kwargs: Any) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Computes the accuracy as the fraction of correctly predicted samples.

        Args:
            X: Test samples. Can be a FeatureSet or numpy array of
               shape (n_samples, n_features).
            y: True labels for X of shape (n_samples,).
            **kwargs: Additional scoring parameters.

        Returns:
            float: Mean accuracy of self.predict(X) wrt. y.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._estimators:
            raise RuntimeError("Model has not been fitted yet. Call 'fit' first.")
        predictions = self.predict(X)
        y_array = np.asarray(y)
        return np.mean(predictions == y_array)

    def get_oob_score(self) -> Optional[float]:
        """
        Get the out-of-bag score if computed during fitting.

        Returns:
            Optional[float]: Out-of-bag score, or None if not computed.
        """
        return self._oob_score

    def get_estimators(self) -> List[BaseModel]:
        """
        Get the list of fitted base estimators.

        Returns:
            List[BaseModel]: List of fitted base estimators.
        """
        return self._estimators.copy()