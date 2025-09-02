from typing import List, Optional, Any, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from general.structures.data_batch import DataBatch
from general.base_classes.model_base import BaseModel, NotFittedError

class BinaryRelevanceClassifier(BaseModel):
    """
    Multi-label classifier using the Binary Relevance approach.
    
    This classifier treats each label independently by training a separate
    binary classifier for each label. It transforms a multi-label problem
    into multiple binary classification problems.
    
    Attributes
    ----------
    estimators_ : List[BaseModel]
        List of binary classifiers, one for each label
    n_labels_ : int
        Number of labels in the multi-label task
    label_names_ : Optional[List[str]]
        Names of the labels if provided during training
    
    Methods
    -------
    fit(X, y) -> 'BinaryRelevanceClassifier'
        Train the binary relevance classifier
    predict(X) -> np.ndarray
        Predict label probabilities for each sample
    predict_labels(X, threshold=0.5) -> np.ndarray
        Predict binary labels based on probability threshold
    """

    def __init__(self, base_classifier: BaseModel, label_names: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the Binary Relevance classifier.
        
        Parameters
        ----------
        base_classifier : BaseModel
            The binary classifier to use for each label. Must implement
            fit, predict, and predict_proba methods.
        label_names : Optional[List[str]], optional
            Names for each label, by default None
        name : Optional[str], optional
            Name for the classifier, by default None
        """
        super().__init__(name=name)
        self.base_classifier = base_classifier
        self.label_names = label_names
        self.estimators_: List[BaseModel] = []
        self.n_labels_: int = 0
        self.label_names_: Optional[List[str]] = label_names

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List[List[int]]], **kwargs: Any) -> 'BinaryRelevanceClassifier':
        """
        Train the binary relevance classifier.
        
        For each label, trains a separate binary classifier using the
        provided base classifier.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training features of shape (n_samples, n_features)
        y : Union[np.ndarray, List[List[int]]]
            Multi-label targets of shape (n_samples, n_labels) with binary values
        **kwargs : Any
            Additional fitting parameters passed to base classifiers
            
        Returns
        -------
        BinaryRelevanceClassifier
            Fitted classifier
            
        Raises
        ------
        ValueError
            If the number of samples in X and y don't match
            If y is not a 2D array/matrix
        """
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        y_array = np.array(y)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f'Number of samples in X ({X_array.shape[0]}) and y ({y_array.shape[0]}) must match')
        if y_array.ndim != 2:
            raise ValueError('y must be a 2D array/matrix')
        self.n_labels_ = y_array.shape[1]
        self.estimators_ = []
        for i in range(self.n_labels_):
            clf = self.base_classifier.clone()
            y_label = y_array[:, i]
            clf.fit(X_array, y_label, **kwargs)
            self.estimators_.append(clf)
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs: Any) -> np.ndarray:
        """
        Predict label probabilities for each sample.
        
        Uses each trained binary classifier to predict the probability
        of each label being positive.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Test features of shape (n_samples, n_features)
        **kwargs : Any
            Additional prediction parameters passed to base classifiers
            
        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n_samples, n_labels)
            
        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet
        """
        if not hasattr(self, 'estimators_') or len(self.estimators_) == 0:
            raise NotFittedError("This BinaryRelevanceClassifier instance is not fitted yet. Call 'fit' before using this estimator.")
        if isinstance(X, FeatureSet):
            X_array = X.features
        else:
            X_array = X
        n_samples = X_array.shape[0]
        probabilities = np.zeros((n_samples, self.n_labels_))
        for (i, estimator) in enumerate(self.estimators_):
            proba = estimator.predict_proba(X_array, **kwargs)[:, 1]
            probabilities[:, i] = proba
        return probabilities

    def predict_labels(self, X: Union[FeatureSet, np.ndarray], threshold: float=0.5, **kwargs: Any) -> np.ndarray:
        """
        Predict binary labels based on probability threshold.
        
        Converts probability predictions to binary labels using
        the specified threshold.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Test features of shape (n_samples, n_features)
        threshold : float, optional
            Probability threshold for positive classification, by default 0.5
        **kwargs : Any
            Additional prediction parameters
            
        Returns
        -------
        np.ndarray
            Predicted binary labels of shape (n_samples, n_labels)
            
        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet
        """
        probabilities = self.predict(X, **kwargs)
        return (probabilities >= threshold).astype(int)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, List[List[int]]], **kwargs: Any) -> float:
        """
        Evaluate model performance using subset accuracy.
        
        Subset accuracy requires all labels for a sample to be correctly
        predicted for that sample to be considered correct.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Test features of shape (n_samples, n_features)
        y : Union[np.ndarray, List[List[int]]]
            True multi-label targets of shape (n_samples, n_labels)
        **kwargs : Any
            Additional scoring parameters
            
        Returns
        -------
        float
            Subset accuracy score
            
        Raises
        ------
        NotFittedError
            If the classifier has not been fitted yet
        """
        if not hasattr(self, 'estimators_') or len(self.estimators_) == 0:
            raise NotFittedError("This BinaryRelevanceClassifier instance is not fitted yet. Call 'fit' before using this estimator.")
        y_true = np.array(y)
        y_pred = self.predict_labels(X, **kwargs)
        correct_predictions = np.all(y_true == y_pred, axis=1)
        subset_accuracy = np.mean(correct_predictions)
        return float(subset_accuracy)