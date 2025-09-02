from typing import Optional, Union
import numpy as np
from general.base_classes.model_base import BaseModel
from general.structures.feature_set import FeatureSet
from src.data_lifecycle.utilities.shuffling.data_shuffling import shuffle_rows

class PerceptronClassifier(BaseModel):
    """
    An online perceptron classifier for binary classification tasks.
    
    This classifier implements the classic perceptron algorithm that updates
    its weights incrementally as new samples arrive. It is particularly suited
    for online learning scenarios where data arrives sequentially.
    
    The perceptron makes predictions based on a linear combination of features
    and applies a step function to produce binary outputs {-1, 1}.
    
    Attributes
    ----------
    weights : np.ndarray
        Weight vector for the linear decision boundary
    bias : float
        Bias term for the decision boundary
    learning_rate : float
        Step size for weight updates
    max_iter : int
        Maximum number of iterations over the training data
    shuffle : bool
        Whether to shuffle training data between epochs
    random_state : Optional[int]
        Random seed for reproducibility
    """

    def __init__(self, learning_rate: float=0.01, max_iter: int=1000, shuffle: bool=True, random_state: Optional[int]=None, name: Optional[str]=None):
        """
        Initialize the PerceptronClassifier.
        
        Parameters
        ----------
        learning_rate : float, default=0.01
            Learning rate for weight updates. Must be positive.
        max_iter : int, default=1000
            Maximum number of iterations over the training data.
            Must be positive.
        shuffle : bool, default=True
            Whether to shuffle training data between epochs.
        random_state : Optional[int], default=None
            Random seed for reproducibility of shuffling.
        name : Optional[str], default=None
            Name identifier for the model instance.
        """
        super().__init__(name=name)
        if learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0

    def fit(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> 'PerceptronClassifier':
        """
        Train the perceptron classifier on the provided data.
        
        The training process iteratively updates weights based on misclassified
        samples until convergence or maximum iterations reached.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Training features. If FeatureSet, uses .features attribute.
            Shape should be (n_samples, n_features).
        y : Union[np.ndarray, list]
            Binary target labels. Should contain values in {-1, 1}.
            Length should be n_samples.
        **kwargs : dict
            Additional training parameters (ignored in this implementation).
            
        Returns
        -------
        PerceptronClassifier
            Trained model instance (self) for method chaining.
            
        Raises
        ------
        ValueError
            If input dimensions don't match or labels aren't binary.
        """
        if isinstance(X, FeatureSet):
            X = X.features
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in X and y must match')
        unique_labels = np.unique(y)
        if not set(unique_labels).issubset({-1, 1}):
            raise ValueError('Labels must be in {-1, 1}')
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.max_iter):
            if self.shuffle:
                indices = rng.permutation(X.shape[0])
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            updated = False
            for i in range(X_shuffled.shape[0]):
                xi = X_shuffled[i]
                yi = y_shuffled[i]
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = np.sign(linear_output)
                if y_pred == 0:
                    y_pred = 1
                if y_pred != yi:
                    self.weights += self.learning_rate * yi * xi
                    self.bias += self.learning_rate * yi
                    updated = True
            if not updated:
                break
        self.is_fitted = True
        return self

    def predict(self, X: Union[FeatureSet, np.ndarray], **kwargs) -> np.ndarray:
        """
        Make binary predictions on new data.
        
        Computes the linear combination of features and weights, then applies
        sign function to produce binary predictions {-1, 1}.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Input features for prediction. If FeatureSet, uses .features attribute.
            Shape should be (n_samples, n_features).
        **kwargs : dict
            Additional prediction parameters (ignored in this implementation).
            
        Returns
        -------
        np.ndarray
            Binary predictions with values in {-1, 1}.
            Shape is (n_samples,).
            
        Raises
        ------
        ValueError
            If model hasn't been fitted or input dimensions mismatch.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before making predictions')
        if isinstance(X, FeatureSet):
            X = X.features
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError('Number of features in X must match number of features used for training')
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.sign(linear_output)
        predictions[predictions == 0] = 1
        return predictions.astype(int)

    def score(self, X: Union[FeatureSet, np.ndarray], y: Union[np.ndarray, list], **kwargs) -> float:
        """
        Calculate accuracy score on test data.
        
        Computes the proportion of correctly classified samples.
        
        Parameters
        ----------
        X : Union[FeatureSet, np.ndarray]
            Test features. If FeatureSet, uses .features attribute.
            Shape should be (n_samples, n_features).
        y : Union[np.ndarray, list]
            True binary labels with values in {-1, 1}.
            Length should be n_samples.
        **kwargs : dict
            Additional scoring parameters (ignored in this implementation).
            
        Returns
        -------
        float
            Accuracy score between 0.0 and 1.0.
            
        Raises
        ------
        ValueError
            If model hasn't been fitted or input dimensions mismatch.
        """
        if not self.is_fitted:
            raise ValueError('Model must be fitted before scoring')
        y_pred = self.predict(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y_pred.shape[0] != y.shape[0]:
            raise ValueError('Number of samples in X and y must match')
        unique_labels = np.unique(y)
        if not set(unique_labels).issubset({-1, 1}):
            raise ValueError('Labels must be in {-1, 1}')
        accuracy = np.mean(y_pred == y)
        return float(accuracy)