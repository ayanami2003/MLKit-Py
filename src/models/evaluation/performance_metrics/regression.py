import numpy as np
from typing import Union, Tuple, Optional
from general.base_classes.model_base import BaseModel
from general.base_classes.scorer_base import BaseScorer

class RegressionErrorMetrics(BaseScorer):
    """
    Implements common regression error metrics: MAE, MSE, and RMSE.
    
    This class computes widely-used regression metrics to evaluate model performance.
    Each metric method validates that input arrays have matching shapes and raises
    a ValueError if they do not. The class inherits from BaseScorer, requiring a
    concrete implementation of the abstract score method.
    """

    def __init__(self):
        """
        Initialize the RegressionErrorMetrics instance.
        
        Sets up tracking variables for the latest computed metrics.
        """
        super().__init__(name='regression_error_metrics')
        self._last_mae = None
        self._last_mse = None
        self._last_rmse = None

    def mean_absolute_error(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Mean Absolute Error (MAE) between true and predicted values.
        
        MAE measures the average magnitude of errors between paired observations,
        without considering their direction. It is a scale-dependent accuracy measure
        that is robust to outliers compared to MSE.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Mean absolute error value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        mae = float(np.mean(np.abs(y_true - y_pred)))
        self._last_mae = mae
        return mae

    def mean_squared_error(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.
        
        MSE measures the average of the squares of the errors, giving higher weight
        to larger errors. It is sensitive to outliers and is differentiable, making
        it suitable for gradient-based optimization.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Mean squared error value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        mse = float(np.mean((y_true - y_pred) ** 2))
        self._last_mse = mse
        return mse

    def root_mean_squared_error(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE) between true and predicted values.
        
        RMSE is the square root of MSE, providing error measurements in the same units
        as the target variable. It maintains sensitivity to outliers while being interpretable.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Root mean squared error value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        self._last_rmse = rmse
        return rmse

    def score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], metric: str='rmse', **kwargs) -> float:
        """
        Compute the specified regression error metric.
        
        This method implements the abstract score method from BaseScorer,
        allowing the class to be instantiated. It supports selecting which
        specific metric to compute via the 'metric' parameter.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            metric (str): Which metric to compute ('mae', 'mse', or 'rmse').
                          Defaults to 'rmse'.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            float: Computed metric value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes or invalid metric.
        """
        metric = metric.lower()
        if metric == 'mae':
            return self.mean_absolute_error(y_true, y_pred)
        elif metric == 'mse':
            return self.mean_squared_error(y_true, y_pred)
        elif metric == 'rmse':
            return self.root_mean_squared_error(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Choose from 'mae', 'mse', or 'rmse'.")


# ...(code omitted)...


class MeanSquaredLogarithmicError(BaseScorer):
    """
    Mean Squared Logarithmic Error (MSLE) metric and loss function.
    
    MSLE measures the ratio between actual and predicted values by comparing
    the natural logarithms of the values. It is particularly useful when
    predicting values with exponential growth trends, as it penalizes
    underestimates more than overestimates.
    
    This class serves both as a performance metric and a loss function for optimization.
    """

    def __init__(self, epsilon: float=1e-08):
        """
        Initialize the MSLE calculator.
        
        Args:
            epsilon (float): Small constant to avoid logarithm of zero. Default is 1e-8.
        """
        super().__init__(name='mean_squared_logarithmic_error')
        self.epsilon = epsilon

    def score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Mean Squared Logarithmic Error.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values (must be non-negative).
            y_pred (Union[np.ndarray, list]): Predicted target values (must be non-negative).
            
        Returns:
            float: Mean squared logarithmic error value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes or contain negative values.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError('All values in y_true and y_pred must be non-negative for MSLE calculation.')
        log_true = np.log(y_true + self.epsilon)
        log_pred = np.log(y_pred + self.epsilon)
        msle = float(np.mean((log_true - log_pred) ** 2))
        return msle

    def loss(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate MSLE as a loss function (identical to score for MSLE).
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values (must be non-negative).
            y_pred (Union[np.ndarray, list]): Predicted target values (must be non-negative).
            
        Returns:
            float: Loss value (same as MSLE score).
        """
        return self.score(y_true, y_pred)

class RSquaredMetrics(BaseScorer):
    """
    R-squared and Adjusted R-squared regression metrics.
    
    R-squared measures the proportion of variance in the dependent variable
    that is predictable from the independent variables. Adjusted R-squared
    adjusts for the number of predictors in the model, penalizing unnecessary
    complexity. Both are commonly used to assess model fit quality.
    """

    def __init__(self):
        """Initialize the R-squared metrics calculator."""
        super().__init__(name='r_squared_metrics')

    def r_squared_score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the R-squared (coefficient of determination) score.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: R-squared score value between -∞ and 1.0, where 1.0 indicates perfect fit.
            
        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape')
        if y_true.size == 0:
            return np.nan
        y_true_var = np.var(y_true)
        if y_true_var == 0:
            if np.allclose(y_pred, y_true):
                return 1.0
            else:
                return 0.0 if np.any(np.isnan(y_pred)) else -np.inf
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else -np.inf
        return float(1 - ss_res / ss_tot)

    def adjusted_r_squared_score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], n_features: Optional[int]=None) -> float:
        """
        Calculate the Adjusted R-squared score.
        
        Adjusted R-squared accounts for the number of features in the model,
        providing a more accurate measure for models with different complexities.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            n_features (Optional[int]): Number of features used in the model.
                                       If None, inferred from y_pred shape when possible.
            
        Returns:
            float: Adjusted R-squared score, which can be negative.

        Raises:
            ValueError: If input arrays have mismatched shapes or if n_features leads to division by zero.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape')
        n_samples = y_true.shape[0]
        if n_features is None:
            if y_pred.ndim == 2:
                n_features = y_pred.shape[1]
            else:
                n_features = 1
        if n_features >= n_samples - 1:
            raise ValueError(f'n_features ({n_features}) must be less than n_samples - 1 ({n_samples - 1}) to compute Adjusted R-squared')
        r2 = self.r_squared_score(y_true, y_pred)
        if n_samples <= 1:
            return np.nan
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        return float(adj_r2)

    def score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], n_features: Optional[int]=None, **kwargs) -> float:
        """
        Compute R-squared or Adjusted R-squared score.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            n_features (Optional[int]): Number of features for Adjusted R-squared.
                                       If provided, computes Adjusted R-squared.
                                       If None, computes regular R-squared.
            **kwargs: Additional keyword arguments (ignored).
            
        Returns:
            float: R-squared or Adjusted R-squared score.
        """
        if n_features is not None:
            return self.adjusted_r_squared_score(y_true, y_pred, n_features)
        else:
            return self.r_squared_score(y_true, y_pred)

    def loss(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], n_features: Optional[int]=None) -> float:
        """
        Calculate loss as 1 - R-squared or 1 - Adjusted R-squared.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            n_features (Optional[int]): Number of features for Adjusted R-squared.
                                       If provided, computes Adjusted R-squared.
                                       If None, computes regular R-squared.
            
        Returns:
            float: Loss value (1 - score).
        """
        return 1.0 - self.score(y_true, y_pred, n_features)

class MedianAbsoluteError(BaseScorer):
    """
    Median Absolute Error (MedAE) regression metric.
    
    MedAE measures the median of absolute differences between predicted and actual values.
    It is robust to outliers since it uses the median rather than the mean, making it
    less sensitive to extreme values compared to Mean Absolute Error.
    """

    def __init__(self):
        """Initialize the Median Absolute Error calculator."""
        super().__init__(name='median_absolute_error')

    def score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Median Absolute Error.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Median absolute error value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        return float(np.median(np.abs(y_true - y_pred)))

class NormalizedMeanSquaredError(BaseScorer):
    """
    Normalized Mean Squared Error (NMSE) regression metric.
    
    NMSE normalizes the mean squared error by the variance of the true values,
    providing a scale-invariant measure of model performance. Values closer to 0
    indicate better model fit, with 0 representing a perfect fit.
    """

    def __init__(self):
        """Initialize the NMSE calculator."""
        super().__init__(name='normalized_mean_squared_error')

    def score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Normalized Mean Squared Error.
        
        NMSE = MSE(y_true, y_pred) / Var(y_true)
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Normalized mean squared error value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes or zero variance in y_true.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        if y_true.size == 0:
            return 0.0
        if y_true.size == 1:
            raise ValueError('Variance of y_true is zero with only one element, cannot compute normalized MSE.')
        mse = np.mean((y_true - y_pred) ** 2)
        y_true_var = np.var(y_true, ddof=0)
        if np.isclose(y_true_var, 0.0):
            raise ValueError('Variance of y_true is zero, cannot compute normalized MSE.')
        nmse = mse / y_true_var
        return float(nmse)

    def loss(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate NMSE as a loss function (identical to score for NMSE).
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Loss value (same as NMSE score).
        """
        return self.score(y_true, y_pred)

class ExplainedVarianceScore(BaseScorer):
    """
    Explained Variance Score regression metric.
    
    Measures the proportion of variance in the target variable that is predictable
    from the input features. Unlike R-squared, it does not center the predictions
    around the mean, making it more sensitive to systematic biases in predictions.
    """

    def __init__(self):
        """Initialize the Explained Variance Score calculator."""
        super().__init__(name='explained_variance_score')

    def score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Explained Variance Score.
        
        The score is computed as 1 - Var(y_true - y_pred) / Var(y_true),
        where Var is the variance. Best possible score is 1.0, and it can
        be negative if the model performs worse than a simple mean predictor.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Explained variance score (can be negative).
            
        Raises:
            ValueError: If input arrays have mismatched shapes or if variance of y_true is zero.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        y_true_var = np.var(y_true)
        if y_true_var == 0:
            raise ValueError('Variance of y_true is zero. Explained variance is undefined.')
        diff_var = np.var(y_true - y_pred)
        explained_var = 1 - diff_var / y_true_var
        return float(explained_var)

class DevianceMetrics(BaseScorer):
    """
    Collection of deviance-based regression metrics.
    
    Implements mean gamma deviance, mean Poisson deviance, and mean Tweedie deviance.
    These metrics are particularly useful for generalized linear models where the
    target variable follows a specific probability distribution from the exponential family.
    """

    def __init__(self):
        """Initialize the Deviance Metrics calculator."""
        super().__init__(name='deviance_metrics')

    def mean_gamma_deviance(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Mean Gamma Deviance.
        
        Suitable for strictly positive target values following a Gamma distribution.
        Commonly used in insurance and finance applications.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values (must be > 0).
            y_pred (Union[np.ndarray, list]): Predicted target values (must be > 0).
            
        Returns:
            float: Mean gamma deviance value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes or contain non-positive values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.')
        if np.any(y_true <= 0):
            raise ValueError('y_true must contain only positive values for Gamma deviance.')
        if np.any(y_pred <= 0):
            raise ValueError('y_pred must contain only positive values for Gamma deviance.')
        ratio = y_pred / y_true
        log_ratio = np.log(ratio)
        deviance = 2 * (log_ratio + 1 / ratio - 1)
        return float(np.mean(deviance))

    def mean_poisson_deviance(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Mean Poisson Deviance.
        
        Suitable for count data following a Poisson distribution.
        Commonly used in modeling event rates or occurrences.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values (must be non-negative integers).
            y_pred (Union[np.ndarray, list]): Predicted target values (must be non-negative).
            
        Returns:
            float: Mean Poisson deviance value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes or contain negative values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.')
        if np.any(y_true < 0):
            raise ValueError('y_true must contain only non-negative values for Poisson deviance.')
        if np.any(y_pred < 0):
            raise ValueError('y_pred must contain only non-negative values for Poisson deviance.')
        safe_log_ratio = np.zeros_like(y_true)
        nonzero_mask = y_true > 0
        safe_log_ratio[nonzero_mask] = np.log(y_true[nonzero_mask] / y_pred[nonzero_mask])
        deviance = 2 * (y_true * safe_log_ratio + y_pred - y_true)
        return float(np.mean(deviance))

    def mean_tweedie_deviance(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list], power: float=0.0) -> float:
        """
        Calculate the Mean Tweedie Deviance.
        
        A flexible deviance metric that generalizes several other deviance measures:
        - power=0: Normal distribution
        - power=1: Poisson distribution
        - power=2: Gamma distribution
        - power=3: Inverse Gaussian distribution
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            power (float): Tweedie power parameter. Defaults to 0.0 (Normal distribution).
            
        Returns:
            float: Mean Tweedie deviance value.
            
        Raises:
            ValueError: If input arrays have mismatched shapes or invalid values for the specified power.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.')
        if power < 0:
            if np.any(y_true <= 0) or np.any(y_pred <= 0):
                raise ValueError('For power < 0, both y_true and y_pred must be strictly positive.')
        elif power <= 1:
            if np.any(y_true < 0) or np.any(y_pred < 0):
                raise ValueError('For 0 <= power <= 1, both y_true and y_pred must be non-negative.')
        elif power < 2:
            if np.any(y_true < 0) or np.any(y_pred <= 0):
                raise ValueError('For 1 < power < 2, y_true must be non-negative and y_pred must be positive.')
        elif np.any(y_true <= 0) or np.any(y_pred <= 0):
            raise ValueError('For power >= 2, both y_true and y_pred must be strictly positive.')
        if power == 0:
            deviance = (y_true - y_pred) ** 2
        elif power == 1:
            safe_log_ratio = np.zeros_like(y_true)
            nonzero_mask = y_true > 0
            safe_log_ratio[nonzero_mask] = np.log(y_true[nonzero_mask] / y_pred[nonzero_mask])
            deviance = 2 * (y_true * safe_log_ratio + y_pred - y_true)
        elif power == 2:
            ratio = y_pred / y_true
            log_ratio = np.log(ratio)
            deviance = 2 * (log_ratio + 1 / ratio - 1)
        else:
            p = power
            term1 = y_true ** (2 - p) / ((1 - p) * (2 - p))
            term2 = -(y_true * y_pred ** (1 - p)) / (1 - p)
            term3 = y_pred ** (2 - p) / (2 - p)
            deviance = 2 * (term1 + term2 + term3)
        return float(np.mean(deviance))

class TheilInequalityCoefficient(BaseScorer):

    def __init__(self):
        """Initialize the Theil Inequality Coefficient calculator."""
        super().__init__(name='theil_inequality_coefficient')

    def score(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the Theil Inequality Coefficient.
        
        The coefficient is defined as sqrt(MSE) / (sqrt(mean(y_true^2)) + sqrt(mean(y_pred^2))),
        where MSE is mean squared error.
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            float: Theil inequality coefficient value between 0 and 1.
            
        Raises:
            ValueError: If input arrays have mismatched shapes.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        mse = np.mean((y_true - y_pred) ** 2)
        mean_sq_true = np.mean(y_true ** 2)
        mean_sq_pred = np.mean(y_pred ** 2)
        if mean_sq_true == 0 and mean_sq_pred == 0:
            return 0.0 if mse == 0 else 1.0
        denom = np.sqrt(mean_sq_true) + np.sqrt(mean_sq_pred)
        if denom == 0:
            return 0.0 if mse == 0 else 1.0
        theil_coeff = np.sqrt(mse) / denom
        return float(max(0.0, min(1.0, theil_coeff)))

    def decompose(self, y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> Tuple[float, float, float]:
        """
        Decompose the Theil coefficient into its three components.
        
        Returns the proportions of inequality due to:
        1. Bias (systematic error)
        2. Variance (difference in variability)
        3. Covariance (unsynchronized fluctuations)
        
        Args:
            y_true (Union[np.ndarray, list]): Ground truth target values.
            y_pred (Union[np.ndarray, list]): Predicted target values.
            
        Returns:
            Tuple[float, float, float]: (bias_proportion, variance_proportion, covariance_proportion)
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
        n = len(y_true)
        if n == 0:
            return (0.0, 0.0, 0.0)
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        mean_sq_true = np.mean(y_true ** 2)
        mean_sq_pred = np.mean(y_pred ** 2)
        if mean_sq_true == 0 and mean_sq_pred == 0:
            return (0.0, 0.0, 0.0)
        denom = np.sqrt(mean_sq_true) + np.sqrt(mean_sq_pred)
        if denom == 0:
            return (0.0, 0.0, 0.0)
        mse = np.mean((y_true - y_pred) ** 2)
        if mse == 0:
            return (0.0, 0.0, 0.0)
        theil_u = np.sqrt(mse) / denom
        std_true = np.sqrt(np.mean((y_true - mean_true) ** 2))
        std_pred = np.sqrt(np.mean((y_pred - mean_pred) ** 2))
        u_bias_sq = (mean_true - mean_pred) ** 2 / denom ** 2
        u_variance_sq = (std_true - std_pred) ** 2 / denom ** 2
        if std_true == 0 or std_pred == 0:
            correlation = 0.0
        else:
            covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
            correlation = covariance / (std_true * std_pred)
        u_covariance_sq = 2 * (1 - correlation) * std_true * std_pred / denom ** 2
        theil_u_sq = theil_u ** 2
        if theil_u_sq == 0:
            return (0.0, 0.0, 0.0)
        bias_prop = u_bias_sq / theil_u_sq
        variance_prop = u_variance_sq / theil_u_sq
        covariance_prop = u_covariance_sq / theil_u_sq
        total = bias_prop + variance_prop + covariance_prop
        if total > 0:
            bias_prop /= total
            variance_prop /= total
            covariance_prop /= total
        return (float(bias_prop), float(variance_prop), float(covariance_prop))


# ...(code omitted)...


def mean_absolute_error(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.
    
    MAE measures the average magnitude of errors between paired observations,
    without considering their direction. It is a scale-dependent accuracy measure
    that is robust to outliers compared to MSE.
    
    Args:
        y_true (Union[np.ndarray, list]): Ground truth target values.
        y_pred (Union[np.ndarray, list]): Predicted target values.
        
    Returns:
        float: Mean absolute error value.
        
    Raises:
        ValueError: If input arrays have mismatched shapes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f'Input arrays must have the same shape. Got {y_true.shape} and {y_pred.shape}')
    return float(np.mean(np.abs(y_true - y_pred)))