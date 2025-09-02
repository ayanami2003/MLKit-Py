from typing import Any, Optional, Union
from general.base_classes.model_base import BaseModel
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np

class ExpectedModelChangeStrategy:
    """
    Implements the Expected Model Change active learning strategy.
    
    This strategy selects samples that are expected to cause the largest change
    in the model's parameters when added to the training set. It estimates the
    impact of each unlabeled sample on the model's future state.
    
    The strategy works by:
    1. Computing the expected change in model parameters for each unlabeled sample
    2. Selecting the samples with the highest expected change values
    3. Returning the indices of the most informative samples for labeling
    
    Attributes
    ----------
    model : BaseModel
        The machine learning model to use for change estimation
    batch_size : int
        Number of samples to select in each query round
    uncertainty_threshold : float, optional
        Minimum uncertainty threshold for sample selection
    """

    def __init__(self, model: BaseModel, batch_size: int=1, uncertainty_threshold: Optional[float]=None, name: Optional[str]=None):
        """
        Initialize the Expected Model Change strategy.
        
        Parameters
        ----------
        model : BaseModel
            The machine learning model to use for change estimation
        batch_size : int, default=1
            Number of samples to select in each query round
        uncertainty_threshold : float, optional
            Minimum uncertainty threshold for sample selection
        name : str, optional
            Name identifier for the strategy
        """
        self.model = model
        self.batch_size = batch_size
        self.uncertainty_threshold = uncertainty_threshold
        self.name = name or self.__class__.__name__

    def query(self, labeled_data: Union[FeatureSet, DataBatch], unlabeled_data: Union[FeatureSet, DataBatch], labeled_labels: Optional[Any]=None, **kwargs) -> list[int]:
        """
        Select the most informative samples using expected model change.
        
        This method computes the expected change in model parameters for
        each unlabeled sample and returns the indices of samples with
        the highest expected change values.
        
        Parameters
        ----------
        labeled_data : Union[FeatureSet, DataBatch]
            Currently labeled training data
        unlabeled_data : Union[FeatureSet, DataBatch]
            Pool of unlabeled samples to select from
        labeled_labels : Any, optional
            Labels for the labeled data (if not included in labeled_data)
        **kwargs : dict
            Additional strategy-specific parameters
            
        Returns
        -------
        list[int]
            Indices of selected samples from unlabeled_data
            
        Raises
        ------
        ValueError
            If model is not fitted or data dimensions don't match
        """
        if not self.model.is_fitted:
            raise ValueError('Model must be fitted before querying')
        if isinstance(labeled_data, FeatureSet):
            labeled_features = labeled_data.features
            if labeled_labels is None:
                if labeled_data.is_labeled():
                    labeled_labels = labeled_data.labels
                else:
                    raise ValueError('Labels must be provided either in labeled_data or as labeled_labels parameter')
        else:
            labeled_features = np.array(labeled_data.data)
            if labeled_labels is None:
                if labeled_data.is_labeled():
                    labeled_labels = np.array(labeled_data.labels)
                else:
                    raise ValueError('Labels must be provided either in labeled_data or as labeled_labels parameter')
        if isinstance(unlabeled_data, FeatureSet):
            unlabeled_features = unlabeled_data.features
        else:
            unlabeled_features = np.array(unlabeled_data.data)
        if labeled_features.shape[1] != unlabeled_features.shape[1]:
            raise ValueError('Feature dimensions of labeled and unlabeled data must match')
        change_estimates = []
        for i in range(len(unlabeled_features)):
            if isinstance(unlabeled_data, FeatureSet):
                candidate_sample = FeatureSet(features=unlabeled_features[i:i + 1])
            else:
                candidate_sample = DataBatch(data=unlabeled_features[i:i + 1])
            change = self.estimate_model_change(labeled_data, candidate_sample, labeled_labels=labeled_labels, **kwargs)
            change_estimates.append(change)
        change_estimates = np.array(change_estimates)
        if self.uncertainty_threshold is not None:
            valid_indices = np.where(change_estimates >= self.uncertainty_threshold)[0]
            if len(valid_indices) == 0:
                valid_indices = np.argsort(change_estimates)[-self.batch_size:]
        else:
            valid_indices = np.arange(len(change_estimates))
        top_indices = valid_indices[np.argsort(change_estimates[valid_indices])[-self.batch_size:]]
        return top_indices.tolist()

    def estimate_model_change(self, labeled_data: Union[FeatureSet, DataBatch], candidate_sample: Union[FeatureSet, DataBatch], labeled_labels: Optional[Any]=None, **kwargs) -> float:
        """
        Estimate the expected change in model parameters for a candidate sample.
        
        Computes how much the model parameters would change if the candidate
        sample was added to the training set and the model was retrained.
        
        Parameters
        ----------
        labeled_data : Union[FeatureSet, DataBatch]
            Currently labeled training data
        candidate_sample : Union[FeatureSet, DataBatch]
            Single sample to evaluate for potential addition
        labeled_labels : Any, optional
            Labels for the labeled data (if not included in labeled_data)
        **kwargs : dict
            Additional computation parameters
            
        Returns
        -------
        float
            Estimated magnitude of model parameter change
            
        Raises
        ------
        ValueError
            If candidate_sample contains more than one sample
        """
        if isinstance(labeled_data, FeatureSet):
            labeled_features = labeled_data.features
            if labeled_labels is None:
                if labeled_data.is_labeled():
                    labeled_labels = labeled_data.labels
                else:
                    raise ValueError('Labels must be provided either in labeled_data or as labeled_labels parameter')
        else:
            labeled_features = np.array(labeled_data.data)
            if labeled_labels is None:
                if labeled_data.is_labeled():
                    labeled_labels = np.array(labeled_data.labels)
                else:
                    raise ValueError('Labels must be provided either in labeled_data or as labeled_labels parameter')
        if isinstance(candidate_sample, FeatureSet):
            candidate_features = candidate_sample.features
        else:
            candidate_features = np.array(candidate_sample.data)
        if len(candidate_features) != 1:
            raise ValueError('Candidate sample must contain exactly one sample')
        if not hasattr(self.model, 'coef_') and (not hasattr(self.model, 'get_params')):
            return self._estimate_change_via_prediction_impact(labeled_features, labeled_labels, candidate_features[0], **kwargs)
        try:
            if hasattr(self.model, 'coef_'):
                current_params = np.copy(self.model.coef_)
                if hasattr(self.model, 'intercept_'):
                    current_intercept = np.copy(self.model.intercept_) if self.model.intercept_ is not None else 0
                else:
                    current_intercept = 0
            else:
                params_dict = self.model.get_params()
                current_params = np.array(list(params_dict.values()))
                current_intercept = 0
            simulated_change = self._simulate_parameter_change(labeled_features, labeled_labels, candidate_features[0], **kwargs)
            return simulated_change
        except Exception:
            return self._estimate_change_via_prediction_impact(labeled_features, labeled_labels, candidate_features[0], **kwargs)

    def _simulate_parameter_change(self, labeled_features: np.ndarray, labeled_labels: Any, candidate_feature: np.ndarray, **kwargs) -> float:
        """
        Simulate the parameter change when adding a candidate sample.
        
        This is a simplified approximation - in practice, this would involve
        actually refitting the model or computing gradients.
        """
        try:
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(labeled_features)
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    predictions = predictions[:, 1]
                else:
                    predictions = predictions.ravel()
            else:
                predictions = self.model.predict(labeled_features)
            residuals = np.array(labeled_labels) - predictions
            gradient_approx = np.dot(residuals, labeled_features) / len(labeled_features)
            candidate_contribution = np.dot(gradient_approx, candidate_feature)
            return abs(float(candidate_contribution))
        except Exception:
            return float(np.linalg.norm(candidate_feature))

    def _estimate_change_via_prediction_impact(self, labeled_features: np.ndarray, labeled_labels: Any, candidate_feature: np.ndarray, **kwargs) -> float:
        """
        Estimate model change by measuring the impact on predictions.
        
        This approach measures how much the candidate sample affects predictions
        on the existing labeled data.
        """
        try:
            if hasattr(self.model, 'predict_proba'):
                current_predictions = self.model.predict_proba(labeled_features)
                if current_predictions.ndim > 1 and current_predictions.shape[1] > 1:
                    current_predictions = current_predictions[:, 1]
            else:
                current_predictions = self.model.predict(labeled_features)
            candidate_prediction = self.model.predict(candidate_feature.reshape(1, -1))
            if hasattr(self.model, 'predict_proba'):
                candidate_proba = self.model.predict_proba(candidate_feature.reshape(1, -1))
                if candidate_proba.ndim > 1 and candidate_proba.shape[1] > 1:
                    probs = candidate_proba[0]
                    uncertainty = -np.sum(probs * np.log(probs + 1e-10))
                    return float(uncertainty)
            distances = []
            for i in range(len(labeled_features)):
                diff = labeled_features[i] - candidate_feature
                distance = np.sqrt(np.sum(diff ** 2))
                distances.append(distance)
            avg_distance = np.mean(distances) if distances else 1.0
            return float(1.0 / (avg_distance + 1e-10))
        except Exception:
            return 1.0