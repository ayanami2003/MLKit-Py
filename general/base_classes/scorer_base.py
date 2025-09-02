from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np

class BaseScorer(ABC):
    """
    Abstract base class for scoring components.
    
    Defines a standard interface for computing metrics and scores,
    supporting both supervised and unsupervised evaluation scenarios.
    """

    def __init__(self, name: Optional[str]=None, greater_is_better: bool=True):
        self.name = name or self.__class__.__name__
        self.greater_is_better = greater_is_better
        self.scores_history: List[float] = []

    @abstractmethod
    def score(self, y_true: Any, y_pred: Any, **kwargs) -> float:
        """
        Compute the score/metric value.
        
        Parameters
        ----------
        y_true : Any
            Ground truth values
        y_pred : Any
            Predicted values
        **kwargs : dict
            Additional scoring parameters
            
        Returns
        -------
        float
            Computed score value
        """
        pass

    def score_unsupervised(self, data: Any, **kwargs) -> float:
        """
        Compute score for unsupervised scenarios.
        
        Parameters
        ----------
        data : Any
            Input data to evaluate
        **kwargs : dict
            Additional scoring parameters
            
        Returns
        -------
        float
            Computed unsupervised score
        """
        raise NotImplementedError(f'Unsupervised scoring not implemented for {self.__class__.__name__}')

    def score_batch(self, y_true_list: List[Any], y_pred_list: List[Any], **kwargs) -> float:
        """
        Compute score across multiple batches/predictions.
        
        Parameters
        ----------
        y_true_list : List[Any]
            List of ground truth values for each batch
        y_pred_list : List[Any]
            List of predicted values for each batch
        **kwargs : dict
            Additional scoring parameters
            
        Returns
        -------
        float
            Aggregated score across all batches
        """
        scores = []
        for (y_true, y_pred) in zip(y_true_list, y_pred_list):
            scores.append(self.score(y_true, y_pred, **kwargs))
        return float(np.mean(scores)) if scores else 0.0

    def record_score(self, score_value: float) -> None:
        """
        Record a score value in history.
        
        Parameters
        ----------
        score_value : float
            Score value to record
        """
        self.scores_history.append(score_value)

    def get_latest_score(self) -> Optional[float]:
        """
        Get the most recently recorded score.
        
        Returns
        -------
        Optional[float]
            Most recent score or None if no scores recorded
        """
        return self.scores_history[-1] if self.scores_history else None

    def get_score_statistics(self) -> Dict[str, float]:
        """
        Get statistics on recorded scores.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with mean, std, min, max of recorded scores
        """
        if not self.scores_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        scores_array = np.array(self.scores_history)
        return {'mean': float(np.mean(scores_array)), 'std': float(np.std(scores_array)), 'min': float(np.min(scores_array)), 'max': float(np.max(scores_array))}