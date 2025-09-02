from typing import Optional, Dict, Any, List
from general.base_classes.validator_base import BaseValidator
from general.structures.data_batch import DataBatch
import numpy as np


# ...(code omitted)...


class EnsembleConceptDriftDetector(BaseValidator):
    """
    Detects concept drift using ensemble methods that combine multiple drift detection algorithms.
    
    This validator monitors incoming data streams for changes in the underlying data distribution
    that indicate a shift in the relationship between features and target variables (concept drift).
    It uses an ensemble approach combining multiple drift detection techniques to improve
    robustness and reduce false positive rates.
    
    Attributes:
        name (str): Name of the detector instance.
        detectors (List[BaseValidator]): Collection of drift detectors in the ensemble.
        voting_strategy (str): How to combine detector results ('majority', 'unanimous', 'weighted').
        threshold (float): Threshold for drift detection decision.
    """

    def __init__(self, name: Optional[str]=None, detectors: Optional[List[BaseValidator]]=None, voting_strategy: str='majority', threshold: float=0.5):
        """
        Initialize the ensemble concept drift detector.
        
        Args:
            name (Optional[str]): Name for the detector instance.
            detectors (Optional[List[BaseValidator]]): List of drift detectors to ensemble.
            voting_strategy (str): Voting strategy to combine detector results. 
                                  Options: 'majority', 'unanimous', 'weighted'.
            threshold (float): Threshold for drift detection decision (used for majority voting).
        """
        super().__init__(name)
        self.detectors = detectors or []
        self.voting_strategy = voting_strategy
        self.threshold = threshold
        self._drift_scores = {}
        if voting_strategy not in ['majority', 'unanimous', 'weighted']:
            raise ValueError("voting_strategy must be one of 'majority', 'unanimous', 'weighted'")

    def fit(self, data: DataBatch, **kwargs) -> 'EnsembleConceptDriftDetector':
        """
        Fit all detectors in the ensemble to reference data.
        
        Args:
            data (DataBatch): Reference data to fit all detectors.
            **kwargs: Additional fitting parameters.
            
        Returns:
            EnsembleConceptDriftDetector: Fitted detector instance.
        """
        for detector in self.detectors:
            detector.fit(data, **kwargs)
        return self

    def validate(self, data: DataBatch, **kwargs) -> bool:
        """
        Check for concept drift using the ensemble of detectors.
        
        Applies all detectors to the new data and combines their binary drift decisions
        based on the configured voting strategy.
        
        Args:
            data (DataBatch): New data to check for concept drift.
            **kwargs: Additional validation parameters.
            
        Returns:
            bool: True if concept drift is detected, False otherwise.
        """
        if not self.detectors:
            return False
        drift_decisions = []
        drift_scores = {}
        for detector in self.detectors:
            try:
                decision = detector.validate(data, **kwargs)
                drift_decisions.append(decision)
                if hasattr(detector, 'get_detection_statistics'):
                    stats = detector.get_detection_statistics()
                    if 'drift_score' in stats:
                        drift_scores[detector.name] = {'score': stats['drift_score'], 'decision': decision}
                    else:
                        drift_scores[detector.name] = {'score': None, 'decision': decision}
                else:
                    drift_scores[detector.name] = {'score': None, 'decision': decision}
            except Exception:
                drift_decisions.append(False)
                drift_scores[detector.name] = {'score': None, 'decision': False}
        self._drift_scores = drift_scores
        if self.voting_strategy == 'unanimous':
            ensemble_decision = all(drift_decisions)
        elif self.voting_strategy == 'weighted':
            drift_count = sum(drift_decisions)
            ensemble_decision = drift_count > len(drift_decisions) / 2
        else:
            drift_count = sum(drift_decisions)
            ensemble_decision = drift_count >= len(drift_decisions) * self.threshold
        self._last_ensemble_decision = ensemble_decision
        return ensemble_decision

    def get_drift_scores(self) -> Dict[str, Any]:
        """
        Get individual drift scores from all detectors and ensemble decision.
        
        Returns:
            Dict[str, Any]: Dictionary containing each detector's name, drift score, 
                           decision, and the ensemble's final verdict.
        """
        ensemble_decision = getattr(self, '_last_ensemble_decision', False)
        result = {'detectors': self._drift_scores, 'ensemble_decision': ensemble_decision, 'voting_strategy': self.voting_strategy}
        return result

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report including drift scores.
        
        Returns:
            Dict[str, Any]: Dictionary containing validation results and drift information.
        """
        base_report = super().get_validation_report()
        base_report['drift_scores'] = self.get_drift_scores()
        return base_report