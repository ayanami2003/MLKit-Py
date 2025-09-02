from typing import Optional, Union, List
import numpy as np
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet

class NonSeasonalAnomalyDetector(BaseTransformer):

    def __init__(self, method: str='zscore', threshold: float=3.0, contamination: float=0.1, feature_columns: Optional[List[str]]=None, name: Optional[str]=None):
        """
        Initialize the NonSeasonalAnomalyDetector.
        
        Parameters
        ----------
        method : str, default='zscore'
            The anomaly detection method to use. Options:
            - 'zscore': Uses z-score thresholding
            - 'iqr': Uses interquartile range method
            - 'isolation_forest': Uses Isolation Forest algorithm
            - 'one_class_svm': Uses One-Class SVM algorithm
        threshold : float, default=3.0
            Threshold for anomaly detection (used by zscore and iqr methods)
        contamination : float, default=0.1
            Expected proportion of anomalies in the data (used by isolation_forest and one_class_svm)
        feature_columns : Optional[List[str]], optional
            Specific columns to analyze for anomalies. If None, all numeric columns are used
        name : Optional[str], optional
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.feature_columns = feature_columns
        self._anomaly_scores = None
        self._is_fitted = False

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'NonSeasonalAnomalyDetector':
        """
        Fit the anomaly detector to the input data.
        
        For statistical methods (zscore, iqr), this computes the necessary parameters.
        For ML-based methods (isolation_forest, one_class_svm), this trains the model.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input time series data to fit the detector on
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        NonSeasonalAnomalyDetector
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If the method is not supported
        """
        if isinstance(data, FeatureSet):
            X = data.features
        elif isinstance(data, DataBatch):
            X = data.data
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError('Input data must be FeatureSet, DataBatch, or numpy array')
        supported_methods = ['zscore', 'iqr', 'isolation_forest', 'one_class_svm']
        if self.method not in supported_methods:
            raise ValueError(f"Method '{self.method}' not supported. Choose from {supported_methods}")
        if self.feature_columns is not None and isinstance(data, FeatureSet):
            col_indices = [data.feature_names.index(col) for col in self.feature_columns if col in data.feature_names]
            if not col_indices:
                raise ValueError('None of the specified feature columns found in data')
            X = X[:, col_indices]
        elif self.feature_columns is not None and (not isinstance(data, FeatureSet)):
            pass
        if self.method == 'zscore':
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            self._std = np.where(self._std == 0, 1e-10, self._std)
        elif self.method == 'iqr':
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self._iqr = q75 - q25
            self._q1 = q25
            self._q3 = q75
            self._iqr = np.where(self._iqr == 0, 1e-10, self._iqr)
        elif self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self._model = IsolationForest(contamination=self.contamination, random_state=42)
            self._model.fit(X)
        elif self.method == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            self._model = OneClassSVM(nu=self.contamination, kernel='rbf')
            self._model.fit(X)
        self._is_fitted = True
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], return_scores: bool=False, **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray, tuple]:
        """
        Detect anomalies in the input data.
        
        Identifies anomalous points in the time series based on the configured method.
        Returns a boolean array indicating which points are anomalies, or the original
        data with anomaly flags added.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input time series data to analyze for anomalies
        return_scores : bool, default=False
            If True, also return the anomaly scores
        **kwargs : dict
            Additional parameters for transformation
            
        Returns
        -------
        Union[FeatureSet, DataBatch, np.ndarray, tuple]
            - If return_scores is False: Boolean array or FeatureSet/DataBatch with anomaly flags
            - If return_scores is True: Tuple of (anomalies, scores)
            
        Raises
        ------
        RuntimeError
            If the detector has not been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call 'fit' first.")
        if isinstance(data, FeatureSet):
            X = data.features
            is_featureset = True
            is_databatch = False
        elif isinstance(data, DataBatch):
            X = data.data
            is_featureset = False
            is_databatch = True
        elif isinstance(data, np.ndarray):
            X = data
            is_featureset = False
            is_databatch = False
        else:
            raise ValueError('Input data must be FeatureSet, DataBatch, or numpy array')
        if self.feature_columns is not None and isinstance(data, FeatureSet):
            col_indices = [data.feature_names.index(col) for col in self.feature_columns if col in data.feature_names]
            if not col_indices:
                raise ValueError('None of the specified feature columns found in data')
            X = X[:, col_indices]
        elif self.feature_columns is not None and (not isinstance(data, FeatureSet)):
            pass
        if self.method == 'zscore':
            z_scores = np.abs((X - self._mean) / self._std)
            anomalies = np.any(z_scores > self.threshold, axis=1)
            self._anomaly_scores = np.max(z_scores, axis=1)
        elif self.method == 'iqr':
            lower_bound = self._q1 - 1.5 * self._iqr
            upper_bound = self._q3 + 1.5 * self._iqr
            anomalies = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            lower_z = (X - self._q1) / self._iqr
            upper_z = (X - self._q3) / self._iqr
            self._anomaly_scores = np.max(np.maximum(-lower_z, upper_z), axis=1)
        elif self.method == 'isolation_forest':
            predictions = self._model.predict(X)
            anomalies = predictions == -1
            self._anomaly_scores = -self._model.decision_function(X)
        elif self.method == 'one_class_svm':
            predictions = self._model.predict(X)
            anomalies = predictions == -1
            self._anomaly_scores = -self._model.decision_function(X)
        if return_scores:
            if is_featureset:
                result = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else {}, quality_scores=data.quality_scores.copy() if data.quality_scores else {})
                result.metadata['anomaly_flags'] = anomalies
                return (result, self._anomaly_scores)
            elif is_databatch:
                result = DataBatch(data=data.data, labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
                result.metadata['anomaly_flags'] = anomalies
                return (result, self._anomaly_scores)
            else:
                return (anomalies, self._anomaly_scores)
        elif is_featureset:
            result = FeatureSet(features=data.features.copy(), feature_names=data.feature_names.copy() if data.feature_names else None, feature_types=data.feature_types.copy() if data.feature_types else None, sample_ids=data.sample_ids.copy() if data.sample_ids else None, metadata=data.metadata.copy() if data.metadata else {}, quality_scores=data.quality_scores.copy() if data.quality_scores else {})
            result.metadata['anomaly_flags'] = anomalies
            return result
        elif is_databatch:
            result = DataBatch(data=data.data, labels=data.labels, metadata=data.metadata.copy() if data.metadata else {}, sample_ids=data.sample_ids, feature_names=data.feature_names, batch_id=data.batch_id)
            result.metadata['anomaly_flags'] = anomalies
            return result
        else:
            return anomalies

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Not implemented for anomaly detectors.
        
        Anomaly detection is a one-way transformation that cannot be inverted.
        
        Parameters
        ----------
        data : Union[FeatureSet, DataBatch, np.ndarray]
            Input data (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not applicable
        """
        raise NotImplementedError('Anomaly detection is a one-way transformation and cannot be inverted.')

    def get_anomaly_scores(self) -> Optional[np.ndarray]:
        """
        Retrieve the anomaly scores computed during the last transform operation.
        
        Returns
        -------
        Optional[np.ndarray]
            Array of anomaly scores, or None if no scores are available
        """
        return self._anomaly_scores