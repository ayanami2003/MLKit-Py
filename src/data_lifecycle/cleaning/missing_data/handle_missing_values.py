from typing import Optional, Union, List
from general.base_classes.transformer_base import BaseTransformer
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from src.data_lifecycle.preprocessing.categorical_encoding.one_hot_encoding import OneHotEncoder
from src.data_lifecycle.preprocessing.categorical_encoding.target_encoding import TargetEncoder


# ...(code omitted)...


class RegressionImputationTransformer(BaseTransformer):
    """
    Impute missing values using regression models.

    This transformer supports fitting regression models to predict and fill missing values
    in specified or all eligible columns. It handles categorical variables via configurable
    encoding strategies and performs iterative imputation to refine imputations.

    Attributes:
        target_columns (Optional[List[str]]): Columns to impute. If None, imputes all columns with missing values.
        regression_method (str): Regression method to use ('linear', 'ridge', 'lasso').
        categorical_handling (str): How to handle categorical variables ('onehot', 'target_mean').
        max_iter (int): Maximum number of imputation iterations.
        _fitted_models (dict): Dictionary storing fitted models for each target column.
        _feature_names (List[str]): Names of features in the dataset.
    """

    def __init__(self, target_columns: Optional[List[str]]=None, regression_method: str='linear', categorical_handling: str='onehot', max_iter: int=10, name: Optional[str]=None):
        super().__init__(name=name)
        self.target_columns = target_columns
        self.regression_method = regression_method
        self.categorical_handling = categorical_handling
        self.max_iter = max_iter
        self._fitted_models = {}
        self._feature_names = None

    def fit(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> 'RegressionImputationTransformer':
        """
        Fit regression models for each target column with missing values.

        Args:
            data: Input data containing features with missing values.
                  Can be FeatureSet, DataBatch, or numpy array.
            **kwargs: Additional fitting parameters.

        Returns:
            RegressionImputationTransformer: Fitted transformer instance.
        """
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            self._feature_names = data.feature_names.copy() if data.feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        elif isinstance(data, DataBatch):
            X = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data.copy()
            self._feature_names = data.feature_names.copy() if data.feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        else:
            X = data.copy()
            self._feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.target_columns is None:
            missing_mask = np.isnan(X) | pd.isna(X)
            self.target_columns = [self._feature_names[i] for i in range(X.shape[1]) if np.any(missing_mask[:, i])]
        else:
            missing_cols = set(self.target_columns) - set(self._feature_names)
            if missing_cols:
                raise ValueError(f'Specified target columns not found in data: {missing_cols}')
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            if np.issubdtype(col_data.dtype, np.number):
                mask = np.isnan(col_data) | pd.isna(col_data)
                if np.any(mask):
                    mean_val = np.nanmean(col_data)
                    X[mask, col_idx] = mean_val
        for iteration in range(self.max_iter):
            X_previous = X.copy()
            if self.categorical_handling == 'onehot':
                encoder = OneHotEncoder(handle_unknown='ignore')
                categorical_columns = []
                for (i, name) in enumerate(self._feature_names):
                    if not np.issubdtype(X[:, i].dtype, np.number):
                        categorical_columns.append(i)
                if categorical_columns:
                    cat_features = X[:, categorical_columns]
                    cat_feature_names = [self._feature_names[i] for i in categorical_columns]
                    cat_fs = FeatureSet(features=cat_features, feature_names=cat_feature_names)
                    encoder.fit(cat_fs)
                    encoded_fs = encoder.transform(cat_fs)
                    encoded_features = encoded_fs.features
                    numerical_columns = [i for i in range(X.shape[1]) if i not in categorical_columns]
                    if numerical_columns:
                        numerical_features = X[:, numerical_columns]
                        X = np.hstack([numerical_features, encoded_features])
                        num_feature_names = [self._feature_names[i] for i in numerical_columns]
                        self._feature_names = num_feature_names + encoded_fs.feature_names
                    else:
                        X = encoded_features
                        self._feature_names = encoded_fs.feature_names
            elif self.categorical_handling == 'target_mean':
                encoder = TargetEncoder()
                categorical_columns = []
                for (i, name) in enumerate(self._feature_names):
                    if not np.issubdtype(X[:, i].dtype, np.number):
                        categorical_columns.append(i)
                if categorical_columns:
                    cat_features = X[:, categorical_columns]
                    cat_feature_names = [self._feature_names[i] for i in categorical_columns]
                    cat_fs = FeatureSet(features=cat_features, feature_names=cat_feature_names)
                    target_indices = [self._feature_names.index(col) for col in self.target_columns if col in self._feature_names]
                    if target_indices:
                        target_values = X[:, target_indices]
                        encoder.fit(cat_fs, target_values)
                        encoded_fs = encoder.transform(cat_fs)
                        encoded_features = encoded_fs.features
                        numerical_columns = [i for i in range(X.shape[1]) if i not in categorical_columns]
                        if numerical_columns:
                            numerical_features = X[:, numerical_columns]
                            X = np.hstack([numerical_features, encoded_features])
                            num_feature_names = [self._feature_names[i] for i in numerical_columns]
                            self._feature_names = num_feature_names + encoded_fs.feature_names
                        else:
                            X = encoded_features
                            self._feature_names = encoded_fs.feature_names
            target_indices = [self._feature_names.index(col) for col in self.target_columns if col in self._feature_names]
            for target_idx in target_indices:
                target_name = self._feature_names[target_idx]
                target_column = X[:, target_idx]
                missing_mask = np.isnan(target_column) | pd.isna(target_column)
                if not np.any(missing_mask):
                    continue
                predictor_indices = [i for i in range(X.shape[1]) if i != target_idx]
                if not predictor_indices:
                    continue
                X_predictors = X[:, predictor_indices]
                y_target = target_column
                valid_mask = ~(np.isnan(y_target) | pd.isna(y_target))
                if not np.any(valid_mask):
                    continue
                X_train = X_predictors[valid_mask]
                y_train = y_target[valid_mask]
                predictor_missing = np.isnan(X_train) | pd.isna(X_train)
                complete_rows = ~np.any(predictor_missing, axis=1)
                if not np.any(complete_rows):
                    continue
                X_train_complete = X_train[complete_rows]
                y_train_complete = y_train[complete_rows]
                if self.regression_method == 'linear':
                    model = LinearRegression()
                elif self.regression_method == 'ridge':
                    model = Ridge()
                elif self.regression_method == 'lasso':
                    model = Lasso()
                else:
                    raise ValueError(f'Unsupported regression method: {self.regression_method}')
                try:
                    model.fit(X_train_complete, y_train_complete)
                    self._fitted_models[target_name] = {'model': model, 'predictor_indices': predictor_indices, 'encoder': encoder if self.categorical_handling in ['onehot', 'target_mean'] else None}
                except Exception:
                    continue
            if iteration > 0 and np.allclose(X, X_previous, equal_nan=True):
                break
        return self

    def transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Impute missing values using fitted regression models.

        Args:
            data: Input data with missing values to impute.
                  Can be FeatureSet, DataBatch, or numpy array.
            **kwargs: Additional transformation parameters.

        Returns:
            Union[FeatureSet, DataBatch, np.ndarray]: Data with imputed values.
        """
        original_type = type(data)
        if isinstance(data, FeatureSet):
            X = data.features.copy()
            feature_names = data.feature_names.copy() if data.feature_names else [f'feature_{i}' for i in range(X.shape[1])]
            sample_ids = data.sample_ids.copy() if data.sample_ids else None
            metadata = data.metadata.copy() if data.metadata else None
            quality_scores = data.quality_scores.copy() if data.quality_scores else None
        elif isinstance(data, DataBatch):
            X = np.array(data.data) if not isinstance(data.data, np.ndarray) else data.data.copy()
            feature_names = data.feature_names.copy() if data.feature_names else [f'feature_{i}' for i in range(X.shape[1])]
            sample_ids = data.sample_ids.copy() if data.sample_ids else None
        else:
            X = data.copy()
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_work = X.copy()
        for col_idx in range(X_work.shape[1]):
            col_data = X_work[:, col_idx]
            if np.issubdtype(col_data.dtype, np.number):
                mask = np.isnan(col_data) | pd.isna(col_data)
                if np.any(mask):
                    mean_val = np.nanmean(col_data)
                    X_work[mask, col_idx] = mean_val
        work_feature_names = feature_names.copy()
        encoders = {}
        if self.categorical_handling == 'onehot':
            categorical_columns = []
            for (i, name) in enumerate(work_feature_names):
                if not np.issubdtype(X_work[:, i].dtype, np.number):
                    categorical_columns.append(i)
            if categorical_columns:
                encoder = OneHotEncoder(handle_unknown='ignore')
                cat_features = X_work[:, categorical_columns]
                cat_feature_names = [work_feature_names[i] for i in categorical_columns]
                cat_fs = FeatureSet(features=cat_features, feature_names=cat_feature_names)
                if hasattr(self, '_fitted_models') and self._fitted_models:
                    for model_info in self._fitted_models.values():
                        if 'encoder' in model_info and model_info['encoder'] is not None:
                            encoder = model_info['encoder']
                            break
                else:
                    encoder.fit(cat_fs)
                encoded_fs = encoder.transform(cat_fs)
                encoded_features = encoded_fs.features
                encoders['main'] = encoder
                numerical_columns = [i for i in range(X_work.shape[1]) if i not in categorical_columns]
                if numerical_columns:
                    numerical_features = X_work[:, numerical_columns]
                    X_work = np.hstack([numerical_features, encoded_features])
                    num_feature_names = [work_feature_names[i] for i in numerical_columns]
                    work_feature_names = num_feature_names + encoded_fs.feature_names
                else:
                    X_work = encoded_features
                    work_feature_names = encoded_fs.feature_names
        elif self.categorical_handling == 'target_mean':
            categorical_columns = []
            for (i, name) in enumerate(work_feature_names):
                if not np.issubdtype(X_work[:, i].dtype, np.number):
                    categorical_columns.append(i)
            if categorical_columns:
                encoder = TargetEncoder()
                cat_features = X_work[:, categorical_columns]
                cat_feature_names = [work_feature_names[i] for i in categorical_columns]
                cat_fs = FeatureSet(features=cat_features, feature_names=cat_feature_names)
                if hasattr(self, '_fitted_models') and self._fitted_models:
                    for model_info in self._fitted_models.values():
                        if 'encoder' in model_info and model_info['encoder'] is not None:
                            encoder = model_info['encoder']
                            break
                else:
                    pass
                try:
                    encoded_fs = encoder.transform(cat_fs)
                    encoded_features = encoded_fs.features
                    encoders['main'] = encoder
                    numerical_columns = [i for i in range(X_work.shape[1]) if i not in categorical_columns]
                    if numerical_columns:
                        numerical_features = X_work[:, numerical_columns]
                        X_work = np.hstack([numerical_features, encoded_features])
                        num_feature_names = [work_feature_names[i] for i in numerical_columns]
                        work_feature_names = num_feature_names + encoded_fs.feature_names
                    else:
                        X_work = encoded_features
                        work_feature_names = encoded_fs.feature_names
                except:
                    pass
        for (target_name, model_info) in self._fitted_models.items():
            if target_name not in work_feature_names:
                continue
            target_idx = work_feature_names.index(target_name)
            target_column = X_work[:, target_idx]
            missing_mask = np.isnan(target_column) | pd.isna(target_column)
            if not np.any(missing_mask):
                continue
            predictor_indices = model_info['predictor_indices']
            if not predictor_indices or max(predictor_indices) >= X_work.shape[1]:
                continue
            X_predictors = X_work[np.ix_(range(X_work.shape[0]), predictor_indices)]
            for col_idx in range(X_predictors.shape[1]):
                col_data = X_predictors[:, col_idx]
                if np.isnan(col_data).any() or pd.isna(col_data).any():
                    mask = np.isnan(col_data) | pd.isna(col_data)
                    if np.any(~mask):
                        mean_val = np.nanmean(col_data)
                        X_predictors[mask, col_idx] = mean_val
                    else:
                        X_predictors[:, col_idx] = 0
            try:
                predictions = model_info['model'].predict(X_predictors[missing_mask])
                X_work[missing_mask, target_idx] = predictions
            except Exception:
                continue
        result_X = X_work[:, :len(feature_names)] if len(work_feature_names) > len(feature_names) else X_work
        if original_type == FeatureSet:
            return FeatureSet(features=result_X, feature_names=feature_names, sample_ids=sample_ids, metadata=metadata, quality_scores=quality_scores)
        elif original_type == DataBatch:
            return DataBatch(data=result_X, feature_names=feature_names, sample_ids=sample_ids)
        else:
            return result_X if result_X.shape[1] > 1 else result_X.ravel()

    def inverse_transform(self, data: Union[FeatureSet, DataBatch, np.ndarray], **kwargs) -> Union[FeatureSet, DataBatch, np.ndarray]:
        """
        Inverse transformation is not supported for regression imputation as it's a lossy operation.

        Args:
            data: Transformed data.
            **kwargs: Additional parameters.

        Returns:
            Union[FeatureSet, DataBatch, np.ndarray]: Original data format (unchanged).

        Raises:
            NotImplementedError: Always raised as inverse transform is not supported.
        """
        raise NotImplementedError('Inverse transform is not supported for regression imputation.')