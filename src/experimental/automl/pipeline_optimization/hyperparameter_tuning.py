from typing import Optional, Dict, Any, Union, List
from general.base_classes.pipeline_base import BasePipelineComponent
from general.structures.component_config import ComponentConfig
from general.structures.data_batch import DataBatch
from general.structures.feature_set import FeatureSet
import time
import numpy as np
from src.data_lifecycle.splitting.cross_validation.stratified_kfold import StratifiedKFold
from src.data_lifecycle.algorithms.hyperparameter_tuning.search_strategies.randomized_search_tuning import RandomizedSearchTuner

def automatically_tune_pipeline_hyperparameters(pipeline: BasePipelineComponent, train_data: Union[FeatureSet, DataBatch], validation_data: Union[FeatureSet, DataBatch], param_grid: Dict[str, List[Any]], scoring_method: str='accuracy', cv_folds: int=5, max_evaluations: int=100, timeout_seconds: Optional[int]=None, random_state: Optional[int]=None) -> ComponentConfig:
    """
    Automatically tune hyperparameters for a given pipeline using cross-validation.

    This function performs hyperparameter optimization for any pipeline component by searching through
    a specified parameter space. It uses cross-validation on the training data to evaluate different
    hyperparameter combinations and selects the best configuration based on the specified scoring method.
    The resulting optimal configuration can then be applied to the pipeline.

    Args:
        pipeline (BasePipelineComponent): The pipeline component to optimize. Must implement the BasePipelineComponent interface.
        train_data (Union[FeatureSet, DataBatch]): Training data to use for cross-validation.
        validation_data (Union[FeatureSet, DataBatch]): Validation data to evaluate final performance.
        param_grid (Dict[str, List[Any]]): Dictionary mapping parameter names to lists of values to try.
        scoring_method (str): Scoring method to use for evaluation (e.g., 'accuracy', 'f1', 'mse'). Defaults to 'accuracy'.
        cv_folds (int): Number of cross-validation folds to use. Defaults to 5.
        max_evaluations (int): Maximum number of parameter combinations to evaluate. Defaults to 100.
        timeout_seconds (Optional[int]): Maximum time in seconds to spend tuning. Defaults to None (no limit).
        random_state (Optional[int]): Random seed for reproducibility. Defaults to None.

    Returns:
        ComponentConfig: Configuration object containing the optimal hyperparameters found during tuning.

    Raises:
        ValueError: If the parameter grid is empty or if train/validation data shapes are incompatible.
        TimeoutError: If the tuning process exceeds the specified timeout.
    """
    if not param_grid:
        raise ValueError('Parameter grid cannot be empty')
    if isinstance(train_data, FeatureSet) and isinstance(validation_data, FeatureSet):
        if train_data.features.shape[1] != validation_data.features.shape[1]:
            raise ValueError('Train and validation data must have the same number of features')
    elif isinstance(train_data, DataBatch) and isinstance(validation_data, DataBatch):
        if train_data.X.shape[1] != validation_data.X.shape[1]:
            raise ValueError('Train and validation data must have the same number of features')
    else:
        train_features = train_data.features if hasattr(train_data, 'features') else train_data.X
        val_features = validation_data.features if hasattr(validation_data, 'features') else validation_data.X
        if train_features.shape[1] != val_features.shape[1]:
            raise ValueError('Train and validation data must have the same number of features')

    def _scoring_function(config: ComponentConfig, X, y):
        pipeline_copy = pipeline.__class__()
        pipeline_copy.set_config(config.parameters)
        if hasattr(X, 'iloc'):
            data_for_cv = DataBatch(X, y)
        else:
            data_for_cv = DataBatch(X.tolist(), y.tolist()) if isinstance(X, np.ndarray) else DataBatch(list(X), list(y))
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        except ValueError:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        start_time = time.time()
        try:
            for (train_idx, val_idx) in cv.split(data_for_cv):
                if timeout_seconds and time.time() - start_time > timeout_seconds:
                    raise TimeoutError('Cross-validation exceeded timeout')
                if hasattr(X, 'iloc'):
                    (X_train_fold, X_val_fold) = (X.iloc[train_idx], X.iloc[val_idx])
                    (y_train_fold, y_val_fold) = (y.iloc[train_idx], y.iloc[val_idx])
                else:
                    (X_train_fold, X_val_fold) = (X[train_idx], X[val_idx])
                    (y_train_fold, y_val_fold) = (y[train_idx], y[val_idx])
                pipeline_copy = pipeline.__class__()
                pipeline_copy.set_config(config.parameters)
                pipeline_copy.process(DataBatch(X_train_fold, y_train_fold))
                try:
                    test_batch = DataBatch(X_val_fold, y_val_fold)
                    processed_data = pipeline_copy.process(test_batch)
                    if scoring_method == 'accuracy':
                        if hasattr(processed_data, 'predictions'):
                            from sklearn.metrics import accuracy_score
                            score = accuracy_score(y_val_fold, processed_data.predictions)
                        else:
                            from sklearn.metrics import accuracy_score
                            score = accuracy_score(y_val_fold, processed_data)
                    elif scoring_method == 'f1':
                        if hasattr(processed_data, 'predictions'):
                            from sklearn.metrics import f1_score
                            score = f1_score(y_val_fold, processed_data.predictions, average='weighted')
                        else:
                            from sklearn.metrics import f1_score
                            score = f1_score(y_val_fold, processed_data, average='weighted')
                    elif scoring_method == 'mse':
                        if hasattr(processed_data, 'predictions'):
                            from sklearn.metrics import mean_squared_error
                            score = -mean_squared_error(y_val_fold, processed_data.predictions)
                        else:
                            from sklearn.metrics import mean_squared_error
                            score = -mean_squared_error(y_val_fold, processed_data)
                    else:
                        raise ValueError(f'Unsupported scoring method: {scoring_method}')
                    scores.append(score)
                except Exception:
                    scores.append(-np.inf)
            return np.mean(scores) if scores else -np.inf
        except TimeoutError:
            raise
        except Exception:
            return -np.inf
    search_space = {}
    for (param_name, param_values) in param_grid.items():
        search_space[param_name] = param_values
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    n_iter = min(max_evaluations, total_combinations)
    tuner = RandomizedSearchTuner(search_space=search_space, n_iter=n_iter, objective_function=_scoring_function, random_state=random_state)
    if isinstance(train_data, FeatureSet):
        X_train = train_data.features
        y_train = train_data.targets
    else:
        X_train = train_data.X
        y_train = train_data.labels
    start_time = time.time()
    try:
        tuner.fit(X_train, y_train)
    except TimeoutError:
        raise
    except Exception as e:
        if timeout_seconds and time.time() - start_time > timeout_seconds:
            raise TimeoutError('Hyperparameter tuning exceeded timeout')
        else:
            raise RuntimeError(f'Error during hyperparameter tuning: {str(e)}')
    if timeout_seconds and time.time() - start_time > timeout_seconds:
        raise TimeoutError('Hyperparameter tuning exceeded timeout')
    best_config = tuner.get_best_configuration()
    if best_config is None:
        best_config = ComponentConfig(component_name=pipeline.name, component_type=pipeline.__class__.__name__, parameters={})
    return best_config