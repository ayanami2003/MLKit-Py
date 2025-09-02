import numpy as np
import pandas as pd
from general.base_classes.transformer_base import BaseTransformer
from general.structures.feature_set import FeatureSet
from typing import List, Optional, Union

class CrossTabulator(BaseTransformer):

    def __init__(self, row_vars: List[str], col_vars: List[str], agg_func: str='count', fill_value: Optional[Union[int, float]]=None, normalize: bool=False, name: Optional[str]=None):
        """
        Initialize the CrossTabulator.
        
        Parameters
        ----------
        row_vars : List[str]
            Names of variables to use as rows in the cross-tabulation
        col_vars : List[str]
            Names of variables to use as columns in the cross-tabulation
        agg_func : str, default='count'
            Aggregation function to apply ('count', 'sum', 'mean', 'median', etc.)
        fill_value : Optional[Union[int, float]], default=None
            Value to replace missing values with
        normalize : bool, default=False
            Whether to normalize the cross-tabulation to show proportions
        name : Optional[str], default=None
            Name of the transformer instance
        """
        super().__init__(name=name)
        self.row_vars = row_vars
        self.col_vars = col_vars
        self.agg_func = agg_func
        self.fill_value = fill_value
        self.normalize = normalize

    def fit(self, data: FeatureSet, **kwargs) -> 'CrossTabulator':
        """
        Fit the cross-tabulator to the input data.
        
        This method validates that the specified variables exist in the data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing categorical variables
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        CrossTabulator
            Self instance for method chaining
        """
        if data.feature_names is None:
            raise ValueError('FeatureSet must have feature names defined')
        missing_vars = []
        for var in self.row_vars + self.col_vars:
            if var not in data.feature_names:
                missing_vars.append(var)
        if missing_vars:
            raise ValueError(f'The following variables were not found in the FeatureSet: {missing_vars}')
        self.is_fitted_ = True
        return self

    def transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Compute the cross-tabulation on the input data.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set containing variables to cross-tabulate
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            Feature set containing the cross-tabulation matrix
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError('Transformer has not been fitted yet. Call fit() first.')
        if data.feature_names is None:
            raise ValueError('FeatureSet must have feature names defined')
        df = pd.DataFrame(data.features, columns=data.feature_names)
        if not self.row_vars or not self.col_vars:
            raise ValueError('Both row_vars and col_vars must be specified and non-empty')
        if self.agg_func == 'count':
            if len(self.row_vars) == 1 and len(self.col_vars) == 1:
                result = pd.crosstab(df[self.row_vars[0]], df[self.col_vars[0]], normalize=self.normalize, dropna=False)
            else:
                if len(self.row_vars) > 1:
                    row_combined = pd.Series(list(zip(*[df[var] for var in self.row_vars])), name='_row_combined')
                else:
                    row_combined = df[self.row_vars[0]]
                if len(self.col_vars) > 1:
                    col_combined = pd.Series(list(zip(*[df[var] for var in self.col_vars])), name='_col_combined')
                else:
                    col_combined = df[self.col_vars[0]]
                result = pd.crosstab(row_combined, col_combined, normalize=self.normalize, dropna=False)
        else:
            group_cols = self.row_vars + self.col_vars
            values_col = None
            if 'values' in df.columns:
                values_col = 'values'
            elif self.agg_func == 'count':
                df['_temp_agg_values'] = 1
                values_col = '_temp_agg_values'
            else:
                raise ValueError(f"Aggregation function '{self.agg_func}' requires a 'values' column in the data")
            grouped = df.groupby(group_cols)[values_col].agg(self.agg_func)
            if len(self.col_vars) == 1:
                result = grouped.unstack(level=-1)
            else:
                unstack_levels = list(range(len(self.row_vars), len(group_cols)))
                result = grouped.unstack(level=unstack_levels)
            if self.agg_func in ['count', 'sum']:
                result = result.fillna(0)
            if self.normalize:
                total = result.sum().sum()
                if total > 0:
                    result = result / total
            if values_col == '_temp_agg_values':
                df.drop('_temp_agg_values', axis=1, inplace=True)
        if self.fill_value is not None:
            result = result.fillna(self.fill_value)
        features = result.values
        if isinstance(result.columns, pd.MultiIndex):
            feature_names = ['_'.join(map(str, col)) for col in result.columns]
        else:
            feature_names = [str(col) for col in result.columns]
        if isinstance(result.index, pd.MultiIndex):
            sample_ids = ['_'.join(map(str, row)) for row in result.index]
        else:
            sample_ids = [str(row) for row in result.index]
        return FeatureSet(features=features, feature_names=feature_names, sample_ids=sample_ids, metadata={'transformer': 'CrossTabulator', 'row_vars': self.row_vars, 'col_vars': self.col_vars, 'agg_func': self.agg_func, 'normalize': self.normalize})

    def inverse_transform(self, data: FeatureSet, **kwargs) -> FeatureSet:
        """
        Inverse transformation is not supported for cross-tabulation.
        
        Parameters
        ----------
        data : FeatureSet
            Input feature set (ignored)
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        FeatureSet
            The same input data (no transformation applied)
            
        Raises
        ------
        NotImplementedError
            Always raised as inverse transformation is not meaningful for cross-tabulation
        """
        raise NotImplementedError('Inverse transformation is not supported for cross-tabulation')