import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

class ExtractCabin(BaseEstimator, TransformerMixin):
    """Connects to the next available port.

    Args:
      minimum: A port value greater or equal to 1024.

    Returns:
      The new minimum port.

    Raises:

    """
    
    def __init__(self):
        self.variable = 'cabin'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['cabin_letter'] = X[self.variable].apply(lambda x: np.nan if x == '?' else x[0][0])
        #X['cabin_letter'] = X[self.variable].apply(lambda x: ''.join(re.findall("[a-zA-Z]+", x)) if type(x)==str else x)
        return X

class ExtractTitle(BaseEstimator, TransformerMixin):

    """Extract the title name.

    Args:
      input_string: string name.

    Returns:
      The title name.

    Raises:

    """
    
    def __init__(self):
        self.variable = 'name'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['title'] = X[self.variable].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        
        return X

class TypeConverter(object):
    """Extract the title name.

    Args:
      numerical_features: List of numerical features
      categorical_features: List of categorical features

    Returns:
      The title name.

    Raises:

    """
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):

        X.replace('?', np.nan, inplace=True)

        # Converting numerical features to float64
        X[self.numerical_features] = X[self.numerical_features].astype('float64', errors='ignore')
        #X.loc[:, self.numerical_features] = X.loc[:, self.numerical_features].astype('float64', errors='ignore')
        # Converting categorical variables to object and cleaning categories
        X[self.categorical_features] = X[self.categorical_features].astype(object, errors='ignore')
        #X.loc[:, self.categorical_features] = X.loc[:, self.categorical_features].astype(object, errors='ignore')
        return X



class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Extract the title name.

    Args:
      variables: List of variables.

    Returns:
      The title name.

    Raises:

    """
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, value = 'MISSING'):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(value)
        return X
