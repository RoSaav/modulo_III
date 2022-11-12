import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

PATH_TO_DATA = "data/"


class ExtractCabin(BaseEstimator, TransformerMixin):
    """Extract the Cabin Letter.

    Args:

    Returns:
      X: dataframe with new variable Cabin_letter, if exist cabin.

    Raises:
      There is not variable cabin
    """
    def __init__(self):
        self.variable = 'cabin'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        try:
          X['cabin_letter'] = X[self.variable].apply(lambda x: np.nan if x == '?' else x[0][0])
        except:
          print('There is not variable cabin')
        return X


class ExtractTitle(BaseEstimator, TransformerMixin):
    """Extract the title name.

    Args:

    Returns:
      X: dataframe with new variable title, if exist name.

    Raises:
      'There is not variable name'

    """
    def __init__(self):
        self.variable = 'name'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        try:
          X['title'] = X[self.variable].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        except:
          print('There is not variable name')        
        return X

class TypeConverter(object):
    """Tranform the data into correct type

    Args:
      numerical_features: List of numerical features
      categorical_features: List of categorical features

    Returns:
      X: datafra tranformed

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
    """Impute a value for nans in categorical vars.

    Args:
      variables: List of variables.

    Returns:
      X: datafra tranformed

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
