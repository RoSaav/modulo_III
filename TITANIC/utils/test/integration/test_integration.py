import os
import sys
import pandas as pd
import numpy as np
import functools
from sklearn.pipeline import Pipeline
import pytest
import shutil
from datetime import datetime
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..'))
from src_utils.utils import ExtractCabin, ExtractTitle, TypeConverter


@pytest.fixture(scope="function")
def df_natural():
    return pd.read_csv(r'..\data\test_utils.csv')

@pytest.fixture(scope="function")
def df_test():

    categorical_features=['sex','embarked', 'title', 'cabin_letter']
    numerical_features=['pclass','age','sibsp','parch','fare']

    df=pd.read_csv(r'..\data\test_utils.csv')

    df['cabin_letter'] = df['cabin'].apply(lambda x: np.nan if x == '?' else x[0][0])
    df['title'] = df['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

    df.replace('?', np.nan, inplace=True)
    # Converting numerical features to float64
    df[numerical_features]=df[numerical_features].astype('float64', errors='ignore')
    # Converting categorical variables to object and cleaning categories
    df[categorical_features]=df[categorical_features].astype(object, errors='ignore')
    return df


def obtener_datos_test_integration():
    return [(True)]


@pytest.mark.parametrize('Bool', obtener_datos_test_integration())
def test_add_to_list(Bool, df_test, df_natural):
    
    categorical_features=['sex','embarked', 'title', 'cabin_letter']
    numerical_features=['pclass','age','sibsp','parch','fare']

    pipeline = Pipeline([ ('extract_cabin', ExtractCabin()),
                          ('extract_letter', ExtractTitle()),
                          ('type_converter', TypeConverter(numerical_features=numerical_features, 
                                                        categorical_features=categorical_features))
    ])

    pd.testing.assert_frame_equal(pipeline.transform(df_natural), df_test)