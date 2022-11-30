import os
import sys
import pandas as pd
import numpy as np
import functools
from sklearn.pipeline import Pipeline
import pytest
import shutil
import joblib
from datetime import datetime
import pdb;pdb.set_trace()
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..'))
from src_train.utils import preprocess_train


@pytest.fixture(scope="function")
def df_natural():
    return pd.read_csv(r'..\data\test_utils.csv')


@pytest.fixture(scope="function")
def model_pipeline_stored():
    model_pipeline_path = r'..\server\models\modelo_pipeline.joblib'
    model_pipeline=joblib.load(model_pipeline_path)

    return model_pipeline


def obtener_datos_test_integration():
    return [(True)]


@pytest.mark.parametrize('Bool', obtener_datos_test_integration())
def test_integration(Bool, model_pipeline_stored, df_natural):

    model_pipeline_new=preprocess_train(
                            data_path='../data/phpMYEkMl.csv',
                            drop_features=['boat','body','home.dest','cabin', 'name', 'ticket'],
                            categorical_features=['title', 'cabin_letter', 'sex','embarked'],
                            numerical_features=['pclass','age','sibsp','parch','fare'],
                            target='survived',
                            random_state=666,
                            classifier='log',
                            save_model=False
                    )
    
    df_natural.pop('survived')

    proba_a=model_pipeline_new.predict_proba(df_natural)[:,1]
    proba_b=model_pipeline_stored.predict_proba(df_natural)[:,1]
    assert functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q, proba_a, proba_b), True)    

    
    

    
