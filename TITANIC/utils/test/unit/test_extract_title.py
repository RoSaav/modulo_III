import os
import sys
import pandas as pd
import numpy as np
import functools
from sklearn.pipeline import Pipeline
import pytest
import shutil
from datetime import datetime
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..') )
from src_utils.utils import ExtractTitle

@pytest.fixture(scope="function")
def df_test():
    return pd.read_csv(r'..\data\test_utils.csv')


def obtener_datos_test_extract_title():
    return [(True, ['Mr', 'Rev', 'Mr', 'Mrs', 'Mrs', 'Mr', 'Mr', 'Mr', 'Rev', 'Mr'])]


@pytest.mark.parametrize('Bool, ls', obtener_datos_test_extract_title())
def test_add_to_list(Bool, ls, df_test):

    pipeline = Pipeline([('extract_letter', ExtractTitle())])
    lst_test = pipeline.transform(df_test).title.map(lambda x:str(x)).to_list()

    assert functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q, ls, lst_test), True)
