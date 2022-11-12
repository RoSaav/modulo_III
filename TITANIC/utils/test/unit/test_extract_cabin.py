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
from src_utils.utils import ExtractCabin


@pytest.fixture(scope="function")
def df_test():
    return pd.read_csv(r'..\data\test_utils.csv')


def obtener_datos_test_extract_cabin():
    return [(True, ['D', 'nan', 'A', 'E', 'nan', 'nan', 'nan', 'nan', 'nan', 'D'])]


@pytest.mark.parametrize('Bool, ls', obtener_datos_test_extract_cabin())
def test_add_to_list(Bool, ls, df_test):

    pipeline = Pipeline([('extract_cabin', ExtractCabin())])
    lst_test = pipeline.transform(df_test).cabin_letter.map(lambda x:str(x)).to_list()

    assert functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q, ls, lst_test), True)
