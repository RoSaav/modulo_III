import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..'))
from src_train.utils import preprocess_train

if __name__ == "__main__":
    try:
        preprocess_train(
         data_path='TITANIC/data/phpMYEkMl.csv',
         drop_features=['boat','body','home.dest','cabin', 'name', 'ticket'],
         categorical_features=['title', 'cabin_letter', 'sex','embarked'],
         numerical_features=['pclass','age','sibsp','parch','fare'],
         target='survived',
         random_state=666,
         classifier='log',
         save_model=True
        )
        
    except Exception as e:
        print('Something went wrong!')