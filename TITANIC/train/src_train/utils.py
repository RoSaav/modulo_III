import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropFeatures
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder 
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
import joblib
from src_utils.utils import TypeConverter, CategoricalImputer, ExtractTitle, ExtractCabin

sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..'))

def preprocess_train(
         data_path : str='TITANIC/data/phpMYEkMl.csv',
         drop_features : list=['boat','body','home.dest','cabin', 'name', 'ticket', 'sex','embarked'],
         categorical_features : list=['title', 'cabin_letter'],
         numerical_features : list=['pclass','age','sibsp','parch','fare'],
         target : str='survived',
         random_state : int=666,
         classifier: str='log',
         save_model: bool=True
         ):

    """Preprocess and train model.

    Args:
      data_path: direction to titanic data
      drop_features: list vars to be droped
      categorical_features : list vars of categorical
      numerical_features : list vars of categorical
      target : str target var to be train 
      random_state : int 
      classifier: str classifier model

    Returns:
      Saved model

    """
    # Loading data 
    df = pd.read_csv(data_path)
    if classifier=='log':
      model=LogisticRegression(random_state=random_state)
    elif  classifier=='ada':
      model=AdaBoostClassifier(random_state=random_state)
    elif  classifier=='rf':
      model=RandomForestClassifier(random_state=random_state)
    else:
      model=HistGradientBoostingClassifier(random_state=random_state)

    pipeline = Pipeline(
                              [ ('extract_cabin', ExtractCabin()),
                                ('extract_letter', ExtractTitle()),
                                ('type_converter', TypeConverter(numerical_features=numerical_features, categorical_features=categorical_features)),
                                ('DropFeatures', DropFeatures(features_to_drop=drop_features)),
                                ('categorical_imputer', CategoricalImputer(variables=categorical_features)),
                                ('simple_imputer', SklearnTransformerWrapper(SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1), variables=numerical_features)),
                                ('rare_label_encoder', RareLabelEncoder(variables=categorical_features, tol=0.01, n_categories=5, max_n_categories=20, replace_with='OTHER', ignore_format=True)),
                                ('ordinal_encoder', OrdinalEncoder(variables=categorical_features, encoding_method='ordered', ignore_format=True)),
                                ('model', model)
                              ])

 
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),df[target],test_size=0.25,random_state=random_state)
    
    pipeline.fit(X_train, y_train)
    
    class_pred = pipeline.predict(X_test)
    proba_pred = pipeline.predict_proba(X_test)[:,1]
    print('test roc-auc : {}'.format(roc_auc_score(y_test, proba_pred)))
    print('test accuracy: {}'.format(accuracy_score(y_test, class_pred)))
    print('---------------------------------------')

    if save_model:
      model_pipeline_path = r'TITANIC\server\models\modelo_pipeline.joblib'
      joblib.dump(pipeline, model_pipeline_path)
      message='Model stored'
    else:
      message='Model not stored'

    print(message)

    return pipeline