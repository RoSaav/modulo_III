import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropFeatures
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder 
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib

from utils import TypeConverter, CategoricalImputer, ExtractTitle, ExtractCabin

def preprocess_train(
         data_path : str='TITANIC/data/phpMYEkMl.csv',
         drop_features : list=['boat','body','home.dest','cabin', 'name', 'ticket'],
         categorical_features : list=['sex','embarked', 'title', 'cabin_letter'],
         numerical_features : list=['pclass','age','sibsp','parch','fare'],
         target : str='survived',
         random_state : int=666
         ):

    """Connects to the next available port.

    Args:
      data_path: direction to titanic data
      vars_drop: vars to be droped

    Returns:
      raw data.

    Raises:

    """
    # Loading data 
    df = pd.read_csv(data_path)

    pipeline = Pipeline(
                              [ ('extract_cabin', ExtractCabin()),
                                ('extract_letter', ExtractTitle()),
                                ('type_converter', TypeConverter(numerical_features=numerical_features, categorical_features=categorical_features)),
                                ('DropFeatures', DropFeatures(features_to_drop=drop_features)),
                                ('categorical_imputer', CategoricalImputer(variables=categorical_features)),
                                ('simple_imputer', SklearnTransformerWrapper(SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1), variables=numerical_features)),
                                ('rare_label_encoder', RareLabelEncoder(variables=categorical_features, tol=0.01, n_categories=5, max_n_categories=20, replace_with='OTHER', ignore_format=True)),
                                ('ordinal_encoder', OrdinalEncoder(variables=categorical_features, encoding_method='ordered', ignore_format=True)),
                                ('log_reg', LogisticRegression(C=0.0005, class_weight='balanced', random_state=random_state))
                              ])

 

    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),df[target],test_size=0.25,random_state=random_state)
    
    pipeline.fit(X_train, y_train)
    
    class_pred = pipeline.predict(X_test)
    proba_pred = pipeline.predict_proba(X_test)[:,1]
    print('test roc-auc : {}'.format(roc_auc_score(y_test, proba_pred)))
    print('test accuracy: {}'.format(accuracy_score(y_test, class_pred)))
    print()

    
    model_pipeline_path = r'TITANIC\models\modelo_pipeline.joblib'
    joblib.dump(pipeline, model_pipeline_path)

    return print('Model stored')

if __name__ == "__main__":
    try:
        preprocess_train(
          data_path = r'TITANIC\data\phpMYEkMl.csv'
        )
        
    except Exception as e:
        print('Something went wrong!')