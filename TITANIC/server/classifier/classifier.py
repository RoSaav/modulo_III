import sys
import os
import pandas as pd
import numpy as np
from src_train.utils import preprocess_train
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../..'))

import pdb;pdb.set_trace()

class TitanicClassifier:
    def __init__(self):
        self.clf = self.train_model()
        self.survive_type = {
            0: 'no_survive',
            1: 'survive',
        }

    def train_model(self) -> preprocess_train:
        model_pipeline = preprocess_train(
                            data_path='TITANIC/data/phpMYEkMl.csv',
                            drop_features=['boat','body','home.dest','cabin', 'name', 'ticket'],
                            categorical_features=['title', 'cabin_letter', 'sex','embarked'],
                            numerical_features=['pclass','age','sibsp','parch','fare'],
                            target='survived',
                            random_state=666,
                            classifier='log',
                            save_model=True
        )
        return model_pipeline

    def classify_iris(self, titanic: Titanic):
        X = [iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]
        prediction = self.clf.predict_proba([X])
        print({'class': self.iris_type[np.argmax(prediction)],
                'probability': round(max(prediction[0]), 2)})
        return {'class': self.iris_type[np.argmax(prediction)],
                'probability': round(max(prediction[0]), 2)}