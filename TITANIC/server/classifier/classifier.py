import sys
import os
import pandas as pd
import numpy as np
from src_train.utils import preprocess_train
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../'))

from server.models.models import Titanic

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

    def classify_titanic(self, titanic: Titanic):
        X = [titanic.pclass, titanic.name, titanic.sex, titanic.age, 
             titanic.sibsp, titanic.parch, titanic.ticket, titanic.fare, 
             titanic.cabin, titanic.embarked, titanic.boat, titanic.body, titanic.home_dest]

        X = pd.DataFrame([X], columns = ['pclass', 'name', 'sex', 'age','sibsp', 'parch', 'ticket', 'fare','cabin', 'embarked', 'boat', 'body', 'home.dest'])

        prediction = self.clf.predict_proba(X)
        print({'class': self.survive_type[np.argmax(prediction)],
                'probability': round(max(prediction[0]), 2)})
        return {'class': self.survive_type[np.argmax(prediction)],
                'probability': round(max(prediction[0]), 2)}