import sys
import os
import joblib
import pandas as pd
import numpy as np
sys.path.append( os.path.abspath(os.path.dirname(__file__)+'/../'))

from server.models.models import Titanic


class TitanicClassifier:
    def __init__(self):
        self.clf = self.train_model()
        self.survive_type = {
            0: 'no_survive',
            1: 'survive',
        }

    def train_model(self):
        model_pipeline_path = sys.path[-1]+'/models/modelo_pipeline.joblib'
        model_pipeline=joblib.load(model_pipeline_path)
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