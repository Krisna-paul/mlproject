import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models,param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            parameters = param[list(models.keys())[i]]

            grids_cv = GridSearchCV(model, parameters, cv=5, n_jobs=-1, verbose=2)
            grids_cv.fit(X_train, y_train)

            model.set_params(**grids_cv.best_params_)
            model.fit(X_train, y_train)
            

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)
            train_model_score = r2_score(y_train, y_train_pred)

            report[list(models.keys())[i]] = test_model_score


        return report

    except Exception as e:
        logging.info("Error in model evaluation")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error in loading object")
        raise CustomException(e, sys)
