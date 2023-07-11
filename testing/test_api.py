import os
import pytest 
import unittest
import pathlib as pl
import app
from app import app
import requests
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib
from joblib import load, dump
import flask
import json
from flask import Flask, render_template, request, jsonify, app, url_for

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
"""
file_path_xtrain = "C:/Users/ADSL/Documents/Projet7/backend/data/X_train.csv"
file_path_ytrain = "C:/Users/ADSL/Documents/Projet7/backend/data/Y_train.csv"
file_path_xtest = "C:/Users/ADSL/Documents/Projet7/backend/data/X_test.csv"
file_path_model = "C:/Users/ADSL/Documents/Projet7/backend/models/model_randforest.pkl"
"""

## with open(os.path.join('C:\\Users\\ADSL\\Documents\\Projet7\\models', 'model_randforest.pkl'), 'rb') as file:

"""
with open(os.path.join('models', 'model_randforest.pkl'), 'rb') as file:
    file_path_model = joblib.load(file)	
threshold = 0.51
"""

file_path_model = os.path.join('models', 'model_randforest.pkl')
file_path_xtrain = os.path.join('data', 'X_train.csv') # concatenation between path and filename

file_path_ytrain = os.path.join('data', 'y_train.csv')

file_path_xtest = os.path.join('data', 'X_test.csv')	

# Description of each feature
#path = os.path.join('data', 'feat_desc.csv')
#feat_desc = pd.read_csv(path, index_col=0)

#TestCase : un scénario de test est créé comme une classe fille
# les tests individuels sont défnis par des méthodes dont les noms commencent par test_: 
    # test_exist_file : exitence de fichier

 # Le coeur de chq test est une appel à assertEqual pour vérifier un résultat attendu : assertTrue ou assertFalse ou 
 # assertRAise pour vérifier qu'une exception particulière est levée
class FileTestCase(unittest.TestCase):
    def test_xtrain_load(self):
        # Check the xtrain file is created/saved in the directory
        path = pl.Path(file_path_xtrain)
        self.assertEqual((str(path), path.is_file()), (str(path), True))
        # Check that the X_train file can be loaded properly 
        loaded_xtrain = pd.read_csv(path, index_col='SK_ID_CURR')
        #print(loaded_xtrain)
        assert isinstance(loaded_xtrain, pd.DataFrame)

    def test_ytrain_load(self):
        # ...
        path = pl.Path(file_path_ytrain)
        self.assertEqual((str(path), path.is_file()), (str(path), True))
        # Check that the y_train file can be loaded properly 
        loaded_ytrain = pd.read_csv(path, index_col='SK_ID_CURR')
        #print(loaded_ytrain)
        assert isinstance(loaded_ytrain, pd.DataFrame)

    def test_xtest_load(self):
        # ...
        path = pl.Path(file_path_xtest)
        self.assertEqual((str(path), path.is_file()), (str(path), True))
        # Check that the X_test file can be loaded properly 
        loaded_xtest = pd.read_csv(path, index_col='SK_ID_CURR')
        #print(loaded_xtest)
        assert isinstance(loaded_xtest, pd.DataFrame)

    def test_model_load(self):
        # Check the model file is created/saved in the directory
        path = pl.Path(file_path_model)
        self.assertEqual((str(path), path.is_file()), (str(path), True))
        # Check that the model file can be loaded properly 
        # (by type checking that it is a sklearn random forest classifier)    
        loaded_model = joblib.load(file_path_model)
        assert isinstance(loaded_model, sklearn.ensemble.RandomForestClassifier)

    # Backlog à faire ultérieurement
    #def test_prediction(self):
        # Check that trained model predicts the y (almost) perfectly given X
        # Note the use of np.testing function instead of standard 'assert'
        # To handle numerical precision issues, we should use the `assert_allclose` function instead of any equality check
        #sk_id_cust = int(request.args.get('SK_ID_CURR'))
        #print(sk_id_cust)

        #prediction = app.scoring_cust()
        #np.testing.assert_allclose(loaded_ytrain,loaded_model.predict(loaded_xtrain))

if __name__ == "__main__":
    unittest.main(verbosity=2)
