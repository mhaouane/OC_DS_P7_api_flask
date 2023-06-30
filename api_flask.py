''' 
-*- coding: utf-8 -*-
To run from the directory 'WEB':
python api/server.py
Author : Mohamed HAOUANE
'''

# Load Librairies
import os
import sys
import joblib
import pandas as pd
import sklearn
import flask
import json
from flask import render_template, request, jsonify, app
import connexion
from markupsafe import escape
import shap
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors

##############################################################################
## Loading model
##############################################################################

# Best model and best threshold	0
with open(os.path.join('models', 'model_randforest.pkl'), 'rb') as file:	
    best_model = joblib.load(file)	
threshold = 0.51
#---------------------------


# Split the steps of the best pipeline
#preproc_step = bestmodel.named_steps['preproc']
#featsel_step = bestmodel.named_steps['featsel']
#clf_step = best_model.named_steps['clf']

# Utilisez l'attribut named_steps ou steps pour inspecter les estimateurs dans le pipeline. La mise en cache des transformateurs est avantageuse lorsque le montage prend du temps.
##############################################################################
## Loading data
##############################################################################
#  Processed data
path = os.path.join('data', 'X_train.csv') # concatenation between path and filename
X_train = pd.read_csv(path, index_col='SK_ID_CURR')
path = os.path.join('data', 'y_train.csv')
y_train = pd.read_csv(path, index_col='SK_ID_CURR')
path = os.path.join('data', 'X_test.csv')	
X_test = pd.read_csv(path, index_col='SK_ID_CURR')

# Description of each feature
path = os.path.join('data', 'feat_desc.csv')
feat_desc = pd.read_csv(path, index_col=0)


# split the steps of the best pipeline
#Use the attribute named_steps or steps to inspect estimators within the pipeline. Caching the transformers is advantageous when fitting is time consuming.
####  ERROR : AttributeError: 'LogisticRegression' object has no attribute 'named_steps
### AttributeError: 'RandomForestClassifier' object has no attribute 'named_steps'
#preproc_step = best_model.named_steps['preproc']
#featsel_step = best_model.named_steps['featsel']
#clf_step = best_model.named_steps['clf']

# compute the preprocessed data (encoding and standardization)
#X_tr_prepro = preproc_step.transform(X_train)
#X_te_prepro = preproc_step.transform(X_test)

#######################################################
# Create the application instance
app = connexion.App(__name__, specification_dir='./')

# Test : http://127.0.0.1:5000/
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/
# Test render: https://oc-api-flask-mh.onrender.com
# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return: the rendered template 'home.html'
    """
    return 'Hello Word'
# If we're running in stand alone mode, run the application

if __name__ == '__main__':
    app.run(debug=True)