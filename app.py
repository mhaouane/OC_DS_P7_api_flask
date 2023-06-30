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
import numpy as np
import sklearn
import flask
import json
from flask import Flask, render_template, request, jsonify, app, url_for
import connexion
from markupsafe import escape
import shap
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

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
# app = connexion.App(__name__, specification_dir='./')
# MH le 30/06 render
app = Flask(__name__)

# Test : http://127.0.0.1:5000/
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/
# Test render: https://oc-api-flask-mh.onrender.com
# Create a URL route in our application for "/"
@app.route('/')
#def index():
#    return render_template('index.html')

def home():
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return: the rendered template 'home.html'
    """
    return 'Hello Word!! I am Mohamed'
# If we're running in stand alone mode, run the application

# Route liste Id
# First 10 lines
# Test local : http://127.0.0.1:5000/api/list_id/
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/list_id/
# Test render : https://oc-api-flask-mh.onrender.com/api/list_id/
@app.route('/api/list_id/')	
def list_id():	
    # Extract list 'SK_ID_CURR' ids in the X_test dataframe	
    list_id = pd.Series(list(X_test.index.sort_values()))
    # Convert pd.Series to JSON	and get oly 10 firts sk_ids
    list_ids_json = json.loads(list_id[:10].to_json())# First 10 lines
    # Returning the processed data	
    return jsonify({'status': 'ok',	
    		        'data': list_ids_json})

# Json object of feature description 
# Test local : http://127.0.0.1:5000/api/feat_desc
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/feat_desc
# Test render : https://oc-api-flask-mh.onrender.com/api/feat_desc
@app.route('/api/feat_desc/')
def send_feat_desc():
    # Convert pd.Series to JSON
    features_desc_json = json.loads(feat_desc.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': features_desc_json})


# Route show data Id : return data of one customer when requested (sk_id_cust)
# Test local : http://127.0.0.1:5000/api/get_data_cust/?SK_ID_CURR=163742
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/get_data_cust/?SK_ID_CURR=163742
# Test render : https://oc-api-flask-mh.onrender.com/api/get_data_cust/?SK_ID_CURR=163742
@app.route('/api/get_data_cust/')	
def get_data_cust():	
    # Parse the http request to get arguments (sk_id_cust)	
    sk_id_cust = int(request.args.get('SK_ID_CURR'))	
    # Get the personal data for the customer (pd.Series)	
    X_cust_ser = X_test.loc[sk_id_cust, :].round(2)	
    X_cust_json = json.loads(X_cust_ser.to_json())	
    # Return the cleaned data	
    return jsonify({'status': 'ok',	
    				'data': X_cust_json})

# Route show data Id : return mean data of dataset 
# Test local : http://127.0.0.1:5000/api/mean/
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/mean/
# Test render : https://oc-api-flask-mh.onrender.com/api/mean/
@app.route('/api/mean/')	
def mean_income():	
    # Parse the http request to get mean income for a group	
    # Parse the http request to get arguments (sk_id_cust)	
    feat_mean = X_test.mean().round(2)#.to_dict()
    # Convert pd.Series to JSON
    feat_mean_json = json.loads(feat_mean.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': feat_mean_json})
    #return f' Bonjour salaire de :  {(sk_id_cust)} et salaire moyen du groupe : '

# find 10 nearest neighbors among the training set
def get_df_neigh(sk_id_cust):
    # get data of 10 nearest neigh in the X_tr_featsel dataframe (pd.DataFrame)
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(X_train)
    X_cust = X_test.loc[sk_id_cust: sk_id_cust]
    idx = neigh.kneighbors(X=X_cust,
                           n_neighbors=10,
                           return_distance=False).ravel()
    nearest_cust_idx = list(X_train.iloc[idx].index)
    X_neigh_df = X_train.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]
    return X_neigh_df, y_neigh

# return data of 10 neighbors of one customer when requested (SK_ID_CURR)
# Test local : http://127.0.0.1:5000/api/neigh_cust/?SK_ID_CURR=100038
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/neigh_cust/?SK_ID_CURR=100128
# Test render : https://oc-api-flask-mh.onrender.com/api/neigh_cust/?SK_ID_CURR=100128
@app.route('/api/neigh_cust/')
def neigh_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh_df, y_neigh = get_df_neigh(sk_id_cust)
    # Convert the customer's data to JSON
    X_neigh_json = json.loads(X_neigh_df.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_neigh': X_neigh_json,
    				'y_neigh': y_neigh_json})


# Test local : http://127.0.0.1:5000/api/make_prediction/163742
# Test local : http://127.0.0.1:5000/api/scoring_cust/?SK_ID_CURR=163742 --> OK
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/scoring_cust/?SK_ID_CURR=319313
# Test render : https://oc-api-flask-mh.onrender.com/api/scoring_cust/?SK_ID_CURR=319313
# http://127.0.0.1:5000/api/scoring_cust/?SK_ID_CURR=319313 OK
# 
'''
1- Charger Dataframe OK
2- Isoler la ligne qui nous intéresse OK
3- Isoler le vecteur de la personne ?? Comment avec ColumnTransformer : transforme
4- Appeler le modèle : model.predict (model) : doit renvoyer 1 ou 0
5- ou mieux le pourcentage ( 0 : proba prédiction 0.03 car si proba de 0.49 ce n’est pas comme 0.03 -> OK pour le X_train
'''
@app.route('/api/scoring_cust/')
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    
    # Get the data for the customer (pd.DataFrame) --> for SK_ID_CURR' in X_train
    #X_cust = X_train.loc[sk_id_cust:sk_id_cust]

    # Get the data for the customer (pd.DataFrame)
    X_cust = X_test.loc[sk_id_cust:sk_id_cust]
	# Compute the score of the customer (using the whole pipeline)   
    score_cust = best_model.predict_proba(X_cust)[:,1][0]
    # Compute decision according to the best threshold (True: loan refused)
    bool_cust = (score_cust >= threshold)
    # Return processed data
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': sk_id_cust,
    		        'score': score_cust,
    		        'answer': str(bool_cust),
                    'thresh': threshold})

# Test avec un elt du X_train:
# Test local   http://127.0.0.1:5000/api/scoring_cust_train/?SK_ID_CURR=296757 --> OK
# Test local  Pas  de crédit http://127.0.0.1:5000/api/scoring_cust_train/?SK_ID_CURR=231811
# Test local  Pas  de crédit http://127.0.0.1:5000/api/scoring_cust_train/?SK_ID_CURR=100002
# http://127.0.0.1:5000/api/scoring_cust_train/?SK_ID_CURR=159992
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/scoring_cust_train/?SK_ID_CURR=100002
# Test render : https://oc-api-flask-mh.onrender.com/api/scoring_cust_train/?SK_ID_CURR=100002
@app.route('/api/scoring_cust_train/')
def scoring_cust_tr():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    
    # Get the data for the customer (pd.DataFrame) --> for SK_ID_CURR' in X_train
    X_cust = X_train.loc[sk_id_cust:sk_id_cust]

    # Get the data for the customer (pd.DataFrame)
    #X_cust = X_test.loc[sk_id_cust:sk_id_cust]
	# Compute the score of the customer (using the whole pipeline)   
    score_cust = best_model.predict_proba(X_cust)[:,1][0]
    # Compute decision according to the best threshold (True: loan refused)
    bool_cust = (score_cust >= threshold)
    # Return processed data
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': sk_id_cust,
    		        'score': score_cust,
                    'answer': str(bool_cust),
                    'thresh': threshold})

# Json object of feature importance ()
# Test local : http://127.0.0.1:5000/api/feat_imp
# Test Heroku : https://oc-api-flask-mh.herokuapp.com/api/feat_imp
# Test render : https://oc-api-flask-mh.onrender.com/api/feat_imp
@app.route('/api/feat_imp/')
def send_feat_imp():
    feat_imp = pd.Series(best_model.feature_importances_.round(2),
                         index=X_train.columns).sort_values(ascending=False)
    # Convert pd.Series to JSON
    feat_imp_json = json.loads(feat_imp.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': feat_imp_json})

#if __name__ == '__main__':

#    app.run(debug=True)