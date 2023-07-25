# OpenClassRooms_Projet7
#### <i>Implémentez un modèle de scoring</i>

## Présentation
Lobjectif principal de ce projet est de prédire le risque de faillite d'un client pour la société Prêt à dépenser. Pour cela, nous devons configurer un modèle de classification binaire et d'évaluer les différentes métriques.

Ce projet consiste à créer une API web avec un Dashboard interactif. Celui-ci devra à minima contenir les fonctionnalités suivantes :

 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
 - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.


## Construction

## OC_DS_P7_api_flask
<u>Dans ce dépôt, vous trouverez :</u>

 - un dossier /data/ qui contient les données d'entrainement, de tests, un fichier avec les labels et la description des features 
 - Un dossier avec le Notebook Jupyter pour l'étude des données, l'entraînement et la configuration du modèle de classification.
 - un dossier /medels/ qui contient le model utilisé
 - un dossier /testing/ qui tete le code avec le déploiement continu
 - Un dossier avec la note technique qui explique en détails la construction et les résultats du modèle.
 - Sous la racine les fichiers de configuration de l'API. Dans le but de comprendre le fonctionnement de Flask, cette "version" de l'API s'appuie sur deux fichiers .py :
    - app.py qui est le fichier Flask contenant la partie backend.
    - requirements.txt : le fichier des version des librairies utilisées pour l'API
 
## Le dépôt OC_DS_Projet7_dashboard_streamlit :
<u>Dans ce dépôt, vous trouverez :</u>

  - un dossier /testing/ qui tete le code avec le déploiement continu
  - le fichier log.png qui représente le logo de l'API et de la société
  - main.py contient la partie Frontend codée avec Streamlit
  - requirements.txt : le fichier de version des librairies utilisées pour l'API Backend
  - streamlit-1.9.0-py2.py3-none-any.whl

## Le dépôt OC_DS_P7_Notebook :
<u>Dans ce dépôt, vous trouverez :</u>

  - P7_1_exploration.ipynb : Notebook d'exploration
  - P7_2_modeling.ipynb : Notebook de modélisation
  - P7_4_mlflow.ipynb : notebook de MLFlow
  - train.py : Fichier python de Data Drift avec Evdently


## Modèle de classification
Le modèle retenu pour cet exercice est le modèle RandomForestClassifier. 

## Dashboard / API
J'ai utilisé deux librairies Python pour ce sujet :
 - Flask
 - Streamlit

## Données d'entrées
 - Lien de téléchargement des données d'entrées : https://www.kaggle.com/c/home-credit-default-risk/data 

