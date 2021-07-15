from flask import render_template, redirect
from app import app
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired
from app.forms import SubmitReplay
from flask import request
from helper import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import layers
from random import randrange, random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from helper import *
import math
import logging
from retrying import retry
import os
import numpy as np
import numpy.random as rand
from itertools import islice
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier,RandomForestClassifier)
from sklearn.svm import SVC
import sklearn.datasets as datasets
import sklearn.model_selection as cv
from sklearn.inspection import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import requests
import joblib
rcParams['figure.figsize'] = (12, 8)
plt.style.use('ggplot')




@app.route('/')
@app.route('/index')
def index():
    return render_template('base.html', title='Home')
    
@app.route('/submit-replay', methods=['POST', 'GET'])
def submit_replay():
    form = SubmitReplay()
    return render_template('submit_replay.html', title='Submit Replay', form=form)

@app.route('/handle_data', methods=['GET', 'POST'])
def handle_data():
    best_model = joblib.load('/home/nicks/Galvanize/projects/rocketleague/best_model')
    rank_dict = {0 :'unranked', 1: 'Bronze 1', 2:'Bronze 2', 3: 'Bronze 3', 4: 'Silver 1', 5:'Silver 2',  6:'Silver 3', 7:
    'Gold 1', 8:'Gold 2', 9:'Gold 3', 10:'Plat 1', 11:'Plat 2', 12:'Plat 3', 13:'Diamond 1', 14:'Diamond 2', 15:'Diamond 3', 16: 'Champ 1', 17:'Champ 2', 18:'Champ 3', 19:'Grand Champ', 20: 'Grand Champ', 21: 'SuperSonic Legend (Liklely Failed Prediction'}
    div_dict = {0.0:'Division 1', 0.25:'Division 2', 0.5:'Division 3', 0.75:'Division 4'}

    a = request.form['replay_id']
    lst = []
    lst.append(a)
    chris_stats = get_stats_from_replays_combined(lst)
    chris_df, names = create_frame_from_jsonv2(chris_stats)
    chris_df = chris_df[best_model.feature_names]
    chris_inp = xgb.DMatrix(chris_df)
    preds = best_model.predict(chris_inp)
    preds_round = [round((math.floor(i))/4) for i in preds]
    div_remainder = [math.floor(i)/4 for i in preds]
    div_round = [i%1 for i in div_remainder]
    names = [names[i] for i in range(len(names))]
    pred_output = [(names[i], rank_dict[preds_round[i]], div_dict[div_round[i]]) for i in range(len(preds))]
    # for i in range(len(preds)):
    #     print(f'Predicted rank for {names.iloc[i]} is {rank_dict[preds_round[i]], div_dict[div_round[i]]}')
    return render_template('results.html', title='Results', results = pred_output)

@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html', title='Home')

@app.route('/feature-importance', methods=['GET'])
def feature_importance():
    return render_template('feature_importance.html')

@app.route('/explanation', methods=['GET'])
def explanation():
    return render_template('explanation.html')