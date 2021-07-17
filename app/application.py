from flask import Flask
from flask import render_template, redirect, request
from forms import SubmitReplay
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
import pandas as pd
import math
import os
import requests
import joblib
import xgboost as xgb
import application

application = Flask(__name__)
# from application import routes


def remove_extra_element(row):
    if 'goals_against_while_last_defender' in row:
        del row['goals_against_while_last_defender']
    return row

def create_frame_from_jsonv2(json_stats):
    df = pd.DataFrame.from_dict(json_stats)
    df['core'] = df.stats.apply(lambda x:x['core'])
    df['boost'] = df.stats.apply(lambda x:x['boost'])
    df['movement'] = df.stats.apply(lambda x:x['movement'])
    df['positioning'] = df.stats.apply(lambda x:x['positioning'])
    df['demo'] = df.stats.apply(lambda x:x['demo'])
    df['mvp'] = df.mvp.apply(lambda x: 1 if x==True else 0)
    camera_keys = list(df.iloc[0]['camera'].keys())
    core_stats_keys = list(df.iloc[0]['stats']['core'].keys())
    boost_keys = list(df.iloc[0]['stats']['boost'].keys())
    movement_keys = list(df.iloc[0]['stats']['movement'].keys())
    positioning_keys = ['avg_distance_to_ball', 'avg_distance_to_ball_possession',
        'avg_distance_to_ball_no_possession', 'avg_distance_to_mates',
        'time_defensive_third', 'time_neutral_third', 'time_offensive_third',
        'time_defensive_half', 'time_offensive_half', 'time_behind_ball',
        'time_infront_ball', 'time_most_back', 'time_most_forward',
        'time_closest_to_ball', 'time_farthest_from_ball',
        'percent_defensive_third', 'percent_offensive_third',
        'percent_neutral_third', 'percent_defensive_half',
        'percent_offensive_half', 'percent_behind_ball', 'percent_infront_ball',
        'percent_most_back', 'percent_most_forward', 'percent_closest_to_ball',
        'percent_farthest_from_ball']
    demo_keys = list(df.iloc[0]['stats']['demo'].keys())
    df[camera_keys] = pd.json_normalize(df['camera'])
    df[core_stats_keys] = pd.json_normalize(df['core'])
    df[boost_keys] = pd.json_normalize(df['boost'])
    df[movement_keys] = pd.json_normalize(df['movement'])
    df[demo_keys] = pd.json_normalize(df['demo'])

    df['positioning'] = df.stats.apply(lambda x: remove_extra_element(x['positioning']))
    df[positioning_keys] = pd.json_normalize(df['positioning'])
    names = df['name']
    df = df.drop(['camera', 'stats', 'camera', 'core', 'boost', 'movement', 'positioning', 'demo', 'start_time', 'end_time', 'id', 'name', 'car_id', 'car_name'], axis=1)
    try:
        df.drop('rank', axis=1)
    except:
        pass
    return df, names

def get_stats_from_replays_combined(replay_ids):
    orange_stats = []
    blue_stats = []
    for replay_id in replay_ids:
        replay_stats = requests.get(f'https://ballchasing.com/api/replays/{replay_id}', headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
        test_stats = replay_stats.json()
        try:
            stats = [test_stats['orange']['players'][player] for player in range(3)]
            orange_stats.append(stats[0])
            orange_stats.append(stats[1])
            orange_stats.append(stats[2])
        except:
            pass
        try:
            stats = [test_stats['blue']['players'][player] for player in range(3)]
            blue_stats.append(stats[0])
            blue_stats.append(stats[1])
            blue_stats.append(stats[2])
        except:
            pass
    stats = blue_stats + orange_stats
    return stats


application.config['SECRET_KEY'] = 'you-will-never-guess'

@application.route('/')
@application.route('/index')
def index():
    return render_template('base.html', title='Home')

@application.route('/submit-replay', methods=['POST', 'GET'])
def submit_replay():
    form = SubmitReplay()
    return render_template('submit_replay.html', title='Submit Replay', form=form)

@application.route('/handle_data', methods=['GET', 'POST'])
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

@application.route('/results', methods=['GET'])
def results():
    return render_template('results.html', title='Home')

@application.route('/feature-importance', methods=['GET'])
def feature_importance():
    return render_template('feature_importance.html')

@application.route('/explanation', methods=['GET'])
def explanation():
    return render_template('explanation.html')

if __name__ == "__main__":
  application.run()