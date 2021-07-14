import pandas as pd
import math
import logging
import requests
import requests
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
import joblib
from sklearn.metrics import mean_absolute_error
rcParams['figure.figsize'] = (12, 8)
plt.style.use('ggplot')

def remove_extra_element(row):
    if 'goals_against_while_last_defender' in row:
        del row['goals_against_while_last_defender']
    return row

def create_frame_from_json(json_stats, rank_target):
    df = pd.DataFrame.from_dict(json_stats)

    df['core'] = df.stats.apply(lambda x:x['core'])
    df['boost'] = df.stats.apply(lambda x:x['boost'])
    df['movement'] = df.stats.apply(lambda x:x['movement'])
    df['positioning'] = df.stats.apply(lambda x:x['positioning'])
    df['demo'] = df.stats.apply(lambda x:x['demo'])
    

    rank_keys = list(df.iloc[0]['rank'].keys())
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

    df[rank_keys] = pd.json_normalize(df['rank'])
    df[camera_keys] = pd.json_normalize(df['camera'])
    df[core_stats_keys] = pd.json_normalize(df['core'])
    df[boost_keys] = pd.json_normalize(df['boost'])
    df[movement_keys] = pd.json_normalize(df['movement'])
    df[demo_keys] = pd.json_normalize(df['demo'])

    df['positioning'] = df.stats.apply(lambda x: remove_extra_element(x['positioning']))
    df[positioning_keys] = pd.json_normalize(df['positioning'])
    if rank_target == 'tier_div':
        df['rank_target'] = df['tier'] * 4 + df['division']
        df = df.drop(['camera', 'stats', 'camera', 'rank', 'core', 'boost', 'movement', 'positioning', 'demo', 'start_time', 'end_time', 'name', 'id', 'car_id', 'tier', 'division'], axis=1)
        return df
    if rank_target == 'tier':
        df['rank_target'] = df['id'] 
        df = df.drop(['camera', 'stats', 'camera', 'rank', 'core', 'boost', 'movement', 'positioning', 'demo', 'start_time', 'end_time', 'name', 'id', 'car_id', 'tier', 'division'], axis=1)
        return df

def get_all_ids():
    season_lst = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','f1', 'f2', 'f3', 'f4']
    replay_ids = []
    for i in range(len(season_lst)):
        r = requests.get(f'https://ballchasing.com/api/replays?player-name=LebaneseNinja&player-id=steam:76561198068157523&count=200&playlist=ranked-standard&season={season_lst[i]}', headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
        d = r.json()
        rank_ids = []
        for i in range(len(d['list'])):
            rank_ids.append(d['list'][i]['id'])
        replay_ids.append(rank_ids)
    flat_list = [item for sublist in replay_ids for item in sublist]
    return flat_list

def get_stats_from_replays(replay_ids):
    orange_ranks = []
    orange_stats = []
    blue_ranks = []
    blue_stats = []
    for replay_id in replay_ids:
        replay_stats = requests.get(f'https://ballchasing.com/api/replays/{replay_id}', headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
        test_stats = replay_stats.json()
        try:
            ranks = [test_stats['orange']['players'][player]['rank']['name'] for player in range(3)]
            orange_ranks.append(ranks)
            stats = [test_stats['orange']['players'][player] for player in range(3)]
            orange_stats.append(stats[0])
            orange_stats.append(stats[1])
            orange_stats.append(stats[2])
        except:
            pass
        try:
            ranks = [test_stats['blue']['players'][player]['rank']['name'] for player in range(3)]
            blue_ranks.append(ranks)
            stats = [test_stats['blue']['players'][player] for player in range(3)]
            blue_stats.append(stats[0])
            blue_stats.append(stats[1])
            blue_stats.append(stats[2])
        except:
            pass
        
    return blue_stats, orange_stats

def combine_clean_dfs(blue_df, orange_df, car_dummies):
    combined_dfs = blue_df.append(orange_df)
    if car_dummies == False:
        combined_dfs['mvp'] = combined_dfs.mvp.apply(lambda x: 1 if x==True else 0)
        combined_dfs = combined_dfs.drop('car_name', axis=1)
        return combined_dfs
    if car_dummies == True:
        dummies = pd.get_dummies(combined_dfs.car_name, drop_first=True)
        combined_concat_w_dummies = pd.concat([combined_dfs, dummies], axis=1)
        combined_concat_w_dummies['mvp'] = combined_concat_w_dummies.mvp.apply(lambda x: 1 if x==True else 0)
        combined_concat_w_dummies = combined_concat_w_dummies.drop('car_name', axis=1)
        return combined_concat_w_dummies


def get_random_ids():
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = 'logging_info.log', level=logging.INFO, force=True, format=log_fmt)
    logger.info('starting pull')
    minmax_rank_lst = [['bronze-1', 'bronze-3'], ['silver-1', 'silver-3'], ['gold-1', 'gold-3'], ['platinum-1', 'platinum-3'], ['diamond-1', 'diamond-3'], ['champion-1', 'champion-3'], ['champion-3', 'grand-champion']]
    replay_ids = []
    ids_pulled = []
    for i in range(len(minmax_rank_lst)):
        logger.info(f'pulling min_rank={minmax_rank_lst[i][0]}, max_rank = {minmax_rank_lst[i][1]}')
        r = requests.get(f'https://ballchasing.com/api/replays?&count=200&playlist=ranked-standard&min-rank={minmax_rank_lst[i][0]}&max-rank={minmax_rank_lst[i][1]}', headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
        d = r.json()
        rank_ids = []
        count = d['count']
        logger.info(f'count = {count}')
        for i in range(len(d['list'])):
            rank_ids.append(d['list'][i]['id'])
        next_link = d['next']
        if i == 6 or i == 5:
            logger.info('i == 5 or 6, moving on')
            replay_ids.append(rank_ids)
            continue 

        logger.info('starting second pull')
        r = requests.get(next_link, headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
        d = r.json()
        for i in range(len(d['list'])):
            rank_ids.append(d['list'][i]['id'])
        next_link = d['next']

        logger.info('starting third pull')
        r = requests.get(next_link, headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
        d = r.json()
        for i in range(len(d['list'])):
            rank_ids.append(d['list'][i]['id'])
        logger.info('rank pull complete.. appending results and moving to next rank') 
        replay_ids.append(rank_ids)

    flat_list = [item for sublist in replay_ids for item in sublist]
    return flat_list

def test_all():
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = 'logging_info.log', level=logging.INFO, force=True, format=log_fmt)
    logger.info('starting pull')
    minmax_rank_lst = [['bronze-1', 'bronze-3'], ['silver-1', 'silver-3'], ['gold-1', 'gold-3'], ['platinum-1', 'platinum-3'], ['diamond-1', 'diamond-3'], ['champion-1', 'champion-3'], ['champion-3', 'grand-champion']]
    replay_ids = []
    ids_pulled = []
    for i in range(len(minmax_rank_lst)):
        logger.info(f'pulling min_rank={minmax_rank_lst[i][0]}, max_rank = {minmax_rank_lst[i][1]}')
        r = requests.get(f'https://ballchasing.com/api/replays?&count=200&playlist=ranked-standard&min-rank={minmax_rank_lst[i][0]}&max-rank={minmax_rank_lst[i][1]}', headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
        d = r.json()
        rank_ids = []
        count = d['count']
        next_link = d['next']
        logger.info(f'count = {count}')
        for pull in range(math.floor(count/200)-2):
            logger.info(f'starting pull number {pull+1}')
            r = requests.get(next_link, headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'})
            d = r.json()
            for i in range(len(d['list'])):
                rank_ids.append(d['list'][i]['id'])
            next_link = d['next']
        logger.info('rank pull complete.. appending results and moving to next rank') 
        replay_ids.append(rank_ids)

    flat_list = [item for sublist in replay_ids for item in sublist]
    return flat_list


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