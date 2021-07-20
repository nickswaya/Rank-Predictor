import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import pandas as pd
import os
from helper import *
import requests
import json
import pprint
import matplotlib.pyplot as plt
import time
plt.style.use('ggplot')


headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'}

def create_url_dates():
    date_list = []
    month_idx = 0
    day_idx = 0
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
    while True:
        url = f'2020-{months[month_idx]}-{days[day_idx]}T15:00:05%2b01:00'
        date_list.append(url)
        day_idx += 1
        if day_idx >= 26:
            day_idx = 0
            month_idx += 1
            if month_idx == 12:
                return date_list

def pull_game_ids(created_before_list, playlist='ranked-standard',min_rank='bronze-1', max_rank='diamond-3'):
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = 'logging_info.log', level=logging.INFO, force=True, format=log_fmt)

    game_ids = []
    # for pull in range(1):
    for pull in range(len(created_before_list)):
        to_append = []
        url =  f'https://ballchasing.com/api/replays?playlist={playlist}&min-rank={min_rank}&max-rank={max_rank}&created-before={created_before_list[pull]}&sort-by=created&sort-dir=desc&count=200'
        logger.info(created_before_list[pull])
        r = requests.get(url, headers=headers).json()
        try:
            for game_id in range(len(r['list'])):
                gid = r['list'][game_id]['id']
                to_append.append(gid)
            game_ids.append(to_append)
        except:
            pass
    flat_list = [item for sublist in game_ids for item in sublist]
    flat_list_res = []
    for i in flat_list:
        if i not in flat_list_res:
            flat_list_res.append(i)
    return flat_list_res

# created_before_list = create_url_dates()
# test_id_list = pull_game_ids(created_before_list)    
# test_id_series = pd.Series(test_id_list)
# test_id_series.to_csv('lower_ranked_ids.csv')

lower_rank_ids = pd.read_csv('lower_ranked_ids.csv', index_col=[0]).values
lower_rank_ids = [item for sublist in lower_rank_ids for item in sublist][36600:]

def get_stats_from_replays(replay_ids):
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = 'logging_info.log', level=logging.INFO, force=True, format=log_fmt)

    orange_ranks = []
    orange_stats = []
    blue_ranks = []
    blue_stats = []
    pull_num = 36600
    for replay_id in range(len(replay_ids)):
        if pull_num % 10 == 0:
            logger.info(f'Starting Pull {pull_num}')
        url = f'https://ballchasing.com/api/replays/{replay_ids[replay_id]}'
        test_stats = requests.get(url, headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'}).json()
        try:
            ranks = [test_stats['orange']['players'][player]['rank']['name'] for player in range(3)]
            orange_ranks.append(ranks)
            stats = [test_stats['orange']['players'][player] for player in range(3)]
            logger.info(stats[0])
            logger.info(stats[1])
            logger.info(stats[2])
            orange_stats.append(stats[0])
            orange_stats.append(stats[1])
            orange_stats.append(stats[2])
        except:
            pass
        try:
            ranks = [test_stats['blue']['players'][player]['rank']['name'] for player in range(3)]
            blue_ranks.append(ranks)
            stats = [test_stats['blue']['players'][player] for player in range(3)]
            logger.info(stats[0])
            logger.info(stats[1])
            logger.info(stats[2])
            blue_stats.append(stats[0])
            blue_stats.append(stats[1])
            blue_stats.append(stats[2])
        except:
            pass
        pull_num += 1
        if pull_num % 100 == 0:
            with open('o4.json', 'w') as f:
                json.dump(orange_stats, f)
            with open('b4.json', 'w') as f:
                json.dump(blue_stats, f) 
    return blue_stats, orange_stats

# b, o = get_stats_from_replays(lower_rank_ids)


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
    df['mvp'] = df.mvp.apply(lambda x: 1 if x==True else 0)
    df['positioning'] = df.stats.apply(lambda x: remove_extra_element(x['positioning']))
    df[positioning_keys] = pd.json_normalize(df['positioning'])
    if rank_target == 'tier_div':
        df['rank_target'] = df['tier'] * 4 + df['division']
        df = df.drop(['camera', 'stats', 'camera', 'rank', 'core', 'boost', 'movement', 'positioning', 'demo', 'start_time', 'end_time', 'name', 'id', 'car_id', 'tier', 'division', 'car_name'], axis=1)
        return df
    if rank_target == 'tier':
        df['rank_target'] = df['id'] 
        df = df.drop(['camera', 'stats', 'camera', 'rank', 'core', 'boost', 'movement', 'positioning', 'demo', 'start_time', 'end_time', 'name', 'id', 'car_id', 'tier', 'division', 'car_name'], axis=1)
        return df

# with open('b.json') as f:
#   b1 = json.load(f)
# with open('b2.json') as f:
#   b2 = json.load(f)
# with open('b3.json') as f:
#   b3 = json.load(f)
# with open('o.json') as f:
#   o1 = json.load(f)
# with open('o2.json') as f:
#   o2 = json.load(f)  
# with open('o3.json') as f:
#   o3 = json.load(f)

# dfb1 = create_frame_from_json(b1, 'tier_div')
# dfb2 = create_frame_from_json(b2, 'tier_div')
# dfb3 = create_frame_from_json(b3, 'tier_div')

# dfo1 = create_frame_from_json(o1, 'tier_div')
# dfo2 = create_frame_from_json(o2, 'tier_div')
# dfo3 = create_frame_from_json(o3, 'tier_div')

# cols_to_use = dfb1.columns

# dfb2 = dfb2[cols_to_use]
# dfb3 = dfb3[cols_to_use]
# dfo1 = dfo1[cols_to_use]
# dfo2 = dfo2[cols_to_use]
# dfo3 = dfo3[cols_to_use]

# test_combine = dfo1.append(dfo2).append(dfo3).append(dfb1).append(dfb2).append(dfb3)
# test_combine.to_csv('lower_rank_to_36_000.csv')


lower_rank = pd.read_csv('lower_rank_to_36_000.csv', index_col=0)
upper_ranks = pd.read_csv('final_full_data.csv', index_col=0)
combined = lower_rank.append(upper_ranks)

# fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(15,10))
# axs = axs.flatten()
# axs[0].hist(lower_rank.rank_target)
# axs[0].set_title('lower rank dist')
# axs[1].hist(upper_ranks.rank_target)
# axs[1].set_title('upper rank dist')
# axs[2].hist(combined.rank_target)
# axs[2].set_title('combined rank dist')
# plt.show()

combined = combined.dropna()
X = combined.iloc[:,:91].values
y = combined['rank_target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


mean_train = np.mean(y_train)
baseline_predictions = np.ones(y_test.shape) * mean_train
mae_baseline = mean_absolute_error(y_test, baseline_predictions)

params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'reg:squarederror',
}

params['gpu_id'] = 0
params['tree_method'] = 'gpu_hist'
params['eval_metric'] = "mae"
num_boost_round = 8000

start_time = time.time()
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)

print(cv_results)
print(cv_results['test-mae-mean'].min())


# gridsearch_params = [
#     (max_depth, min_child_weight)
#     for max_depth in range(2,3)
#     for min_child_weight in range(1,3)
# ]
# # Define initial best params and MAE
# min_mae = float("Inf")
# best_params = None
# for max_depth, min_child_weight in gridsearch_params:
#     print("CV with max_depth={}, min_child_weight={}".format(
#                              max_depth,
#                              min_child_weight))
#     # Update our parameters
#     params['max_depth'] = max_depth
#     params['min_child_weight'] = min_child_weight
#     # Run CV
#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=5,
#         metrics={'mae'},
#         early_stopping_rounds=10
#     )
#     # Update best MAE
#     mean_mae = cv_results['test-mae-mean'].min()
#     boost_rounds = cv_results['test-mae-mean'].argmin()
#     print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
#     if mean_mae < min_mae:
#         min_mae = mean_mae
#         best_params = (max_depth,min_child_weight)
# print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

params['max_depth'] = 3
params['min_child_weight'] = 2


# min_mae = float("Inf")
# best_params = None
# for eta in [.3, .2, .1, .05, .01, .005]:
#     print("CV with eta={}".format(eta))    # We update our parameters
#     params['eta'] = eta
#     cv_results = xgb.cv(
#             params,
#             dtrain,
#             num_boost_round=num_boost_round,
#             seed=42,
#             nfold=5,
#             metrics=['mae'],
#             early_stopping_rounds=10)
#     # Update best score
#     mean_mae = cv_results['test-mae-mean'].min()
#     boost_rounds = cv_results['test-mae-mean'].argmin()
#     print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
#     if mean_mae < min_mae:
#         min_mae = mean_mae
#         best_params = eta

# print("Best params: {}, MAE: {}".format(best_params, min_mae))

params['eta'] = .2
params['num_estimators'] = 1000

