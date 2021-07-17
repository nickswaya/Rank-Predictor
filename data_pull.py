import requests
import numpy as np
import pandas as pd
import os
from helper import *
import requests
import json
import pprint

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

created_before_list = create_url_dates()

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

# test_id_list = pull_game_ids(created_before_list)    
# test_id_series = pd.Series(test_id_list)
# test_id_series.to_csv('lower_ranked_ids.csv')

lower_rank_ids = pd.read_csv('lower_ranked_ids.csv', index_col=[0]).values
lower_rank_ids = [item for sublist in lower_rank_ids for item in sublist]

def get_stats_from_replays(replay_ids):
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = 'logging_info.log', level=logging.INFO, force=True, format=log_fmt)

    orange_ranks = []
    orange_stats = []
    blue_ranks = []
    blue_stats = []
    c = []
    pull_num = 0
    for replay_id in range(len(replay_ids)):
        logger.info(replay_ids[replay_id])
        logger.info(f'starting pull {pull_num}')
        url = f'https://ballchasing.com/api/replays/{replay_ids[replay_id]}'
        test_stats = requests.get(url, headers={'Authorization': 'gMXy4BUhXt0OQhc37kJV5KP0GUyLzhJeZhogpa94'}).json()
        c.append(test_stats)
        try:
            ranks = [test_stats['orange']['players'][player]['rank']['name'] for player in range(3)]
            orange_ranks.append(ranks)
            stats = [test_stats['orange']['players'][player] for player in range(3)]
            logger.info(stats)
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
        pull_num += 1
        logger.info(f'stats extracted for pull num {pull_num}')
    logger.info(len(blue_stats))    
    logger.info(len(orange_stats))    
    return blue_stats, orange_stats, c

b, o, c = get_stats_from_replays(lower_rank_ids)

with open('c.json', 'w') as f:
    json.dump(c, f)
with open('o.json', 'w') as f:
    json.dump(o, f)
with open('b.json', 'w') as f:
    json.dump(b, f)

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
with open('b.json') as f:
  b = json.load(f)
with open('o.json') as f:
  o = json.load(f)


b_test_df = create_frame_from_json(b, 'tier_div')
o_test_df = create_frame_from_json(o, 'tier_div')
o_test_df = o_test_df[b_test_df.columns]

b_test_df.to_csv('lower_rank_b_full.csv')
o_test_df.to_csv('lower_rank_o_full.csv')

