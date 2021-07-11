import pandas as pd


def remove_extra_element(row):
    if 'goals_against_while_last_defender' in row:
        del row['goals_against_while_last_defender']
    return row

def create_frame_from_json(json_stats):
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

    df = df.drop(['camera', 'stats', 'camera', 'rank', 'core', 'boost', 'movement', 'positioning', 'demo', 'start_time', 'end_time'], axis=1)
    return df