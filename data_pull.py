import requests
import numpy as np
import pandas as pd
import os
from helper import *
import requests
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


def pull_data(created_before_list, playlist='ranked-standard',min_rank='bronze-1', max_rank='champion-3'):
    for pull in range(1):
    # for pull in range(len(created_before_list)):
        url =  f'https://ballchasing.com/api/replays?playlist={playlist}&min-rank={min_rank}&max-rank={max_rank}&created-before={created_before_list[pull]}&sort-by=created&sort-dir=desc&count=2'
        r = requests.get(url, headers=headers).json()
        return url

dates = create_url_dates()

example_json = pull_data(dates)

pd.
