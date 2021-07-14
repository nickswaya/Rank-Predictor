def get_stats_from_replays(replay_ids):
    logger = logging.getLogger(__name__)
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = 'logging_info.log', level=logging.INFO, force=True, format=log_fmt)

    orange_ranks = []
    orange_stats = []
    blue_ranks = []
    blue_stats = []
    pull_num = 0
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
        pull_num += 1
        logger.info(f'stats extracted for pull num {pull_num}')
        
    return blue_stats, orange_stats

up_to_d1 = pd.read_csv('up_to_diamond1.csv')
d1uptochamp = pd.read_csv('diamond1_uptochamp1.csv')
champtogc = pd.read_csv('champ_to_gc.csv')


d1uptochampa = d1uptochamp.iloc[0]['0']
da = d1uptochampa.split(',')
da = [i.replace('"', '').replace('[', '').replace('""', '').replace('\'', '').replace(']', '') for i in da]
champtogca = champtogc.iloc[0]['0']
gca = champtogca.split(',')
gca = [i.replace('"', '').replace('[', '').replace('""', '').replace('\'', '').replace(']', '') for i in gca]
champtogcb = champtogc.iloc[1]['0']
gcb = champtogcb.split(',')
gcb = [i.replace('"', '').replace('[', '').replace('""', '').replace('\'', '').replace(']', '') for i in gcb]
up_to_d1_a = up_to_d1.iloc[0]['0']
a = up_to_d1_a.split(',')
a = [i.replace('"', '').replace('[', '').replace('""', '').replace('\'', '').replace(']', '') for i in a]
up_to_d1_b = up_to_d1.iloc[1]['0']
b = up_to_d1_b.split(',')
b = [i.replace('"', '').replace('[', '').replace('""', '').replace('\'', '').replace(']', '') for i in b]
up_to_d1_c = up_to_d1.iloc[2]['0']
c = up_to_d1_c.split(',')
c = [i.replace('"', '').replace('[', '').replace('""', '').replace('\'', '').replace(']', '') for i in c]
up_to_d1_d = up_to_d1.iloc[3]['0']
d = up_to_d1_d.split(',')
d = [i.replace('"', '').replace('[', '').replace('""', '').replace('\'', '').replace(']', '') for i in d]

res = []
res.append(a)
res.append(b)
res.append(c)
res.append(d)
res.append(da)
res.append(gca)
res.append(gcb)

flat_list = [item for sublist in res for item in sublist]
res1 = []
for i in flat_list:
    if i not in res1:
        res1.append(i)
res1 = [i.replace(' ', '') for i in res1]

blue_stats, orange_stats = get_stats_from_replays(res1)


