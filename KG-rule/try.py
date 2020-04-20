from scipy import stats
import random
import numpy as np

# # times = np.random.randint(1,2,1000) + np.random.randint(0,4,1000)
# # times = [1 for i in range(10)] + [2 for i in range(10)]
# times = np.random.randint(2,4,2000)
# times_1 = np.random.randint(2,4,1600)
# print(type(times))
# print(stats.ks_2samp(times, times_1))
# # print(stats.kstest(times, 'uniform'))
# # print(stats.uniform(loc=1, scale=1).cdf(6))

import json

with open('/media/becky/GNOME-p3/monopoly_game_schema_v1-2.json') as f:
  data = json.load(f)

remain_list = []
# for i in data['cards']['chance']['card_states']:
#     if i['name'] == 'go_to_jail' or i['name'] =='get_out_of_jail_free' or i['name'] =='pay_poor_tax':
#         remain_list.append(i)
# data['cards']['chance']['card_states'] = remain_list
# data['cards']['chance']['card_count'] = len(remain_list)
# with open('/media/becky/GNOME-p3/monopoly_game_schema_v1-2.json', 'w') as json_file:
#     json.dump(data, json_file)
print(data['cards'].keys())
print('Chance:', data['cards']['chance']['card_states'])
print('Community', data['cards']['community_chest']['card_states'])