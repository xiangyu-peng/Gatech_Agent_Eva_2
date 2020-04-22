#ignore this file!

from scipy import stats
import random
import numpy as np
import os
from configparser import ConfigParser



def params_read(config_data, keys):
    """
    Read config.ini file
    :param config_data:
    :param keys (string): sections in config file
    :return: a dict with info in config file
    """
    params = {}
    for key in config_data[keys]:
        v = eval(config_data[keys][key])
        params[key] = v
        print(v)
    return params

config_file = '/media/becky/GNOME-p3/monopoly_simulator/config.ini'
config_data = ConfigParser()
config_data.read(config_file)
params = params_read(config_data, keys='kg')


# times = np.random.randint(1,2,1000) + np.random.randint(0,4,1000)
# times = [1 for i in range(10)] + [2 for i in range(10)]
# p = 1
# total = 2000
# rate = 0
# while p > 0.001:
#   rate += 1
#   # print(rate)
#   times = np.random.randint(0,7,total)
#
#   i = 0
#   p_l = []
#   while i < 3:
#     i += 1
#     times_1 = np.random.randint(0,7,total)
#     # print(max(times_1))
#   # print(type(times))
#   # print(stats.ks_2samp(times, times_1))
#     p = stats.ks_2samp(times, times_1).pvalue
#     p_l.append(p)
#   p = max(p_l)
# print(rate)
# print(p)

# print(stats.kstest(times, 'uniform'))
# print(stats.uniform(loc=1, scale=1).cdf(6))

# import json
#
# with open('/media/becky/GNOME-p3/monopoly_game_schema_v1-2.json') as f:
#   data = json.load(f)
#
# remain_list = []
# # for i in data['cards']['chance']['card_states']:
# #     if i['name'] == 'go_to_jail' or i['name'] =='get_out_of_jail_free' or i['name'] =='pay_poor_tax':
# #         remain_list.append(i)
# # data['cards']['chance']['card_states'] = remain_list
# # data['cards']['chance']['card_count'] = len(remain_list)
# # with open('/media/becky/GNOME-p3/monopoly_game_schema_v1-2.json', 'w') as json_file:
# #     json.dump(data, json_file)
# print(data['cards'].keys())
# print('Chance:', data['cards']['chance']['card_states'])
# print('Community', data['cards']['community_chest']['card_states'])