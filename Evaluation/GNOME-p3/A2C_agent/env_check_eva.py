#####Evaluation#####
import sys, os
upper_path = os.path.abspath('..').replace('/Evaluation/GNOME-p3','')
upper_path_eva = upper_path + '/Evaluation/GNOME-p3'
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation/GNOME-p3')
#####################################

from monopoly_simulator_background.vanilla_A2C import *
from configparser import ConfigParser
import os, sys
import time
from monopoly_simulator_background.simple_background_agent_becky_p1 import P1Agent
# from monopoly_simulator_background.background_agent_v2_agent import P2Agent_v2
from monopoly_simulator import background_agent_v3
from monopoly_simulator.agent import Agent
from monopoly_simulator_background.gameplay_tf import *
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# n = 8
# sum_win = 0
# while n < 9:
#     done_num = 0
#     env = gym.make('monopoly_simple-v1')
#     env.seed(seed=n)
#     with HiddenPrints():
#         s,mask = env.reset()
#     done = 0
#     while done_num < 3:
#         # s, rew, done, info = env.step_nochange(0)
#         # s, rew, done, info = env.step_after_nochange(0)
#
#         with HiddenPrints():
#             s, rew, done, info = env.step_nochange(0)
#         # print('s', s, rew)
#         # print('info',info)
#         # break
#         with HiddenPrints():
#             s_later, rew_later, _, info = env.step_after_nochange(0)
#         # print('s =>', s)
#         # print('s_later', s_later)
#         if done_num ==1:
#             done_num+=1
#             print('s =>',s,rew)
#             print('s_later', s_later,rew_later)
#             print('info=>',info)
#             break
#         if done:
#             done_num += 1
#             print(s_later)
#
#
#     n += 1
#     sum_win += done - 1
# print(sum_win)
start = time.time()
n = 1
env = gym.make('monopoly_simple-v1')
# print(env.MonopolyWorld.upper_path)

####set up board####
player_decision_agents = dict()
player_decision_agents['player_1'] = None

# Assign the beckground agents to the players
name_num = 1
num_player = 4
while name_num < num_player:
    name_num += 1
    player_decision_agents['player_' + str(name_num)] = None

gameboard_initial = set_up_board('/media/becky/GNOME-p3/monopoly_game_schema_v1-1.json',
                                      player_decision_agents)
############

env.set_board(gameboard_initial)
env.set_kg(False)
# env.set_board()
env.seed(seed=n)
# with HiddenPrints():
s,mask = env.reset()
print(s,mask)
done_num = 0
done_num_total = 5000
# i = 0
# while i < 30000:
#     i += 1
#     s, rew, done_after, info = env.step_after_nochange(0)
s_old = [1,2,2]
while done_num < done_num_total:

    with HiddenPrints():
        s, rew, done, info = env.step_nochange(1)
    print('s-nochange',s, rew, info)

    if done > 0:
        # print('Done')
        done_num += 1
        print('s', s, rew)

    # if done_num == done_num_total:
    #     break


    # done_after = False
    # while done_after == False:
    #     with HiddenPrints():
    #         s, rew, done_after, info = env.step_hyp(0)
    #     print('s-hyp0', s, rew, info, done_after)

        # with HiddenPrints():
        #     s, rew, done, info = env.step_nochange(0)
        # print('s-nochange', s, rew, info)

        # with HiddenPrints():
        #     s, rew, done_after, info = env.step_hyp(0)
        # print('s-hyp0', s, rew, info, done_after)
        #
        # with HiddenPrints():
        #     s, rew, done_after, info = env.step_hyp(1)
        # print('s-hyp1', s, rew, info, done_after)

    with HiddenPrints():
        s, rew, done_after, info = env.step_after_nochange(1)
    print('s_after', s, rew, info)
    if s_old[-1] <= 0 and s_old[-2] <= 0 and s_old[-3] <= 0 and s[-1] <= 0 and s[-2] <= 0 and s[-3] <= 0:
        break

    s_old = s
    print('===========')

    done_num += 1
    # print('s',s,rew)
end = time.time()
# print('Run %d environments need %f' % (done_num_total, float(str(end-start))))
