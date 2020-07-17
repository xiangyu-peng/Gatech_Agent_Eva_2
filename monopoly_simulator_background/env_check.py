#####Evaluation#####
import sys, os
upper_path = os.path.abspath('..')
sys.path.append(upper_path + '/KG_rule')
sys.path.append(upper_path)
sys.path.append(upper_path + '/Evaluation/GNOME-p3')
####################
from monopoly_simulator_background.vanilla_A2C import *
from configparser import ConfigParser
import os, sys
import time
from monopoly_simulator_background.simple_background_agent_becky_p1 import P1Agent
from monopoly_simulator import background_agent_v3
from monopoly_simulator_background.gameplay_tf import *
from monopoly_simulator.agent import Agent
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

i = 40
while i < 50:
    i += 1
    env = gym.make('monopoly_simple-v1')
    env.set_config_file('/monopoly_simulator_background/config.ini')

    exp_dict = dict()
    exp_dict['novelty_num'] = (30,1)
    exp_dict['novelty_inject_num'] = 0
    exp_dict['exp_type'] = 'kg'
    env.set_exp(exp_dict)
    env.set_kg(True)

    env.set_board()
    env.seed(seed=i)

    # with HiddenPrints():
    s,mask = env.reset()
    # print('reset',s)
    done_num = 0
    done_num_total = 10
    # i = 0
    # while i < 30000:
    #     i += 1
    #     s, rew, done_after, info = env.step_after_nochange(0)

    bool = False
    num = 0
    while done_num < done_num_total:
        num += 1

        # with HiddenPrints():
        #     s, rew, done, info = env.step_nochange(1)
        # print('s-nochange',s, rew, info, len(s))
        #
        # with HiddenPrints():
        #     s, rew, done, info = env.step_nochange(0)
        # print('s-nochange', s, rew, info, len(s))

        # print('money', s.tolist()[-12:-6].index(1), s.tolist()[-12:-6])
        # loc = s.tolist()[-52:-12].index(1)
        # print('loc', loc)
        # print('who owned?',s.tolist()[loc])
        # print('=====')

        # with HiddenPrints():
        s, rew, done, info = env.step(1)
        # print('reward =>', rew)
        # loc = s.tolist()[-52:-12].index(1)
        # print('s-step', rew, 'loc =>', loc)
        # print('s',s)



        if bool:
            # print('reset', s.tolist().index(1))

            bool = False


            # print('reset', s, len(s), len(s))

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

        # with HiddenPrints():
        #     s, rew, done_after, info = env.step_after_nochange(0)
        #
        # print(info)


        if done > 0:
            done_num += 1
            bool = True
            # print('===========')
            # print('done', s)

        # print('s',s,rew)
    end = time.time()
    # print('Run %d environments need %f' % (done_num_total, float(str(end-start))))
