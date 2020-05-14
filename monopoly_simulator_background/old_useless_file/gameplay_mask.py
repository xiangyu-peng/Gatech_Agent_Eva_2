import initialize_game_elements
from action_choices import roll_die
import numpy as np
from simple_background_agent_becky_p1 import P1Agent
from simple_background_agent_becky_p2 import P2Agent
from gameplay_tf import *
# import simple_decision_agent_1
import json
import diagnostics
from interface import Interface
import sys, os
from card_utility_actions import move_player_after_die_roll

import xlsxwriter
import logging

from log_setting import set_log_level, ini_log_level

if __name__ == '__main__':
    # this is where everything begins. Assign decision agents to your players, set up the board and start simulating! You can
    # control any number of players you like, and assign the rest to the simple agent. We plan to release a more sophisticated
    # but still relatively simple agent soon.
    # file_path = '/media/becky/GNOME-p3/monopoly_simulator/gameplay.log'
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    ini_log_level()
    logger = set_log_level()

    logger.debug('tf_file')
    player_decision_agents = dict()
    num_active_players = 2

    player_decision_agents['player_1'] = P1Agent()
    player_decision_agents['player_2'] = P2Agent()

    game_elements = set_up_board('/media/becky/GNOME/monopoly_game_schema_v1-2.json',
                                 player_decision_agents, num_active_players)
    print('1')
    simulate_game_instance(game_elements, num_active_players, np_seed=5)

    ############
    logger.debug('tf_file')
    print('222222222222222222222')
    os.remove('/media/becky/GNOME-p3/monopoly_simulator/gameplay.log')
    # ini_log_level()
    logger = set_log_level()
    player_decision_agents = dict()
    num_active_players = 2

    player_decision_agents['player_1'] = P1Agent()
    player_decision_agents['player_2'] = P2Agent()
    game_elements = set_up_board('/media/becky/GNOME/monopoly_game_schema_v1-2.json',
                                 player_decision_agents, num_active_players)
    simulate_game_instance(game_elements, num_active_players, np_seed=1)
    ########

    #just testing history.
    # print len(game_elements['history']['function'])
    # print len(game_elements['history']['param'])
    # print len(game_elements['history']['return'])